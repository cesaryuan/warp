//! Streaming response handling for the local OpenAI-compatible Responses backend.

use anyhow::{anyhow, Context as _};
use serde_json::{json, Value};
use uuid::Uuid;
use warp_multi_agent_api as api;

use crate::ai::agent::task::TaskId;

use super::request::{assistant_output_item, function_call_history_item};
use super::tool_calls::parse_tool_call;
use super::types::{
    ParsedFunctionCall, ResponsesApiResponse, ResponsesCompletedEvent,
    ResponsesFunctionCallArgumentsDeltaEvent, ResponsesFunctionCallArgumentsDoneEvent,
    ResponsesOutputItem, ResponsesOutputItemDoneEvent, ResponsesTextDeltaEvent,
    StreamMessageResult, StreamingFunctionCallState, StreamingResponsesAccumulator,
    StreamingTextMessageState,
};
use super::{
    add_messages_event, conversation_state_store, finished_reason_for_error, stream_finished_event,
    user_visible_error_event, Event, RequestParams,
};

/// Parses the assistant output returned by the Responses API into text and tool calls.
pub(super) fn parse_responses_output(
    output: Vec<ResponsesOutputItem>,
) -> anyhow::Result<(Vec<String>, Vec<ParsedFunctionCall>)> {
    let mut assistant_messages = Vec::new();
    let mut function_calls = Vec::new();

    for item in output {
        match item.item_type.as_str() {
            "message" if item.role.as_deref() == Some("assistant") => {
                let text = item
                    .content
                    .into_iter()
                    .filter(|content| matches!(content.item_type.as_str(), "output_text" | "text"))
                    .filter_map(|content| content.text)
                    .collect::<Vec<_>>()
                    .join("");
                if !text.is_empty() {
                    assistant_messages.push(text);
                }
            }
            "function_call" => {
                let name = item.name.context("Missing function call name")?;
                let call_id = item
                    .call_id
                    .or(item.id)
                    .unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple()));
                let arguments = item
                    .arguments
                    .as_deref()
                    .map(serde_json::from_str)
                    .transpose()
                    .context("Failed to parse function call arguments")?
                    .unwrap_or_else(|| json!({}));
                function_calls.push(ParsedFunctionCall {
                    name,
                    call_id,
                    arguments,
                });
            }
            _ => {}
        }
    }

    if assistant_messages.is_empty() && function_calls.is_empty() {
        return Err(anyhow!(
            "Local OpenAI Responses output contained no assistant messages or tool calls"
        ));
    }

    Ok((assistant_messages, function_calls))
}

/// Translates a streamed Responses SSE message into Warp client events and updates stream state.
pub(super) fn handle_responses_stream_message(
    params: &RequestParams,
    task_id: &TaskId,
    request_id: &str,
    event_name: &str,
    data: &str,
    accumulator: &mut StreamingResponsesAccumulator,
) -> anyhow::Result<StreamMessageResult> {
    let payload = serde_json::from_str::<Value>(data)
        .with_context(|| format!("Failed to decode streamed Responses event payload: {data}"))?;
    let event_type = streamed_event_type(event_name, &payload);

    match event_type.as_str() {
        "response.output_text.delta" | "response.text.delta" => {
            let delta_event: ResponsesTextDeltaEvent = serde_json::from_value(payload)?;
            let Some(event) =
                handle_streamed_text_delta(task_id, request_id, accumulator, delta_event)?
            else {
                return Ok(StreamMessageResult::default());
            };

            Ok(StreamMessageResult {
                events: vec![Ok(event)],
                is_terminal: false,
            })
        }
        "response.function_call_arguments.delta" => {
            let delta_event: ResponsesFunctionCallArgumentsDeltaEvent =
                serde_json::from_value(payload)?;
            let function_call_id = streamed_function_call_id(
                delta_event.call_id.as_deref(),
                delta_event.item_id.as_deref(),
            )?;
            accumulator
                .function_calls_by_call_id
                .entry(function_call_id)
                .or_default()
                .arguments
                .push_str(&delta_event.delta);
            Ok(StreamMessageResult::default())
        }
        "response.function_call_arguments.done" => {
            let done_event: ResponsesFunctionCallArgumentsDoneEvent =
                serde_json::from_value(payload)?;
            let Some(function_call) = finalize_streamed_function_call(accumulator, done_event)?
            else {
                return Ok(StreamMessageResult::default());
            };
            Ok(StreamMessageResult {
                events: vec![Ok(add_messages_event(
                    task_id,
                    vec![tool_call_message(task_id, request_id, function_call)?],
                ))],
                is_terminal: false,
            })
        }
        "response.output_item.done" => {
            let done_event: ResponsesOutputItemDoneEvent = serde_json::from_value(payload)?;
            let Some(function_call) =
                handle_streamed_output_item_done(accumulator, done_event.item)?
            else {
                return Ok(StreamMessageResult::default());
            };

            Ok(StreamMessageResult {
                events: vec![Ok(add_messages_event(
                    task_id,
                    vec![tool_call_message(task_id, request_id, function_call)?],
                ))],
                is_terminal: false,
            })
        }
        "response.completed" => {
            let completed_event: ResponsesCompletedEvent = serde_json::from_value(payload)?;
            Ok(StreamMessageResult {
                events: finalize_stream_state(
                    params,
                    std::mem::take(accumulator),
                    request_id,
                    Some(completed_event.response),
                )?,
                is_terminal: true,
            })
        }
        "response.failed" | "error" => {
            let message = streamed_error_message(&payload);
            let error = anyhow!("Local OpenAI Responses stream failed: {message}");
            Ok(StreamMessageResult {
                events: vec![
                    Ok(user_visible_error_event(
                        task_id,
                        request_id,
                        &error.to_string(),
                    )),
                    Ok(stream_finished_event(finished_reason_for_error(
                        &error,
                        params.model.to_string(),
                    ))),
                ],
                is_terminal: true,
            })
        }
        _ => Ok(StreamMessageResult::default()),
    }
}

/// Creates or appends to a streamed assistant text message based on a Responses text delta.
fn handle_streamed_text_delta(
    task_id: &TaskId,
    request_id: &str,
    accumulator: &mut StreamingResponsesAccumulator,
    delta_event: ResponsesTextDeltaEvent,
) -> anyhow::Result<Option<api::ResponseEvent>> {
    if delta_event.delta.is_empty() {
        return Ok(None);
    }

    if let Some(existing_message) = accumulator
        .text_messages_by_item_id
        .get_mut(&delta_event.item_id)
    {
        existing_message.text.push_str(&delta_event.delta);
        return Ok(Some(update_agent_output_text_event(
            task_id,
            request_id,
            &existing_message.message_id,
            existing_message.text.clone(),
        )));
    }

    let message_id = Uuid::new_v4().to_string();
    accumulator
        .emitted_text_item_ids
        .push(delta_event.item_id.clone());
    accumulator.text_messages_by_item_id.insert(
        delta_event.item_id,
        StreamingTextMessageState {
            message_id: message_id.clone(),
            text: delta_event.delta.clone(),
        },
    );

    Ok(Some(add_messages_event(
        task_id,
        vec![agent_output_message_with_id(
            message_id,
            task_id,
            request_id,
            delta_event.delta,
        )],
    )))
}

/// Finalizes a streamed function call and returns the parsed Warp tool call payload.
pub(super) fn finalize_streamed_function_call(
    accumulator: &mut StreamingResponsesAccumulator,
    done_event: ResponsesFunctionCallArgumentsDoneEvent,
) -> anyhow::Result<Option<ParsedFunctionCall>> {
    let function_call_id = reconcile_streamed_function_call_state_key(
        accumulator,
        done_event.call_id.as_deref(),
        done_event.item_id.as_deref(),
    )?;
    let state = accumulator
        .function_calls_by_call_id
        .entry(function_call_id.clone())
        .or_default();
    state.provider_call_id = done_event
        .call_id
        .clone()
        .or(state.provider_call_id.clone());
    state.output_item_id = done_event.item_id.clone().or(state.output_item_id.clone());
    state.name = done_event.name.clone().or(state.name.clone());
    let final_arguments = if done_event.arguments.is_empty() {
        state.arguments.clone()
    } else {
        done_event.arguments.clone()
    };
    state.arguments = final_arguments.clone();

    maybe_emit_streamed_function_call(accumulator, &function_call_id, false)
}

/// Uses `response.output_item.done` to enrich streamed function call metadata with authoritative IDs.
pub(super) fn handle_streamed_output_item_done(
    accumulator: &mut StreamingResponsesAccumulator,
    item: ResponsesOutputItem,
) -> anyhow::Result<Option<ParsedFunctionCall>> {
    if item.item_type != "function_call" {
        return Ok(None);
    }

    let function_call_id = reconcile_streamed_function_call_state_key(
        accumulator,
        item.call_id.as_deref(),
        item.id.as_deref(),
    )?;
    let state = accumulator
        .function_calls_by_call_id
        .entry(function_call_id.clone())
        .or_default();
    state.provider_call_id = item.call_id.clone().or(state.provider_call_id.clone());
    state.output_item_id = item.id.clone().or(state.output_item_id.clone());
    state.name = item.name.clone().or(state.name.clone());
    if state.arguments.is_empty() {
        state.arguments = item.arguments.clone().unwrap_or_default();
    }

    maybe_emit_streamed_function_call(accumulator, &function_call_id, true)
}

/// Emits a completed streamed function call once its metadata is sufficiently populated.
fn maybe_emit_streamed_function_call(
    accumulator: &mut StreamingResponsesAccumulator,
    function_call_id: &str,
    allow_item_id_fallback: bool,
) -> anyhow::Result<Option<ParsedFunctionCall>> {
    let state = accumulator
        .function_calls_by_call_id
        .get_mut(function_call_id)
        .ok_or_else(|| anyhow!("Missing streamed function call state for id {function_call_id}"))?;
    if state.emitted {
        return Ok(None);
    }

    let Some(name) = state.name.clone() else {
        log::debug!(
            "Deferring streamed function call emission until name is available for id {}",
            function_call_id
        );
        return Ok(None);
    };
    let Some(canonical_call_id) = state.provider_call_id.clone().or_else(|| {
        allow_item_id_fallback
            .then(|| state.output_item_id.clone())
            .flatten()
    }) else {
        log::debug!(
            "Deferring streamed function call emission until call_id metadata is available for id {}",
            function_call_id
        );
        return Ok(None);
    };
    let arguments = if state.arguments.is_empty() {
        json!({})
    } else {
        serde_json::from_str(&state.arguments)
            .context("Failed to parse streamed function call arguments")?
    };

    state.emitted = true;
    if !accumulator
        .emitted_function_call_ids
        .iter()
        .any(|existing_id| existing_id == function_call_id)
    {
        accumulator
            .emitted_function_call_ids
            .push(function_call_id.to_string());
    }

    Ok(Some(ParsedFunctionCall {
        name,
        call_id: canonical_call_id,
        arguments,
    }))
}

/// Reconciles a streamed function call state key, promoting item_id-backed state to a real call_id when available.
fn reconcile_streamed_function_call_state_key(
    accumulator: &mut StreamingResponsesAccumulator,
    call_id: Option<&str>,
    item_id: Option<&str>,
) -> anyhow::Result<String> {
    let Some(resolved_key) = call_id.or(item_id).filter(|value| !value.is_empty()) else {
        return Err(anyhow!(
            "Streaming function call event did not include either call_id or item_id"
        ));
    };

    if let (Some(call_id), Some(item_id)) = (
        call_id.filter(|value| !value.is_empty()),
        item_id.filter(|value| !value.is_empty()),
    ) {
        if call_id != item_id {
            let existing_call_state = accumulator.function_calls_by_call_id.remove(item_id);
            let merged_state = merge_streaming_function_call_state(
                existing_call_state,
                accumulator.function_calls_by_call_id.remove(call_id),
            );
            if let Some(merged_state) = merged_state {
                accumulator
                    .function_calls_by_call_id
                    .insert(call_id.to_string(), merged_state);
            }
            return Ok(call_id.to_string());
        }
    }

    Ok(resolved_key.to_string())
}

/// Merges two streamed function call states, preferring the more authoritative populated fields.
fn merge_streaming_function_call_state(
    first: Option<StreamingFunctionCallState>,
    second: Option<StreamingFunctionCallState>,
) -> Option<StreamingFunctionCallState> {
    match (first, second) {
        (None, None) => None,
        (Some(state), None) | (None, Some(state)) => Some(state),
        (Some(first), Some(second)) => Some(StreamingFunctionCallState {
            output_item_id: second.output_item_id.or(first.output_item_id),
            provider_call_id: second.provider_call_id.or(first.provider_call_id),
            name: second.name.or(first.name),
            arguments: if second.arguments.is_empty() {
                first.arguments
            } else {
                second.arguments
            },
            emitted: first.emitted || second.emitted,
        }),
    }
}

/// Finalizes stream state, backfills any non-streamed outputs, and records conversation history.
pub(super) fn finalize_stream_state(
    params: &RequestParams,
    accumulator: StreamingResponsesAccumulator,
    request_id: &str,
    completed_response: Option<ResponsesApiResponse>,
) -> anyhow::Result<Vec<Event>> {
    let mut events = Vec::new();

    if let Some(response) = completed_response {
        let output = response.output;
        let history_items = if output.is_empty() {
            history_items_from_accumulator(&accumulator)?
        } else {
            match parse_responses_output(output.clone()) {
                Ok((assistant_messages, function_calls)) => {
                    if assistant_messages.is_empty()
                        && function_calls.is_empty()
                        && has_streamed_output(&accumulator)
                    {
                        history_items_from_accumulator(&accumulator)?
                    } else {
                        let mut history_items = assistant_messages
                            .iter()
                            .map(|text| assistant_output_item(text))
                            .collect::<Vec<_>>();
                        history_items.extend(function_calls.iter().map(function_call_history_item));
                        history_items
                    }
                }
                Err(error) if has_streamed_output(&accumulator) => {
                    log::debug!(
                        "Falling back to streamed local Responses history because completed payload could not be parsed: {error:#}"
                    );
                    history_items_from_accumulator(&accumulator)?
                }
                Err(error) => return Err(error),
            }
        };
        if !history_items.is_empty() {
            let mut state_store = conversation_state_store().lock();
            let state = state_store.entry(params.conversation_id).or_default();
            state.items.extend(history_items);
        }

        let backfill_messages = build_backfill_messages(
            &accumulator,
            params.target_task_id.as_ref(),
            request_id,
            output,
        )?;
        if let Some(task_id) = params.target_task_id.as_ref() {
            if !backfill_messages.is_empty() {
                events.push(Ok(add_messages_event(task_id, backfill_messages)));
            }
        }
    }

    events.push(Ok(stream_finished_event(
        api::response_event::stream_finished::Reason::Done(
            api::response_event::stream_finished::Done {},
        ),
    )));
    Ok(events)
}

/// Returns whether the accumulator already captured streamed text or tool calls.
fn has_streamed_output(accumulator: &StreamingResponsesAccumulator) -> bool {
    !accumulator.text_messages_by_item_id.is_empty()
        || !accumulator.emitted_function_call_ids.is_empty()
}

/// Reconstructs conversation history items from streamed deltas when the completed payload is empty.
fn history_items_from_accumulator(
    accumulator: &StreamingResponsesAccumulator,
) -> anyhow::Result<Vec<Value>> {
    let mut items = Vec::new();

    for item_id in &accumulator.emitted_text_item_ids {
        let Some(message_state) = accumulator.text_messages_by_item_id.get(item_id) else {
            continue;
        };
        if !message_state.text.is_empty() {
            items.push(assistant_output_item(&message_state.text));
        }
    }

    for call_id in &accumulator.emitted_function_call_ids {
        let Some(function_call_state) = accumulator.function_calls_by_call_id.get(call_id) else {
            continue;
        };
        let Some(name) = function_call_state.name.clone() else {
            continue;
        };
        let arguments = if function_call_state.arguments.is_empty() {
            json!({})
        } else {
            serde_json::from_str(&function_call_state.arguments)
                .context("Failed to parse streamed function call arguments from accumulator")?
        };
        items.push(function_call_history_item(&ParsedFunctionCall {
            name,
            call_id: call_id.clone(),
            arguments,
        }));
    }

    Ok(items)
}

/// Builds fallback messages for any completed outputs that were not already streamed to the UI.
fn build_backfill_messages(
    accumulator: &StreamingResponsesAccumulator,
    task_id: Option<&TaskId>,
    request_id: &str,
    output: Vec<ResponsesOutputItem>,
) -> anyhow::Result<Vec<api::Message>> {
    let Some(task_id) = task_id else {
        return Ok(Vec::new());
    };

    let mut messages = Vec::new();
    for item in output {
        match item.item_type.as_str() {
            "message" if item.role.as_deref() == Some("assistant") => {
                let Some(item_id) = item.id.as_ref() else {
                    continue;
                };
                if accumulator
                    .emitted_text_item_ids
                    .iter()
                    .any(|existing_id| existing_id == item_id)
                {
                    continue;
                }

                let text = item
                    .content
                    .iter()
                    .filter(|content| content.item_type == "output_text")
                    .filter_map(|content| content.text.clone())
                    .collect::<String>();
                if !text.is_empty() {
                    messages.push(agent_output_message(task_id, request_id, text));
                }
            }
            "function_call" => {
                let Some(call_id) = item.call_id.as_ref() else {
                    continue;
                };
                if accumulator
                    .emitted_function_call_ids
                    .iter()
                    .any(|existing_id| existing_id == call_id)
                {
                    continue;
                }

                let Some(name) = item.name.clone() else {
                    continue;
                };
                let arguments = item.arguments.unwrap_or_else(|| "{}".to_string());
                messages.push(tool_call_message(
                    task_id,
                    request_id,
                    ParsedFunctionCall {
                        name,
                        call_id: call_id.clone(),
                        arguments: serde_json::from_str(&arguments)
                            .context("Failed to parse backfilled function call arguments")?,
                    },
                )?);
            }
            _ => {}
        }
    }

    Ok(messages)
}

/// Resolves the canonical event type for a streamed SSE message.
fn streamed_event_type(event_name: &str, payload: &Value) -> String {
    if !event_name.is_empty() {
        return event_name.to_string();
    }

    payload
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Extracts the most useful error message from a streamed Responses failure payload.
fn streamed_error_message(payload: &Value) -> String {
    if let Some(message) = payload.get("message").and_then(Value::as_str) {
        return message.to_string();
    }

    if let Some(message) = payload
        .get("error")
        .and_then(Value::as_object)
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
    {
        return message.to_string();
    }

    payload.to_string()
}

/// Resolves the function call identifier from a streamed Responses event.
fn streamed_function_call_id(
    call_id: Option<&str>,
    item_id: Option<&str>,
) -> anyhow::Result<String> {
    if let Some(call_id) = call_id.filter(|value| !value.is_empty()) {
        return Ok(call_id.to_string());
    }

    if let Some(item_id) = item_id.filter(|value| !value.is_empty()) {
        log::debug!("Streaming Responses event omitted call_id, falling back to item_id");
        return Ok(item_id.to_string());
    }

    Err(anyhow!(
        "Streaming function call event did not include either call_id or item_id"
    ))
}

/// Converts a parsed Responses function call into a Warp tool call message.
fn tool_call_message(
    task_id: &TaskId,
    request_id: &str,
    function_call: ParsedFunctionCall,
) -> anyhow::Result<api::Message> {
    Ok(api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.to_string(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::ToolCall(api::message::ToolCall {
            tool_call_id: function_call.call_id,
            tool: Some(parse_tool_call(
                function_call.name.as_str(),
                function_call.arguments,
            )?),
        })),
        request_id: request_id.to_string(),
        timestamp: None,
    })
}

/// Converts assistant text into a Warp agent output message.
fn agent_output_message(task_id: &TaskId, request_id: &str, text: String) -> api::Message {
    agent_output_message_with_id(Uuid::new_v4().to_string(), task_id, request_id, text)
}

/// Converts assistant text into a Warp agent output message with a stable message ID.
pub(super) fn agent_output_message_with_id(
    message_id: String,
    task_id: &TaskId,
    request_id: &str,
    text: String,
) -> api::Message {
    api::Message {
        id: message_id,
        task_id: task_id.to_string(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::AgentOutput(
            api::message::AgentOutput { text },
        )),
        request_id: request_id.to_string(),
        timestamp: None,
    }
}

/// Builds an update client action that replaces an existing streamed assistant text message.
pub(super) fn update_agent_output_text_event(
    task_id: &TaskId,
    request_id: &str,
    message_id: &str,
    full_text: String,
) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::UpdateTaskMessage(
                        api::client_action::UpdateTaskMessage {
                            task_id: task_id.to_string(),
                            message: Some(agent_output_message_with_id(
                                message_id.to_string(),
                                task_id,
                                request_id,
                                full_text,
                            )),
                            mask: Some(prost_types::FieldMask {
                                paths: vec!["agent_output.text".to_string()],
                            }),
                        },
                    )),
                }],
            },
        )),
    }
}
