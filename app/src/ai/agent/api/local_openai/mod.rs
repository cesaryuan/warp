//! Local OpenAI-compatible Responses API backend for Warp Agent.

mod request;
mod stream;
#[cfg(test)]
mod tests;
mod tool_calls;
mod tool_schemas;
mod types;

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::anyhow;
use async_stream::stream;
use futures::channel::oneshot;
use futures::future::{select, Either};
use futures::StreamExt as _;
use parking_lot::FairMutex;
use reqwest_eventsource::Event as EventSourceEvent;
use uuid::Uuid;
use warp_multi_agent_api as api;

use crate::ai::agent::conversation::AIConversationId;
use crate::ai::agent::task::TaskId;
use crate::server::server_api::ServerApi;

use super::{Event, RequestParams, ResponseStream};
use request::{start_local_responses_eventsource, stream_error_to_anyhow};
use stream::handle_responses_stream_message;
use types::{LocalConversationState, StreamingResponsesAccumulator};

/// Minimal system instructions that teach a local Responses model how to behave like Warp Agent.
pub(super) const LOCAL_OPENAI_SYSTEM_PROMPT: &str = concat!(
    "You are Warp Agent running locally inside the user's terminal workspace. ",
    "Help with coding and shell tasks. ",
    "Use the provided tools whenever you need to inspect files, search code, run commands, ",
    "or apply edits. ",
    "When you do not need a tool, answer directly and concisely. ",
    "When returning tool calls, provide valid JSON arguments that exactly match the tool schema."
);

/// Returns the global in-memory state used to preserve local conversation history.
fn conversation_state_store(
) -> &'static FairMutex<HashMap<AIConversationId, LocalConversationState>> {
    static STORE: OnceLock<FairMutex<HashMap<AIConversationId, LocalConversationState>>> =
        OnceLock::new();
    STORE.get_or_init(|| FairMutex::new(HashMap::new()))
}

/// Generates a local OpenAI Responses-backed event stream for a Warp Agent request.
pub fn generate_local_openai_responses_output(
    server_api: std::sync::Arc<ServerApi>,
    params: RequestParams,
    cancellation_rx: oneshot::Receiver<()>,
) -> ResponseStream {
    let request_id = Uuid::new_v4().to_string();
    let conversation_token = params
        .conversation_token
        .as_ref()
        .map(|token| token.as_str().to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let task_id = params.target_task_id.clone();
    let model_name = params.model.clone().to_string();

    Box::pin(stream! {
        yield Ok(stream_init_event(conversation_token, request_id.clone()));

        let Some(task_id) = task_id else {
            let error = anyhow!("Missing task ID for local OpenAI backend");
            yield Ok(user_visible_error_event(
                &TaskId::new("missing-task".to_string()),
                &request_id,
                &error.to_string(),
            ));
            yield Ok(stream_finished_event(finished_reason_for_error(&error, model_name)));
            return;
        };

        if should_emit_create_task(&params) {
            yield Ok(create_task_event(&task_id));
        }

        let mut response_stream = match start_local_responses_eventsource(&server_api, &params).await {
            Ok(stream) => stream,
            Err(error) => {
                log::warn!("Local OpenAI Responses backend failed before streaming began: {error:#}");
                yield Ok(user_visible_error_event(
                    &task_id,
                    &request_id,
                    &error.to_string(),
                ));
                yield Ok(stream_finished_event(
                    finished_reason_for_error(&error, model_name),
                ));
                return;
            }
        };

        let mut accumulator = StreamingResponsesAccumulator::default();
        let mut cancellation_future = Box::pin(cancellation_rx);
        let mut stream_finished = false;

        loop {
            let next_event_future = Box::pin(response_stream.next());
            match select(next_event_future, cancellation_future).await {
                Either::Left((next_event, next_cancellation_future)) => {
                    cancellation_future = next_cancellation_future;
                    let Some(next_event) = next_event else {
                        break;
                    };

                    match next_event {
                        Ok(EventSourceEvent::Open) => {}
                        Ok(EventSourceEvent::Message(message)) => {
                            match handle_responses_stream_message(
                                &params,
                                &task_id,
                                &request_id,
                                message.event.as_str(),
                                message.data.as_str(),
                                &mut accumulator,
                            ) {
                                Ok(result) => {
                                    for event in result.events {
                                        yield event;
                                    }
                                    if result.is_terminal {
                                        stream_finished = true;
                                        break;
                                    }
                                }
                                Err(error) => {
                                    log::warn!("Failed to translate local OpenAI Responses event: {error:#}");
                                    yield Ok(user_visible_error_event(
                                        &task_id,
                                        &request_id,
                                        &error.to_string(),
                                    ));
                                    yield Ok(stream_finished_event(
                                        finished_reason_for_error(&error, params.model.to_string()),
                                    ));
                                    return;
                                }
                            }
                        }
                        Err(err) => {
                            let error = stream_error_to_anyhow(err).await;
                            log::warn!("Local OpenAI Responses stream failed: {error:#}");
                            yield Ok(user_visible_error_event(
                                &task_id,
                                &request_id,
                                &error.to_string(),
                            ));
                            yield Ok(stream_finished_event(
                                finished_reason_for_error(&error, params.model.to_string()),
                            ));
                            return;
                        }
                    }
                }
                Either::Right((_, _)) => {
                    log::info!("Local OpenAI Responses stream cancelled");
                    return;
                }
            }
        }

        if !stream_finished {
            log::debug!("Local OpenAI Responses stream ended without a terminal event");
            yield Ok(stream_finished_event(
                api::response_event::stream_finished::Reason::Done(
                    api::response_event::stream_finished::Done {},
                ),
            ));
        }
    })
}

/// Builds the initial stream event for a local backend request.
fn stream_init_event(conversation_id: String, request_id: String) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::Init(
            api::response_event::StreamInit {
                conversation_id,
                request_id,
                run_id: String::new(),
            },
        )),
    }
}

/// Builds a finished event with the provided terminal reason.
fn stream_finished_event(
    reason: api::response_event::stream_finished::Reason,
) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::Finished(
            api::response_event::StreamFinished {
                reason: Some(reason),
                ..Default::default()
            },
        )),
    }
}

/// Builds an `AddMessagesToTask` client action event.
fn add_messages_event(task_id: &TaskId, messages: Vec<api::Message>) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::AddMessagesToTask(
                        api::client_action::AddMessagesToTask {
                            task_id: task_id.to_string(),
                            messages,
                        },
                    )),
                }],
            },
        )),
    }
}

/// Builds a `CreateTask` client action event so the local root task is upgraded before messages arrive.
fn create_task_event(task_id: &TaskId) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::CreateTask(
                        api::client_action::CreateTask {
                            task: Some(api::Task {
                                id: task_id.to_string(),
                                messages: vec![],
                                dependencies: None,
                                description: String::new(),
                                summary: String::new(),
                                server_data: String::new(),
                            }),
                        },
                    )),
                }],
            },
        )),
    }
}

/// Returns whether the local backend should emit a `CreateTask` action for this request.
fn should_emit_create_task(params: &RequestParams) -> bool {
    params.conversation_token.is_none()
}

/// Builds a user-visible error message event so the UI never gets stuck without output.
fn user_visible_error_event(
    task_id: &TaskId,
    request_id: &str,
    error_message: &str,
) -> api::ResponseEvent {
    add_messages_event(
        task_id,
        vec![stream::agent_output_message_with_id(
            Uuid::new_v4().to_string(),
            task_id,
            request_id,
            format!("Local OpenAI backend error: {error_message}"),
        )],
    )
}

/// Maps a local backend provider failure into the closest Warp finished reason.
fn finished_reason_for_error(
    error: &anyhow::Error,
    model_name: String,
) -> api::response_event::stream_finished::Reason {
    if let Some(provider_error) = error.downcast_ref::<ProviderError>() {
        match provider_error.status_code {
            401 | 403 => {
                return api::response_event::stream_finished::Reason::InvalidApiKey(
                    api::response_event::stream_finished::InvalidApiKey {
                        provider: api::LlmProvider::Openai.into(),
                        model_name,
                    },
                );
            }
            429 => {
                return api::response_event::stream_finished::Reason::QuotaLimit(
                    api::response_event::stream_finished::QuotaLimit {},
                );
            }
            _ => {}
        }
    }

    api::response_event::stream_finished::Reason::InternalError(
        api::response_event::stream_finished::InternalError {
            message: error.to_string(),
        },
    )
}

/// Simple typed wrapper for HTTP provider failures that need status-aware handling.
#[derive(Debug)]
struct ProviderError {
    status_code: u16,
    message: String,
}

impl ProviderError {
    /// Creates a new provider error with the given status and message.
    fn new(status_code: u16, message: String) -> Self {
        Self {
            status_code,
            message,
        }
    }
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "provider returned HTTP {}: {}",
            self.status_code, self.message
        )
    }
}

impl std::error::Error for ProviderError {}
