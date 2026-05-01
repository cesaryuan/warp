use std::borrow::Cow;
use std::sync::Arc;

use ai::agent::action_result::{AIAgentActionResultType, ReadFilesResult};
use warp_multi_agent_api as api;

use super::request::{
    build_tools_payload, convert_inputs_to_response_items, convert_inputs_to_task_messages,
    mcp_tool_schema, normalize_openai_model_and_reasoning, normalize_responses_endpoint,
    prepare_local_responses_request, task_history_response_items,
};
use super::stream::{
    agent_output_message_with_id, finalize_stream_state, finalize_streamed_function_call,
    handle_responses_stream_message, handle_streamed_output_item_done,
    update_agent_output_text_event,
};
use super::tool_calls::parse_read_file;
use super::types::{
    ResponsesApiResponse, ResponsesContentItem, ResponsesFunctionCallArgumentsDoneEvent,
    ResponsesOutputItem, ResponsesReasoningSummaryPart, StreamingResponsesAccumulator,
    StreamingTextMessageState,
};
use super::*;
use crate::ai::agent::conversation::AIConversationId;
use crate::ai::agent::task::TaskId;
use crate::ai::agent::AIAgentActionResult;
use crate::ai::agent::AIAgentContext;
use crate::ai::agent::AIAgentInput;
use crate::ai::agent::MCPContext;
use crate::ai::agent::MCPServer;
use crate::ai::blocklist::SessionContext;
use crate::ai::llms::{LLMId, LLMProvider};

/// Builds a minimal RequestParams instance for local backend tests.
fn request_params_for_local_backend_tests() -> RequestParams {
    let model = LLMId::from("gpt-5-4-low");
    RequestParams {
        conversation_id: AIConversationId::new(),
        input: vec![],
        target_task_id: Some(TaskId::new("task-id".to_string())),
        conversation_token: None,
        forked_from_conversation_token: None,
        ambient_agent_task_id: None,
        tasks: vec![],
        existing_suggestions: None,
        metadata: None,
        session_context: SessionContext::new_for_test(),
        model: model.clone(),
        coding_model: model.clone(),
        cli_agent_model: model.clone(),
        computer_use_model: model,
        is_memory_enabled: false,
        warp_drive_context_enabled: false,
        mcp_context: None,
        planning_enabled: true,
        should_redact_secrets: false,
        api_keys: None,
        allow_use_of_warp_credits_with_byok: false,
        local_openai_responses_backend_enabled: true,
        local_openai_api_key: None,
        local_openai_base_url: None,
        model_provider: LLMProvider::OpenAI,
        autonomy_level: api::AutonomyLevel::Supervised,
        isolation_level: api::IsolationLevel::None,
        web_search_enabled: false,
        computer_use_enabled: false,
        ask_user_question_enabled: false,
        research_agent_enabled: false,
        orchestration_enabled: false,
        supported_tools_override: None,
        parent_agent_id: None,
        agent_name: None,
    }
}

/// Builds a minimal MCP tool used for schema passthrough tests.
fn test_mcp_tool() -> rmcp::model::Tool {
    rmcp::model::Tool {
        name: Cow::Owned("lookup_weather".to_string()),
        title: Some("Lookup Weather".to_string()),
        description: Some(Cow::Owned("Look up weather by city.".to_string())),
        input_schema: Arc::new(
            serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name to look up."
                    }
                },
                "required": ["city"]
            })
            .as_object()
            .expect("schema should be object")
            .clone(),
        ),
        output_schema: None,
        annotations: None,
        icons: None,
        meta: None,
    }
}

/// Verifies that a base URL without `/v1` is normalized correctly.
#[test]
fn normalize_responses_endpoint_appends_v1() {
    assert_eq!(
        normalize_responses_endpoint("https://api.openai.com"),
        "https://api.openai.com/v1/responses"
    );
}

/// Verifies that an existing `/v1` suffix is preserved when building the endpoint.
#[test]
fn normalize_responses_endpoint_preserves_v1() {
    assert_eq!(
        normalize_responses_endpoint("https://example.com/v1/"),
        "https://example.com/v1/responses"
    );
}

/// Verifies that Warp-style GPT-5.4 reasoning variants are converted to Responses model plus effort.
#[test]
fn normalize_openai_model_and_reasoning_extracts_effort() {
    let (model, reasoning) = normalize_openai_model_and_reasoning("gpt-5-4-low");
    assert_eq!(model, "gpt-5.4");
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref()),
        Some("low")
    );
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.summary.as_deref()),
        Some("auto")
    );
}

/// Verifies that Warp-style GPT-5.5 reasoning variants are converted to Responses model plus effort.
#[test]
fn normalize_openai_model_and_reasoning_extracts_effort_for_gpt_5_5() {
    let (model, reasoning) = normalize_openai_model_and_reasoning("gpt-5-5-low");
    assert_eq!(model, "gpt-5.5");
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref()),
        Some("low")
    );
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.summary.as_deref()),
        Some("auto")
    );
}

/// Verifies that dotted GPT-5.2 reasoning variants preserve the exact Responses model name.
#[test]
fn normalize_openai_model_and_reasoning_extracts_effort_for_gpt_5_2() {
    let (model, reasoning) = normalize_openai_model_and_reasoning("gpt-5.2-low");
    assert_eq!(model, "gpt-5.2");
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref()),
        Some("low")
    );
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.summary.as_deref()),
        Some("auto")
    );
}

/// Verifies that GPT-5.2 Codex reasoning variants preserve the exact Responses model name.
#[test]
fn normalize_openai_model_and_reasoning_extracts_effort_for_gpt_5_2_codex() {
    let (model, reasoning) = normalize_openai_model_and_reasoning("gpt-5.2-codex-low");
    assert_eq!(model, "gpt-5.2-codex");
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref()),
        Some("low")
    );
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.summary.as_deref()),
        Some("auto")
    );
}

/// Verifies that GPT-5.3 Codex reasoning variants preserve the exact Responses model name.
#[test]
fn normalize_openai_model_and_reasoning_extracts_effort_for_gpt_5_3_codex() {
    let (model, reasoning) = normalize_openai_model_and_reasoning("gpt-5.3-codex-low");
    assert_eq!(model, "gpt-5.3-codex");
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref()),
        Some("low")
    );
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.summary.as_deref()),
        Some("auto")
    );
}

/// Verifies that the rendered local OpenAI prompt includes the actual request model name.
#[test]
fn build_local_openai_system_prompt_injects_model_name() {
    let prompt = build_local_openai_system_prompt("qwen3:32b");
    assert!(prompt.contains(r#"You are powered by the "qwen3:32b" model."#));
    assert!(!prompt.contains("__LOCAL_OPENAI_MODEL__"));
}

/// Verifies that base GPT numeric variants opt into reasoning summaries without inventing an effort.
#[test]
fn normalize_openai_model_and_reasoning_preserves_base_variant() {
    let (model, reasoning) = normalize_openai_model_and_reasoning("gpt-5-4");
    assert_eq!(model, "gpt-5.4");
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref()),
        None
    );
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.summary.as_deref()),
        Some("auto")
    );
}

/// Verifies that base GPT-5.5 variants opt into reasoning summaries without inventing an effort.
#[test]
fn normalize_openai_model_and_reasoning_preserves_gpt_5_5_base_variant() {
    let (model, reasoning) = normalize_openai_model_and_reasoning("gpt-5-5");
    assert_eq!(model, "gpt-5.5");
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref()),
        None
    );
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.summary.as_deref()),
        Some("auto")
    );
}

/// Verifies that base GPT-5.2 Codex variants opt into reasoning summaries without inventing an effort.
#[test]
fn normalize_openai_model_and_reasoning_preserves_gpt_5_2_codex_base_variant() {
    let (model, reasoning) = normalize_openai_model_and_reasoning("gpt-5.2-codex");
    assert_eq!(model, "gpt-5.2-codex");
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref()),
        None
    );
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.summary.as_deref()),
        Some("auto")
    );
}

/// Verifies that base GPT-5.3 Codex variants opt into reasoning summaries without inventing an effort.
#[test]
fn normalize_openai_model_and_reasoning_preserves_gpt_5_3_codex_base_variant() {
    let (model, reasoning) = normalize_openai_model_and_reasoning("gpt-5.3-codex");
    assert_eq!(model, "gpt-5.3-codex");
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref()),
        None
    );
    assert_eq!(
        reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.summary.as_deref()),
        Some("auto")
    );
}

/// Verifies that MCP tool schemas reuse the server-provided description and JSON Schema.
#[test]
fn mcp_tool_schema_reuses_existing_metadata() {
    let schema = mcp_tool_schema(Some("server-1"), &test_mcp_tool()).expect("schema should exist");
    assert_eq!(schema["type"], "function");
    assert_eq!(schema["strict"], false);
    assert_eq!(schema["description"], "Look up weather by city.");
    assert_eq!(
        schema["parameters"]["properties"]["city"]["description"],
        "City name to look up."
    );
}

/// Verifies that build_tools_payload merges built-in schemas with MCP tool schemas.
#[test]
fn build_tools_payload_includes_mcp_tools() {
    let mut params = request_params_for_local_backend_tests();
    params.mcp_context = Some(MCPContext {
        #[allow(deprecated)]
        resources: vec![],
        #[allow(deprecated)]
        tools: vec![],
        servers: vec![MCPServer {
            id: "server-1".to_string(),
            name: "Test MCP".to_string(),
            description: "Test server".to_string(),
            resources: vec![],
            tools: vec![test_mcp_tool()],
        }],
    });

    let payload = build_tools_payload(&params);
    assert!(payload.iter().any(|tool| {
        tool["name"]
            .as_str()
            .is_some_and(|name| name.starts_with("warp_mcp_tool__"))
    }));
}

/// Verifies that supported-tools overrides are respected when building local tool payloads.
#[test]
fn build_tools_payload_respects_supported_tools_override() {
    let mut params = request_params_for_local_backend_tests();
    params.supported_tools_override = Some(vec![api::ToolType::SuggestPrompt]);

    let payload = build_tools_payload(&params);
    assert_eq!(payload.len(), 1);
    assert_eq!(payload[0]["name"], "suggest_prompt");
}

/// Verifies that unsupported override tools are omitted rather than silently broadening the payload.
#[test]
fn build_tools_payload_omits_unsupported_override_tools() {
    let mut params = request_params_for_local_backend_tests();
    params.supported_tools_override = Some(vec![api::ToolType::Subagent]);

    let payload = build_tools_payload(&params);
    assert!(payload.is_empty());
}

/// Verifies that built-in tool schemas now carry parameter descriptions.
#[test]
fn built_in_tool_schemas_include_property_descriptions() {
    let payload = build_tools_payload(&request_params_for_local_backend_tests());
    let run_shell_schema = payload
        .iter()
        .find(|tool| tool["name"] == "run_shell_command")
        .expect("run_shell_command schema should be present");
    assert_eq!(run_shell_schema["type"], "function");
    assert_eq!(run_shell_schema["strict"], false);
    assert!(
        run_shell_schema["parameters"]["properties"]["command"]["description"]
            .as_str()
            .is_some()
    );
}

/// Verifies that the run_shell_command schema exposes proto-backed risk fields.
#[test]
fn run_shell_command_schema_includes_proto_risk_fields() {
    let payload = build_tools_payload(&request_params_for_local_backend_tests());
    let run_shell_schema = payload
        .iter()
        .find(|tool| tool["name"] == "run_shell_command")
        .expect("run_shell_command schema should be present");
    assert_eq!(
        run_shell_schema["parameters"]["additionalProperties"],
        serde_json::json!(false)
    );
    assert!(run_shell_schema["parameters"]["properties"]["uses_pager"].is_object());
    assert!(run_shell_schema["parameters"]["properties"]["risk_category"].is_object());
}

/// Verifies that local Responses requests opt into parallel tool calls without storing provider-side state.
#[test]
fn prepare_local_responses_request_configures_parallel_tool_calls_and_store_policy() {
    let mut params = request_params_for_local_backend_tests();
    params.local_openai_api_key = Some("test-key".to_string());
    params.local_openai_base_url = Some("https://example.com".to_string());

    let prepared_request =
        prepare_local_responses_request(&params).expect("request should prepare successfully");
    let request_body = serde_json::to_value(&prepared_request.request_body)
        .expect("request body should serialize");

    assert_eq!(request_body["parallel_tool_calls"], serde_json::json!(true));
    assert_eq!(request_body["tool_choice"], serde_json::json!("auto"));
    assert_eq!(request_body["store"], serde_json::json!(false));
    assert_eq!(
        request_body["reasoning"]["effort"],
        serde_json::json!("low")
    );
    assert_eq!(
        request_body["reasoning"]["summary"],
        serde_json::json!("auto")
    );
    assert_eq!(
        request_body["include"],
        serde_json::json!(["reasoning.encrypted_content"])
    );
    assert_eq!(
        request_body["prompt_cache_key"],
        serde_json::json!(params.conversation_id.to_string())
    );
    assert_eq!(
        prepared_request.session_id_header,
        Some(params.conversation_id.to_string())
    );
}

/// Verifies that prompt cache keys stay stable across per-turn input changes within the same conversation.
#[test]
fn prepare_local_responses_request_keeps_prompt_cache_key_stable_for_same_conversation() {
    let conversation_id = AIConversationId::new();
    let mut params_a = request_params_for_local_backend_tests();
    params_a.conversation_id = conversation_id;
    params_a.local_openai_api_key = Some("test-key".to_string());
    params_a.local_openai_base_url = Some("https://example.com".to_string());
    params_a.input = vec![AIAgentInput::UserQuery {
        query: "first question".to_string(),
        context: std::sync::Arc::new([]),
        static_query_type: None,
        referenced_attachments: std::collections::HashMap::new(),
        user_query_mode: crate::ai::agent::UserQueryMode::Normal,
        running_command: None,
        intended_agent: None,
    }];

    let mut params_b = request_params_for_local_backend_tests();
    params_b.conversation_id = conversation_id;
    params_b.local_openai_api_key = Some("test-key".to_string());
    params_b.local_openai_base_url = Some("https://example.com".to_string());
    params_b.input = vec![AIAgentInput::UserQuery {
        query: "second question".to_string(),
        context: std::sync::Arc::new([]),
        static_query_type: None,
        referenced_attachments: std::collections::HashMap::new(),
        user_query_mode: crate::ai::agent::UserQueryMode::Normal,
        running_command: None,
        intended_agent: None,
    }];

    let prepared_a =
        prepare_local_responses_request(&params_a).expect("first request should prepare");
    let prepared_b =
        prepare_local_responses_request(&params_b).expect("second request should prepare");

    assert_eq!(
        prepared_a.request_body.prompt_cache_key,
        prepared_b.request_body.prompt_cache_key
    );
    assert_eq!(prepared_a.session_id_header, prepared_b.session_id_header);
}

/// Verifies that prompt cache keys change when Warp assigns a different conversation identity.
#[test]
fn prepare_local_responses_request_changes_prompt_cache_key_when_conversation_changes() {
    let mut params_a = request_params_for_local_backend_tests();
    params_a.local_openai_api_key = Some("test-key".to_string());
    params_a.local_openai_base_url = Some("https://example.com".to_string());

    let mut params_b = request_params_for_local_backend_tests();
    params_b.local_openai_api_key = Some("test-key".to_string());
    params_b.local_openai_base_url = Some("https://example.com".to_string());

    let prepared_a =
        prepare_local_responses_request(&params_a).expect("first request should prepare");
    let prepared_b =
        prepare_local_responses_request(&params_b).expect("second request should prepare");

    assert_ne!(
        prepared_a.request_body.prompt_cache_key,
        prepared_b.request_body.prompt_cache_key
    );
    assert_ne!(prepared_a.session_id_header, prepared_b.session_id_header);
}

/// Verifies that the Session_id request header mirrors the prompt cache key exactly.
#[test]
fn prepare_local_responses_request_sets_session_id_header_from_prompt_cache_key() {
    let mut params = request_params_for_local_backend_tests();
    params.local_openai_api_key = Some("test-key".to_string());
    params.local_openai_base_url = Some("https://example.com".to_string());

    let prepared_request =
        prepare_local_responses_request(&params).expect("request should prepare successfully");

    assert_eq!(
        prepared_request.session_id_header,
        prepared_request.request_body.prompt_cache_key
    );
}

/// Verifies that streaming text deltas update the existing agent output field.
#[test]
fn update_agent_output_text_event_replaces_text() {
    let task_id = TaskId::new("task-id".to_string());
    let initial_message = agent_output_message_with_id(
        "message-id".to_string(),
        &task_id,
        "request-id",
        "hello".to_string(),
    );
    let update_event = update_agent_output_text_event(
        &task_id,
        "request-id",
        "message-id",
        "hello world".to_string(),
    );

    let Some(api::response_event::Type::ClientActions(actions)) = update_event.r#type else {
        panic!("expected client actions");
    };
    let Some(api::client_action::Action::UpdateTaskMessage(update)) =
        actions.actions[0].action.as_ref()
    else {
        panic!("expected update action");
    };
    let merged = field_mask::FieldMaskOperation::update(
        &api::MESSAGE_DESCRIPTOR,
        &initial_message,
        update
            .message
            .as_ref()
            .expect("update message should exist"),
        update.mask.clone().expect("update mask should exist"),
    )
    .apply()
    .expect("update should succeed");

    let Some(api::message::Message::AgentOutput(agent_output)) = merged.message else {
        panic!("expected agent output message");
    };
    assert_eq!(agent_output.text, "hello world");
}

/// Verifies that streamed reasoning deltas update the existing agent reasoning message and finalize its duration.
#[test]
fn streamed_reasoning_events_update_reasoning_text_and_duration() {
    let params = request_params_for_local_backend_tests();
    let task_id = TaskId::new("task-id".to_string());
    let mut accumulator = StreamingResponsesAccumulator::default();

    let delta_result = handle_responses_stream_message(
        &params,
        &task_id,
        "request-id",
        "response.reasoning_summary_text.delta",
        r#"{"item_id":"rs_1","summary_index":0,"delta":"First pass"}"#,
        &mut accumulator,
    )
    .expect("reasoning delta should parse");
    assert_eq!(delta_result.events.len(), 1);

    let delta_event = delta_result.events[0]
        .as_ref()
        .expect("delta event should be ok");
    let Some(api::response_event::Type::ClientActions(delta_actions)) = &delta_event.r#type else {
        panic!("expected client actions");
    };
    let Some(api::client_action::Action::AddMessagesToTask(delta_add)) =
        delta_actions.actions[0].action.as_ref()
    else {
        panic!("expected add-messages action");
    };
    let initial_message = delta_add.messages[0].clone();
    let Some(api::message::Message::AgentReasoning(initial_reasoning)) = &initial_message.message
    else {
        panic!("expected initial reasoning message");
    };
    assert_eq!(initial_reasoning.reasoning, "First pass");
    assert!(initial_reasoning.finished_duration.is_none());

    let done_result = handle_responses_stream_message(
        &params,
        &task_id,
        "request-id",
        "response.reasoning_summary_text.done",
        r#"{"item_id":"rs_1","summary_index":0,"text":"First pass complete"}"#,
        &mut accumulator,
    )
    .expect("reasoning done should parse");
    assert_eq!(done_result.events.len(), 1);

    let done_event = done_result.events[0]
        .as_ref()
        .expect("done event should be ok");
    let Some(api::response_event::Type::ClientActions(done_actions)) = &done_event.r#type else {
        panic!("expected client actions");
    };
    let Some(api::client_action::Action::UpdateTaskMessage(update)) =
        done_actions.actions[0].action.as_ref()
    else {
        panic!("expected update action");
    };
    let merged = field_mask::FieldMaskOperation::update(
        &api::MESSAGE_DESCRIPTOR,
        &initial_message,
        update
            .message
            .as_ref()
            .expect("update message should exist"),
        update.mask.clone().expect("update mask should exist"),
    )
    .apply()
    .expect("reasoning update should succeed");

    let Some(api::message::Message::AgentReasoning(reasoning)) = merged.message else {
        panic!("expected reasoning message");
    };
    assert_eq!(reasoning.reasoning, "First pass complete");
    assert!(reasoning.finished_duration.is_some());
}

/// Verifies that completed reasoning output items are backfilled into task messages.
#[test]
fn finalize_stream_state_backfills_reasoning_messages_from_completed_output() {
    let params = request_params_for_local_backend_tests();
    conversation_state_store()
        .lock()
        .insert(params.conversation_id, LocalConversationState::default());

    let events = finalize_stream_state(
        &params,
        StreamingResponsesAccumulator::default(),
        "request-id",
        Some(ResponsesApiResponse {
            output: vec![
                ResponsesOutputItem {
                    id: Some("rs_1".to_string()),
                    item_type: "reasoning".to_string(),
                    role: None,
                    content: vec![],
                    summary: vec![ResponsesReasoningSummaryPart {
                        item_type: "summary_text".to_string(),
                        text: Some("Checked the repository wiring first.".to_string()),
                    }],
                    encrypted_content: Some("enc_reasoning_payload".to_string()),
                    name: None,
                    call_id: None,
                    arguments: None,
                },
                ResponsesOutputItem {
                    id: Some("msg_1".to_string()),
                    item_type: "message".to_string(),
                    role: Some("assistant".to_string()),
                    content: vec![ResponsesContentItem {
                        item_type: "output_text".to_string(),
                        text: Some("Done".to_string()),
                    }],
                    summary: vec![],
                    encrypted_content: None,
                    name: None,
                    call_id: None,
                    arguments: None,
                },
            ],
        }),
    )
    .expect("completed payload should finalize");

    assert_eq!(events.len(), 2);
    let add_messages_event = events[0].as_ref().expect("add-messages event should be ok");
    let Some(api::response_event::Type::ClientActions(actions)) = &add_messages_event.r#type else {
        panic!("expected client actions");
    };
    let Some(api::client_action::Action::AddMessagesToTask(add_messages)) =
        actions.actions[0].action.as_ref()
    else {
        panic!("expected add-messages action");
    };
    let reasoning_message = add_messages
        .messages
        .iter()
        .find_map(|message| match &message.message {
            Some(api::message::Message::AgentReasoning(reasoning)) => Some(reasoning),
            _ => None,
        })
        .expect("expected a backfilled reasoning message");
    assert_eq!(
        reasoning_message.reasoning,
        "Checked the repository wiring first."
    );
    assert!(reasoning_message.finished_duration.is_some());
    let stored_items = conversation_state_store()
        .lock()
        .get(&params.conversation_id)
        .cloned()
        .expect("conversation state should exist")
        .items;
    let reasoning_item = stored_items
        .iter()
        .find(|item| item["type"] == "reasoning")
        .expect("expected stored reasoning history item");
    assert_eq!(
        reasoning_item["encrypted_content"],
        serde_json::json!("enc_reasoning_payload")
    );
    assert_eq!(
        reasoning_item["summary"],
        serde_json::json!([
            {
                "type": "summary_text",
                "text": "Checked the repository wiring first."
            }
        ])
    );
    assert!(
        reasoning_item.get("id").is_none(),
        "replayed reasoning items should not send ids back"
    );

    conversation_state_store()
        .lock()
        .remove(&params.conversation_id);
}

/// Verifies that reasoning encrypted content captured from `output_item.done` survives even if `response.completed` omits the reasoning item.
#[test]
fn finalize_stream_state_preserves_reasoning_history_from_output_item_done() {
    let params = request_params_for_local_backend_tests();
    let task_id = TaskId::new("task-id".to_string());
    conversation_state_store()
        .lock()
        .insert(params.conversation_id, LocalConversationState::default());
    let mut accumulator = StreamingResponsesAccumulator::default();

    let result = handle_responses_stream_message(
        &params,
        &task_id,
        "request-id",
        "response.output_item.done",
        r#"{"item":{"id":"rs_stream_1","type":"reasoning","content":[],"summary":[{"type":"summary_text","text":"Need to keep this reasoning context."}],"encrypted_content":"enc_from_output_item_done"}}"#,
        &mut accumulator,
    )
    .expect("reasoning output_item.done should parse");
    assert_eq!(result.events.len(), 1);

    finalize_stream_state(
        &params,
        accumulator,
        "request-id",
        Some(ResponsesApiResponse {
            output: vec![ResponsesOutputItem {
                id: Some("msg_1".to_string()),
                item_type: "message".to_string(),
                role: Some("assistant".to_string()),
                content: vec![ResponsesContentItem {
                    item_type: "output_text".to_string(),
                    text: Some("Done".to_string()),
                }],
                summary: vec![],
                encrypted_content: None,
                name: None,
                call_id: None,
                arguments: None,
            }],
        }),
    )
    .expect("completed payload should finalize");

    let stored_items = conversation_state_store()
        .lock()
        .get(&params.conversation_id)
        .cloned()
        .expect("conversation state should exist")
        .items;
    let reasoning_item = stored_items
        .iter()
        .find(|item| item["type"] == "reasoning")
        .expect("expected stored reasoning history item");
    assert_eq!(
        reasoning_item["encrypted_content"],
        serde_json::json!("enc_from_output_item_done")
    );
    assert_eq!(
        reasoning_item["summary"],
        serde_json::json!([
            {
                "type": "summary_text",
                "text": "Need to keep this reasoning context."
            }
        ])
    );
    assert!(
        reasoning_item.get("id").is_none(),
        "replayed reasoning items should not send ids back"
    );

    conversation_state_store()
        .lock()
        .remove(&params.conversation_id);
}

/// Verifies that replayed reasoning items preserve an explicit empty summary array.
#[test]
fn finalize_stream_state_preserves_empty_reasoning_summary_array() {
    let params = request_params_for_local_backend_tests();
    conversation_state_store()
        .lock()
        .insert(params.conversation_id, LocalConversationState::default());

    finalize_stream_state(
        &params,
        StreamingResponsesAccumulator::default(),
        "request-id",
        Some(ResponsesApiResponse {
            output: vec![ResponsesOutputItem {
                id: Some("rs_empty_summary".to_string()),
                item_type: "reasoning".to_string(),
                role: None,
                content: vec![],
                summary: vec![],
                encrypted_content: Some("enc_empty_summary".to_string()),
                name: None,
                call_id: None,
                arguments: None,
            }],
        }),
    )
    .expect("completed payload should finalize");

    let stored_items = conversation_state_store()
        .lock()
        .get(&params.conversation_id)
        .cloned()
        .expect("conversation state should exist")
        .items;
    let reasoning_item = stored_items
        .iter()
        .find(|item| item["type"] == "reasoning")
        .expect("expected stored reasoning history item");
    assert_eq!(reasoning_item["summary"], serde_json::json!([]));

    conversation_state_store()
        .lock()
        .remove(&params.conversation_id);
}

/// Verifies that an empty completed payload does not fail if text was already streamed.
#[test]
fn finalize_stream_state_accepts_empty_completed_payload_after_streamed_text() {
    let params = request_params_for_local_backend_tests();
    conversation_state_store()
        .lock()
        .insert(params.conversation_id, LocalConversationState::default());
    let mut accumulator = StreamingResponsesAccumulator::default();
    accumulator.text_messages_by_item_id.insert(
        "msg_1".to_string(),
        StreamingTextMessageState {
            message_id: "message-id".to_string(),
            text: "hello".to_string(),
        },
    );
    accumulator.emitted_text_item_ids.push("msg_1".to_string());

    let events = finalize_stream_state(
        &params,
        accumulator,
        "request-id",
        Some(ResponsesApiResponse { output: vec![] }),
    )
    .expect("empty completed payload should be accepted after streamed text");

    assert_eq!(events.len(), 1);
    let stored_items = conversation_state_store()
        .lock()
        .get(&params.conversation_id)
        .cloned()
        .expect("conversation state should exist")
        .items;
    assert_eq!(stored_items.len(), 1);
    assert_eq!(stored_items[0]["content"][0]["text"], "hello");

    conversation_state_store()
        .lock()
        .remove(&params.conversation_id);
}

/// Verifies that streamed function calls wait for output_item metadata when call_id is missing.
#[test]
fn streamed_function_calls_use_output_item_done_to_get_real_call_id() {
    let mut accumulator = StreamingResponsesAccumulator::default();
    accumulator
        .function_calls_by_call_id
        .entry("item_123".to_string())
        .or_default()
        .arguments
        .push_str(r#"{"path":"."}"#);

    let function_call = finalize_streamed_function_call(
        &mut accumulator,
        ResponsesFunctionCallArgumentsDoneEvent {
            call_id: None,
            item_id: Some("item_123".to_string()),
            name: Some("file_glob".to_string()),
            arguments: String::new(),
        },
    )
    .expect("missing call_id should not error");
    assert!(function_call.is_none());

    let function_call = handle_streamed_output_item_done(
        &mut accumulator,
        ResponsesOutputItem {
            id: Some("item_123".to_string()),
            item_type: "function_call".to_string(),
            role: None,
            content: vec![],
            summary: vec![],
            encrypted_content: None,
            name: Some("file_glob".to_string()),
            call_id: Some("call_real_123".to_string()),
            arguments: Some(r#"{"path":"."}"#.to_string()),
        },
    )
    .expect("output_item.done should not error")
    .expect("output_item.done should emit the completed tool call");

    assert_eq!(function_call.call_id, "call_real_123");
    assert_eq!(function_call.name, "file_glob");
    assert_eq!(function_call.arguments["path"], ".");
}

/// Verifies that streamed function calls tolerate missing names until output_item metadata arrives.
#[test]
fn streamed_function_calls_wait_for_output_item_name() {
    let mut accumulator = StreamingResponsesAccumulator::default();
    accumulator
        .function_calls_by_call_id
        .entry("item_456".to_string())
        .or_default()
        .arguments
        .push_str(r#"{"path":"."}"#);

    let function_call = finalize_streamed_function_call(
        &mut accumulator,
        ResponsesFunctionCallArgumentsDoneEvent {
            call_id: None,
            item_id: Some("item_456".to_string()),
            name: None,
            arguments: String::new(),
        },
    )
    .expect("missing name should not error");
    assert!(function_call.is_none());

    let function_call = handle_streamed_output_item_done(
        &mut accumulator,
        ResponsesOutputItem {
            id: Some("item_456".to_string()),
            item_type: "function_call".to_string(),
            role: None,
            content: vec![],
            summary: vec![],
            encrypted_content: None,
            name: Some("file_glob".to_string()),
            call_id: Some("call_real_456".to_string()),
            arguments: Some(r#"{"path":"."}"#.to_string()),
        },
    )
    .expect("output_item.done should not error")
    .expect("output_item.done should emit the completed tool call");

    assert_eq!(function_call.call_id, "call_real_456");
    assert_eq!(function_call.name, "file_glob");
    assert_eq!(function_call.arguments["path"], ".");
}

/// Verifies that streamed function calls can still fall back to item_id from output_item metadata.
#[test]
fn streamed_function_calls_fall_back_to_item_id_when_output_item_has_no_call_id() {
    let mut accumulator = StreamingResponsesAccumulator::default();
    accumulator
        .function_calls_by_call_id
        .entry("item_123".to_string())
        .or_default()
        .arguments
        .push_str(r#"{"path":"."}"#);

    let function_call = handle_streamed_output_item_done(
        &mut accumulator,
        ResponsesOutputItem {
            id: Some("item_123".to_string()),
            item_type: "function_call".to_string(),
            role: None,
            content: vec![],
            summary: vec![],
            encrypted_content: None,
            name: Some("file_glob".to_string()),
            call_id: None,
            arguments: Some(r#"{"path":"."}"#.to_string()),
        },
    )
    .expect("output_item.done should not error")
    .expect("output_item.done should emit the completed tool call");

    assert_eq!(function_call.call_id, "item_123");
    assert_eq!(function_call.name, "file_glob");
    assert_eq!(function_call.arguments["path"], ".");
}

/// Verifies that tools with optional parameters are exposed with strict mode disabled.
#[test]
fn tools_with_optional_fields_disable_strict_mode() {
    let payload = build_tools_payload(&request_params_for_local_backend_tests());
    let file_glob_schema = payload
        .iter()
        .find(|tool| tool["name"] == "file_glob")
        .expect("file_glob schema should be present");
    let read_documents_schema = payload
        .iter()
        .find(|tool| tool["name"] == "read_documents")
        .expect("read_documents schema should be present");

    assert_eq!(file_glob_schema["strict"], false);
    assert_eq!(read_documents_schema["strict"], false);
}

/// Verifies that read_files parsing accepts the proto-shaped line_ranges field.
#[test]
fn parse_read_file_supports_line_ranges() {
    let parsed = parse_read_file(&serde_json::json!({
        "name": "README.md",
        "line_ranges": [
            { "start": 1, "end": 5 }
        ]
    }))
    .expect("read_files payload should parse");

    assert_eq!(parsed.name, "README.md");
    assert_eq!(parsed.line_ranges.len(), 1);
    assert_eq!(parsed.line_ranges[0].start, 1);
    assert_eq!(parsed.line_ranges[0].end, 5);
}

/// Verifies that the local backend upgrades the optimistic task on the first turn.
#[test]
fn should_emit_create_task_for_first_turn() {
    let mut params = request_params_for_local_backend_tests();
    params.conversation_token = None;
    assert!(should_emit_create_task(&params));
}

/// Verifies that follow-up turns reuse the existing task without re-emitting CreateTask.
#[test]
fn should_not_emit_create_task_for_follow_up_turn() {
    let mut params = request_params_for_local_backend_tests();
    params.conversation_token = Some(crate::ai::agent::api::ServerConversationToken::new(
        "conversation-token".to_string(),
    ));
    assert!(!should_emit_create_task(&params));
}

/// Verifies that user queries and tool outputs map to Responses input items.
#[test]
fn convert_inputs_to_response_items_supports_user_query_and_action_result() {
    let inputs = vec![
        AIAgentInput::UserQuery {
            query: "hello".to_string(),
            context: std::sync::Arc::new([AIAgentContext::SelectedText("world".to_string())]),
            static_query_type: None,
            referenced_attachments: std::collections::HashMap::new(),
            user_query_mode: crate::ai::agent::UserQueryMode::Normal,
            running_command: None,
            intended_agent: None,
        },
        AIAgentInput::ActionResult {
            result: AIAgentActionResult {
                id: "call_123".to_string().into(),
                task_id: TaskId::new("task".to_string()),
                result: AIAgentActionResultType::ReadFiles(ReadFilesResult::Error(
                    "missing".to_string(),
                )),
            },
            context: std::sync::Arc::new([]),
        },
    ];

    let items = convert_inputs_to_response_items(&inputs).expect("inputs should convert");
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["role"], "user");
    assert_eq!(items[1]["type"], "function_call_output");
}

/// Verifies that local request inputs are mirrored into persisted task messages for restore.
#[test]
fn convert_inputs_to_task_messages_supports_user_query_and_action_result() {
    let task_id = TaskId::new("task".to_string());
    let request_id = "request-123";
    let inputs = vec![
        AIAgentInput::UserQuery {
            query: "hello".to_string(),
            context: std::sync::Arc::new([AIAgentContext::SelectedText("world".to_string())]),
            static_query_type: None,
            referenced_attachments: std::collections::HashMap::new(),
            user_query_mode: crate::ai::agent::UserQueryMode::Normal,
            running_command: None,
            intended_agent: None,
        },
        AIAgentInput::ActionResult {
            result: AIAgentActionResult {
                id: "call_123".to_string().into(),
                task_id: task_id.clone(),
                result: AIAgentActionResultType::ReadFiles(ReadFilesResult::Error(
                    "missing".to_string(),
                )),
            },
            context: std::sync::Arc::new([]),
        },
    ];

    let messages = convert_inputs_to_task_messages(&inputs, &task_id, request_id)
        .expect("inputs should convert into task messages");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].task_id, task_id.to_string());
    assert_eq!(messages[0].request_id, request_id);
    assert!(matches!(
        messages[0].message,
        Some(api::message::Message::UserQuery(_))
    ));
    assert!(matches!(
        messages[1].message,
        Some(api::message::Message::ToolCallResult(_))
    ));
}

/// Verifies that persisted task history seeds local Responses input items for conversation resumes.
#[test]
fn task_history_response_items_restore_prior_messages() {
    let mut params = request_params_for_local_backend_tests();
    params.tasks = vec![api::Task {
        id: params
            .target_task_id
            .as_ref()
            .expect("task id should exist")
            .to_string(),
        messages: vec![
            api::Message {
                id: "message-user".to_string(),
                task_id: "task-id".to_string(),
                server_message_data: String::new(),
                citations: vec![],
                request_id: "request-1".to_string(),
                timestamp: None,
                message: Some(api::message::Message::UserQuery(api::message::UserQuery {
                    query: "prior question".to_string(),
                    context: None,
                    referenced_attachments: std::collections::HashMap::new(),
                    mode: None,
                    intended_agent: Default::default(),
                })),
            },
            api::Message {
                id: "message-tool".to_string(),
                task_id: "task-id".to_string(),
                server_message_data: String::new(),
                citations: vec![],
                request_id: "request-1".to_string(),
                timestamp: None,
                message: Some(api::message::Message::ToolCall(api::message::ToolCall {
                    tool_call_id: "call_1".to_string(),
                    tool: Some(api::message::tool_call::Tool::ReadFiles(
                        api::message::tool_call::ReadFiles {
                            files: vec![api::message::tool_call::read_files::File {
                                name: "README.md".to_string(),
                                line_ranges: vec![],
                            }],
                        },
                    )),
                })),
            },
            api::Message {
                id: "message-result".to_string(),
                task_id: "task-id".to_string(),
                server_message_data: String::new(),
                citations: vec![],
                request_id: "request-1".to_string(),
                timestamp: None,
                message: Some(api::message::Message::ToolCallResult(
                    api::message::ToolCallResult {
                        tool_call_id: "call_1".to_string(),
                        context: None,
                        result: Some(api::message::tool_call_result::Result::ReadFiles(
                            api::ReadFilesResult {
                                result: Some(api::read_files_result::Result::Error(
                                    api::read_files_result::Error {
                                        message: "missing".to_string(),
                                    },
                                )),
                            },
                        )),
                    },
                )),
            },
            api::Message {
                id: "message-output".to_string(),
                task_id: "task-id".to_string(),
                server_message_data: String::new(),
                citations: vec![],
                request_id: "request-1".to_string(),
                timestamp: None,
                message: Some(api::message::Message::AgentOutput(
                    api::message::AgentOutput {
                        text: "prior answer".to_string(),
                    },
                )),
            },
        ],
        dependencies: None,
        description: String::new(),
        summary: String::new(),
        server_data: String::new(),
    }];

    let items = task_history_response_items(&params).expect("task history should convert");
    assert_eq!(items.len(), 4);
    assert_eq!(items[0]["role"], "user");
    assert_eq!(items[1]["type"], "function_call");
    assert_eq!(items[1]["name"], "read_files");
    assert_eq!(items[2]["type"], "function_call_output");
    assert_eq!(items[2]["call_id"], "call_1");
    assert_eq!(items[3]["role"], "assistant");
    assert_eq!(items[3]["content"][0]["text"], "prior answer");
}

/// Verifies that transient provider status codes are retried by the local backend.
#[test]
fn local_backend_retries_transient_provider_errors() {
    let transient_statuses = [408_u16, 429_u16, 500_u16, 503_u16];

    for status in transient_statuses {
        let error = anyhow::anyhow!(ProviderError::new(status, "retry me".to_string()));
        assert!(
            is_retryable_local_backend_error(&error),
            "status {status} should be retryable"
        );
        assert!(should_retry_local_backend_error(&error, 1, false));
    }
}

/// Verifies that permanent provider status codes fail fast instead of retrying.
#[test]
fn local_backend_does_not_retry_permanent_provider_errors() {
    let permanent_statuses = [400_u16, 401_u16, 403_u16, 404_u16];

    for status in permanent_statuses {
        let error = anyhow::anyhow!(ProviderError::new(status, "do not retry".to_string()));
        assert!(
            !is_retryable_local_backend_error(&error),
            "status {status} should not be retryable"
        );
        assert!(!should_retry_local_backend_error(&error, 1, false));
    }
}

/// Verifies that retries stop once output has started or the attempt budget is exhausted.
#[test]
fn local_backend_retry_stops_after_output_or_max_attempts() {
    let error = anyhow::anyhow!("connection reset");

    assert!(should_retry_local_backend_error(&error, 1, false));
    assert!(!should_retry_local_backend_error(
        &error,
        MAX_ATTEMPTS,
        false
    ));
    assert!(!should_retry_local_backend_error(&error, 1, true));
}
