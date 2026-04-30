//! Request-building helpers for the local OpenAI-compatible Responses backend.

use std::collections::{HashMap, HashSet};

use anyhow::anyhow;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use prost::Message as _;
use serde_json::{json, Value};
use uuid::Uuid;
use warp_multi_agent_api as api;

use crate::ai::agent::api::{convert_to::convert_input, user_inputs_from_messages};
use crate::ai::agent::{AIAgentContext, AIAgentInput, MCPContext, MCPServer};
use crate::server::server_api::ServerApi;

use super::tool_schemas::built_in_tool_schema;
use super::types::{
    ParsedFunctionCall, ResponsesErrorEnvelope, ResponsesReasoningConfig, ResponsesRequestBody,
};
use super::{
    build_local_openai_system_prompt, conversation_state_store, ProviderError, RequestParams,
};
use crate::ai::agent::api::r#impl::get_supported_tools;

/// Prepared request data that can be reused across bounded local backend retries.
pub(super) struct PreparedLocalResponsesRequest {
    pub(super) api_key: String,
    pub(super) endpoint: String,
    pub(super) request_body: ResponsesRequestBody,
}

/// Prepares a local Responses request after recording the new inputs in conversation state once.
pub(super) fn prepare_local_responses_request(
    params: &RequestParams,
) -> anyhow::Result<PreparedLocalResponsesRequest> {
    let api_key = params
        .local_openai_api_key
        .as_ref()
        .filter(|value| !value.trim().is_empty())
        .cloned()
        .ok_or_else(|| anyhow!("OpenAI API key is required for the local OpenAI backend"))?;
    let base_url = params
        .local_openai_base_url
        .as_ref()
        .filter(|value| !value.trim().is_empty())
        .cloned()
        .ok_or_else(|| anyhow!("OpenAI base URL is required for the local OpenAI backend"))?;
    let endpoint = normalize_responses_endpoint(&base_url);

    ensure_conversation_state_initialized(params)?;
    let new_input_items = convert_inputs_to_response_items(&params.input)?;
    {
        let mut state_store = conversation_state_store().lock();
        let state = state_store.entry(params.conversation_id).or_default();
        state.items.extend(new_input_items.clone());
    }

    let request_body = {
        let state_store = conversation_state_store().lock();
        let state = state_store
            .get(&params.conversation_id)
            .cloned()
            .unwrap_or_default();
        let (normalized_model, reasoning) =
            normalize_openai_model_and_reasoning(&params.model.to_string());
        ResponsesRequestBody {
            instructions: build_local_openai_system_prompt(&normalized_model),
            model: normalized_model,
            reasoning,
            input: state.items,
            tools: build_tools_payload(params),
            tool_choice: "auto",
            parallel_tool_calls: true,
            store: false,
            stream: true,
        }
    };

    Ok(PreparedLocalResponsesRequest {
        api_key,
        endpoint,
        request_body,
    })
}

/// Opens a local Responses event stream from a prepared request payload.
pub(super) async fn open_local_responses_eventsource(
    server_api: &ServerApi,
    prepared_request: &PreparedLocalResponsesRequest,
) -> anyhow::Result<http_client::EventSourceStream> {
    Ok(server_api
        .http_client()
        .post(prepared_request.endpoint.clone())
        .bearer_auth(prepared_request.api_key.clone())
        .json(&prepared_request.request_body)
        .eventsource())
}

/// Converts an SSE stream error into the closest local backend error shape we can expose.
pub(super) async fn stream_error_to_anyhow(err: reqwest_eventsource::Error) -> anyhow::Error {
    match err {
        reqwest_eventsource::Error::InvalidStatusCode(status, response) => {
            let response_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown provider error".to_string());
            let provider_message = serde_json::from_str::<ResponsesErrorEnvelope>(&response_text)
                .map(|body| body.error.message)
                .unwrap_or(response_text);
            anyhow!(ProviderError::new(status.as_u16(), provider_message))
        }
        other => anyhow!("Failed to read local OpenAI Responses stream: {other}"),
    }
}

/// Converts the current request inputs into Responses API conversation items.
pub(super) fn convert_inputs_to_response_items(
    inputs: &[AIAgentInput],
) -> anyhow::Result<Vec<Value>> {
    let mut items = Vec::new();
    for input in inputs {
        match input {
            AIAgentInput::UserQuery {
                query,
                context,
                referenced_attachments,
                ..
            } => items.push(user_message_item(
                query,
                context,
                Some(referenced_attachments),
            )),
            AIAgentInput::ActionResult { result, .. } => {
                items.push(function_call_output_item(
                    result.id.to_string(),
                    result.to_string(),
                ));
            }
            AIAgentInput::AutoCodeDiffQuery { query, context }
            | AIAgentInput::CreateNewProject { query, context } => {
                items.push(user_message_item(query, context, None));
            }
            AIAgentInput::CloneRepository {
                clone_repo_url,
                context,
            } => {
                let query = format!("Clone {}", clone_repo_url.clone().into_url());
                items.push(user_message_item(&query, context, None));
            }
            AIAgentInput::FetchReviewComments { repo_path, context } => {
                let query = format!("Fetch review comments for {repo_path}");
                items.push(user_message_item(&query, context, None));
            }
            unsupported => {
                return Err(anyhow!(
                    "Local OpenAI backend does not support {:?} inputs yet",
                    unsupported
                ));
            }
        }
    }
    Ok(items)
}

/// Converts request inputs into synthetic task messages so local conversations survive restore.
pub(super) fn convert_inputs_to_task_messages(
    inputs: &[AIAgentInput],
    task_id: &crate::ai::agent::task::TaskId,
    request_id: &str,
) -> anyhow::Result<Vec<api::Message>> {
    inputs
        .iter()
        .cloned()
        .map(|input| convert_input_to_task_message(input, task_id, request_id))
        .collect()
}

/// Converts one persisted local request input into the matching task message representation.
fn convert_input_to_task_message(
    input: AIAgentInput,
    task_id: &crate::ai::agent::task::TaskId,
    request_id: &str,
) -> anyhow::Result<api::Message> {
    let converted_input = convert_input(vec![input]).map_err(|error| {
        anyhow!("Failed to convert local request input into task message: {error}")
    })?;
    let context = converted_input.context;
    let Some(api::request::input::Type::UserInputs(user_inputs)) = converted_input.r#type else {
        return Err(anyhow!(
            "Local request input did not convert into user_inputs for task persistence"
        ));
    };
    let Some(user_input) = user_inputs
        .inputs
        .into_iter()
        .next()
        .and_then(|input| input.input)
    else {
        return Err(anyhow!(
            "Local request input converted into an empty user_inputs payload"
        ));
    };

    let message = match user_input {
        api::request::input::user_inputs::user_input::Input::UserQuery(user_query) => {
            api::message::Message::UserQuery(api::message::UserQuery {
                query: user_query.query,
                context,
                referenced_attachments: user_query.referenced_attachments,
                mode: user_query.mode,
                intended_agent: user_query.intended_agent,
            })
        }
        api::request::input::user_inputs::user_input::Input::ToolCallResult(tool_call_result) => {
            api::message::Message::ToolCallResult(request_tool_call_result_to_message(
                tool_call_result,
                context,
            )?)
        }
        unsupported => {
            return Err(anyhow!(
                "Local request input converted into unsupported persisted input variant: {:?}",
                unsupported
            ));
        }
    };

    Ok(api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.to_string(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(message),
        request_id: request_id.to_string(),
        timestamp: None,
    })
}

/// Transcodes a request-side tool-call result into the persisted task-message shape.
fn request_tool_call_result_to_message(
    tool_call_result: api::request::input::ToolCallResult,
    context: Option<api::InputContext>,
) -> anyhow::Result<api::message::ToolCallResult> {
    // These protobuf messages intentionally share the same wire shape for `tool_call_id`
    // and `result`, so we can transcode instead of hand-mapping every result variant.
    let mut message_tool_call_result = api::message::ToolCallResult::decode(
        tool_call_result.encode_to_vec().as_slice(),
    )
    .map_err(|error| {
        anyhow!("Failed to transcode request tool_call_result into persisted task message: {error}")
    })?;
    message_tool_call_result.context = context;
    Ok(message_tool_call_result)
}

/// Creates a user message item including serialized Warp-specific context.
fn user_message_item(
    query: &str,
    context: &[AIAgentContext],
    referenced_attachments: Option<&HashMap<String, crate::ai::agent::AIAgentAttachment>>,
) -> Value {
    let mut parts = vec![json!({
        "type": "input_text",
        "text": render_user_query_with_context(query, context, referenced_attachments),
    })];
    for context_item in context {
        if let AIAgentContext::Image(image) = context_item {
            parts.push(json!({
                "type": "input_image",
                "image_url": format!("data:{};base64,{}", image.mime_type, image.data),
            }));
        }
    }

    json!({
        "type": "message",
        "role": "user",
        "content": parts,
    })
}

/// Renders a user query with the most important attached Warp context inline as text.
fn render_user_query_with_context(
    query: &str,
    context: &[AIAgentContext],
    referenced_attachments: Option<&HashMap<String, crate::ai::agent::AIAgentAttachment>>,
) -> String {
    let mut rendered = query.to_string();
    let context_text = render_context_block(context, referenced_attachments);
    if !context_text.is_empty() {
        rendered.push_str("\n\nContext:\n");
        rendered.push_str(&context_text);
    }
    rendered
}

/// Serializes a compact text representation of the Warp context available to the request.
fn render_context_block(
    context: &[AIAgentContext],
    referenced_attachments: Option<&HashMap<String, crate::ai::agent::AIAgentAttachment>>,
) -> String {
    let mut lines = Vec::new();
    for item in context {
        match item {
            AIAgentContext::Directory {
                pwd,
                home_dir,
                are_file_symbols_indexed,
            } => {
                if let Some(pwd) = pwd {
                    lines.push(format!("Working directory: {pwd}"));
                }
                if let Some(home_dir) = home_dir {
                    lines.push(format!("Home directory: {home_dir}"));
                }
                lines.push(format!(
                    "Codebase index available: {}",
                    if *are_file_symbols_indexed {
                        "yes"
                    } else {
                        "no"
                    }
                ));
            }
            AIAgentContext::SelectedText(text) => {
                lines.push(format!("Selected text:\n{text}"));
            }
            AIAgentContext::ExecutionEnvironment(env) => {
                lines.push(format!(
                    "Execution environment: os={:?}/{:?}, shell={} {:?}",
                    env.os.category, env.os.distribution, env.shell_name, env.shell_version
                ));
            }
            AIAgentContext::CurrentTime { current_time } => {
                lines.push(format!("Current local time: {current_time}"));
            }
            AIAgentContext::Codebase { path, name } => {
                lines.push(format!("Indexed codebase: {name} at {path}"));
            }
            AIAgentContext::ProjectRules {
                root_path,
                active_rules,
                additional_rule_paths,
            } => {
                lines.push(format!("Project rules root: {root_path}"));
                if !active_rules.is_empty() {
                    lines.push(format!("Active rule files: {}", active_rules.len()));
                }
                if !additional_rule_paths.is_empty() {
                    lines.push(format!(
                        "Additional rule paths: {}",
                        additional_rule_paths.join(", ")
                    ));
                }
            }
            AIAgentContext::File(file) => {
                lines.push(format!("Attached file context: {}", file.file_name));
            }
            AIAgentContext::Git { head, branch } => {
                lines.push(format!("Git HEAD: {head}"));
                if let Some(branch) = branch {
                    lines.push(format!("Git branch: {branch}"));
                }
            }
            AIAgentContext::Skills { skills } => {
                lines.push(format!("Available skills: {}", skills.len()));
            }
            AIAgentContext::Block(block) => {
                lines.push(format!("Shell block command: {}", block.command));
                if !block.output.is_empty() {
                    lines.push(format!("Shell block output:\n{}", block.output));
                }
            }
            AIAgentContext::Image(image) => {
                lines.push(format!("Attached image: {}", image.file_name));
            }
        }
    }

    if let Some(attachments) = referenced_attachments {
        for (name, attachment) in attachments {
            match attachment {
                crate::ai::agent::AIAgentAttachment::PlainText(text) => {
                    lines.push(format!("Referenced attachment {name}:\n{text}"));
                }
                crate::ai::agent::AIAgentAttachment::DocumentContent { content, .. } => {
                    lines.push(format!("Referenced document {name}:\n{content}"));
                }
                crate::ai::agent::AIAgentAttachment::Block(block) => {
                    lines.push(format!("Referenced block {name}: {}", block.command));
                }
                _ => {}
            }
        }
    }

    lines.join("\n")
}

/// Converts a successful assistant text output into a conversation history item.
pub(super) fn assistant_output_item(text: &str) -> Value {
    json!({
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": text,
        }],
    })
}

/// Converts a function call into a history item that can be replayed on later turns.
pub(super) fn function_call_history_item(function_call: &ParsedFunctionCall) -> Value {
    json!({
        "type": "function_call",
        "call_id": function_call.call_id,
        "name": function_call.name,
        "arguments": function_call.arguments.to_string(),
    })
}

/// Converts a completed tool result into a Responses `function_call_output` item.
fn function_call_output_item(call_id: String, output: String) -> Value {
    json!({
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    })
}

/// Builds the list of tool definitions exposed to the local Responses model.
pub(super) fn build_tools_payload(params: &RequestParams) -> Vec<Value> {
    let requested_tools = requested_local_tool_types(params);
    let supports_mcp_tools = requested_tools.contains(&api::ToolType::CallMcpTool);
    let mut tools = requested_tools
        .into_iter()
        .filter_map(built_in_tool_schema)
        .collect::<Vec<_>>();
    if supports_mcp_tools {
        tools.extend(mcp_tool_schemas(params.mcp_context.as_ref()));
    }
    tools
}

/// Returns the tool set the local backend should expose after applying request overrides.
fn requested_local_tool_types(params: &RequestParams) -> Vec<api::ToolType> {
    params
        .supported_tools_override
        .clone()
        .unwrap_or_else(|| get_supported_tools(params))
}

/// Converts MCP tool metadata already present in the request context into OpenAI function schemas.
fn mcp_tool_schemas(mcp_context: Option<&MCPContext>) -> Vec<Value> {
    let Some(mcp_context) = mcp_context else {
        return Vec::new();
    };

    if !mcp_context.servers.is_empty() {
        return mcp_context
            .servers
            .iter()
            .flat_map(mcp_server_tool_schemas)
            .collect();
    }

    #[allow(deprecated)]
    mcp_context
        .tools
        .iter()
        .filter_map(|tool| mcp_tool_schema(None, tool))
        .collect()
}

/// Converts every tool for a grouped MCP server into OpenAI function schemas.
fn mcp_server_tool_schemas(server: &MCPServer) -> Vec<Value> {
    server
        .tools
        .iter()
        .filter_map(|tool| mcp_tool_schema(Some(server.id.as_str()), tool))
        .collect()
}

/// Converts a single MCP tool into an OpenAI function schema using the server-provided JSON Schema.
pub(super) fn mcp_tool_schema(server_id: Option<&str>, tool: &rmcp::model::Tool) -> Option<Value> {
    let input_schema = Value::Object(tool.input_schema.as_ref().clone());
    let description = tool
        .description
        .as_deref()
        .map(str::to_string)
        .or_else(|| tool.title.clone())
        .unwrap_or_else(|| format!("Call MCP tool '{}'.", tool.name));

    Some(json!({
        "type": "function",
        "name": mcp_function_name(server_id, tool.name.as_ref()),
        "description": description,
        "parameters": input_schema,
        "strict": false,
    }))
}

/// Encodes a unique OpenAI function name for an MCP tool.
fn mcp_function_name(server_id: Option<&str>, tool_name: &str) -> String {
    let encoded_server_id = URL_SAFE_NO_PAD.encode(server_id.unwrap_or_default());
    let encoded_tool_name = URL_SAFE_NO_PAD.encode(tool_name);
    format!("warp_mcp_tool__{encoded_server_id}__{encoded_tool_name}")
}

/// Decodes a synthetic MCP tool function name back into its server ID and tool name.
pub(super) fn parse_mcp_function_name(name: &str) -> Option<(Option<String>, String)> {
    let suffix = name.strip_prefix("warp_mcp_tool__")?;
    let (encoded_server_id, encoded_tool_name) = suffix.split_once("__")?;
    let server_id = URL_SAFE_NO_PAD.decode(encoded_server_id).ok()?;
    let tool_name = URL_SAFE_NO_PAD.decode(encoded_tool_name).ok()?;

    let server_id = String::from_utf8(server_id).ok()?;
    let tool_name = String::from_utf8(tool_name).ok()?;

    Some(((!server_id.is_empty()).then_some(server_id), tool_name))
}

/// Normalizes the user-provided base URL into the exact `/v1/responses` endpoint.
pub(super) fn normalize_responses_endpoint(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        format!("{trimmed}/responses")
    } else {
        format!("{trimmed}/v1/responses")
    }
}

/// Normalizes OpenAI-compatible model IDs and extracts a Responses reasoning effort when present.
pub(super) fn normalize_openai_model_and_reasoning(
    model_id: &str,
) -> (String, Option<ResponsesReasoningConfig>) {
    let (base_model_id, reasoning) = split_openai_reasoning_suffix(model_id);
    if let Some(normalized_model) = normalize_openai_model_base(base_model_id) {
        let reasoning = reasoning
            .map(enable_reasoning_summary)
            .or_else(|| build_reasoning_summary_config(&normalized_model));
        return (normalized_model, reasoning);
    }

    (model_id.to_string(), None)
}

/// Splits a supported Responses reasoning effort suffix from the provided model ID.
fn split_openai_reasoning_suffix(model_id: &str) -> (&str, Option<ResponsesReasoningConfig>) {
    let Some((base_model_id, effort)) = model_id.rsplit_once('-') else {
        return (model_id, None);
    };
    if is_supported_reasoning_effort(effort) {
        return (
            base_model_id,
            Some(ResponsesReasoningConfig {
                effort: Some(effort.to_string()),
                summary: None,
            }),
        );
    }

    (model_id, None)
}

/// Enables reasoning summaries on a Responses reasoning config so Warp can render thinking blocks.
fn enable_reasoning_summary(mut reasoning: ResponsesReasoningConfig) -> ResponsesReasoningConfig {
    reasoning.summary = Some("auto".to_string());
    reasoning
}

/// Builds a summary-only reasoning config for supported OpenAI reasoning models.
fn build_reasoning_summary_config(model_id: &str) -> Option<ResponsesReasoningConfig> {
    should_request_reasoning_summary(model_id).then(|| ResponsesReasoningConfig {
        effort: None,
        summary: Some("auto".to_string()),
    })
}

/// Returns whether Warp should opt into reasoning summaries for the given normalized model.
fn should_request_reasoning_summary(model_id: &str) -> bool {
    model_id.starts_with("gpt-5")
        || model_id.starts_with("gpt-oss")
        || matches!(model_id.split('-').next(), Some("o1" | "o3" | "o4"))
}

/// Normalizes the base model ID into the exact Responses API model name when we recognize it.
fn normalize_openai_model_base(model_id: &str) -> Option<String> {
    let parts = model_id.split('-').collect::<Vec<_>>();
    if parts.len() == 3
        && parts[0] == "gpt"
        && parts[1].chars().all(|c| c.is_ascii_digit())
        && parts[2].chars().all(|c| c.is_ascii_digit())
    {
        return Some(format!("gpt-{}.{}", parts[1], parts[2]));
    }

    if parts.len() == 4
        && parts[0] == "gpt"
        && parts[1].chars().all(|c| c.is_ascii_digit())
        && parts[2].chars().all(|c| c.is_ascii_digit())
        && parts[3] == "codex"
    {
        return Some(format!("gpt-{}.{}-codex", parts[1], parts[2]));
    }

    if matches!(model_id, "gpt-5.2" | "gpt-5.2-codex" | "gpt-5.3-codex") {
        return Some(model_id.to_string());
    }

    None
}

/// Returns whether the provided suffix is a Responses reasoning effort we can forward directly.
fn is_supported_reasoning_effort(value: &str) -> bool {
    matches!(
        value,
        "none" | "minimal" | "low" | "medium" | "high" | "xhigh"
    )
}

/// Seeds local conversation state from task history the first time a conversation uses the local backend.
fn ensure_conversation_state_initialized(params: &RequestParams) -> anyhow::Result<()> {
    let mut state_store = conversation_state_store().lock();
    let state = state_store.entry(params.conversation_id).or_default();
    if !state.items.is_empty() {
        return Ok(());
    }

    state.items = task_history_response_items(params)?;
    Ok(())
}

/// Converts the current task's persisted server messages into Responses history items.
pub(super) fn task_history_response_items(params: &RequestParams) -> anyhow::Result<Vec<Value>> {
    let Some(task) = task_for_history(params) else {
        return Ok(Vec::new());
    };

    let mut items = Vec::new();
    let mut replayable_tool_call_ids = HashSet::new();
    for message in &task.messages {
        let Some(inner) = &message.message else {
            continue;
        };

        match inner {
            api::message::Message::UserQuery(_) | api::message::Message::SystemQuery(_) => {
                items.extend(history_items_from_message_inputs(message)?);
            }
            api::message::Message::AgentOutput(output) => {
                if !output.text.is_empty() {
                    items.push(assistant_output_item(&output.text));
                }
            }
            api::message::Message::ToolCall(tool_call) => {
                let Some(history_item) = tool_call_history_item_from_api(tool_call)? else {
                    continue;
                };
                replayable_tool_call_ids.insert(tool_call.tool_call_id.clone());
                items.push(history_item);
            }
            api::message::Message::ToolCallResult(tool_call_result) => {
                if !replayable_tool_call_ids.contains(&tool_call_result.tool_call_id) {
                    continue;
                }
                items.extend(history_items_from_message_inputs(message)?);
            }
            _ => {}
        }
    }

    Ok(items)
}

/// Returns the task whose messages should seed the local conversation history.
fn task_for_history<'a>(params: &'a RequestParams) -> Option<&'a api::Task> {
    let target_task_id = params.target_task_id.as_ref().map(ToString::to_string);
    target_task_id
        .as_deref()
        .and_then(|task_id| params.tasks.iter().find(|task| task.id == task_id))
        .or_else(|| params.tasks.first())
}

/// Converts persisted input-like server messages into Responses history items.
fn history_items_from_message_inputs(message: &api::Message) -> anyhow::Result<Vec<Value>> {
    let inputs = user_inputs_from_messages(std::slice::from_ref(message));
    convert_inputs_to_response_items(&inputs)
}

/// Converts a persisted tool call into a Responses `function_call` history item when the tool is locally understood.
fn tool_call_history_item_from_api(
    tool_call: &api::message::ToolCall,
) -> anyhow::Result<Option<Value>> {
    let Some(tool) = tool_call.tool.as_ref() else {
        return Ok(None);
    };
    let Some((name, arguments)) = serialize_api_tool_call(tool)? else {
        return Ok(None);
    };

    Ok(Some(json!({
        "type": "function_call",
        "call_id": tool_call.tool_call_id,
        "name": name,
        "arguments": arguments.to_string(),
    })))
}

/// Converts a supported proto tool call into the local backend's function name plus JSON arguments.
fn serialize_api_tool_call(
    tool: &api::message::tool_call::Tool,
) -> anyhow::Result<Option<(String, Value)>> {
    let serialized = match tool {
        api::message::tool_call::Tool::RunShellCommand(command) => Some((
            "run_shell_command".to_string(),
            json!({
                "command": command.command,
                "is_read_only": command.is_read_only,
                "uses_pager": command.uses_pager,
                "is_risky": command.is_risky,
                "risk_category": risk_category_name(command.risk_category),
            }),
        )),
        api::message::tool_call::Tool::ReadFiles(read_files) => Some((
            "read_files".to_string(),
            json!({
                "files": read_files.files.iter().map(serialize_read_file).collect::<Vec<_>>(),
            }),
        )),
        api::message::tool_call::Tool::SearchCodebase(search_codebase) => Some((
            "search_codebase".to_string(),
            json!({
                "query": search_codebase.query,
                "path_filters": search_codebase.path_filters,
                "codebase_path": search_codebase.codebase_path,
            }),
        )),
        api::message::tool_call::Tool::Grep(grep) => Some((
            "grep".to_string(),
            json!({
                "queries": grep.queries,
                "path": grep.path,
            }),
        )),
        #[allow(deprecated)]
        api::message::tool_call::Tool::FileGlob(glob) => Some((
            "file_glob".to_string(),
            json!({
                "patterns": glob.patterns,
                "path": glob.path,
            }),
        )),
        api::message::tool_call::Tool::FileGlobV2(glob) => Some((
            "file_glob_v2".to_string(),
            json!({
                "patterns": glob.patterns,
                "search_dir": glob.search_dir,
                "max_matches": glob.max_matches,
                "max_depth": glob.max_depth,
                "min_depth": glob.min_depth,
            }),
        )),
        api::message::tool_call::Tool::ApplyFileDiffs(diffs) => Some((
            "apply_file_diffs".to_string(),
            json!({
                "summary": diffs.summary,
                "diffs": diffs.diffs.iter().map(serialize_file_diff).collect::<Vec<_>>(),
                "new_files": diffs.new_files.iter().map(serialize_new_file).collect::<Vec<_>>(),
                "deleted_files": diffs.deleted_files.iter().map(serialize_deleted_file).collect::<Vec<_>>(),
                "v4a_updates": diffs.v4a_updates.iter().map(serialize_v4a_update).collect::<Vec<_>>(),
            }),
        )),
        api::message::tool_call::Tool::ReadMcpResource(resource) => Some((
            "read_mcp_resource".to_string(),
            json!({
                "uri": resource.uri,
                "server_id": resource.server_id,
            }),
        )),
        api::message::tool_call::Tool::CallMcpTool(tool_call) => Some((
            "call_mcp_tool".to_string(),
            json!({
                "name": tool_call.name,
                "server_id": tool_call.server_id,
                "args": tool_call.args.as_ref().map(prost_struct_to_json).transpose()?,
            }),
        )),
        api::message::tool_call::Tool::WriteToLongRunningShellCommand(write) => Some((
            "write_to_long_running_shell_command".to_string(),
            json!({
                "command_id": write.command_id,
                "input": String::from_utf8_lossy(&write.input).to_string(),
                "mode": write.mode.as_ref().and_then(write_mode_name),
            }),
        )),
        api::message::tool_call::Tool::ReadShellCommandOutput(read) => Some((
            "read_shell_command_output".to_string(),
            json!({
                "command_id": read.command_id,
                "delay_seconds": read.delay.as_ref().and_then(read_shell_output_delay_seconds),
                "on_completion": read.delay.as_ref().is_some_and(is_read_shell_output_on_completion),
            }),
        )),
        api::message::tool_call::Tool::SuggestNewConversation(suggest) => Some((
            "suggest_new_conversation".to_string(),
            json!({
                "message_id": suggest.message_id,
            }),
        )),
        api::message::tool_call::Tool::ReadDocuments(read) => Some((
            "read_documents".to_string(),
            json!({
                "documents": read.documents.iter().map(serialize_read_document).collect::<Vec<_>>(),
            }),
        )),
        api::message::tool_call::Tool::EditDocuments(edit) => Some((
            "edit_documents".to_string(),
            json!({
                "diffs": edit.diffs.iter().map(serialize_document_diff).collect::<Vec<_>>(),
            }),
        )),
        api::message::tool_call::Tool::CreateDocuments(create) => Some((
            "create_documents".to_string(),
            json!({
                "new_documents": create.new_documents.iter().map(serialize_new_document).collect::<Vec<_>>(),
            }),
        )),
        api::message::tool_call::Tool::SuggestPrompt(suggest_prompt) => Some((
            "suggest_prompt".to_string(),
            serialize_suggest_prompt(suggest_prompt),
        )),
        api::message::tool_call::Tool::OpenCodeReview(_) => {
            Some(("open_code_review".to_string(), json!({})))
        }
        api::message::tool_call::Tool::InitProject(_) => {
            Some(("init_project".to_string(), json!({})))
        }
        api::message::tool_call::Tool::FetchConversation(fetch) => Some((
            "fetch_conversation".to_string(),
            json!({
                "conversation_id": fetch.conversation_id,
            }),
        )),
        api::message::tool_call::Tool::ReadSkill(read_skill) => {
            Some(("read_skill".to_string(), serialize_read_skill(read_skill)))
        }
        _ => None,
    };

    Ok(serialized)
}

/// Converts a read-files request into the JSON argument format expected by the local backend.
fn serialize_read_file(file: &api::message::tool_call::read_files::File) -> Value {
    json!({
        "name": file.name,
        "line_ranges": file.line_ranges.iter().map(serialize_line_range).collect::<Vec<_>>(),
    })
}

/// Converts a single file line range into the local JSON format.
fn serialize_line_range(range: &api::FileContentLineRange) -> Value {
    json!({
        "start": range.start,
        "end": range.end,
    })
}

/// Converts a file diff entry into the local JSON format.
fn serialize_file_diff(diff: &api::message::tool_call::apply_file_diffs::FileDiff) -> Value {
    json!({
        "file_path": diff.file_path,
        "search": diff.search,
        "replace": diff.replace,
    })
}

/// Converts a new-file creation entry into the local JSON format.
fn serialize_new_file(new_file: &api::message::tool_call::apply_file_diffs::NewFile) -> Value {
    json!({
        "file_path": new_file.file_path,
        "content": new_file.content,
    })
}

/// Converts a delete-file entry into the local JSON format.
fn serialize_deleted_file(
    deleted_file: &api::message::tool_call::apply_file_diffs::DeleteFile,
) -> Value {
    json!({
        "file_path": deleted_file.file_path,
    })
}

/// Converts a structured V4A update into the local JSON format.
fn serialize_v4a_update(
    update: &api::message::tool_call::apply_file_diffs::V4aFileUpdate,
) -> Value {
    json!({
        "file_path": update.file_path,
        "move_to": update.move_to,
        "hunks": update.hunks.iter().map(serialize_v4a_hunk).collect::<Vec<_>>(),
    })
}

/// Converts a structured V4A hunk into the local JSON format.
fn serialize_v4a_hunk(
    hunk: &api::message::tool_call::apply_file_diffs::v4a_file_update::Hunk,
) -> Value {
    json!({
        "change_context": hunk.change_context,
        "pre_context": hunk.pre_context,
        "old": hunk.old,
        "new": hunk.new,
        "post_context": hunk.post_context,
    })
}

/// Converts a read-documents request entry into the local JSON format.
fn serialize_read_document(document: &api::message::tool_call::read_documents::Document) -> Value {
    json!({
        "document_id": document.document_id,
        "line_ranges": document.line_ranges.iter().map(serialize_line_range).collect::<Vec<_>>(),
    })
}

/// Converts an edit-documents diff into the local JSON format.
fn serialize_document_diff(diff: &api::message::tool_call::edit_documents::DocumentDiff) -> Value {
    json!({
        "document_id": diff.document_id,
        "search": diff.search,
        "replace": diff.replace,
    })
}

/// Converts a create-documents entry into the local JSON format.
fn serialize_new_document(
    document: &api::message::tool_call::create_documents::NewDocument,
) -> Value {
    json!({
        "content": document.content,
        "title": document.title,
    })
}

/// Converts a suggest-prompt tool call into the local JSON argument format.
fn serialize_suggest_prompt(suggest_prompt: &api::message::tool_call::SuggestPrompt) -> Value {
    let mut payload = serde_json::Map::new();
    payload.insert(
        "is_trigger_irrelevant".to_string(),
        Value::Bool(suggest_prompt.is_trigger_irrelevant),
    );

    if let Some(display_mode) = suggest_prompt.display_mode.as_ref() {
        match display_mode {
            api::message::tool_call::suggest_prompt::DisplayMode::InlineQueryBanner(banner) => {
                payload.insert(
                    "display_mode".to_string(),
                    Value::String("inline_query_banner".to_string()),
                );
                payload.insert("title".to_string(), Value::String(banner.title.clone()));
                payload.insert(
                    "description".to_string(),
                    Value::String(banner.description.clone()),
                );
                payload.insert("query".to_string(), Value::String(banner.query.clone()));
            }
            api::message::tool_call::suggest_prompt::DisplayMode::PromptChip(chip) => {
                payload.insert(
                    "display_mode".to_string(),
                    Value::String("prompt_chip".to_string()),
                );
                payload.insert("prompt".to_string(), Value::String(chip.prompt.clone()));
                payload.insert("label".to_string(), Value::String(chip.label.clone()));
            }
        }
    }

    Value::Object(payload)
}

/// Converts a read-skill tool call into the local JSON argument format.
fn serialize_read_skill(read_skill: &api::message::tool_call::ReadSkill) -> Value {
    let mut payload = serde_json::Map::new();
    if !read_skill.name.is_empty() {
        payload.insert("name".to_string(), Value::String(read_skill.name.clone()));
    }

    if let Some(skill_reference) = read_skill.skill_reference.as_ref() {
        match skill_reference {
            api::message::tool_call::read_skill::SkillReference::SkillPath(path) => {
                payload.insert("skill_path".to_string(), Value::String(path.clone()));
            }
            api::message::tool_call::read_skill::SkillReference::BundledSkillId(id) => {
                payload.insert("bundled_skill_id".to_string(), Value::String(id.clone()));
            }
        }
    }

    Value::Object(payload)
}

/// Converts a protobuf `Struct` into serde JSON for historical MCP tool arguments.
fn prost_struct_to_json(structure: &prost_types::Struct) -> anyhow::Result<Value> {
    let mut object = serde_json::Map::new();
    for (key, value) in &structure.fields {
        object.insert(key.clone(), prost_value_to_json(value)?);
    }
    Ok(Value::Object(object))
}

/// Converts a protobuf `Value` into serde JSON for historical MCP tool arguments.
fn prost_value_to_json(value: &prost_types::Value) -> anyhow::Result<Value> {
    use prost_types::value::Kind;

    let Some(kind) = value.kind.as_ref() else {
        return Ok(Value::Null);
    };

    let json_value = match kind {
        Kind::NullValue(_) => Value::Null,
        Kind::BoolValue(value) => Value::Bool(*value),
        Kind::NumberValue(value) => serde_json::Number::from_f64(*value)
            .map(Value::Number)
            .ok_or_else(|| anyhow!("Failed to serialize non-finite protobuf number"))?,
        Kind::StringValue(value) => Value::String(value.clone()),
        Kind::StructValue(value) => prost_struct_to_json(value)?,
        Kind::ListValue(value) => Value::Array(
            value
                .values
                .iter()
                .map(prost_value_to_json)
                .collect::<anyhow::Result<Vec<_>>>()?,
        ),
    };

    Ok(json_value)
}

/// Returns the string name for a persisted shell-command risk category when it is known.
fn risk_category_name(risk_category: i32) -> Option<&'static str> {
    match api::RiskCategory::try_from(risk_category).ok()? {
        api::RiskCategory::Unspecified => None,
        api::RiskCategory::ReadOnly => Some("read_only"),
        api::RiskCategory::TrivialLocalChange => Some("trivial_local_change"),
        api::RiskCategory::NontrivialLocalChange => Some("nontrivial_local_change"),
        api::RiskCategory::ExternalChange => Some("external_change"),
        api::RiskCategory::Risky => Some("risky"),
    }
}

/// Returns the serialized write mode name for long-running shell input.
fn write_mode_name(
    mode: &api::message::tool_call::write_to_long_running_shell_command::Mode,
) -> Option<&'static str> {
    use api::message::tool_call::write_to_long_running_shell_command::mode::Mode;

    match mode.mode.as_ref()? {
        Mode::Raw(_) => Some("raw"),
        Mode::Line(_) => Some("line"),
        Mode::Block(_) => Some("block"),
    }
}

/// Returns the whole-second delay configured for a read-shell-command-output request, if any.
fn read_shell_output_delay_seconds(
    delay: &api::message::tool_call::read_shell_command_output::Delay,
) -> Option<i64> {
    match delay {
        api::message::tool_call::read_shell_command_output::Delay::Duration(duration) => {
            Some(duration.seconds)
        }
        api::message::tool_call::read_shell_command_output::Delay::OnCompletion(_) => None,
    }
}

/// Returns whether a read-shell-command-output request waits for completion.
fn is_read_shell_output_on_completion(
    delay: &api::message::tool_call::read_shell_command_output::Delay,
) -> bool {
    matches!(
        delay,
        api::message::tool_call::read_shell_command_output::Delay::OnCompletion(_)
    )
}
