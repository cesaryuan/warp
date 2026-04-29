//! Local OpenAI-compatible Responses API backend for Warp Agent.

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{anyhow, Context as _};
use async_stream::stream;
use futures::channel::oneshot;
use futures::future::{select, Either};
use parking_lot::FairMutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;
use warp_multi_agent_api as api;

use crate::ai::agent::conversation::AIConversationId;
use crate::ai::agent::task::TaskId;
use crate::ai::agent::{AIAgentContext, AIAgentInput};
use crate::server::server_api::ServerApi;

use super::{Event, RequestParams, ResponseStream};
use crate::ai::agent::api::r#impl::get_supported_tools;

/// Minimal system instructions that teach a local Responses model how to behave like Warp Agent.
const LOCAL_OPENAI_SYSTEM_PROMPT: &str = concat!(
    "You are Warp Agent running locally inside the user's terminal workspace. ",
    "Help with coding and shell tasks. ",
    "Use the provided tools whenever you need to inspect files, search code, run commands, ",
    "or apply edits. ",
    "When you do not need a tool, answer directly and concisely. ",
    "When returning tool calls, provide valid JSON arguments that exactly match the tool schema."
);

/// In-memory history for a local OpenAI-backed conversation.
#[derive(Debug, Default, Clone)]
struct LocalConversationState {
    items: Vec<Value>,
}

/// Parsed subset of a Responses API payload used by the local backend.
#[derive(Debug, Deserialize)]
struct ResponsesApiResponse {
    #[serde(default)]
    output: Vec<ResponsesOutputItem>,
}

/// Parsed subset of a single Responses API output item.
#[derive(Debug, Deserialize)]
struct ResponsesOutputItem {
    #[serde(rename = "type")]
    item_type: String,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Vec<ResponsesContentItem>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Parsed subset of a message content item returned by Responses.
#[derive(Debug, Deserialize)]
struct ResponsesContentItem {
    #[serde(rename = "type")]
    item_type: String,
    #[serde(default)]
    text: Option<String>,
}

/// Generic error envelope returned by OpenAI-compatible APIs.
#[derive(Debug, Deserialize)]
struct ResponsesErrorEnvelope {
    error: ResponsesErrorBody,
}

/// Minimal error body used for user-visible provider failures.
#[derive(Debug, Deserialize)]
struct ResponsesErrorBody {
    #[serde(default)]
    message: String,
}

/// Single function tool call extracted from a Responses output.
#[derive(Debug)]
struct ParsedFunctionCall {
    name: String,
    call_id: String,
    arguments: Value,
}

/// Request body sent to the local OpenAI-compatible `/v1/responses` endpoint.
#[derive(Debug, Serialize)]
struct ResponsesRequestBody<'a> {
    model: &'a str,
    instructions: &'a str,
    input: Vec<Value>,
    tools: Vec<Value>,
    tool_choice: &'static str,
}

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

        let request_future = {
            let server_api = server_api.clone();
            let params = params.clone();
            let task_id = task_id.clone();
            let request_id = request_id.clone();
            async move {
                execute_local_responses_request(&server_api, &params, &task_id, &request_id).await
            }
        };
        match select(Box::pin(request_future), Box::pin(cancellation_rx)).await {
            Either::Left((result, _)) => {
                match result {
                    Ok(output_events) => {
                        for event in output_events {
                            yield event;
                        }
                    }
                    Err(error) => {
                        log::warn!("Local OpenAI Responses backend failed: {error:#}");
                        yield Ok(user_visible_error_event(
                            &task_id,
                            &request_id,
                            &error.to_string(),
                        ));
                        yield Ok(stream_finished_event(
                            finished_reason_for_error(&error, model_name),
                        ));
                    }
                }
            }
            Either::Right((_cancelled, _)) => {}
        }
    })
}

/// Executes a single local Responses API turn and converts it into Warp ResponseEvents.
async fn execute_local_responses_request(
    server_api: &ServerApi,
    params: &RequestParams,
    task_id: &TaskId,
    request_id: &str,
) -> anyhow::Result<Vec<Event>> {
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
        ResponsesRequestBody {
            model: &params.model.to_string(),
            instructions: LOCAL_OPENAI_SYSTEM_PROMPT,
            input: state.items,
            tools: build_tools_payload(params),
            tool_choice: "auto",
        }
    };

    let response = server_api
        .http_client()
        .post(endpoint)
        .bearer_auth(api_key)
        .json(&request_body)
        .send()
        .await
        .context("Failed to send local OpenAI Responses request")?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown provider error".to_string());
        let provider_message = serde_json::from_str::<ResponsesErrorEnvelope>(&error_text)
            .map(|body| body.error.message)
            .unwrap_or(error_text);
        return Err(anyhow!(ProviderError::new(
            status.as_u16(),
            provider_message
        )));
    }

    let response: ResponsesApiResponse = response
        .json()
        .await
        .context("Failed to decode local OpenAI Responses payload")?;
    let (assistant_messages, function_calls) = parse_responses_output(response.output)?;

    let mut history_items = assistant_messages
        .iter()
        .map(|text| assistant_output_item(text))
        .collect::<Vec<_>>();
    history_items.extend(function_calls.iter().map(function_call_history_item));
    if !history_items.is_empty() {
        let mut state_store = conversation_state_store().lock();
        let state = state_store.entry(params.conversation_id).or_default();
        state.items.extend(history_items);
    }

    let mut messages = assistant_messages
        .into_iter()
        .map(|text| agent_output_message(task_id, request_id, text))
        .collect::<Vec<_>>();
    messages.extend(
        function_calls
            .into_iter()
            .map(|tool_call| tool_call_message(task_id, request_id, tool_call))
            .collect::<anyhow::Result<Vec<_>>>()?,
    );

    let mut events = Vec::new();
    if !messages.is_empty() {
        events.push(Ok(add_messages_event(task_id, messages)));
    }
    events.push(Ok(stream_finished_event(
        api::response_event::stream_finished::Reason::Done(
            api::response_event::stream_finished::Done {},
        ),
    )));
    Ok(events)
}

/// Converts the current request inputs into Responses API conversation items.
fn convert_inputs_to_response_items(inputs: &[AIAgentInput]) -> anyhow::Result<Vec<Value>> {
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
fn assistant_output_item(text: &str) -> Value {
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
fn function_call_history_item(function_call: &ParsedFunctionCall) -> Value {
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
fn build_tools_payload(params: &RequestParams) -> Vec<Value> {
    get_supported_tools(params)
        .into_iter()
        .filter_map(tool_schema_for_type)
        .collect()
}

/// Maps a supported Warp tool to a Responses function schema.
fn tool_schema_for_type(tool_type: api::ToolType) -> Option<Value> {
    match tool_type {
        api::ToolType::RunShellCommand => Some(json!({
            "name": "run_shell_command",
            "description": "Run a shell command in the user's workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "is_read_only": { "type": "boolean" },
                    "wait_until_complete": { "type": "boolean" }
                },
                "required": ["command"]
            }
        })),
        api::ToolType::ReadFiles => Some(json!({
            "name": "read_files",
            "description": "Read one or more files from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "start_line": { "type": "integer" },
                                "end_line": { "type": "integer" }
                            },
                            "required": ["name"]
                        }
                    }
                },
                "required": ["files"]
            }
        })),
        api::ToolType::SearchCodebase => Some(json!({
            "name": "search_codebase",
            "description": "Search the indexed codebase for relevant files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "path_filters": { "type": "array", "items": { "type": "string" } },
                    "codebase_path": { "type": "string" }
                },
                "required": ["query"]
            }
        })),
        api::ToolType::Grep => Some(json!({
            "name": "grep",
            "description": "Run grep-like text searches in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": { "type": "array", "items": { "type": "string" } },
                    "path": { "type": "string" }
                },
                "required": ["queries", "path"]
            }
        })),
        api::ToolType::FileGlob => Some(json!({
            "name": "file_glob",
            "description": "Find files by glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patterns": { "type": "array", "items": { "type": "string" } },
                    "path": { "type": "string" }
                },
                "required": ["patterns"]
            }
        })),
        api::ToolType::FileGlobV2 => Some(json!({
            "name": "file_glob_v2",
            "description": "Find files by glob pattern with directory controls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patterns": { "type": "array", "items": { "type": "string" } },
                    "search_dir": { "type": "string" },
                    "max_matches": { "type": "integer" },
                    "max_depth": { "type": "integer" },
                    "min_depth": { "type": "integer" }
                },
                "required": ["patterns"]
            }
        })),
        api::ToolType::ApplyFileDiffs => Some(json!({
            "name": "apply_file_diffs",
            "description": "Create, edit, move, or delete files in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": { "type": "string" },
                    "diffs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": { "type": "string" },
                                "search": { "type": "string" },
                                "replace": { "type": "string" }
                            },
                            "required": ["file_path", "search", "replace"]
                        }
                    },
                    "new_files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": { "type": "string" },
                                "content": { "type": "string" }
                            },
                            "required": ["file_path", "content"]
                        }
                    },
                    "deleted_files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": { "type": "string" }
                            },
                            "required": ["file_path"]
                        }
                    },
                    "v4a_updates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": { "type": "string" },
                                "move_to": { "type": "string" },
                                "hunks": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "change_context": { "type": "array", "items": { "type": "string" } },
                                            "pre_context": { "type": "string" },
                                            "old": { "type": "string" },
                                            "new": { "type": "string" },
                                            "post_context": { "type": "string" }
                                        }
                                    }
                                }
                            },
                            "required": ["file_path", "hunks"]
                        }
                    }
                }
            }
        })),
        api::ToolType::ReadMcpResource => Some(json!({
            "name": "read_mcp_resource",
            "description": "Read a resource exposed by an MCP server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "uri": { "type": "string" },
                    "server_id": { "type": "string" }
                },
                "required": ["uri"]
            }
        })),
        api::ToolType::CallMcpTool => Some(json!({
            "name": "call_mcp_tool",
            "description": "Call a tool exposed by an MCP server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "args": { "type": "object" },
                    "server_id": { "type": "string" }
                },
                "required": ["name", "args"]
            }
        })),
        api::ToolType::WriteToLongRunningShellCommand => Some(json!({
            "name": "write_to_long_running_shell_command",
            "description": "Send input to a running shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command_id": { "type": "string" },
                    "input": { "type": "string" },
                    "mode": { "type": "string", "enum": ["raw", "line", "block"] }
                },
                "required": ["command_id", "input"]
            }
        })),
        api::ToolType::SuggestNewConversation => Some(json!({
            "name": "suggest_new_conversation",
            "description": "Suggest starting a new conversation from a specific message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": { "type": "string" }
                },
                "required": ["message_id"]
            }
        })),
        _ => None,
    }
}

/// Parses the assistant output returned by the Responses API into text and tool calls.
fn parse_responses_output(
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
    api::Message {
        id: Uuid::new_v4().to_string(),
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

#[allow(deprecated)]
/// Converts a function tool name plus JSON arguments into the corresponding Warp tool call variant.
fn parse_tool_call(name: &str, arguments: Value) -> anyhow::Result<api::message::tool_call::Tool> {
    match name {
        "run_shell_command" | "shell" => Ok(api::message::tool_call::Tool::RunShellCommand(
            api::message::tool_call::RunShellCommand {
                command: required_string(&arguments, "command")?,
                is_read_only: arguments
                    .get("is_read_only")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                uses_pager: arguments
                    .get("uses_pager")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                citations: vec![],
                is_risky: arguments
                    .get("is_risky")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                wait_until_complete_value: arguments
                    .get("wait_until_complete")
                    .and_then(Value::as_bool)
                    .map(
                        api::message::tool_call::run_shell_command::WaitUntilCompleteValue::WaitUntilComplete,
                    ),
                risk_category: api::RiskCategory::Unspecified.into(),
            },
        )),
        "read_files" => Ok(api::message::tool_call::Tool::ReadFiles(
            api::message::tool_call::ReadFiles {
                files: required_array(&arguments, "files")?
                    .iter()
                    .map(parse_read_file)
                    .collect::<anyhow::Result<Vec<_>>>()?,
            },
        )),
        "search_codebase" => Ok(api::message::tool_call::Tool::SearchCodebase(
            api::message::tool_call::SearchCodebase {
                query: required_string(&arguments, "query")?,
                path_filters: optional_string_array(&arguments, "path_filters"),
                codebase_path: optional_string(&arguments, "codebase_path").unwrap_or_default(),
            },
        )),
        "grep" => Ok(api::message::tool_call::Tool::Grep(api::message::tool_call::Grep {
            queries: required_string_array(&arguments, "queries")?,
            path: required_string(&arguments, "path")?,
        })),
        "file_glob" => Ok(api::message::tool_call::Tool::FileGlob(
            api::message::tool_call::FileGlob {
                patterns: required_string_array(&arguments, "patterns")?,
                path: optional_string(&arguments, "path").unwrap_or_default(),
            },
        )),
        "file_glob_v2" => Ok(api::message::tool_call::Tool::FileGlobV2(
            api::message::tool_call::FileGlobV2 {
                patterns: required_string_array(&arguments, "patterns")?,
                search_dir: optional_string(&arguments, "search_dir").unwrap_or_default(),
                max_matches: optional_i32(&arguments, "max_matches").unwrap_or_default(),
                max_depth: optional_i32(&arguments, "max_depth").unwrap_or_default(),
                min_depth: optional_i32(&arguments, "min_depth").unwrap_or_default(),
            },
        )),
        "apply_file_diffs" => Ok(api::message::tool_call::Tool::ApplyFileDiffs(
            parse_apply_file_diffs(arguments)?,
        )),
        "read_mcp_resource" => Ok(api::message::tool_call::Tool::ReadMcpResource(
            api::message::tool_call::ReadMcpResource {
                uri: required_string(&arguments, "uri")?,
                server_id: optional_string(&arguments, "server_id").unwrap_or_default(),
            },
        )),
        "call_mcp_tool" => Ok(api::message::tool_call::Tool::CallMcpTool(
            api::message::tool_call::CallMcpTool {
                name: required_string(&arguments, "name")?,
                args: optional_object(&arguments, "args").map(serde_json_object_to_prost_struct).transpose()?,
                server_id: optional_string(&arguments, "server_id").unwrap_or_default(),
            },
        )),
        "write_to_long_running_shell_command" | "write_to_lrc" => {
            Ok(api::message::tool_call::Tool::WriteToLongRunningShellCommand(
                api::message::tool_call::WriteToLongRunningShellCommand {
                    input: required_string(&arguments, "input")?.into_bytes(),
                    mode: optional_string(&arguments, "mode")
                        .map(parse_write_mode)
                        .transpose()?,
                    command_id: required_string(&arguments, "command_id")?,
                },
            ))
        }
        "suggest_new_conversation" => Ok(api::message::tool_call::Tool::SuggestNewConversation(
            api::message::tool_call::SuggestNewConversation {
                message_id: required_string(&arguments, "message_id")?,
            },
        )),
        unsupported => Err(anyhow!("Unsupported local OpenAI tool call: {unsupported}")),
    }
}

/// Parses the file arguments used by the `read_files` tool.
fn parse_read_file(value: &Value) -> anyhow::Result<api::message::tool_call::read_files::File> {
    let name = required_string(value, "name")?;
    let mut line_ranges = Vec::new();
    if let (Some(start), Some(end)) = (
        optional_u32(value, "start_line"),
        optional_u32(value, "end_line"),
    ) {
        line_ranges.push(api::FileContentLineRange { start, end });
    }

    Ok(api::message::tool_call::read_files::File { name, line_ranges })
}

/// Parses the complex `apply_file_diffs` argument payload.
fn parse_apply_file_diffs(
    arguments: Value,
) -> anyhow::Result<api::message::tool_call::ApplyFileDiffs> {
    let diffs = optional_array(&arguments, "diffs")
        .into_iter()
        .flatten()
        .map(|value| {
            Ok(api::message::tool_call::apply_file_diffs::FileDiff {
                file_path: required_string(value, "file_path")?,
                search: required_string(value, "search")?,
                replace: required_string(value, "replace")?,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let new_files = optional_array(&arguments, "new_files")
        .into_iter()
        .flatten()
        .map(|value| {
            Ok(api::message::tool_call::apply_file_diffs::NewFile {
                file_path: required_string(value, "file_path")?,
                content: required_string(value, "content")?,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let deleted_files = optional_array(&arguments, "deleted_files")
        .into_iter()
        .flatten()
        .map(|value| {
            Ok(api::message::tool_call::apply_file_diffs::DeleteFile {
                file_path: required_string(value, "file_path")?,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let v4a_updates = optional_array(&arguments, "v4a_updates")
        .into_iter()
        .flatten()
        .map(parse_v4a_update)
        .collect::<anyhow::Result<Vec<_>>>()?;

    Ok(api::message::tool_call::ApplyFileDiffs {
        summary: optional_string(&arguments, "summary").unwrap_or_default(),
        diffs,
        new_files,
        deleted_files,
        v4a_updates,
    })
}

/// Parses a single V4A file update definition for `apply_file_diffs`.
fn parse_v4a_update(
    value: &Value,
) -> anyhow::Result<api::message::tool_call::apply_file_diffs::V4aFileUpdate> {
    let hunks = required_array(value, "hunks")?
        .iter()
        .map(|hunk| {
            Ok(
                api::message::tool_call::apply_file_diffs::v4a_file_update::Hunk {
                    change_context: optional_string_array(hunk, "change_context"),
                    pre_context: optional_string(hunk, "pre_context").unwrap_or_default(),
                    old: optional_string(hunk, "old").unwrap_or_default(),
                    new: optional_string(hunk, "new").unwrap_or_default(),
                    post_context: optional_string(hunk, "post_context").unwrap_or_default(),
                },
            )
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    Ok(api::message::tool_call::apply_file_diffs::V4aFileUpdate {
        file_path: required_string(value, "file_path")?,
        move_to: optional_string(value, "move_to").unwrap_or_default(),
        hunks,
    })
}

/// Parses the write mode used for long-running shell input.
fn parse_write_mode(
    mode: String,
) -> anyhow::Result<api::message::tool_call::write_to_long_running_shell_command::Mode> {
    let mode = match mode.as_str() {
        "raw" => api::message::tool_call::write_to_long_running_shell_command::mode::Mode::Raw(()),
        "line" => {
            api::message::tool_call::write_to_long_running_shell_command::mode::Mode::Line(())
        }
        "block" => {
            api::message::tool_call::write_to_long_running_shell_command::mode::Mode::Block(())
        }
        _ => {
            return Err(anyhow!(
                "Unsupported write_to_long_running_shell_command mode: {mode}"
            ))
        }
    };

    Ok(api::message::tool_call::write_to_long_running_shell_command::Mode { mode: Some(mode) })
}

/// Converts a serde JSON object into a protobuf Struct for MCP tool arguments.
fn serde_json_object_to_prost_struct(
    object: serde_json::Map<String, Value>,
) -> anyhow::Result<prost_types::Struct> {
    Ok(prost_types::Struct {
        fields: object
            .into_iter()
            .map(|(key, value)| serde_json_to_prost_value(value).map(|value| (key, value)))
            .collect::<anyhow::Result<_>>()?,
    })
}

/// Converts a serde JSON value into a protobuf Value.
fn serde_json_to_prost_value(value: Value) -> anyhow::Result<prost_types::Value> {
    use prost_types::value::Kind;

    let kind = match value {
        Value::Null => Kind::NullValue(0),
        Value::Bool(value) => Kind::BoolValue(value),
        Value::Number(value) => Kind::NumberValue(
            value
                .as_f64()
                .ok_or_else(|| anyhow!("Failed to convert JSON number to f64"))?,
        ),
        Value::String(value) => Kind::StringValue(value),
        Value::Array(values) => Kind::ListValue(prost_types::ListValue {
            values: values
                .into_iter()
                .map(serde_json_to_prost_value)
                .collect::<anyhow::Result<Vec<_>>>()?,
        }),
        Value::Object(object) => Kind::StructValue(serde_json_object_to_prost_struct(object)?),
    };

    Ok(prost_types::Value { kind: Some(kind) })
}

/// Extracts a required string field from a JSON object.
fn required_string(value: &Value, key: &str) -> anyhow::Result<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("Missing required string field '{key}'"))
}

/// Extracts an optional string field from a JSON object.
fn optional_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

/// Extracts a required array of strings from a JSON object.
fn required_string_array(value: &Value, key: &str) -> anyhow::Result<Vec<String>> {
    required_array(value, key)?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| anyhow!("Field '{key}' must contain only strings"))
        })
        .collect()
}

/// Extracts an optional array of strings from a JSON object.
fn optional_string_array(value: &Value, key: &str) -> Vec<String> {
    optional_array(value, key)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

/// Extracts a required array field from a JSON object.
fn required_array<'a>(value: &'a Value, key: &str) -> anyhow::Result<&'a [Value]> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| anyhow!("Missing required array field '{key}'"))
}

/// Extracts an optional array field from a JSON object.
fn optional_array<'a>(value: &'a Value, key: &str) -> Option<&'a [Value]> {
    value.get(key).and_then(Value::as_array).map(Vec::as_slice)
}

/// Extracts an optional object field from a JSON object.
fn optional_object(value: &Value, key: &str) -> Option<serde_json::Map<String, Value>> {
    value.get(key).and_then(Value::as_object).cloned()
}

/// Extracts an optional `u32` field from a JSON object.
fn optional_u32(value: &Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

/// Extracts an optional `i32` field from a JSON object.
fn optional_i32(value: &Value, key: &str) -> Option<i32> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
}

/// Normalizes the user-provided base URL into the exact `/v1/responses` endpoint.
fn normalize_responses_endpoint(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        format!("{trimmed}/responses")
    } else {
        format!("{trimmed}/v1/responses")
    }
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

/// Builds a user-visible error message event so the UI never gets stuck without output.
fn user_visible_error_event(
    task_id: &TaskId,
    request_id: &str,
    error_message: &str,
) -> api::ResponseEvent {
    add_messages_event(
        task_id,
        vec![agent_output_message(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::agent::AIAgentActionResult;
    use crate::ai::agent::AIAgentContext;
    use ai::agent::action_result::{AIAgentActionResultType, ReadFilesResult};

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

    /// Verifies that user queries and tool outputs map to Responses input items.
    #[test]
    fn convert_inputs_to_response_items_supports_user_query_and_action_result() {
        let inputs = vec![
            AIAgentInput::UserQuery {
                query: "hello".to_string(),
                context: std::sync::Arc::new([AIAgentContext::SelectedText("world".to_string())]),
                static_query_type: None,
                referenced_attachments: HashMap::new(),
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

    /// Verifies that message text and function calls are extracted from Responses output.
    #[test]
    fn parse_responses_output_extracts_text_and_function_calls() {
        let output = vec![
            ResponsesOutputItem {
                item_type: "message".to_string(),
                role: Some("assistant".to_string()),
                content: vec![ResponsesContentItem {
                    item_type: "output_text".to_string(),
                    text: Some("done".to_string()),
                }],
                name: None,
                call_id: None,
                arguments: None,
            },
            ResponsesOutputItem {
                item_type: "function_call".to_string(),
                role: None,
                content: vec![],
                name: Some("read_files".to_string()),
                call_id: Some("call_1".to_string()),
                arguments: Some(r#"{"files":[{"name":"README.md"}]}"#.to_string()),
            },
        ];

        let (messages, calls) = parse_responses_output(output).expect("output should parse");
        assert_eq!(messages, vec!["done".to_string()]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_files");
    }
}
