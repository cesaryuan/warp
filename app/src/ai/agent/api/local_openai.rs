//! Local OpenAI-compatible Responses API backend for Warp Agent.

mod tool_schemas;

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{anyhow, Context as _};
use async_stream::stream;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use futures::channel::oneshot;
use futures::future::{select, Either};
use parking_lot::FairMutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;
use warp_multi_agent_api as api;

use crate::ai::agent::conversation::AIConversationId;
use crate::ai::agent::task::TaskId;
use crate::ai::agent::{AIAgentContext, AIAgentInput, MCPContext, MCPServer};
use crate::server::server_api::ServerApi;

use super::{Event, RequestParams, ResponseStream};
use crate::ai::agent::api::r#impl::get_supported_tools;
use tool_schemas::built_in_tool_schema;

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
struct ResponsesRequestBody {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ResponsesReasoningConfig>,
    instructions: &'static str,
    input: Vec<Value>,
    tools: Vec<Value>,
    tool_choice: &'static str,
}

/// Reasoning configuration supported by the Responses API.
#[derive(Debug, Clone, Serialize)]
struct ResponsesReasoningConfig {
    effort: String,
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

        if should_emit_create_task(&params) {
            yield Ok(create_task_event(&task_id));
        }

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
        let (normalized_model, reasoning) =
            normalize_openai_model_and_reasoning(&params.model.to_string());
        ResponsesRequestBody {
            model: normalized_model,
            reasoning,
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
    let mut tools = get_supported_tools(params)
        .into_iter()
        .filter_map(built_in_tool_schema)
        .collect::<Vec<_>>();
    tools.extend(mcp_tool_schemas(params.mcp_context.as_ref()));
    tools
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
fn mcp_tool_schema(server_id: Option<&str>, tool: &rmcp::model::Tool) -> Option<Value> {
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
fn parse_mcp_function_name(name: &str) -> Option<(Option<String>, String)> {
    let suffix = name.strip_prefix("warp_mcp_tool__")?;
    let (encoded_server_id, encoded_tool_name) = suffix.split_once("__")?;
    let server_id = URL_SAFE_NO_PAD.decode(encoded_server_id).ok()?;
    let tool_name = URL_SAFE_NO_PAD.decode(encoded_tool_name).ok()?;

    let server_id = String::from_utf8(server_id).ok()?;
    let tool_name = String::from_utf8(tool_name).ok()?;

    Some(((!server_id.is_empty()).then_some(server_id), tool_name))
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
    if let Some((server_id, tool_name)) = parse_mcp_function_name(name) {
        return Ok(api::message::tool_call::Tool::CallMcpTool(
            api::message::tool_call::CallMcpTool {
                name: tool_name,
                args: optional_object(&arguments, "args")
                    .or_else(|| arguments.as_object().cloned())
                    .map(serde_json_object_to_prost_struct)
                    .transpose()?,
                server_id: server_id.unwrap_or_default(),
            },
        ));
    }

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
                risk_category: optional_string(&arguments, "risk_category")
                    .and_then(|value| parse_risk_category(&value))
                    .unwrap_or(api::RiskCategory::Unspecified)
                    .into(),
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
        "read_shell_command_output" => Ok(api::message::tool_call::Tool::ReadShellCommandOutput(
            api::message::tool_call::ReadShellCommandOutput {
                command_id: required_string(&arguments, "command_id")?,
                delay: parse_read_shell_command_output_delay(&arguments)?,
            },
        )),
        "suggest_new_conversation" => Ok(api::message::tool_call::Tool::SuggestNewConversation(
            api::message::tool_call::SuggestNewConversation {
                message_id: required_string(&arguments, "message_id")?,
            },
        )),
        "read_documents" => Ok(api::message::tool_call::Tool::ReadDocuments(
            api::message::tool_call::ReadDocuments {
                documents: required_array(&arguments, "documents")?
                    .iter()
                    .map(parse_read_document)
                    .collect::<anyhow::Result<Vec<_>>>()?,
            },
        )),
        "edit_documents" => Ok(api::message::tool_call::Tool::EditDocuments(
            api::message::tool_call::EditDocuments {
                diffs: required_array(&arguments, "diffs")?
                    .iter()
                    .map(parse_document_diff)
                    .collect::<anyhow::Result<Vec<_>>>()?,
            },
        )),
        "create_documents" => Ok(api::message::tool_call::Tool::CreateDocuments(
            api::message::tool_call::CreateDocuments {
                new_documents: required_array(&arguments, "new_documents")?
                    .iter()
                    .map(parse_new_document)
                    .collect::<anyhow::Result<Vec<_>>>()?,
            },
        )),
        "suggest_prompt" => Ok(api::message::tool_call::Tool::SuggestPrompt(
            parse_suggest_prompt(arguments)?,
        )),
        "open_code_review" => Ok(api::message::tool_call::Tool::OpenCodeReview(
            api::message::tool_call::OpenCodeReview {},
        )),
        "init_project" => Ok(api::message::tool_call::Tool::InitProject(
            api::message::tool_call::InitProject {},
        )),
        "fetch_conversation" => Ok(api::message::tool_call::Tool::FetchConversation(
            api::message::tool_call::FetchConversation {
                conversation_id: optional_string(&arguments, "conversation_id").unwrap_or_default(),
            },
        )),
        "read_skill" => Ok(api::message::tool_call::Tool::ReadSkill(
            parse_read_skill(arguments)?,
        )),
        unsupported => Err(anyhow!("Unsupported local OpenAI tool call: {unsupported}")),
    }
}

/// Parses the file arguments used by the `read_files` tool.
fn parse_read_file(value: &Value) -> anyhow::Result<api::message::tool_call::read_files::File> {
    let name = required_string(value, "name")?;
    let line_ranges = parse_line_ranges(value, "line_ranges")?;

    Ok(api::message::tool_call::read_files::File { name, line_ranges })
}

/// Parses the document arguments used by the `read_documents` tool.
fn parse_read_document(
    value: &Value,
) -> anyhow::Result<api::message::tool_call::read_documents::Document> {
    Ok(api::message::tool_call::read_documents::Document {
        document_id: required_string(value, "document_id")?,
        line_ranges: parse_line_ranges(value, "line_ranges")?,
    })
}

/// Parses a single document diff entry for `edit_documents`.
fn parse_document_diff(
    value: &Value,
) -> anyhow::Result<api::message::tool_call::edit_documents::DocumentDiff> {
    Ok(api::message::tool_call::edit_documents::DocumentDiff {
        document_id: required_string(value, "document_id")?,
        search: required_string(value, "search")?,
        replace: required_string(value, "replace")?,
    })
}

/// Parses a single new document entry for `create_documents`.
fn parse_new_document(
    value: &Value,
) -> anyhow::Result<api::message::tool_call::create_documents::NewDocument> {
    Ok(api::message::tool_call::create_documents::NewDocument {
        content: required_string(value, "content")?,
        title: optional_string(value, "title").unwrap_or_default(),
    })
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

/// Parses the optional line ranges used by read_files and read_documents.
fn parse_line_ranges(value: &Value, key: &str) -> anyhow::Result<Vec<api::FileContentLineRange>> {
    if let Some(line_ranges) = optional_array(value, key) {
        return line_ranges
            .iter()
            .map(|value| {
                Ok(api::FileContentLineRange {
                    start: required_u32(value, "start")?,
                    end: required_u32(value, "end")?,
                })
            })
            .collect();
    }

    // Backwards-compatible fallback for the earlier start_line/end_line shape.
    if let (Some(start), Some(end)) = (
        optional_u32(value, "start_line"),
        optional_u32(value, "end_line"),
    ) {
        return Ok(vec![api::FileContentLineRange { start, end }]);
    }

    Ok(Vec::new())
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

/// Parses the optional delay configuration for `read_shell_command_output`.
fn parse_read_shell_command_output_delay(
    arguments: &Value,
) -> anyhow::Result<Option<api::message::tool_call::read_shell_command_output::Delay>> {
    if arguments
        .get("on_completion")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Ok(Some(
            api::message::tool_call::read_shell_command_output::Delay::OnCompletion(()),
        ));
    }

    if let Some(delay_seconds) = optional_i64(arguments, "delay_seconds") {
        return Ok(Some(
            api::message::tool_call::read_shell_command_output::Delay::Duration(
                prost_types::Duration {
                    seconds: delay_seconds,
                    nanos: 0,
                },
            ),
        ));
    }

    Ok(None)
}

/// Parses the prompt suggestion oneof shape into Warp's proto tool call.
fn parse_suggest_prompt(
    arguments: Value,
) -> anyhow::Result<api::message::tool_call::SuggestPrompt> {
    let display_mode = required_string(&arguments, "display_mode")?;
    let display_mode = match display_mode.as_str() {
        "inline_query_banner" => {
            api::message::tool_call::suggest_prompt::DisplayMode::InlineQueryBanner(
                api::message::tool_call::suggest_prompt::InlineQueryBanner {
                    title: required_string(&arguments, "title")?,
                    description: required_string(&arguments, "description")?,
                    query: required_string(&arguments, "query")?,
                },
            )
        }
        "prompt_chip" => api::message::tool_call::suggest_prompt::DisplayMode::PromptChip(
            api::message::tool_call::suggest_prompt::PromptChip {
                prompt: required_string(&arguments, "prompt")?,
                label: optional_string(&arguments, "label").unwrap_or_default(),
            },
        ),
        _ => {
            return Err(anyhow!(
                "Unsupported suggest_prompt display_mode: {display_mode}"
            ))
        }
    };

    Ok(api::message::tool_call::SuggestPrompt {
        display_mode: Some(display_mode),
        is_trigger_irrelevant: arguments
            .get("is_trigger_irrelevant")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

/// Parses the read_skill oneof shape into Warp's proto tool call.
fn parse_read_skill(arguments: Value) -> anyhow::Result<api::message::tool_call::ReadSkill> {
    let skill_reference = if let Some(skill_path) = optional_string(&arguments, "skill_path") {
        Some(api::message::tool_call::read_skill::SkillReference::SkillPath(skill_path))
    } else {
        optional_string(&arguments, "bundled_skill_id")
            .map(api::message::tool_call::read_skill::SkillReference::BundledSkillId)
    };

    Ok(api::message::tool_call::ReadSkill {
        skill_reference,
        name: optional_string(&arguments, "name").unwrap_or_default(),
    })
}

/// Parses a string risk category into the protobuf enum.
fn parse_risk_category(value: &str) -> Option<api::RiskCategory> {
    match value {
        "unspecified" => Some(api::RiskCategory::Unspecified),
        "read_only" => Some(api::RiskCategory::ReadOnly),
        "trivial_local_change" => Some(api::RiskCategory::TrivialLocalChange),
        "nontrivial_local_change" => Some(api::RiskCategory::NontrivialLocalChange),
        "external_change" => Some(api::RiskCategory::ExternalChange),
        "risky" => Some(api::RiskCategory::Risky),
        _ => None,
    }
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

/// Extracts a required `u32` field from a JSON object.
fn required_u32(value: &Value, key: &str) -> anyhow::Result<u32> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| anyhow!("Missing required u32 field '{key}'"))
}

/// Extracts an optional `i64` field from a JSON object.
fn optional_i64(value: &Value, key: &str) -> Option<i64> {
    value.get(key).and_then(Value::as_i64)
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

/// Normalizes Warp-style OpenAI GPT numeric model IDs and extracts a Responses reasoning effort when present.
fn normalize_openai_model_and_reasoning(
    model_id: &str,
) -> (String, Option<ResponsesReasoningConfig>) {
    let parts = model_id.split('-').collect::<Vec<_>>();
    if parts.len() == 4
        && parts[0] == "gpt"
        && parts[1].chars().all(|c| c.is_ascii_digit())
        && parts[2].chars().all(|c| c.is_ascii_digit())
        && is_supported_reasoning_effort(parts[3])
    {
        return (
            format!("gpt-{}.{}", parts[1], parts[2]),
            Some(ResponsesReasoningConfig {
                effort: parts[3].to_string(),
            }),
        );
    }

    if parts.len() == 3
        && parts[0] == "gpt"
        && parts[1].chars().all(|c| c.is_ascii_digit())
        && parts[2].chars().all(|c| c.is_ascii_digit())
    {
        return (format!("gpt-{}.{}", parts[1], parts[2]), None);
    }

    (model_id.to_string(), None)
}

/// Returns whether the provided suffix is a Responses reasoning effort we can forward directly.
fn is_supported_reasoning_effort(value: &str) -> bool {
    matches!(
        value,
        "none" | "minimal" | "low" | "medium" | "high" | "xhigh"
    )
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
    use std::borrow::Cow;
    use std::sync::Arc;

    use crate::ai::agent::conversation::AIConversationId;
    use crate::ai::agent::task::TaskId;
    use crate::ai::agent::AIAgentActionResult;
    use crate::ai::agent::AIAgentContext;
    use crate::ai::blocklist::SessionContext;
    use crate::ai::llms::{LLMId, LLMProvider};
    use ai::agent::action_result::{AIAgentActionResultType, ReadFilesResult};
    use warp_multi_agent_api as api;

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
                .map(|reasoning| reasoning.effort.as_str()),
            Some("low")
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
                .map(|reasoning| reasoning.effort.as_str()),
            Some("low")
        );
    }

    /// Verifies that base GPT numeric variants are normalized without inventing a reasoning effort.
    #[test]
    fn normalize_openai_model_and_reasoning_preserves_base_variant() {
        let (model, reasoning) = normalize_openai_model_and_reasoning("gpt-5-4");
        assert_eq!(model, "gpt-5.4");
        assert!(reasoning.is_none());
    }

    /// Verifies that base GPT-5.5 variants are normalized without inventing a reasoning effort.
    #[test]
    fn normalize_openai_model_and_reasoning_preserves_gpt_5_5_base_variant() {
        let (model, reasoning) = normalize_openai_model_and_reasoning("gpt-5-5");
        assert_eq!(model, "gpt-5.5");
        assert!(reasoning.is_none());
    }

    /// Verifies that MCP tool schemas reuse the server-provided description and JSON Schema.
    #[test]
    fn mcp_tool_schema_reuses_existing_metadata() {
        let schema =
            mcp_tool_schema(Some("server-1"), &test_mcp_tool()).expect("schema should exist");
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
