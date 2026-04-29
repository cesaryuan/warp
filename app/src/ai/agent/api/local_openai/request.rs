//! Request-building helpers for the local OpenAI-compatible Responses backend.

use std::collections::HashMap;

use anyhow::anyhow;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde_json::{json, Value};

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

/// Starts a local Responses event stream after recording the new request inputs in conversation state.
pub(super) async fn start_local_responses_eventsource(
    server_api: &ServerApi,
    params: &RequestParams,
) -> anyhow::Result<http_client::EventSourceStream> {
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
            instructions: build_local_openai_system_prompt(&normalized_model),
            model: normalized_model,
            reasoning,
            input: state.items,
            tools: build_tools_payload(params),
            tool_choice: "auto",
            stream: true,
        }
    };

    Ok(server_api
        .http_client()
        .post(endpoint)
        .bearer_auth(api_key)
        .json(&request_body)
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
                effort: effort.to_string(),
            }),
        );
    }

    (model_id, None)
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
