//! Shared data types for the local OpenAI-compatible Responses backend.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::Event;

/// In-memory history for a local OpenAI-backed conversation.
#[derive(Debug, Default, Clone)]
pub(super) struct LocalConversationState {
    pub(super) items: Vec<Value>,
}

/// Parsed subset of a Responses API payload used by the local backend.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct ResponsesApiResponse {
    #[serde(default)]
    pub(super) output: Vec<ResponsesOutputItem>,
}

/// Parsed subset of a single Responses API output item.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct ResponsesOutputItem {
    #[serde(default)]
    pub(super) id: Option<String>,
    #[serde(rename = "type")]
    pub(super) item_type: String,
    #[serde(default)]
    pub(super) role: Option<String>,
    #[serde(default)]
    pub(super) content: Vec<ResponsesContentItem>,
    #[serde(default)]
    pub(super) name: Option<String>,
    #[serde(default)]
    pub(super) call_id: Option<String>,
    #[serde(default)]
    pub(super) arguments: Option<String>,
}

/// Parsed subset of a message content item returned by Responses.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct ResponsesContentItem {
    #[serde(rename = "type")]
    pub(super) item_type: String,
    #[serde(default)]
    pub(super) text: Option<String>,
}

/// Generic error envelope returned by OpenAI-compatible APIs.
#[derive(Debug, Deserialize)]
pub(super) struct ResponsesErrorEnvelope {
    pub(super) error: ResponsesErrorBody,
}

/// Minimal error body used for user-visible provider failures.
#[derive(Debug, Deserialize)]
pub(super) struct ResponsesErrorBody {
    #[serde(default)]
    pub(super) message: String,
}

/// Single function tool call extracted from a Responses output.
#[derive(Debug)]
pub(super) struct ParsedFunctionCall {
    pub(super) name: String,
    pub(super) call_id: String,
    pub(super) arguments: Value,
}

/// Request body sent to the local OpenAI-compatible `/v1/responses` endpoint.
#[derive(Debug, Serialize)]
pub(super) struct ResponsesRequestBody {
    pub(super) model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reasoning: Option<ResponsesReasoningConfig>,
    pub(super) instructions: String,
    pub(super) input: Vec<Value>,
    pub(super) tools: Vec<Value>,
    pub(super) tool_choice: &'static str,
    pub(super) parallel_tool_calls: bool,
    pub(super) stream: bool,
}

/// Reasoning configuration supported by the Responses API.
#[derive(Debug, Clone, Serialize)]
pub(super) struct ResponsesReasoningConfig {
    pub(super) effort: String,
}

/// Tracks streamed assistant text for a single Responses output item.
#[derive(Debug, Clone)]
pub(super) struct StreamingTextMessageState {
    pub(super) message_id: String,
    pub(super) text: String,
}

/// Tracks streamed function call arguments for a single Responses tool call.
#[derive(Debug, Default, Clone)]
pub(super) struct StreamingFunctionCallState {
    pub(super) output_item_id: Option<String>,
    pub(super) provider_call_id: Option<String>,
    pub(super) name: Option<String>,
    pub(super) arguments: String,
    pub(super) emitted: bool,
}

/// Aggregates streaming state until the Responses stream completes.
#[derive(Debug, Default)]
pub(super) struct StreamingResponsesAccumulator {
    pub(super) text_messages_by_item_id: HashMap<String, StreamingTextMessageState>,
    pub(super) emitted_text_item_ids: Vec<String>,
    pub(super) function_calls_by_call_id: HashMap<String, StreamingFunctionCallState>,
    pub(super) emitted_function_call_ids: Vec<String>,
}

/// Minimal typed view over a streamed text delta event.
#[derive(Debug, Deserialize)]
pub(super) struct ResponsesTextDeltaEvent {
    pub(super) item_id: String,
    pub(super) delta: String,
}

/// Minimal typed view over a streamed function call arguments delta event.
#[derive(Debug, Deserialize)]
pub(super) struct ResponsesFunctionCallArgumentsDeltaEvent {
    #[serde(default)]
    pub(super) call_id: Option<String>,
    #[serde(default)]
    pub(super) item_id: Option<String>,
    pub(super) delta: String,
}

/// Minimal typed view over a streamed function call arguments done event.
#[derive(Debug, Deserialize)]
pub(super) struct ResponsesFunctionCallArgumentsDoneEvent {
    #[serde(default)]
    pub(super) call_id: Option<String>,
    #[serde(default)]
    pub(super) item_id: Option<String>,
    #[serde(default)]
    pub(super) name: Option<String>,
    #[serde(default)]
    pub(super) arguments: String,
}

/// Minimal typed view over a streamed output item completion event.
#[derive(Debug, Deserialize)]
pub(super) struct ResponsesOutputItemDoneEvent {
    pub(super) item: ResponsesOutputItem,
}

/// Minimal typed view over a streamed completed event.
#[derive(Debug, Deserialize)]
pub(super) struct ResponsesCompletedEvent {
    pub(super) response: ResponsesApiResponse,
}

/// Result of translating a single streamed Responses SSE message into Warp client events.
#[derive(Debug, Default)]
pub(super) struct StreamMessageResult {
    pub(super) events: Vec<Event>,
    pub(super) is_terminal: bool,
}
