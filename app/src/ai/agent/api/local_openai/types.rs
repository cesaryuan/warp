//! Shared data types for the local OpenAI-compatible Responses backend.

use std::collections::HashMap;
use std::time::Instant;

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
    pub(super) summary: Vec<ResponsesReasoningSummaryPart>,
    #[serde(default)]
    pub(super) encrypted_content: Option<String>,
    #[serde(default)]
    pub(super) name: Option<String>,
    #[serde(default)]
    pub(super) call_id: Option<String>,
    #[serde(default)]
    pub(super) arguments: Option<String>,
    #[serde(default)]
    pub(super) status: Option<String>,
    #[serde(default)]
    pub(super) action: Option<ResponsesWebSearchAction>,
}

/// Parsed subset of a message content item returned by Responses.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct ResponsesContentItem {
    #[serde(rename = "type")]
    pub(super) item_type: String,
    #[serde(default)]
    pub(super) text: Option<String>,
    #[serde(default)]
    pub(super) annotations: Vec<ResponsesOutputTextAnnotation>,
}

/// Parsed subset of an `output_text` citation annotation returned by Responses.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct ResponsesOutputTextAnnotation {
    #[serde(rename = "type", default)]
    pub(super) item_type: String,
    #[serde(default)]
    pub(super) url: Option<String>,
    #[serde(default)]
    pub(super) title: Option<String>,
    #[serde(default)]
    pub(super) url_citation: Option<ResponsesNestedUrlCitation>,
}

/// Parsed subset of a nested `url_citation` payload used by some compatible providers.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct ResponsesNestedUrlCitation {
    #[serde(rename = "type", default)]
    pub(super) item_type: String,
    #[serde(default)]
    pub(super) url: Option<String>,
    #[serde(default)]
    pub(super) title: Option<String>,
}

/// Parsed subset of a reasoning summary part returned by Responses.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct ResponsesReasoningSummaryPart {
    #[serde(rename = "type")]
    pub(super) item_type: String,
    #[serde(default)]
    pub(super) text: Option<String>,
}

/// Parsed subset of a `web_search_call` action returned by Responses.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct ResponsesWebSearchAction {
    #[serde(rename = "type", default)]
    pub(super) action_type: String,
    #[serde(default)]
    pub(super) query: Option<String>,
    #[serde(default)]
    pub(super) sources: Vec<ResponsesWebSearchSource>,
}

/// Parsed subset of a single web-search source returned by Responses when included explicitly.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct ResponsesWebSearchSource {
    #[serde(rename = "type", default)]
    pub(super) source_type: String,
    #[serde(default)]
    pub(super) url: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) include: Vec<String>,
    pub(super) instructions: String,
    pub(super) input: Vec<Value>,
    pub(super) tools: Vec<Value>,
    pub(super) tool_choice: &'static str,
    pub(super) parallel_tool_calls: bool,
    pub(super) store: bool,
    pub(super) stream: bool,
}

/// Reasoning configuration supported by the Responses API.
#[derive(Debug, Clone, Serialize)]
pub(super) struct ResponsesReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) summary: Option<String>,
}

/// Tracks streamed assistant text for a single Responses output item.
#[derive(Debug, Clone)]
pub(super) struct StreamingTextMessageState {
    pub(super) message_id: String,
    pub(super) text: String,
}

/// Tracks a streamed web-search status message keyed by the Responses output item ID.
#[derive(Debug, Clone)]
pub(super) struct StreamingWebSearchState {
    pub(super) message_id: String,
}

/// Tracks streamed reasoning text for a single Responses reasoning item.
#[derive(Debug, Clone)]
pub(super) struct StreamingReasoningMessageState {
    pub(super) message_id: String,
    pub(super) text: String,
    pub(super) started_at: Instant,
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
#[derive(Debug)]
pub(super) struct StreamingResponsesAccumulator {
    pub(super) stream_started_at: Instant,
    pub(super) text_messages_by_item_id: HashMap<String, StreamingTextMessageState>,
    pub(super) emitted_text_item_ids: Vec<String>,
    pub(super) finalized_text_item_ids: Vec<String>,
    pub(super) reasoning_messages_by_key: HashMap<String, StreamingReasoningMessageState>,
    pub(super) emitted_reasoning_keys: Vec<String>,
    pub(super) reasoning_history_items_by_key: HashMap<String, Value>,
    pub(super) reasoning_history_item_keys_in_order: Vec<String>,
    pub(super) function_calls_by_call_id: HashMap<String, StreamingFunctionCallState>,
    pub(super) emitted_function_call_ids: Vec<String>,
    pub(super) web_search_states_by_item_id: HashMap<String, StreamingWebSearchState>,
    pub(super) emitted_web_search_item_ids: Vec<String>,
    pub(super) replayable_history_items_by_key: HashMap<String, Value>,
    pub(super) replayable_history_item_keys_in_order: Vec<String>,
}

impl Default for StreamingResponsesAccumulator {
    fn default() -> Self {
        Self {
            stream_started_at: Instant::now(),
            text_messages_by_item_id: HashMap::new(),
            emitted_text_item_ids: Vec::new(),
            finalized_text_item_ids: Vec::new(),
            reasoning_messages_by_key: HashMap::new(),
            emitted_reasoning_keys: Vec::new(),
            reasoning_history_items_by_key: HashMap::new(),
            reasoning_history_item_keys_in_order: Vec::new(),
            function_calls_by_call_id: HashMap::new(),
            emitted_function_call_ids: Vec::new(),
            web_search_states_by_item_id: HashMap::new(),
            emitted_web_search_item_ids: Vec::new(),
            replayable_history_items_by_key: HashMap::new(),
            replayable_history_item_keys_in_order: Vec::new(),
        }
    }
}

/// Minimal typed view over a streamed text delta event.
#[derive(Debug, Deserialize)]
pub(super) struct ResponsesTextDeltaEvent {
    pub(super) item_id: String,
    pub(super) delta: String,
}

/// Minimal typed view over a streamed reasoning text delta event.
#[derive(Debug, Deserialize)]
pub(super) struct ResponsesReasoningTextDeltaEvent {
    pub(super) item_id: String,
    #[serde(default)]
    pub(super) summary_index: Option<usize>,
    #[serde(default)]
    pub(super) content_index: Option<usize>,
    pub(super) delta: String,
}

/// Minimal typed view over a streamed reasoning text completion event.
#[derive(Debug, Deserialize)]
pub(super) struct ResponsesReasoningTextDoneEvent {
    pub(super) item_id: String,
    #[serde(default)]
    pub(super) summary_index: Option<usize>,
    #[serde(default)]
    pub(super) content_index: Option<usize>,
    #[serde(default)]
    pub(super) text: String,
    #[serde(default)]
    pub(super) part: Option<ResponsesContentItem>,
}

/// Minimal typed view over a streamed reasoning-part creation event.
#[derive(Debug, Deserialize)]
pub(super) struct ResponsesReasoningPartAddedEvent {
    pub(super) item_id: String,
    #[serde(default)]
    pub(super) summary_index: Option<usize>,
    #[serde(default)]
    pub(super) content_index: Option<usize>,
    pub(super) part: ResponsesContentItem,
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

/// Minimal typed view over a streamed web-search lifecycle event.
#[derive(Debug, Deserialize)]
pub(super) struct ResponsesWebSearchCallEvent {
    pub(super) item_id: String,
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
