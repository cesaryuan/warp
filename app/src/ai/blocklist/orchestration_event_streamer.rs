use super::history_model::{BlocklistAIHistoryEvent, BlocklistAIHistoryModel};
use super::orchestration_events::{
    build_lifecycle_event, LifecycleEventDetailPayload, LifecycleEventDetailStage,
    OrchestrationEventService, PendingEvent, PendingEventDetail,
};
use crate::ai::agent::{
    conversation::AIConversationId, AIAgentExchangeId, AIAgentOutputMessageType,
    ReceivedMessageInput,
};
use crate::ai::agent_events::{
    run_agent_event_driver, AgentEventConsumer, AgentEventConsumerControlFlow,
    AgentEventDriverConfig, MessageHydrator, ServerApiAgentEventSource,
};
use crate::server::server_api::ai::{AIClient, AgentRunEvent};
use crate::server::server_api::{ServerApi, ServerApiProvider};
use anyhow::anyhow;
use async_trait::async_trait;
use futures::channel::mpsc;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;
use warp_core::features::FeatureFlag;
use warp_multi_agent_api as api;
use warpui::r#async::Timer;
use warpui::{
    Entity, EntityId, GetSingletonModelHandle, ModelContext, SingletonEntity, UpdateModel,
};

/// Backoff schedule (seconds) for the post-restore
/// `get_ambient_agent_task` retry: 1s, 2s, 5s, then 10s max.
const RESTORE_FETCH_BACKOFF_STEPS: &[u64] = &[1, 2, 5, 10];
/// How often (milliseconds) the drain timer checks for SSE events.
const SSE_DRAIN_INTERVAL_MS: u64 = 500;

/// Tracks messages awaiting server-side delivery confirmation.
struct PendingDeliveryConfirmation {
    message_ids: Vec<String>,
}

/// Per-event item delivered from the SSE background task to the entity.
struct SseStreamItem {
    event: AgentRunEvent,
    fetched_message: Option<ReceivedMessageInput>,
}

/// State for a single active SSE connection.
struct SseConnectionState {
    /// Receives parsed events from the background SSE task.
    event_receiver: mpsc::UnboundedReceiver<SseStreamItem>,
    /// Generation counter; used to discard stale callbacks after reconnect.
    generation: u64,
}

struct SseForwardingConsumer {
    tx: mpsc::UnboundedSender<SseStreamItem>,
    self_run_id: String,
    hydrator: MessageHydrator,
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl AgentEventConsumer for SseForwardingConsumer {
    async fn on_event(
        &mut self,
        event: AgentRunEvent,
    ) -> anyhow::Result<AgentEventConsumerControlFlow> {
        let fetched_message = self
            .hydrator
            .hydrate_event_for_recipient(&event, &self.self_run_id)
            .await;

        self.tx
            .unbounded_send(SseStreamItem {
                event,
                fetched_message,
            })
            .map_err(|_| anyhow!("SSE event receiver dropped"))?;

        Ok(AgentEventConsumerControlFlow::Continue)
    }
}

/// Async network coordinator for v2 orchestration event delivery via SSE.
///
/// Holds a long-lived SSE connection per *eligible* conversation. A
/// conversation is eligible iff there is an active consumer for it AND
/// the conversation has at least one role in an orchestration tree:
///
/// ```text
/// has_active_consumer()
///   AND (is_child_agent_conversation() OR has_at_least_one_watched_child_run_id)
/// ```
///
/// The active-consumer requirement applies to both roles. A child still
/// has its `self_run_id` watched as soon as the server token arrives,
/// but no SSE opens until something in this process actually consumes
/// the events — either an open agent view or an `agent_sdk` driver
/// (in CLI / cloud worker processes). Without a local consumer the
/// events would have nowhere to go, and any state the consumer cares
/// about can be backfilled via the cursor when one registers later.
pub struct OrchestrationEventStreamer {
    ai_client: Arc<dyn AIClient>,
    server_api: Arc<ServerApi>,
    /// Set of run_ids being watched on behalf of each conversation. For a
    /// child-role conversation this contains its `self_run_id` (its
    /// inbox); for a parent-role conversation this contains the run_ids
    /// of registered children. A dual-role conversation contains both.
    watched_run_ids: HashMap<AIConversationId, HashSet<String>>,
    /// Last fully handled event sequence per conversation.
    event_cursor: HashMap<AIConversationId, i64>,
    /// Messages awaiting server-side mark-delivered confirmation,
    /// triggered when the recipient streams a `MessagesReceivedFromAgents`
    /// chunk through `BlocklistAIHistoryEvent::UpdatedStreamingExchange`.
    pending_delivery: HashMap<AIConversationId, PendingDeliveryConfirmation>,
    /// Registered consumers per conversation, keyed by the consumer
    /// entity's `EntityId` (the terminal pane for an agent view, the
    /// driver model itself for `agent_sdk`). A non-empty set is what
    /// satisfies the parent-role gate of the eligibility predicate.
    consumers: HashMap<AIConversationId, HashSet<EntityId>>,
    /// Active SSE connections keyed by conversation.
    sse_connections: HashMap<AIConversationId, SseConnectionState>,
    /// Monotonic counter for SSE connection generations. Ensures stale
    /// callbacks from replaced connections are discarded.
    next_sse_generation: u64,
    /// Consecutive failure count for the post-restore
    /// `get_ambient_agent_task` fetch (resets on success). Drives
    /// exponential backoff for retries.
    restore_fetch_failures: HashMap<AIConversationId, usize>,
}

pub enum OrchestrationEventStreamerEvent {
    // Reserved for future use (e.g., status signals to the controller).
}

impl OrchestrationEventStreamer {
    pub fn new(ctx: &mut ModelContext<Self>) -> Self {
        let provider = ServerApiProvider::as_ref(ctx);
        let ai_client = provider.get_ai_client();
        let server_api = provider.get();
        let history_model = BlocklistAIHistoryModel::handle(ctx);
        ctx.subscribe_to_model(&history_model, |me, event, ctx| {
            me.handle_history_event(event, ctx);
        });
        Self {
            ai_client,
            server_api,
            watched_run_ids: HashMap::new(),
            event_cursor: HashMap::new(),
            pending_delivery: HashMap::new(),
            consumers: HashMap::new(),
            sse_connections: HashMap::new(),
            next_sse_generation: 0,
            restore_fetch_failures: HashMap::new(),
        }
    }

    /// Constructs a streamer wired to the supplied (mock) clients instead of
    /// looking them up via `ServerApiProvider`. Lets unit tests inject a
    /// `MockAIClient` while still subscribing to `BlocklistAIHistoryModel`.
    #[cfg(test)]
    pub(super) fn new_with_clients_for_test(
        ai_client: Arc<dyn AIClient>,
        server_api: Arc<ServerApi>,
        ctx: &mut ModelContext<Self>,
    ) -> Self {
        let history_model = BlocklistAIHistoryModel::handle(ctx);
        ctx.subscribe_to_model(&history_model, |me, event, ctx| {
            me.handle_history_event(event, ctx);
        });
        Self {
            ai_client,
            server_api,
            watched_run_ids: HashMap::new(),
            event_cursor: HashMap::new(),
            pending_delivery: HashMap::new(),
            consumers: HashMap::new(),
            sse_connections: HashMap::new(),
            next_sse_generation: 0,
            restore_fetch_failures: HashMap::new(),
        }
    }

    // ---- Public consumer registry API ---------------------------------

    /// Register a consumer for a conversation. Re-evaluates eligibility
    /// and opens the SSE connection if the conversation is newly
    /// eligible. Idempotent: re-registering an existing consumer is a
    /// no-op for the registry, but still triggers eligibility
    /// re-evaluation (which is itself idempotent).
    pub fn register_consumer(
        &mut self,
        conversation_id: AIConversationId,
        consumer_id: EntityId,
        ctx: &mut ModelContext<Self>,
    ) {
        let inserted = self
            .consumers
            .entry(conversation_id)
            .or_default()
            .insert(consumer_id);
        if inserted {
            log::info!(
                "register_consumer for {conversation_id:?}: {consumer_id:?} \
                 (total={})",
                self.consumers
                    .get(&conversation_id)
                    .map(|s| s.len())
                    .unwrap_or(0)
            );
        }
        // Driver-hosted callers stamp `parent_agent_id` immediately
        // before this call; if the server-token event fired earlier, the
        // helper picks up the just-stamped child role here.
        self.try_insert_self_run_id_if_in_tree(conversation_id, ctx);
        self.reevaluate_eligibility(conversation_id, ctx);
    }

    /// Unregister a consumer for a conversation. Re-evaluates eligibility
    /// and tears down the SSE connection if the conversation is no longer
    /// eligible (and the conversation is not also in the child role).
    pub fn unregister_consumer(
        &mut self,
        conversation_id: AIConversationId,
        consumer_id: EntityId,
        ctx: &mut ModelContext<Self>,
    ) {
        let removed = self
            .consumers
            .get_mut(&conversation_id)
            .map(|set| set.remove(&consumer_id))
            .unwrap_or(false);
        if removed {
            let remaining = self
                .consumers
                .get(&conversation_id)
                .map(|s| s.len())
                .unwrap_or(0);
            log::info!(
                "unregister_consumer for {conversation_id:?}: {consumer_id:?} \
                 (remaining={remaining})"
            );
            if let Some(set) = self.consumers.get(&conversation_id) {
                if set.is_empty() {
                    self.consumers.remove(&conversation_id);
                }
            }
        }
        self.reevaluate_eligibility(conversation_id, ctx);
    }

    /// Registers a run_id to watch for events on a conversation. Called
    /// by the start_agent executor for child run_ids and by the
    /// streamer's own helpers for self_run_id (child / parent inbox).
    pub fn register_watched_run_id(
        &mut self,
        conversation_id: AIConversationId,
        run_id: String,
        ctx: &mut ModelContext<Self>,
    ) {
        let inserted = self
            .watched_run_ids
            .entry(conversation_id)
            .or_default()
            .insert(run_id);
        // Adding the first child flips the conversation into the parent
        // role; ensure self_run_id is also watched so child→parent
        // messages match the SSE filter (without it the parent only sees
        // child lifecycle events).
        let self_inserted = self.try_insert_self_run_id_if_in_tree(conversation_id, ctx);
        if inserted || self_inserted {
            self.reevaluate_eligibility(conversation_id, ctx);
        }
    }

    // ---- Event subscriptions from BlocklistAIHistoryModel -------------

    fn handle_history_event(
        &mut self,
        event: &BlocklistAIHistoryEvent,
        ctx: &mut ModelContext<Self>,
    ) {
        match event {
            BlocklistAIHistoryEvent::ConversationServerTokenAssigned {
                conversation_id, ..
            } => self.on_server_token_assigned(*conversation_id, ctx),
            BlocklistAIHistoryEvent::UpdatedStreamingExchange {
                conversation_id,
                exchange_id,
                ..
            } => self.on_streaming_exchange_updated(*conversation_id, *exchange_id, ctx),
            BlocklistAIHistoryEvent::RemoveConversation {
                conversation_id, ..
            }
            | BlocklistAIHistoryEvent::DeletedConversation {
                conversation_id, ..
            } => {
                self.on_conversation_removed(*conversation_id, ctx);
            }
            BlocklistAIHistoryEvent::RestoredConversations {
                conversation_ids, ..
            } => {
                self.on_restored_conversations(conversation_ids.clone(), ctx);
            }
            BlocklistAIHistoryEvent::StartedNewConversation { .. }
            | BlocklistAIHistoryEvent::CreatedSubtask { .. }
            | BlocklistAIHistoryEvent::UpgradedTask { .. }
            | BlocklistAIHistoryEvent::AppendedExchange { .. }
            | BlocklistAIHistoryEvent::ReassignedExchange { .. }
            | BlocklistAIHistoryEvent::SetActiveConversation { .. }
            | BlocklistAIHistoryEvent::ClearedActiveConversation { .. }
            | BlocklistAIHistoryEvent::ClearedConversationsInTerminalView { .. }
            | BlocklistAIHistoryEvent::UpdatedTodoList { .. }
            | BlocklistAIHistoryEvent::UpdatedAutoexecuteOverride { .. }
            | BlocklistAIHistoryEvent::SplitConversation { .. }
            | BlocklistAIHistoryEvent::UpdatedConversationStatus { .. }
            | BlocklistAIHistoryEvent::UpdatedConversationMetadata { .. }
            | BlocklistAIHistoryEvent::UpdatedConversationArtifacts { .. } => {}
        }
    }

    fn on_server_token_assigned(
        &mut self,
        conversation_id: AIConversationId,
        ctx: &mut ModelContext<Self>,
    ) {
        if self.try_insert_self_run_id_if_in_tree(conversation_id, ctx) {
            self.reevaluate_eligibility(conversation_id, ctx);
        }
    }

    /// Inserts `self_run_id` into `watched_run_ids` if the conversation
    /// has any orchestration role (child or parent) and is not a passive
    /// remote-run view. Returns whether anything was inserted; callers
    /// reevaluate eligibility on `true`. Idempotent.
    fn try_insert_self_run_id_if_in_tree(
        &mut self,
        conversation_id: AIConversationId,
        ctx: &warpui::AppContext,
    ) -> bool {
        let (run_id, is_child) = {
            let history = BlocklistAIHistoryModel::as_ref(ctx);
            let Some(conversation) = history.conversation(&conversation_id) else {
                return false;
            };
            // Passive views of agent runs hosted elsewhere (shared-session
            // viewers and remote-child placeholders) must not subscribe —
            // the actual agent (in another process) is the inbox.
            if conversation.is_viewing_shared_session() || conversation.is_remote_child() {
                return false;
            }
            let Some(run_id) = conversation.run_id() else {
                return false;
            };
            // Child role: parent has a local placeholder
            // (parent_conversation_id) or we know the parent's run_id
            // (parent_agent_id under v2).
            let is_child = conversation.is_child_agent_conversation()
                || conversation.parent_agent_id().is_some();
            (run_id, is_child)
        };

        // Parent role: any watched run_id that isn't this conversation's
        // own self_run_id (i.e. a registered child).
        let is_parent = self
            .watched_run_ids
            .get(&conversation_id)
            .is_some_and(|set| set.iter().any(|id| id != &run_id));

        if !is_child && !is_parent {
            return false;
        }

        self.watched_run_ids
            .entry(conversation_id)
            .or_default()
            .insert(run_id)
    }

    fn on_streaming_exchange_updated(
        &mut self,
        conversation_id: AIConversationId,
        exchange_id: AIAgentExchangeId,
        ctx: &mut ModelContext<Self>,
    ) {
        let Some(pending) = self.pending_delivery.get(&conversation_id) else {
            return;
        };

        let Some(conversation) =
            BlocklistAIHistoryModel::as_ref(ctx).conversation(&conversation_id)
        else {
            return;
        };
        let Some(exchange) = conversation.exchange_with_id(exchange_id) else {
            return;
        };

        // Check if the exchange output contains any of the messages we're
        // waiting to confirm.
        let pending_ids: HashSet<&str> = pending.message_ids.iter().map(String::as_str).collect();
        let mut confirmed_ids = Vec::new();
        if let Some(output) = exchange.output_status.output() {
            for msg in &output.get().messages {
                if let AIAgentOutputMessageType::MessagesReceivedFromAgents { messages } =
                    &msg.message
                {
                    for received in messages {
                        if pending_ids.contains(received.message_id.as_str()) {
                            confirmed_ids.push(received.message_id.clone());
                        }
                    }
                }
            }
        }

        if confirmed_ids.is_empty() {
            return;
        }

        // Remove confirmed messages from pending.
        if let Some(pending) = self.pending_delivery.get_mut(&conversation_id) {
            pending.message_ids.retain(|id| !confirmed_ids.contains(id));
            if pending.message_ids.is_empty() {
                self.pending_delivery.remove(&conversation_id);
            }
        }

        let hydrator = MessageHydrator::new(self.ai_client.clone());
        ctx.spawn(
            async move {
                hydrator
                    .mark_messages_delivered_best_effort(confirmed_ids.iter().map(String::as_str))
                    .await
            },
            |_, failures, _| {
                for (message_id, err) in failures {
                    log::warn!("Failed to confirm message delivery for {message_id}: {err:#}");
                }
            },
        );
    }

    /// Cleans up local state for a removed/deleted conversation, then
    /// prunes the removed conversation's run_id from any *other*
    /// tracked conversation's `watched_run_ids` (in case it was a child
    /// of another parent we're still tracking) and re-evaluates
    /// eligibility for those parents.
    fn on_conversation_removed(
        &mut self,
        conversation_id: AIConversationId,
        ctx: &mut ModelContext<Self>,
    ) {
        // Capture the removed conversation's run_id BEFORE we clean up,
        // since the history model still holds the conversation record at
        // event-emit time.
        let removed_run_id = BlocklistAIHistoryModel::as_ref(ctx)
            .conversation(&conversation_id)
            .and_then(|c| c.run_id());

        // Local cleanup for the removed conversation itself.
        self.watched_run_ids.remove(&conversation_id);
        self.event_cursor.remove(&conversation_id);
        self.pending_delivery.remove(&conversation_id);
        self.consumers.remove(&conversation_id);
        self.restore_fetch_failures.remove(&conversation_id);
        // Dropping the SSE receiver causes the driver task's next send
        // to fail and exit; the drain timer's `is_current` check then
        // no-ops on its next tick.
        self.sse_connections.remove(&conversation_id);

        // Prune the removed conversation's run_id from every other
        // tracked conversation's watched set. If a parent's set becomes
        // empty (or its remaining state no longer makes it eligible),
        // tear down its SSE connection.
        if let Some(run_id) = removed_run_id.as_deref() {
            let mut affected = Vec::new();
            for (other_id, run_ids) in self.watched_run_ids.iter_mut() {
                if run_ids.remove(run_id) {
                    affected.push(*other_id);
                }
            }
            for other_id in affected {
                if self
                    .watched_run_ids
                    .get(&other_id)
                    .is_some_and(|s| s.is_empty())
                {
                    self.watched_run_ids.remove(&other_id);
                }
                self.reevaluate_eligibility(other_id, ctx);
            }
        }
    }

    // ---- Restore-on-startup ------------------------------------------

    /// Re-establishes orchestration event delivery state for conversations
    /// loaded from disk on startup. Initializes the in-memory cursor from
    /// the SQLite-persisted `last_event_sequence`, registers each
    /// conversation's own run_id as watched, and (when a run_id is
    /// available) issues `GET /agent/runs/{run_id}` to repopulate child
    /// run_ids and merge the server-side cursor. SSE eligibility is then
    /// re-evaluated through the standard predicate — it opens an SSE iff
    /// a consumer registers and the conversation has a role in the tree.
    fn on_restored_conversations(
        &mut self,
        conversation_ids: Vec<AIConversationId>,
        ctx: &mut ModelContext<Self>,
    ) {
        // Orchestration v2 owns the events endpoints and the cursor model.
        // V1 conversations may carry a run_id but the v2-only event APIs
        // would return spurious 4xx responses, so skip restore entirely
        // when V2 is disabled.
        if !FeatureFlag::OrchestrationV2.is_enabled() {
            return;
        }

        for conv_id in conversation_ids {
            let (run_id, cursor, is_remote_view) = {
                let history = BlocklistAIHistoryModel::as_ref(ctx);
                let Some(conversation) = history.conversation(&conv_id) else {
                    continue;
                };
                let is_remote_view =
                    conversation.is_viewing_shared_session() || conversation.is_remote_child();
                let run_id = conversation.run_id();
                let cursor = conversation.last_event_sequence().unwrap_or(0);
                (run_id, cursor, is_remote_view)
            };

            // Passive views of remote runs (shared-session viewers,
            // remote-child placeholders) must not subscribe — the actual
            // agent in another process owns the inbox.
            if is_remote_view {
                continue;
            }

            // Initialize the in-memory cursor from the persisted SQLite
            // value. A later server `GET /agent/runs/{run_id}` response
            // may advance it to `max(SQLite, server)` before delivery
            // starts.
            self.event_cursor.insert(conv_id, cursor);

            // Register the conversation's own run_id so lifecycle events
            // for self are correctly filtered and the SSE filter has a
            // run_id to open against once eligibility is met.
            if let Some(ref own) = run_id {
                self.watched_run_ids
                    .entry(conv_id)
                    .or_default()
                    .insert(own.clone());
            }

            // No run_id means we can't query the server for children or
            // for the canonical cursor. Re-evaluate eligibility based on
            // current state; a run_id assigned later flows through
            // `on_server_token_assigned`.
            let Some(run_id) = run_id else {
                self.reevaluate_eligibility(conv_id, ctx);
                continue;
            };

            let Ok(task_id) = run_id.parse::<crate::ai::ambient_agents::AmbientAgentTaskId>()
            else {
                log::warn!("could not parse run_id {run_id:?} for {conv_id:?}");
                self.reevaluate_eligibility(conv_id, ctx);
                continue;
            };

            self.spawn_restore_fetch(conv_id, task_id, cursor, ctx);
        }
    }

    /// Issues `GET /agent/runs/{task_id}` and routes the result through
    /// `finish_restore_fetch`. Used both for the initial post-restore
    /// fetch and for backoff-driven retries.
    fn spawn_restore_fetch(
        &mut self,
        conv_id: AIConversationId,
        task_id: crate::ai::ambient_agents::AmbientAgentTaskId,
        sqlite_cursor: i64,
        ctx: &mut ModelContext<Self>,
    ) {
        let ai_client = self.ai_client.clone();
        ctx.spawn(
            async move { ai_client.get_ambient_agent_task(&task_id).await },
            move |me, run_result, ctx| {
                me.finish_restore_fetch(conv_id, task_id, sqlite_cursor, run_result, ctx);
            },
        );
    }

    /// Completes the post-restore async fetch by merging the server cursor
    /// and installing the server-reported child run_ids. On a server-fetch
    /// failure, schedules a retry with exponential backoff: V2 children
    /// always have a server-side `ai_tasks` row, so the server is the
    /// authoritative source for the watched run_id set, and any local
    /// fallback would be incomplete. Without network connectivity event
    /// delivery wouldn't function anyway, so retrying is the right
    /// behavior. SSE is started/reconnected through `reevaluate_eligibility`
    /// so the standard consumer-and-role predicate gates delivery.
    fn finish_restore_fetch(
        &mut self,
        conv_id: AIConversationId,
        task_id: crate::ai::ambient_agents::AmbientAgentTaskId,
        sqlite_cursor: i64,
        run_result: anyhow::Result<crate::ai::ambient_agents::task::AmbientAgentTask>,
        ctx: &mut ModelContext<Self>,
    ) {
        match run_result {
            Ok(task) => {
                // If the conversation was removed while the fetch was
                // in-flight, the removal handler already cleaned up all
                // streamer state. Return early to avoid recreating
                // watched_run_ids for a deleted conversation.
                if !self.event_cursor.contains_key(&conv_id) {
                    self.restore_fetch_failures.remove(&conv_id);
                    return;
                }

                // Reset the retry counter on success.
                self.restore_fetch_failures.remove(&conv_id);

                // Merge the server cursor: use the max of SQLite and
                // server values so we don't re-deliver events the client
                // already acknowledged locally.
                let server_seq = task.last_event_sequence.unwrap_or(0);
                let merged = sqlite_cursor.max(server_seq);
                self.event_cursor.insert(conv_id, merged);

                // The server response includes `children` inline on
                // `AmbientAgentTask`; this is the authoritative set of
                // direct child run_ids for the parent.
                //
                // Insert children. If any new run_ids were added and an
                // SSE connection is already open (e.g. a status race
                // opened SSE with only the parent's own run_id), reconnect
                // so the new run_ids are included in the filter; otherwise
                // re-evaluate eligibility through the standard predicate.
                let had_sse = self.sse_connections.contains_key(&conv_id);
                let watched = self.watched_run_ids.entry(conv_id).or_default();
                let mut any_new_children = false;
                for child in task.children {
                    if watched.insert(child) {
                        any_new_children = true;
                    }
                }
                if any_new_children && had_sse {
                    self.reconnect_sse(conv_id, ctx);
                } else {
                    self.reevaluate_eligibility(conv_id, ctx);
                }
            }
            Err(err) => {
                log::warn!("Restore: get_agent_run failed for {conv_id:?}: {err:#}; will retry");
                self.start_restore_fetch_retry_timer(conv_id, task_id, sqlite_cursor, ctx);
            }
        }
    }

    /// Schedules a retry of the post-restore `get_ambient_agent_task`
    /// fetch after an exponential backoff (1s, 2s, 5s, 10s capped) keyed
    /// on a per-conversation failure counter. The counter resets on
    /// success.
    fn start_restore_fetch_retry_timer(
        &mut self,
        conv_id: AIConversationId,
        task_id: crate::ai::ambient_agents::AmbientAgentTaskId,
        sqlite_cursor: i64,
        ctx: &mut ModelContext<Self>,
    ) {
        let failures = self
            .restore_fetch_failures
            .entry(conv_id)
            .and_modify(|c| *c += 1)
            .or_insert(1);
        let step_index = failures
            .saturating_sub(1)
            .min(RESTORE_FETCH_BACKOFF_STEPS.len() - 1);
        let backoff = Duration::from_secs(RESTORE_FETCH_BACKOFF_STEPS[step_index]);
        ctx.spawn(
            async move { Timer::after(backoff).await },
            move |me, _, ctx| {
                // The conversation may have been removed in the meantime;
                // if so, drop the retry. Otherwise re-issue the fetch.
                if !me.event_cursor.contains_key(&conv_id) {
                    me.restore_fetch_failures.remove(&conv_id);
                    return;
                }
                me.spawn_restore_fetch(conv_id, task_id, sqlite_cursor, ctx);
            },
        );
    }

    // ---- Eligibility predicate ---------------------------------------

    /// Child role: the conversation either has a local placeholder for
    /// the parent (`parent_conversation_id`, set in the GUI parent) or
    /// knows the parent's server-side run_id (`parent_agent_id`,
    /// stamped by the agent_sdk driver in driver-hosted processes).
    fn is_child_agent_conversation(
        &self,
        conversation_id: AIConversationId,
        ctx: &warpui::AppContext,
    ) -> bool {
        BlocklistAIHistoryModel::as_ref(ctx)
            .conversation(&conversation_id)
            .is_some_and(|c| c.is_child_agent_conversation() || c.parent_agent_id().is_some())
    }

    fn self_run_id(
        &self,
        conversation_id: AIConversationId,
        ctx: &warpui::AppContext,
    ) -> Option<String> {
        BlocklistAIHistoryModel::as_ref(ctx)
            .conversation(&conversation_id)
            .and_then(|c| c.run_id())
    }

    /// True iff a parent has at least one watched child run_id (i.e. a
    /// run_id that is not the conversation's own self_run_id).
    fn has_watched_child_run_id(
        &self,
        conversation_id: AIConversationId,
        ctx: &warpui::AppContext,
    ) -> bool {
        let Some(set) = self.watched_run_ids.get(&conversation_id) else {
            return false;
        };
        let self_run_id = self.self_run_id(conversation_id, ctx);
        set.iter()
            .any(|id| Some(id.as_str()) != self_run_id.as_deref())
    }

    fn has_active_consumer(&self, conversation_id: AIConversationId) -> bool {
        self.consumers
            .get(&conversation_id)
            .is_some_and(|s| !s.is_empty())
    }

    /// True iff this conversation is a passive view of an agent run that
    /// is actually executing in another process — either a shared-session
    /// viewer or a placeholder for a remote child run spawned via
    /// `start_agent` with cloud `execution_mode`. Either way the actual
    /// run lives elsewhere (and that process owns the inbox), so this
    /// process should not open its own SSE for the conversation.
    fn is_remote_run_view(
        &self,
        conversation_id: AIConversationId,
        ctx: &warpui::AppContext,
    ) -> bool {
        BlocklistAIHistoryModel::as_ref(ctx)
            .conversation(&conversation_id)
            .is_some_and(|c| c.is_viewing_shared_session() || c.is_remote_child())
    }

    /// True iff this conversation should currently hold an SSE connection.
    /// A subscription is needed only when there is an active consumer in
    /// this process (an open agent view or an agent_sdk driver) AND the
    /// conversation has a real role to consume events for. Passive views
    /// of agent runs hosted elsewhere (shared-session viewers, remote
    /// child placeholders) are excluded regardless of their other state.
    fn is_eligible(&self, conversation_id: AIConversationId, ctx: &warpui::AppContext) -> bool {
        if !self.has_active_consumer(conversation_id) {
            return false;
        }
        if self.is_remote_run_view(conversation_id, ctx) {
            return false;
        }
        self.is_child_agent_conversation(conversation_id, ctx)
            || self.has_watched_child_run_id(conversation_id, ctx)
    }

    /// Returns the list of run_ids to subscribe to for `conversation_id`.
    /// Includes both the conversation's own `self_run_id` (when it is a
    /// child) and any registered child run_ids (when the conversation
    /// is a parent). Both contributions live in `watched_run_ids`
    /// already, so this is a straight clone.
    fn run_ids_for_sse(&self, conversation_id: AIConversationId) -> Vec<String> {
        self.watched_run_ids
            .get(&conversation_id)
            .into_iter()
            .flat_map(|set| set.iter().cloned())
            .collect()
    }

    /// Re-evaluates eligibility and either opens / reconnects or tears
    /// down the SSE connection for the given conversation.
    fn reevaluate_eligibility(
        &mut self,
        conversation_id: AIConversationId,
        ctx: &mut ModelContext<Self>,
    ) {
        let eligible = self.is_eligible(conversation_id, ctx);
        let connected = self.sse_connections.contains_key(&conversation_id);

        match (eligible, connected) {
            (true, false) => self.start_sse_connection(conversation_id, ctx),
            (true, true) => {
                // Already connected; reconnect with the current run_ids
                // list (in case the parent role's contribution
                // changed).
                self.reconnect_sse(conversation_id, ctx);
            }
            (false, true) => self.teardown_sse(conversation_id, ctx),
            (false, false) => {}
        }
    }

    /// Opens a long-lived SSE connection for `conversation_id`. Events
    /// are sent through an mpsc channel and drained by a periodic timer.
    fn start_sse_connection(
        &mut self,
        conversation_id: AIConversationId,
        ctx: &mut ModelContext<Self>,
    ) {
        let run_ids = self.run_ids_for_sse(conversation_id);
        if run_ids.is_empty() {
            return;
        }

        let cursor = self
            .event_cursor
            .get(&conversation_id)
            .copied()
            .unwrap_or(0);

        let server_api = self.server_api.clone();
        let ai_client = self.ai_client.clone();

        let self_run_id = self.self_run_id(conversation_id, ctx).unwrap_or_default();

        let (tx, rx) = mpsc::unbounded();
        let generation = self.next_sse_generation;
        self.next_sse_generation += 1;

        self.sse_connections.insert(
            conversation_id,
            SseConnectionState {
                event_receiver: rx,
                generation,
            },
        );

        log::info!(
            "Opening SSE stream for {conversation_id:?} (gen={generation}, \
             run_ids={run_ids:?}, since={cursor})"
        );

        let config = AgentEventDriverConfig::retry_forever(run_ids.clone(), cursor);
        let source = ServerApiAgentEventSource::new(server_api);
        let hydrator = MessageHydrator::new(ai_client);

        ctx.spawn(
            async move {
                let mut consumer = SseForwardingConsumer {
                    tx,
                    self_run_id,
                    hydrator,
                };
                run_agent_event_driver(source, config, &mut consumer).await
            },
            move |me, result, ctx| {
                let is_current = me
                    .sse_connections
                    .get(&conversation_id)
                    .is_some_and(|s| s.generation == generation);
                if !is_current {
                    return;
                }

                me.drain_sse_events(conversation_id, ctx);

                if let Err(err) = result {
                    log::warn!(
                        "SSE driver exited for {conversation_id:?} (gen={generation}): {err:#}"
                    );
                    me.reconnect_sse(conversation_id, ctx);
                }
            },
        );

        // Start periodic event drain.
        self.start_sse_drain_timer(conversation_id, generation, ctx);
    }

    /// Periodically fires to drain buffered SSE events into the event
    /// service.
    fn start_sse_drain_timer(
        &self,
        conversation_id: AIConversationId,
        generation: u64,
        ctx: &mut ModelContext<Self>,
    ) {
        ctx.spawn(
            async move {
                Timer::after(Duration::from_millis(SSE_DRAIN_INTERVAL_MS)).await;
            },
            move |me, _, ctx| {
                let is_current = me
                    .sse_connections
                    .get(&conversation_id)
                    .is_some_and(|s| s.generation == generation);
                if !is_current {
                    return;
                }
                me.drain_sse_events(conversation_id, ctx);
                me.start_sse_drain_timer(conversation_id, generation, ctx);
            },
        );
    }

    /// Drains all buffered SSE events and feeds them through the
    /// `handle_event_batch` sink.
    fn drain_sse_events(
        &mut self,
        conversation_id: AIConversationId,
        ctx: &mut ModelContext<Self>,
    ) {
        let Some(sse) = self.sse_connections.get_mut(&conversation_id) else {
            return;
        };

        let cursor = self
            .event_cursor
            .get(&conversation_id)
            .copied()
            .unwrap_or(0);

        let mut events = Vec::new();
        let mut messages = Vec::new();

        while let Ok(Some(item)) = sse.event_receiver.try_next() {
            // Deduplicate: discard events at or below the cursor.
            if item.event.sequence > cursor {
                if let Some(msg) = item.fetched_message {
                    messages.push(msg);
                }
                events.push(item.event);
            }
        }

        if events.is_empty() {
            return;
        }

        let self_run_id = self.self_run_id(conversation_id, ctx).unwrap_or_default();

        self.handle_event_batch(conversation_id, &self_run_id, cursor, events, messages, ctx);
    }

    /// Feeds a batch of fetched events through the OrchestrationEventService,
    /// updating the in-memory and persisted cursors and tracking message
    /// IDs awaiting delivery confirmation.
    fn handle_event_batch(
        &mut self,
        conversation_id: AIConversationId,
        self_run_id: &str,
        previous_cursor: i64,
        events: Vec<AgentRunEvent>,
        messages: Vec<ReceivedMessageInput>,
        ctx: &mut ModelContext<Self>,
    ) {
        let max_seq = events
            .iter()
            .map(|e| e.sequence)
            .max()
            .unwrap_or(previous_cursor);
        self.event_cursor.insert(conversation_id, max_seq);

        // Persist the cursor to SQLite so that after a restart we can
        // resume event delivery from this sequence number without
        // re-delivering events the parent has already acted on.
        BlocklistAIHistoryModel::handle(ctx).update(ctx, |model, ctx| {
            model.update_event_sequence(conversation_id, max_seq, ctx);
        });

        // Also persist the cursor to the server so driver / cloud
        // restarts can resume without local SQLite state. Fire-and-forget:
        // log on failure, don't block event delivery. The server persists
        // the cursor on `ai_tasks.last_event_sequence`.
        let own_run_id = BlocklistAIHistoryModel::as_ref(ctx)
            .conversation(&conversation_id)
            .and_then(|c| c.run_id());
        if let Some(run_id) = own_run_id {
            // TODO: consider debouncing this server write (see
            // specs/replay-agent-events-on-restore/TECH.md Risks).
            let ai_client = self.ai_client.clone();
            ctx.spawn(
                async move {
                    ai_client
                        .update_event_sequence_on_server(&run_id, max_seq)
                        .await
                },
                move |_, result, _| {
                    if let Err(err) = result {
                        log::warn!(
                            "Failed to persist event cursor to server for {conversation_id:?}: {err:#}"
                        );
                    }
                },
            );
        }

        // Track message IDs for server-side mark_delivered calls.
        let message_ids: Vec<String> = events
            .iter()
            .filter(|e| e.event_type == "new_message" && e.run_id == self_run_id)
            .filter_map(|e| e.ref_id.clone())
            .collect();
        if !message_ids.is_empty() {
            self.pending_delivery
                .entry(conversation_id)
                .or_insert_with(|| PendingDeliveryConfirmation {
                    message_ids: Vec::new(),
                })
                .message_ids
                .extend(message_ids);
        }

        let lifecycle_events = convert_lifecycle_events(&events, self_run_id);
        if messages.is_empty() && lifecycle_events.is_empty() {
            return;
        }

        let pending = build_pending_events(messages, lifecycle_events);
        OrchestrationEventService::handle(ctx).update(ctx, |svc, ctx| {
            svc.enqueue_event_batch(conversation_id, pending, ctx);
        });
    }

    /// Tears down the current SSE connection and (if still eligible)
    /// opens a new one with the latest run_ids list and cursor.
    fn reconnect_sse(&mut self, conversation_id: AIConversationId, ctx: &mut ModelContext<Self>) {
        // Drain buffered events before dropping the channel so we don't
        // discard already-fetched message bodies.
        self.drain_sse_events(conversation_id, ctx);
        self.sse_connections.remove(&conversation_id);

        if self.is_eligible(conversation_id, ctx) {
            self.start_sse_connection(conversation_id, ctx);
        }
    }

    /// Drops the SSE connection for a no-longer-eligible conversation.
    /// Leaves `watched_run_ids` and `consumers` alone — those reflect
    /// external state and are pruned through their own paths.
    fn teardown_sse(&mut self, conversation_id: AIConversationId, ctx: &mut ModelContext<Self>) {
        // Drain anything buffered so we don't lose hydrated messages.
        self.drain_sse_events(conversation_id, ctx);
        if self.sse_connections.remove(&conversation_id).is_some() {
            log::info!("Tearing down SSE for {conversation_id:?} (no longer eligible)");
        }
    }
}

impl Entity for OrchestrationEventStreamer {
    type Event = OrchestrationEventStreamerEvent;
}

impl SingletonEntity for OrchestrationEventStreamer {}

fn parse_occurred_at(s: &str) -> prost_types::Timestamp {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| prost_types::Timestamp {
            seconds: dt.timestamp(),
            nanos: dt.timestamp_subsec_nanos() as i32,
        })
        .unwrap_or_else(|_| {
            let now = chrono::Utc::now();
            prost_types::Timestamp {
                seconds: now.timestamp(),
                nanos: now.timestamp_subsec_nanos() as i32,
            }
        })
}

fn convert_lifecycle_events(events: &[AgentRunEvent], self_run_id: &str) -> Vec<api::AgentEvent> {
    events
        .iter()
        .filter(|e| e.event_type != "new_message" && e.run_id != self_run_id)
        .filter_map(|event| {
            let lifecycle_type = match event.event_type.as_str() {
                // New canonical event types aligned with task states.
                "run_in_progress" => api::LifecycleEventType::InProgress,
                "run_succeeded" => api::LifecycleEventType::Succeeded,
                "run_failed" => api::LifecycleEventType::Failed,
                // Legacy event types mapped to new variants for backward compat.
                #[allow(deprecated)]
                "run_started" => api::LifecycleEventType::InProgress,
                #[allow(deprecated)]
                "run_idle" => api::LifecycleEventType::Succeeded,
                #[allow(deprecated)]
                "run_restarted" => api::LifecycleEventType::InProgress,
                "run_errored" => api::LifecycleEventType::Errored,
                "run_cancelled" => api::LifecycleEventType::Cancelled,
                "run_blocked" => api::LifecycleEventType::Blocked,
                _ => return None,
            };
            let timestamp = parse_occurred_at(&event.occurred_at);
            // TODO: Parse richer detail payloads (reason, error_message) from
            // the server event log once the schema supports them.
            let detail = match lifecycle_type {
                api::LifecycleEventType::Errored => LifecycleEventDetailPayload {
                    stage: Some(LifecycleEventDetailStage::Runtime),
                    reason: event.ref_id.clone(),
                    ..Default::default()
                },
                _ => LifecycleEventDetailPayload::default(),
            };
            let event_id = Uuid::new_v4().to_string();
            Some(build_lifecycle_event(
                event_id,
                event.run_id.clone(),
                lifecycle_type,
                timestamp,
                &detail,
            ))
        })
        .collect()
}

fn build_pending_events(
    messages: Vec<ReceivedMessageInput>,
    lifecycle_events: Vec<api::AgentEvent>,
) -> Vec<PendingEvent> {
    let mut pending = Vec::with_capacity(messages.len() + lifecycle_events.len());
    for msg in &messages {
        pending.push(PendingEvent {
            event_id: msg.message_id.clone(),
            source_agent_id: msg.sender_agent_id.clone(),
            attempt_count: 0,
            detail: PendingEventDetail::Message {
                message_id: msg.message_id.clone(),
                addresses: msg.addresses.clone(),
                subject: msg.subject.clone(),
                message_body: msg.message_body.clone(),
            },
        });
    }
    for event in lifecycle_events {
        pending.push(PendingEvent {
            event_id: event.event_id.clone(),
            source_agent_id: String::new(),
            attempt_count: 0,
            detail: PendingEventDetail::Lifecycle { event },
        });
    }
    pending
}

// ---- Free-function consumer registration helpers ---------------------
//
// Wrap the feature-flag check + singleton handle update so call sites
// in `ActiveAgentViewsModel` and the agent_sdk driver don't have to
// repeat the boilerplate. The generic bound covers both
// `&mut AppContext` and `&mut ModelContext<T>` / `&mut ViewContext<T>`.
//
// Consumers are identified by an `EntityId` — the terminal pane's id
// for an agent view, the driver model's id for `agent_sdk`. The
// streamer never branches on consumer kind, so a single pair of helpers
// covers both call sites.

/// Registers a consumer of orchestration agent events for
/// `conversation_id`. No-op when `OrchestrationV2` is disabled.
pub fn register_agent_event_consumer<C>(
    conversation_id: AIConversationId,
    consumer_id: EntityId,
    ctx: &mut C,
) where
    C: GetSingletonModelHandle + UpdateModel,
{
    if !FeatureFlag::OrchestrationV2.is_enabled() {
        return;
    }
    OrchestrationEventStreamer::handle(ctx).update(ctx, |streamer, ctx| {
        streamer.register_consumer(conversation_id, consumer_id, ctx);
    });
}

/// Pair to [`register_agent_event_consumer`].
pub fn unregister_agent_event_consumer<C>(
    conversation_id: AIConversationId,
    consumer_id: EntityId,
    ctx: &mut C,
) where
    C: GetSingletonModelHandle + UpdateModel,
{
    if !FeatureFlag::OrchestrationV2.is_enabled() {
        return;
    }
    OrchestrationEventStreamer::handle(ctx).update(ctx, |streamer, ctx| {
        streamer.unregister_consumer(conversation_id, consumer_id, ctx);
    });
}

#[cfg(test)]
#[path = "orchestration_event_streamer_tests.rs"]
mod tests;
