//! OpenAI function schema registry for Warp's built-in local backend tools.

use serde_json::{json, Map, Value};
use warp_multi_agent_api as api;

/// Returns the OpenAI function schema for a built-in Warp tool, if one is exposed locally.
pub(super) fn built_in_tool_schema(tool_type: api::ToolType) -> Option<Value> {
    match tool_type {
        api::ToolType::RunShellCommand => Some(function_schema(
            "run_shell_command",
            "Run a shell command in the user's workspace.",
            object_schema(
                vec![
                    (
                        "command",
                        string_schema(
                            "The exact shell command to execute in the current workspace.",
                        ),
                    ),
                    (
                        "is_read_only",
                        boolean_schema(
                            "Whether the command is expected to be read-only and avoid filesystem changes.",
                        ),
                    ),
                    (
                        "uses_pager",
                        boolean_schema(
                            "Whether the command is expected to launch a pager-like interface.",
                        ),
                    ),
                    (
                        "is_risky",
                        boolean_schema(
                            "Whether the agent believes the command is risky and should be user-approved.",
                        ),
                    ),
                    (
                        "wait_until_complete",
                        boolean_schema(
                            "Whether Warp should wait for the command to finish before returning a result.",
                        ),
                    ),
                    (
                        "risk_category",
                        enum_schema(
                            "A finer-grained risk classification that clients should prefer over the deprecated read-only and risky booleans.",
                            [
                                "unspecified",
                                "read_only",
                                "trivial_local_change",
                                "nontrivial_local_change",
                                "external_change",
                                "risky",
                            ],
                        ),
                    ),
                ],
                ["command"],
            ),
            true,
        )),
        api::ToolType::ReadFiles => Some(function_schema(
            "read_files",
            "Read one or more files from the workspace.",
            object_schema(
                vec![(
                    "files",
                    array_schema(
                        "The files and optional line ranges to read.",
                        object_schema(
                            vec![
                                (
                                    "name",
                                    string_schema(
                                        "The absolute or workspace-relative path of the file to read.",
                                    ),
                                ),
                                (
                                    "line_ranges",
                                    array_schema(
                                        "Optional 1-based line ranges to read. If omitted, Warp reads the entire file.",
                                        object_schema(
                                            vec![
                                                (
                                                    "start",
                                                    integer_schema("The inclusive 1-based starting line number."),
                                                ),
                                                (
                                                    "end",
                                                    integer_schema("The inclusive 1-based ending line number."),
                                                ),
                                            ],
                                            ["start", "end"],
                                        ),
                                    ),
                                ),
                            ],
                            ["name"],
                        ),
                    ),
                )],
                ["files"],
            ),
            true,
        )),
        api::ToolType::SearchCodebase => Some(function_schema(
            "search_codebase",
            "Search the indexed codebase for relevant files.",
            object_schema(
                vec![
                    (
                        "query",
                        string_schema("The semantic search query to run against the indexed codebase."),
                    ),
                    (
                        "path_filters",
                        array_schema(
                            "Optional path filters used to narrow the search to specific areas.",
                            string_schema("A path prefix or partial path filter."),
                        ),
                    ),
                    (
                        "codebase_path",
                        string_schema(
                            "Optional workspace path identifying which indexed codebase to search.",
                        ),
                    ),
                ],
                ["query"],
            ),
            true,
        )),
        api::ToolType::Grep => Some(function_schema(
            "grep",
            "Run grep-like text searches in the workspace.",
            object_schema(
                vec![
                    (
                        "queries",
                        array_schema(
                            "One or more literal or regex-like search terms to run.",
                            string_schema("A search query to match in file contents."),
                        ),
                    ),
                    (
                        "path",
                        string_schema(
                            "The directory path within the workspace where the grep search should run.",
                        ),
                    ),
                ],
                ["queries", "path"],
            ),
            true,
        )),
        api::ToolType::FileGlob => Some(function_schema(
            "file_glob",
            "Find files by glob pattern.",
            object_schema(
                vec![
                    (
                        "patterns",
                        array_schema(
                            "One or more glob patterns to evaluate.",
                            string_schema("A glob pattern such as src/**/*.rs."),
                        ),
                    ),
                    (
                        "path",
                        string_schema(
                            "Optional directory path that scopes the glob search.",
                        ),
                    ),
                ],
                ["patterns"],
            ),
            true,
        )),
        api::ToolType::FileGlobV2 => Some(function_schema(
            "file_glob_v2",
            "Find files by glob pattern with directory controls.",
            object_schema(
                vec![
                    (
                        "patterns",
                        array_schema(
                            "One or more glob patterns to evaluate.",
                            string_schema("A glob pattern such as src/**/*.rs."),
                        ),
                    ),
                    (
                        "search_dir",
                        string_schema("Optional directory path where the glob search should begin."),
                    ),
                    (
                        "max_matches",
                        integer_schema("Optional maximum number of matching files to return."),
                    ),
                    (
                        "max_depth",
                        integer_schema("Optional maximum directory traversal depth."),
                    ),
                    (
                        "min_depth",
                        integer_schema("Optional minimum directory traversal depth."),
                    ),
                ],
                ["patterns"],
            ),
            true,
        )),
        api::ToolType::ApplyFileDiffs => Some(function_schema(
            "apply_file_diffs",
            "Create, edit, move, or delete files in the workspace.",
            object_schema(
                vec![
                    (
                        "summary",
                        string_schema("A short summary of the intended file changes."),
                    ),
                    (
                        "diffs",
                        array_schema(
                            "Search-and-replace edits to apply to existing files.",
                            object_schema(
                                vec![
                                    (
                                        "file_path",
                                        string_schema("The file to update."),
                                    ),
                                    (
                                        "search",
                                        string_schema("The exact text to find in the file."),
                                    ),
                                    (
                                        "replace",
                                        string_schema("The replacement text to write."),
                                    ),
                                ],
                                ["file_path", "search", "replace"],
                            ),
                        ),
                    ),
                    (
                        "new_files",
                        array_schema(
                            "New files to create.",
                            object_schema(
                                vec![
                                    (
                                        "file_path",
                                        string_schema("The path of the new file to create."),
                                    ),
                                    (
                                        "content",
                                        string_schema("The full file contents to write."),
                                    ),
                                ],
                                ["file_path", "content"],
                            ),
                        ),
                    ),
                    (
                        "deleted_files",
                        array_schema(
                            "Existing files to delete.",
                            object_schema(
                                vec![(
                                    "file_path",
                                    string_schema("The path of the file to delete."),
                                )],
                                ["file_path"],
                            ),
                        ),
                    ),
                    (
                        "v4a_updates",
                        array_schema(
                            "Structured V4A patch updates for advanced file edits and moves.",
                            object_schema(
                                vec![
                                    (
                                        "file_path",
                                        string_schema("The file to update."),
                                    ),
                                    (
                                        "move_to",
                                        string_schema("Optional new destination path if the file should be moved."),
                                    ),
                                    (
                                        "hunks",
                                        array_schema(
                                            "Patch hunks to apply to the target file.",
                                            object_schema(
                                                vec![
                                                    (
                                                        "change_context",
                                                        array_schema(
                                                            "Optional contextual lines associated with the change.",
                                                            string_schema("A context line."),
                                                        ),
                                                    ),
                                                    (
                                                        "pre_context",
                                                        string_schema("Context immediately before the changed block."),
                                                    ),
                                                    (
                                                        "old",
                                                        string_schema("The original text being replaced."),
                                                    ),
                                                    (
                                                        "new",
                                                        string_schema("The new text that should replace the original."),
                                                    ),
                                                    (
                                                        "post_context",
                                                        string_schema("Context immediately after the changed block."),
                                                    ),
                                                ],
                                                [],
                                            ),
                                        ),
                                    ),
                                ],
                                ["file_path", "hunks"],
                            ),
                        ),
                    ),
                ],
                [],
            ),
            true,
        )),
        api::ToolType::ReadMcpResource => Some(function_schema(
            "read_mcp_resource",
            "Read a resource exposed by an MCP server.",
            object_schema(
                vec![
                    (
                        "uri",
                        string_schema("The MCP resource URI to read."),
                    ),
                    (
                        "server_id",
                        string_schema("Optional MCP server identifier when multiple servers are active."),
                    ),
                ],
                ["uri"],
            ),
            true,
        )),
        api::ToolType::WriteToLongRunningShellCommand => Some(function_schema(
            "write_to_long_running_shell_command",
            "Send input to a running shell command.",
            object_schema(
                vec![
                    (
                        "command_id",
                        string_schema("The long-running command identifier previously returned by Warp."),
                    ),
                    (
                        "input",
                        string_schema("The raw text or bytes to send to the running command."),
                    ),
                    (
                        "mode",
                        enum_schema(
                            "How Warp should write the provided input to the running command.",
                            ["raw", "line", "block"],
                        ),
                    ),
                ],
                ["command_id", "input"],
            ),
            true,
        )),
        api::ToolType::ReadShellCommandOutput => Some(function_schema(
            "read_shell_command_output",
            "Read the output of a running or completed shell command.",
            object_schema(
                vec![
                    (
                        "command_id",
                        string_schema("The command identifier previously returned by Warp."),
                    ),
                    (
                        "delay_seconds",
                        integer_schema("Optional delay in whole seconds before Warp returns command output."),
                    ),
                    (
                        "on_completion",
                        boolean_schema("If true, wait until the command completes before returning output."),
                    ),
                ],
                ["command_id"],
            ),
            true,
        )),
        api::ToolType::SuggestNewConversation => Some(function_schema(
            "suggest_new_conversation",
            "Suggest starting a new conversation from a specific message.",
            object_schema(
                vec![(
                    "message_id",
                    string_schema("The message that should become the split point for a new conversation."),
                )],
                ["message_id"],
            ),
            true,
        )),
        api::ToolType::ReadDocuments => Some(function_schema(
            "read_documents",
            "Read one or more Warp AI documents.",
            object_schema(
                vec![(
                    "documents",
                    array_schema(
                        "The documents and optional line ranges to read.",
                        object_schema(
                            vec![
                                (
                                    "document_id",
                                    string_schema("The document identifier to read."),
                                ),
                                (
                                    "line_ranges",
                                    array_schema(
                                        "Optional 1-based line ranges to read. If omitted, Warp reads the entire document.",
                                        object_schema(
                                            vec![
                                                ("start", integer_schema("The inclusive 1-based starting line number.")),
                                                ("end", integer_schema("The inclusive 1-based ending line number.")),
                                            ],
                                            ["start", "end"],
                                        ),
                                    ),
                                ),
                            ],
                            ["document_id"],
                        ),
                    ),
                )],
                ["documents"],
            ),
            true,
        )),
        api::ToolType::EditDocuments => Some(function_schema(
            "edit_documents",
            "Edit one or more existing Warp AI documents.",
            object_schema(
                vec![(
                    "diffs",
                    array_schema(
                        "Search-and-replace edits to apply to existing documents.",
                        object_schema(
                            vec![
                                ("document_id", string_schema("The document identifier to update.")),
                                ("search", string_schema("The exact text to find in the document.")),
                                ("replace", string_schema("The replacement text to write.")),
                            ],
                            ["document_id", "search", "replace"],
                        ),
                    ),
                )],
                ["diffs"],
            ),
            true,
        )),
        api::ToolType::CreateDocuments => Some(function_schema(
            "create_documents",
            "Create one or more new Warp AI documents.",
            object_schema(
                vec![(
                    "new_documents",
                    array_schema(
                        "The documents to create.",
                        object_schema(
                            vec![
                                ("content", string_schema("The full contents of the new document.")),
                                ("title", string_schema("An optional human-readable title for the new document.")),
                            ],
                            ["content"],
                        ),
                    ),
                )],
                ["new_documents"],
            ),
            true,
        )),
        api::ToolType::SuggestPrompt => Some(function_schema(
            "suggest_prompt",
            "Suggest a prompt for the user to run.",
            object_schema(
                vec![
                    (
                        "display_mode",
                        enum_schema(
                            "The UI presentation mode for the suggestion.",
                            ["inline_query_banner", "prompt_chip"],
                        ),
                    ),
                    (
                        "title",
                        string_schema("The title shown for an inline query banner suggestion."),
                    ),
                    (
                        "description",
                        string_schema("The descriptive text shown for an inline query banner suggestion."),
                    ),
                    (
                        "query",
                        string_schema("The query used when display_mode is inline_query_banner."),
                    ),
                    (
                        "prompt",
                        string_schema("The prompt used when display_mode is prompt_chip."),
                    ),
                    (
                        "label",
                        string_schema("An optional shorter UI label used for a prompt chip."),
                    ),
                    (
                        "is_trigger_irrelevant",
                        boolean_schema("Whether the original trigger is unrelated to the suggestion itself."),
                    ),
                ],
                ["display_mode"],
            ),
            true,
        )),
        api::ToolType::OpenCodeReview => Some(function_schema(
            "open_code_review",
            "Trigger the client to open the code review pane.",
            object_schema(vec![], []),
            true,
        )),
        api::ToolType::InitProject => Some(function_schema(
            "init_project",
            "Initialize the project setup flow on the client.",
            object_schema(vec![], []),
            true,
        )),
        api::ToolType::FetchConversation => Some(function_schema(
            "fetch_conversation",
            "Fetch tasks from another or the current conversation.",
            object_schema(
                vec![(
                    "conversation_id",
                    string_schema("Optional conversation identifier to fetch. Leave empty to target the current conversation."),
                )],
                [],
            ),
            true,
        )),
        api::ToolType::ReadSkill => Some(function_schema(
            "read_skill",
            "Read a skill from disk or from the bundled client skills.",
            object_schema(
                vec![
                    (
                        "skill_path",
                        string_schema("The path to a SKILL.md file to load."),
                    ),
                    (
                        "bundled_skill_id",
                        string_schema("The identifier of a skill bundled with the client."),
                    ),
                    (
                        "name",
                        string_schema("The human-readable name of the skill."),
                    ),
                ],
                [],
            ),
            true,
        )),
        // MCP tools are exposed through their own rich per-server schemas instead of this generic shell.
        api::ToolType::CallMcpTool => None,
        _ => None,
    }
}

/// Builds an OpenAI function tool definition for the Responses API.
fn function_schema(name: &str, description: &str, parameters: Value, strict: bool) -> Value {
    json!({
        "type": "function",
        "name": name,
        "description": description,
        "parameters": parameters,
        "strict": strict,
    })
}

/// Builds a JSON Schema object type with the provided properties and required keys.
fn object_schema<const N: usize>(properties: Vec<(&str, Value)>, required: [&str; N]) -> Value {
    let properties = properties
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect::<Map<String, Value>>();
    let required = required.into_iter().map(str::to_string).collect::<Vec<_>>();

    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

/// Builds a JSON Schema string field with a description.
fn string_schema(description: &str) -> Value {
    json!({
        "type": "string",
        "description": description,
    })
}

/// Builds a JSON Schema boolean field with a description.
fn boolean_schema(description: &str) -> Value {
    json!({
        "type": "boolean",
        "description": description,
    })
}

/// Builds a JSON Schema integer field with a description.
fn integer_schema(description: &str) -> Value {
    json!({
        "type": "integer",
        "description": description,
    })
}

/// Builds a JSON Schema enum field with a description.
fn enum_schema<const N: usize>(description: &str, values: [&str; N]) -> Value {
    let values = values.into_iter().map(str::to_string).collect::<Vec<_>>();
    json!({
        "type": "string",
        "description": description,
        "enum": values,
    })
}

/// Builds a JSON Schema array field with a description.
fn array_schema(description: &str, items: Value) -> Value {
    json!({
        "type": "array",
        "description": description,
        "items": items,
    })
}
