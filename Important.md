# 常用命令

```pwsh
.\script\windows\bundle.ps1 -Channel oss -Arch x64 -ReleaseTag "$(git describe --tags --always)"

.\script\create_github_release.ps1 -AssetPath .\script\windows\Output\WarpOssSetup.exe -DryRun
```

# Local OpenAI Web Search 约定

`app/src/ai/agent/api/local_openai` 现在支持 OpenAI Responses API 的 built-in `web_search` tool，但这条能力不是通过 Warp 的 `ToolType` / `tool_calls.rs` 内建函数分发实现的，而是直接把 `{ "type": "web_search" }` 放进 Responses 请求的 `tools` 里。后续如果再看这块，不要去 `built_in_tool_schema()` 或 `parse_tool_call()` 里找 `web_search` 的 function tool 分支。

`response.completed.output` 不能当成 web search 的主数据源。实际接入里，大部分关键 output item，尤其是 `web_search_call`，可能只会在 SSE 过程中通过 `response.output_item.done` 出现；`response.completed` 更适合作为兜底，而不是主路径。以后如果发现“UI 有流式输出但 completed 里是空的”，先检查 SSE 处理，不要先怀疑上层逻辑。

对 assistant 文本和 citations，最终真相也应该以 `response.output_item.done` 为准，而不是只依赖 `response.output_text.delta`。`delta` 负责让 UI 尽早看到文本，`output_item.done` 负责补齐最终文本、annotations、citations；否则很容易出现正文有了但网页引用丢失的情况。

如果继续使用 `store: false` 的本地手动回放模式，上一轮的 `web_search_call` 应该回传给 OpenAI。原因不是“没有它一定报错”，而是为了尽量复刻 Responses API 的真实会话状态，避免模型丢失“上一轮已经搜索过什么”的轨迹，减少重复搜索和上下文漂移。

回放时不要只保存 Warp UI 层抽象后的消息，尽量保留可重建的 Responses output item 语义。当前已经把 assistant `message`、`reasoning`、`function_call`、`web_search_call` 都纳入了 replayable history；其中 assistant `message` 还需要连 `output_text.annotations` 一起保留，否则网页 citations 无法在下一轮完整回放。

当 `response.completed.output` 为空时，不能只靠已有的文本缓存兜底，因为那样会丢 output item 的类型信息和顺序。现在的做法是按 SSE 到达顺序记录 replayable history item，再在 finalize 阶段优先从这份记录回放；后续如果新增新的 Responses output item，也优先接入这套 replay 机制，而不是临时拼字符串。

`response.web_search_call.searching` 目前主要用于 UI 上先展示 searching 状态，真正需要持久化回放的是最终完成态或失败态的 `web_search_call`。也就是说，searching 阶段的即时状态可以更新 UI，但长期上下文里更重要的是最终 `output_item.done` 产出的结果。

# Windows OSS 安装包约定

本地如果要产出和 `.github/workflows/release_master_from_upstream_stable.yml` 那条 OSS release 流程同类型的 Windows 安装包，不要直接猜 `cargo bundle` 或手搓 `ISCC` 参数，优先走 `script/windows/bundle.ps1 -Channel oss -Arch x64`。这条链路和 CI 对齐，最终产物名就是 `script/windows/Output/WarpOssSetup.exe`。

CI 的 Windows release 实际是两段式调用同一个脚本：先 `-SkipBuildInstaller` 编二进制，再 `-SkipBuildBinary` 组装 installer；但本地如果只是想拿到可安装的 `WarpOssSetup.exe`，单次运行 `bundle.ps1` 就够了。是否签名不是这条 OSS 本地构建的重点，默认不签名也和当前 OSS release workflow 保持一致。
