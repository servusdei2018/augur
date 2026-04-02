//! Agent orchestration: manifest prompt, tool loop, structured findings.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionToolArgs,
    FunctionObjectArgs,
};
use serde::Deserialize;

use crate::diff::ParsedDiff;
use crate::llm::{LlmConfig, ToolLoopConfig};
use crate::tools::{run_tool, ToolContext};

/// One line-targeted review comment (GitHub inline comment body + coordinates).
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct ReviewFinding {
    pub path: String,
    pub line: u32,
    pub body: String,
}

#[derive(Debug, Deserialize)]
struct FindingsPayload {
    findings: Vec<ReviewFinding>,
}

/// Final agent output: Markdown summary + validated per-line findings.
#[derive(Debug, Clone)]
pub struct ReviewOutput {
    pub markdown: String,
    pub findings: Vec<ReviewFinding>,
}

/// Tunables for the review agent.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_rounds: u32,
    pub max_tool_calls: u32,
    pub max_tool_output_chars: usize,
    pub max_patch_chars: usize,
    pub max_file_lines: usize,
    pub max_grep_matches: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_rounds: 24,
            max_tool_calls: 48,
            max_tool_output_chars: 400_000,
            max_patch_chars: 32_000,
            max_file_lines: 400,
            max_grep_matches: 80,
        }
    }
}

pub fn system_prompt_agent() -> &'static str {
    "You are a senior software engineer reviewing a patch. The full unified diff is NOT inlined; \
you MUST use the provided tools to list changed files, read per-file patches, read file contents at base/head when a local repo is available, and grep when needed. \
Work iteratively: start with the file list, then inspect high-risk or large changes. \
Be concise and actionable. \
When you are finished with exploration, write a Markdown review with sections: Summary, Strengths, Issues, Suggestions. \
After the Markdown, output a single JSON code block (fenced with ```json) containing an object: \
{\"findings\":[{\"path\":\"relative/path\",\"line\":42,\"body\":\"comment text\"}]} \
Use `line` as the line number on the NEW (right) side of the diff for additions/context you care about. \
Only include findings you can anchor to specific lines; omit the findings array if none."
}

/// Build OpenAI tool definitions for the review agent.
pub fn review_chat_tools() -> Vec<ChatCompletionTool> {
    vec![
        ChatCompletionToolArgs::default()
            .function(
                FunctionObjectArgs::default()
                    .name("list_changed_files")
                    .description("List files changed in this review with change kind and rough size.")
                    .parameters(serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }))
                    .build()
                    .expect("tool schema"),
            )
            .build()
            .expect("tool"),
        ChatCompletionToolArgs::default()
            .function(
                FunctionObjectArgs::default()
                    .name("read_patch")
                    .description("Return the unified diff text for one file, optionally one hunk by index (0-based).")
                    .parameters(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "Repository-relative file path (new path)" },
                            "hunk_index": { "type": "integer", "description": "Optional 0-based hunk index" },
                            "max_chars": { "type": "integer", "description": "Optional max characters of patch text" }
                        },
                        "required": ["path"],
                        "additionalProperties": false
                    }))
                    .build()
                    .expect("tool schema"),
            )
            .build()
            .expect("tool"),
        ChatCompletionToolArgs::default()
            .function(
                FunctionObjectArgs::default()
                    .name("read_file_at_ref")
                    .description("Read UTF-8 file content at merge-base (base) or PR head (head) from the local clone.")
                    .parameters(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "git_ref": { "type": "string", "description": "Either \"base\" or \"head\"" },
                            "path": { "type": "string" },
                            "start_line": { "type": "integer" },
                            "end_line": { "type": "integer" }
                        },
                        "required": ["git_ref", "path"],
                        "additionalProperties": false
                    }))
                    .build()
                    .expect("tool schema"),
            )
            .build()
            .expect("tool"),
        ChatCompletionToolArgs::default()
            .function(
                FunctionObjectArgs::default()
                    .name("grep_repo")
                    .description("Search with a regex (or literal if invalid regex) in changed files or the whole repo.")
                    .parameters(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "pattern": { "type": "string" },
                            "scope": { "type": "string", "enum": ["changed", "all"], "description": "Default changed" }
                        },
                        "required": ["pattern"],
                        "additionalProperties": false
                    }))
                    .build()
                    .expect("tool schema"),
            )
            .build()
            .expect("tool"),
    ]
}

/// Run the tool-using review agent.
pub async fn run_review_agent(
    llm: &LlmConfig,
    agent_cfg: &AgentConfig,
    tool_ctx: ToolContext,
    manifest_user_message: &str,
) -> Result<ReviewOutput> {
    let tools = Arc::new(review_chat_tools());
    let messages: Vec<ChatCompletionRequestMessage> = vec![
        ChatCompletionRequestSystemMessageArgs::default()
            .content(system_prompt_agent())
            .build()?
            .into(),
        ChatCompletionRequestUserMessageArgs::default()
            .content(manifest_user_message)
            .build()?
            .into(),
    ];

    let loop_cfg = ToolLoopConfig {
        max_rounds: agent_cfg.max_rounds,
        max_tool_calls: agent_cfg.max_tool_calls,
        max_tool_output_chars: agent_cfg.max_tool_output_chars,
    };

    let text = llm
        .chat_with_tools(
            messages,
            tools,
            {
                let ctx = tool_ctx.clone();
                move |name, args| run_tool(&ctx, name, args)
            },
            loop_cfg,
        )
        .await
        .context("agent chat_with_tools")?;

    let (markdown, mut findings) = split_markdown_and_findings(&text);
    findings = validate_findings(&tool_ctx.parsed, findings);

    Ok(ReviewOutput { markdown, findings })
}

fn split_markdown_and_findings(text: &str) -> (String, Vec<ReviewFinding>) {
    const FENCE: &str = "```json";
    let mut last_findings: Option<Vec<ReviewFinding>> = None;
    let mut last_fence_start: Option<usize> = None;
    let mut search = 0usize;

    while let Some(rel) = text.get(search..).and_then(|s| s.find(FENCE)) {
        let pos = search + rel;
        let after_fence = text.get(pos + FENCE.len()..).unwrap_or("");
        let after_fence = after_fence.trim_start();
        if let Some(end_rel) = after_fence.find("```") {
            let json_s = after_fence[..end_rel].trim();
            if let Ok(payload) = serde_json::from_str::<FindingsPayload>(json_s) {
                last_findings = Some(payload.findings);
                last_fence_start = Some(pos);
            }
            search = pos + FENCE.len() + end_rel + "```".len();
        } else {
            search = pos + FENCE.len();
        }
    }

    if let (Some(findings), Some(pos)) = (last_findings, last_fence_start) {
        return (text[..pos].trim().to_string(), findings);
    }

    if let Some((json_start, findings)) = try_parse_bare_findings_suffix(text) {
        let md = text[..json_start].trim().to_string();
        return (md, findings);
    }

    (text.trim().to_string(), vec![])
}

/// If the message ends with a bare `{"findings": ...}` object (no fence), parse it.
fn try_parse_bare_findings_suffix(text: &str) -> Option<(usize, Vec<ReviewFinding>)> {
    let t = text.trim_end();
    let key_pos = t.rfind("\"findings\"")?;
    let json_start = t[..key_pos].rfind('{')?;
    let payload = serde_json::from_str::<FindingsPayload>(&t[json_start..]).ok()?;
    Some((json_start, payload.findings))
}

/// Keep only findings that land on commentable new-side lines from the parsed diff.
pub fn validate_findings(parsed: &ParsedDiff, findings: Vec<ReviewFinding>) -> Vec<ReviewFinding> {
    let ok: HashSet<(String, u32)> = parsed.commentable_line_set();
    findings
        .into_iter()
        .filter(|f| ok.contains(&(f.path.clone(), f.line)))
        .collect()
}

/// Format findings as Markdown bullets for local output.
pub fn findings_to_markdown(findings: &[ReviewFinding]) -> String {
    if findings.is_empty() {
        return String::new();
    }
    let mut s = String::from("\n\n### Line comments\n\n");
    for f in findings {
        s.push_str(&format!("- `{}:{}` — {}\n", f.path, f.line, f.body));
    }
    s
}

#[cfg(test)]
mod tests {
    #[test]
    fn split_prefers_last_valid_json_fence() {
        let text = r#"Summary

```json
{"findings": [{"path": "a.rs", "line": 1, "body": "old"}]}
```

More text

```json
{"findings": [{"path": "b.rs", "line": 2, "body": "new"}]}
```
"#;
        let (md, findings) = super::split_markdown_and_findings(text);
        assert!(md.contains("Summary"));
        assert!(md.contains("More text"));
        assert!(!md.contains("b.rs"));
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].path, "b.rs");
        assert_eq!(findings[0].line, 2);
    }

    #[test]
    fn split_skips_invalid_fence_then_uses_valid() {
        let text = r#"Ok

```json
not json
```

```json
{"findings": [{"path": "x.rs", "line": 1, "body": "y"}]}
```
"#;
        let (_, findings) = super::split_markdown_and_findings(text);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].path, "x.rs");
    }

    #[test]
    fn split_bare_findings_at_end() {
        let text = r#"## Review

Good stuff.

{"findings": [{"path": "z.rs", "line": 3, "body": "note"}]}"#;
        let (md, findings) = super::split_markdown_and_findings(text);
        assert!(md.contains("Good stuff"));
        assert!(!md.contains("\"findings\""));
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].path, "z.rs");
    }

    #[test]
    fn validate_findings_keeps_only_commentable_lines() {
        let diff = r#"diff --git a/src/foo.rs b/src/foo.rs
index 111..222 100644
--- a/src/foo.rs
+++ b/src/foo.rs
@@ -1,3 +1,4 @@
 fn main() {
-    let x = 1;
+    let x = 2;
+    let y = 3;
 }
"#;
        let parsed = crate::diff::parse_unified_diff(diff).expect("parse");
        let findings = vec![
            super::ReviewFinding {
                path: "src/foo.rs".into(),
                line: 3,
                body: "on diff".into(),
            },
            super::ReviewFinding {
                path: "src/foo.rs".into(),
                line: 999,
                body: "bogus".into(),
            },
        ];
        let kept = super::validate_findings(&parsed, findings);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].line, 3);
    }
}
