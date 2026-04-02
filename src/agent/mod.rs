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
    /// Recommended review action: `"APPROVE"`, `"REQUEST_CHANGES"`, or `"COMMENT"` (default).
    action: Option<String>,
}

/// Final agent output: Markdown summary + validated per-line findings.
#[derive(Debug, Clone)]
pub struct ReviewOutput {
    pub markdown: String,
    pub findings: Vec<ReviewFinding>,
    /// Suggested review action produced by the agent (`APPROVE`, `REQUEST_CHANGES`, `COMMENT`).
    /// `None` means the agent did not provide a recommendation.
    pub suggested_action: Option<String>,
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
    r#"You are an expert staff software engineer performing a thorough, meticulous code review.
The full unified diff is NOT inlined - you MUST use the provided tools extensively:
start by listing all changed files, then read each file's patch carefully, and use read_file_at_ref
and grep_repo to understand surrounding context before drawing any conclusions.
Do NOT summarise what the patch does without reading it first.

## Review standards - follow these strictly

**Specificity is mandatory.** Every issue or suggestion MUST:
  - Name the exact file path and line number(s) it applies to.
  - Quote or paraphrase the relevant code fragment.
  - Explain precisely WHY it is a problem (e.g. what invariant breaks, what edge case fails,
    what performance cliff is hit, what security boundary is crossed).
  - Propose a concrete fix or alternative.

**Forbidden content** - do NOT write:
  - Vague statements like "this is complex", "consider improving documentation",
    "ensure thorough testing", "this may cause issues", or any comment that could apply to any PR.
  - Meta-observations about the size or scope of the diff.
  - Generic encouragements like "good job" or "looks fine overall".
  - Suggestions without a specific location in the code.

**Always look for**:
  - Logic errors, off-by-one errors, incorrect conditionals.
  - Unchecked error paths, missing error propagation, swallowed exceptions.
  - Race conditions, incorrect use of locks/atomics, shared-state bugs.
  - Security issues: injection, unvalidated input, leaked secrets, privilege escalation paths.
  - Performance regressions: unnecessary allocations, O(n^2) loops, blocking calls in async context.
  - API misuse: wrong argument order, missing required parameters, deprecated functions.
  - Test gaps: changed behaviour with no corresponding test update.
  - Broken or missing error messages, panics on bad input.

## Output format

Write a detailed Markdown review with the following sections. Each section must be substantive.

### Summary
One to three paragraphs describing what the PR actually does (based on reading the diff),
its architectural impact, and any overarching concerns.

### Strengths
Bullet list. Each bullet must reference a specific file/approach and explain concretely why it is good.

### Issues
Bullet list. This is the most important section. Each bullet must follow this format:

  **[Severity: Critical|High|Medium|Low]** `path/to/file.ext:LINE` - <concise title>
  2-5 sentences: what the code does, why it is wrong, and the exact fix.

If there are no genuine issues, say so explicitly - do not invent problems.

### Suggestions
Non-blocking improvements. Same specificity rules as Issues, but labelled as optional polish.

---
After the Markdown, output a single JSON code block (fenced with triple backticks and the word json) containing:
  "findings": array of {"path":"relative/path","line":42,"body":"comment"}
    - one entry per Issue/Suggestion anchored to a specific diff line; omit or use [] if none.
  "action": "APPROVE" | "REQUEST_CHANGES" | "COMMENT"
    - APPROVE only when there are zero Critical or High issues and the code is ready to merge;
    REQUEST_CHANGES when Critical or High issues exist; COMMENT otherwise.
Use "line" as the line number on the NEW (right) side of the diff."#
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

    let (markdown, mut findings, suggested_action) = split_markdown_and_findings(&text);
    findings = validate_findings(&tool_ctx.parsed, findings);

    Ok(ReviewOutput {
        markdown,
        findings,
        suggested_action,
    })
}

fn split_markdown_and_findings(text: &str) -> (String, Vec<ReviewFinding>, Option<String>) {
    const FENCE: &str = "```json";
    let mut last_findings: Option<Vec<ReviewFinding>> = None;
    let mut last_action: Option<String> = None;
    let mut last_fence_start: Option<usize> = None;
    let mut search = 0usize;

    while let Some(rel) = text.get(search..).and_then(|s| s.find(FENCE)) {
        let pos = search + rel;
        let after_fence = text.get(pos + FENCE.len()..).unwrap_or("");
        let after_fence = after_fence.trim_start();
        if let Some(end_rel) = after_fence.find("```") {
            let json_s = after_fence[..end_rel].trim();
            if let Ok(payload) = serde_json::from_str::<FindingsPayload>(json_s) {
                last_action = payload.action;
                last_findings = Some(payload.findings);
                last_fence_start = Some(pos);
            }
            search = pos + FENCE.len() + end_rel + "```".len();
        } else {
            search = pos + FENCE.len();
        }
    }

    if let (Some(findings), Some(pos)) = (last_findings, last_fence_start) {
        return (text[..pos].trim().to_string(), findings, last_action);
    }

    if let Some((json_start, findings, action)) = try_parse_bare_findings_suffix(text) {
        let md = text[..json_start].trim().to_string();
        return (md, findings, action);
    }

    (text.trim().to_string(), vec![], None)
}

/// If the message ends with a bare `{"findings": ...}` object (no fence), parse it.
fn try_parse_bare_findings_suffix(
    text: &str,
) -> Option<(usize, Vec<ReviewFinding>, Option<String>)> {
    let t = text.trim_end();
    let key_pos = t.rfind("\"findings\"")?;
    let json_start = t[..key_pos].rfind('{')?;
    let payload = serde_json::from_str::<FindingsPayload>(&t[json_start..]).ok()?;
    Some((json_start, payload.findings, payload.action))
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
        let (md, findings, _action) = super::split_markdown_and_findings(text);
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
        let (_, findings, _action) = super::split_markdown_and_findings(text);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].path, "x.rs");
    }

    #[test]
    fn split_bare_findings_at_end() {
        let text = r#"## Review

Good stuff.

{"findings": [{"path": "z.rs", "line": 3, "body": "note"}]}"#;
        let (md, findings, _action) = super::split_markdown_and_findings(text);
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
