//! Prompt construction for code review.

use crate::diff::{FileChangeKind, ParsedDiff};

/// System prompt shared by local and PR single-shot flows.
pub fn system_prompt() -> &'static str {
    "You are a senior software engineer performing a code review. \
Be concise and actionable. Structure your answer in Markdown with sections: \
Summary, Strengths, Issues (bugs, security, performance, maintainability), and Suggestions. \
If the diff is truncated or empty, say so briefly."
}

/// User content for a local branch review.
pub fn user_prompt_local(diff: &str, base: &str, head: &str) -> String {
    format!(
        "Review the following unified diff for the git range `{base}..{head}` (changes on `head` since merge-base with `base`).\n\n\
```diff\n{diff}\n```\n"
    )
}

/// User content for a GitHub pull request.
pub fn user_prompt_pr(
    diff: &str,
    owner: &str,
    repo: &str,
    number: u64,
    title: &str,
    author: &str,
) -> String {
    format!(
        "Review the following unified diff for GitHub pull request {owner}/{repo}#{number}.\n\
PR title: {title}\n\
Author: {author}\n\n\
```diff\n{diff}\n```\n"
    )
}

/// Serialize changed files for the agent manifest.
pub fn changed_files_summary(parsed: &ParsedDiff) -> String {
    #[derive(serde::Serialize)]
    struct Row<'a> {
        path: &'a str,
        kind: &'static str,
    }

    let rows: Vec<Row<'_>> = parsed
        .files
        .iter()
        .map(|f| {
            let kind = match f.kind {
                FileChangeKind::Added => "added",
                FileChangeKind::Deleted => "deleted",
                FileChangeKind::Modified => "modified",
                FileChangeKind::Renamed => "renamed",
            };
            Row {
                path: f.path.as_str(),
                kind,
            }
        })
        .collect();

    serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".to_string())
}

/// Agent manifest: no full diff inlined.
pub fn user_prompt_manifest_local(
    base: &str,
    head: &str,
    base_sha: &str,
    head_sha: &str,
    summary: &str,
) -> String {
    format!(
        "Review request (agent mode). The full unified diff is NOT inlined — use tools to inspect.\n\n\
Git range: `{base}..{head}`\n\
Merge-base (base) SHA: {base_sha}\n\
Head SHA: {head_sha}\n\n\
Changed files (parsed from patch):\n```json\n{summary}\n```\n\n\
When finished, follow the system instructions for Markdown + JSON findings.\n"
    )
}

/// Agent manifest for a GitHub PR.
#[allow(clippy::too_many_arguments)]
pub fn user_prompt_manifest_pr(
    owner: &str,
    repo: &str,
    number: u64,
    title: &str,
    author: &str,
    base_sha: &str,
    head_sha: &str,
    summary: &str,
) -> String {
    format!(
        "Review request (agent mode). The full unified diff is NOT inlined — use tools to inspect.\n\n\
Repository: {owner}/{repo} PR #{number}\n\
Title: {title}\n\
Author: {author}\n\
Base SHA: {base_sha}\n\
Head SHA: {head_sha}\n\n\
Changed files (parsed from patch):\n```json\n{summary}\n```\n\n\
When finished, follow the system instructions for Markdown + JSON findings.\n"
    )
}

/// Truncate diff by whole-file boundaries when possible; fallback to raw char cap.
pub fn maybe_truncate_diff(diff: &str, max_chars: usize) -> (String, bool) {
    crate::diff::truncate_diff_by_files(diff, max_chars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_adds_notice() {
        let huge = "x".repeat(100);
        let (t, truncated) = maybe_truncate_diff(&huge, 20);
        assert!(truncated);
        assert!(t.contains("truncated"));
        assert!(t.len() > 20);
        assert!(t.starts_with("xxxxxxxxxxxxxxxxxxxx"));
    }

    #[test]
    fn no_truncate_when_small() {
        let (t, truncated) = maybe_truncate_diff("small", 100);
        assert!(!truncated);
        assert_eq!(t, "small");
    }
}
