//! Sandboxed tool implementations for the review agent.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;

use crate::diff::{FileChangeKind, ParsedDiff};
use crate::git::repo::read_file_at_commit;

/// Budgets and paths for tool execution.
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub repo_root: Option<std::path::PathBuf>,
    pub base_sha: String,
    pub head_sha: String,
    pub parsed: ParsedDiff,
    pub max_patch_chars: usize,
    pub max_file_lines: usize,
    pub max_grep_matches: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ListChangedFilesArgs {}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ReadPatchArgs {
    pub path: String,
    #[serde(default)]
    pub hunk_index: Option<usize>,
    #[serde(default)]
    pub max_chars: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ReadFileAtRefArgs {
    /// `"base"` or `"head"` (resolved SHAs in context).
    pub git_ref: String,
    pub path: String,
    #[serde(default)]
    pub start_line: Option<usize>,
    #[serde(default)]
    pub end_line: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GrepRepoArgs {
    pub pattern: String,
    /// `"changed"` (default) or `"all"`.
    #[serde(default = "default_scope")]
    pub scope: String,
}

fn default_scope() -> String {
    "changed".to_string()
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LookupHunkArgs {
    pub path: String,
    pub line: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ReadHunkContextArgs {
    pub path: String,
    pub line: u32,
    #[serde(default)]
    pub radius: Option<usize>,
}

/// Run a tool by name with JSON arguments; returns JSON string for the assistant.
pub fn run_tool(ctx: &ToolContext, name: &str, args_json: &str) -> String {
    match name {
        "list_changed_files" => list_changed_files(ctx),
        "read_patch" => match serde_json::from_str::<ReadPatchArgs>(args_json) {
            Ok(a) => read_patch(ctx, a),
            Err(e) => json!({ "error": format!("invalid arguments: {e}") }).to_string(),
        },
        "read_file_at_ref" => match serde_json::from_str::<ReadFileAtRefArgs>(args_json) {
            Ok(a) => read_file_at_ref(ctx, a),
            Err(e) => json!({ "error": format!("invalid arguments: {e}") }).to_string(),
        },
        "grep_repo" => match serde_json::from_str::<GrepRepoArgs>(args_json) {
            Ok(a) => grep_repo(ctx, a),
            Err(e) => json!({ "error": format!("invalid arguments: {e}") }).to_string(),
        },
        "lookup_hunk" => match serde_json::from_str::<LookupHunkArgs>(args_json) {
            Ok(a) => lookup_hunk(ctx, a),
            Err(e) => json!({ "error": format!("invalid arguments: {e}") }).to_string(),
        },
        "read_hunk_context" => match serde_json::from_str::<ReadHunkContextArgs>(args_json) {
            Ok(a) => read_hunk_context(ctx, a),
            Err(e) => json!({ "error": format!("invalid arguments: {e}") }).to_string(),
        },
        _ => json!({ "error": format!("unknown tool: {name}") }).to_string(),
    }
}

fn list_changed_files(ctx: &ToolContext) -> String {
    #[derive(Serialize)]
    struct Row<'a> {
        path: &'a str,
        kind: &'static str,
        line_delta_estimate: i64,
    }

    let mut rows = Vec::new();
    for f in &ctx.parsed.files {
        let kind = match f.kind {
            FileChangeKind::Added => "added",
            FileChangeKind::Deleted => "deleted",
            FileChangeKind::Modified => "modified",
            FileChangeKind::Renamed => "renamed",
        };
        let mut add = 0i64;
        let mut del = 0i64;
        for h in &f.hunks {
            for ln in &h.lines {
                use crate::diff::LineKind;
                match ln.kind {
                    LineKind::Addition => add += 1,
                    LineKind::Removal => del += 1,
                    _ => {}
                }
            }
        }
        rows.push(Row {
            path: f.path.as_str(),
            kind,
            line_delta_estimate: add + del,
        });
    }

    let total = ctx.parsed.line_delta_estimate();
    json!({
        "files": rows,
        "total_line_delta_estimate": total,
    })
    .to_string()
}

fn read_patch(ctx: &ToolContext, args: ReadPatchArgs) -> String {
    let max = args
        .max_chars
        .unwrap_or(ctx.max_patch_chars)
        .min(ctx.max_patch_chars);
    let patch = match ctx.parsed.file_patch(&args.path) {
        Some(p) => p,
        None => {
            return json!({ "error": format!("unknown path in diff: {}", args.path) }).to_string();
        }
    };

    let mut text = patch.to_string();
    if let Some(hi) = args.hunk_index {
        match ctx.parsed.files.iter().find(|f| f.path == args.path) {
            Some(f) if hi < f.hunks.len() => {
                let h = &f.hunks[hi];
                let mut chunk = String::new();
                chunk.push_str(&format!(
                    "@@ -{},{} +{},{} @@\n",
                    h.old_start, h.old_count, h.new_start, h.new_count
                ));
                for ln in &h.lines {
                    let prefix = match ln.kind {
                        crate::diff::LineKind::Context => ' ',
                        crate::diff::LineKind::Addition => '+',
                        crate::diff::LineKind::Removal => '-',
                        crate::diff::LineKind::NoNewline => '\\',
                    };
                    chunk.push(prefix);
                    chunk.push_str(&ln.text);
                    chunk.push('\n');
                }
                text = chunk;
            }
            Some(f) => {
                return json!({ "error": format!("hunk_index {hi} out of range ({} hunks)", f.hunks.len()) }).to_string();
            }
            None => {
                return json!({ "error": format!("unknown path in diff: {}", args.path) })
                    .to_string();
            }
        }
    }

    if text.len() > max {
        let mut t: String = text.chars().take(max).collect();
        t.push_str("\n[truncated]\n");
        text = t;
    }

    json!({ "path": args.path, "patch": text }).to_string()
}

fn read_file_at_ref(ctx: &ToolContext, args: ReadFileAtRefArgs) -> String {
    let repo = match &ctx.repo_root {
        Some(r) => r,
        None => {
            return json!({ "error": "no local repository path configured for this review; use read_patch only" })
                .to_string();
        }
    };

    if let Err(e) = validate_repo_rel_path(&args.path) {
        return json!({ "error": e.to_string() }).to_string();
    }

    let sha = match args.git_ref.to_lowercase().as_str() {
        "base" => ctx.base_sha.clone(),
        "head" => ctx.head_sha.clone(),
        _ => {
            return json!({ "error": "git_ref must be \"base\" or \"head\"" }).to_string();
        }
    };

    match read_file_at_commit(repo, &sha, &args.path) {
        Ok(mut s) => {
            let lines: Vec<&str> = s.lines().collect();
            let start = args.start_line.unwrap_or(1).saturating_sub(1);
            let end = args.end_line.unwrap_or(lines.len()).min(lines.len());
            if start >= lines.len() {
                return json!({ "error": "start_line past end of file" }).to_string();
            }
            let take_end = end.min(lines.len()).max(start);
            let slice: Vec<&str> = lines[start..take_end].to_vec();
            if slice.len() > ctx.max_file_lines {
                let truncated: Vec<&str> = slice.iter().take(ctx.max_file_lines).copied().collect();
                s = truncated.join("\n");
                s.push_str("\n[truncated]\n");
            } else {
                s = slice.join("\n");
            }
            json!({
                "path": args.path,
                "git_ref": args.git_ref,
                "start_line": start + 1,
                "content": s,
            })
            .to_string()
        }
        Err(e) => json!({ "error": e.to_string() }).to_string(),
    }
}

fn validate_repo_rel_path(path: &str) -> Result<()> {
    if path.is_empty() || path.contains("..") || path.starts_with('/') {
        anyhow::bail!("invalid path");
    }
    Ok(())
}

fn grep_repo(ctx: &ToolContext, args: GrepRepoArgs) -> String {
    let repo = match &ctx.repo_root {
        Some(r) => r,
        None => {
            return json!({ "error": "no local repository path for grep" }).to_string();
        }
    };

    if !repo.is_dir() {
        return json!({ "error": "repository path is not a directory" }).to_string();
    }

    let re = match regex::Regex::new(&args.pattern) {
        Ok(r) => r,
        Err(_) => {
            let escaped = regex::escape(&args.pattern);
            match regex::Regex::new(&escaped) {
                Ok(r) => r,
                Err(e) => return json!({ "error": e.to_string() }).to_string(),
            }
        }
    };

    let changed: HashSet<String> = ctx.parsed.files.iter().map(|f| f.path.clone()).collect();

    let paths: Vec<std::path::PathBuf> = if args.scope == "all" {
        let mut out = Vec::new();
        let walker = ignore::WalkBuilder::new(repo)
            .standard_filters(true)
            .build();
        for w in walker {
            let entry = match w {
                Ok(e) => e,
                Err(_) => continue,
            };
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            out.push(entry.path().to_path_buf());
        }
        out
    } else {
        changed
            .iter()
            .filter_map(|rel| {
                let p = repo.join(rel);
                if p.is_file() {
                    Some(p)
                } else {
                    None
                }
            })
            .collect()
    };

    #[derive(Serialize)]
    struct Hit {
        path: String,
        line: usize,
        text: String,
    }

    let mut hits: Vec<Hit> = Vec::new();
    let mut skipped_non_utf8_files: u32 = 0;
    'outer: for path in paths {
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let text = match std::str::from_utf8(&bytes) {
            Ok(s) => s,
            Err(_) => {
                skipped_non_utf8_files += 1;
                continue;
            }
        };
        let rel = path.strip_prefix(repo).unwrap_or(&path);
        let rel_s = rel.to_string_lossy().to_string();
        for (i, line) in text.lines().enumerate() {
            if re.is_match(line) {
                hits.push(Hit {
                    path: rel_s.clone(),
                    line: i + 1,
                    text: line.chars().take(500).collect(),
                });
                if hits.len() >= ctx.max_grep_matches {
                    break 'outer;
                }
            }
        }
    }

    json!({
        "matches": hits,
        "truncated": hits.len() >= ctx.max_grep_matches,
        "skipped_non_utf8_files": skipped_non_utf8_files,
    })
    .to_string()
}

/// Return hunk coordinates and ±5 context diff lines for a commentable new-side line.
fn lookup_hunk(ctx: &ToolContext, args: LookupHunkArgs) -> String {
    use crate::diff::LineKind;
    match ctx.parsed.lookup_line(&args.path, args.line) {
        Some(info) => {
            let kind = match info.kind {
                LineKind::Addition => "addition",
                LineKind::Context => "context",
                _ => "other",
            };
            json!({
                "path": info.path,
                "hunk_index": info.hunk_index,
                "hunk_header": info.hunk_header,
                "new_line": info.new_line,
                "old_line": info.old_line,
                "kind": kind,
                "context_lines": info.context_lines,
            })
            .to_string()
        }
        None => json!({
            "error": format!(
                "line {} in '{}' is not a commentable new-side line in this diff",
                args.line, args.path
            )
        })
        .to_string(),
    }
}

/// Return file lines centred on a new-side line; uses head checkout when available,
/// falls back to diff-hunk context otherwise.
fn read_hunk_context(ctx: &ToolContext, args: ReadHunkContextArgs) -> String {
    let radius = args.radius.unwrap_or(10);

    let info = match ctx.parsed.lookup_line(&args.path, args.line) {
        Some(i) => i,
        None => {
            return json!({
                "error": format!(
                    "line {} in '{}' is not a commentable new-side line in this diff",
                    args.line, args.path
                )
            })
            .to_string();
        }
    };

    // Prefer live file content from the head checkout.
    if let Some(repo) = &ctx.repo_root {
        if let Ok(content) = read_file_at_commit(repo, &ctx.head_sha, &args.path) {
            let all_lines: Vec<&str> = content.lines().collect();
            let center_idx = (args.line as usize).saturating_sub(1);
            let start = center_idx.saturating_sub(radius);
            let end = (center_idx + radius + 1).min(all_lines.len());
            let lines: Vec<serde_json::Value> = all_lines[start..end]
                .iter()
                .enumerate()
                .map(|(i, text)| json!({ "n": start + i + 1, "text": text }))
                .collect();
            return json!({
                "path": args.path,
                "source": "head",
                "center_line": args.line,
                "lines": lines,
            })
            .to_string();
        }
    }

    // Fallback: hunk-bounded context from the parsed diff.
    match ctx
        .parsed
        .hunk_context(&args.path, info.hunk_index, args.line, radius)
    {
        Some(hunk_ctx) => {
            let lines: Vec<serde_json::Value> = hunk_ctx
                .lines
                .iter()
                .map(|(n, text)| json!({ "n": n, "text": text }))
                .collect();
            json!({
                "path": args.path,
                "source": "diff_hunk",
                "center_line": args.line,
                "hunk_header": hunk_ctx.hunk_header,
                "note": "local repository unavailable; null `n` values indicate removed lines",
                "lines": lines,
            })
            .to_string()
        }
        None => json!({
            "error": format!(
                "could not retrieve hunk context for '{}' line {}",
                args.path, args.line
            )
        })
        .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::parse_unified_diff;
    use serde_json::Value;
    use std::path::PathBuf;

    const SAMPLE_DIFF: &str = r#"diff --git a/src/foo.rs b/src/foo.rs
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

    fn sample_ctx(repo_root: Option<PathBuf>) -> ToolContext {
        let parsed = parse_unified_diff(SAMPLE_DIFF).expect("parse");
        ToolContext {
            repo_root,
            base_sha: "deadbeef".into(),
            head_sha: "cafebabe".into(),
            parsed,
            max_patch_chars: 16_000,
            max_file_lines: 256,
            max_grep_matches: 32,
        }
    }

    fn parse_json(s: &str) -> Value {
        serde_json::from_str(s).expect("json")
    }

    #[test]
    fn list_changed_files_lists_path_and_total() {
        let ctx = sample_ctx(None);
        let out = run_tool(&ctx, "list_changed_files", "{}");
        let v = parse_json(&out);
        assert_eq!(v["files"][0]["path"], "src/foo.rs");
        assert_eq!(v["files"][0]["kind"], "modified");
        assert!(v["total_line_delta_estimate"].as_i64().unwrap() > 0);
    }

    #[test]
    fn read_patch_unknown_path_errors() {
        let ctx = sample_ctx(None);
        let out = run_tool(&ctx, "read_patch", r#"{"path": "missing.rs"}"#);
        let v = parse_json(&out);
        assert!(v["error"].as_str().unwrap().contains("unknown path"));
    }

    #[test]
    fn read_patch_hunk_out_of_range_errors() {
        let ctx = sample_ctx(None);
        let out = run_tool(
            &ctx,
            "read_patch",
            r#"{"path": "src/foo.rs", "hunk_index": 99}"#,
        );
        let v = parse_json(&out);
        assert!(v["error"].as_str().unwrap().contains("out of range"));
    }

    #[test]
    fn read_patch_truncates_when_max_chars_small() {
        let ctx = sample_ctx(None);
        let out = run_tool(
            &ctx,
            "read_patch",
            r#"{"path": "src/foo.rs", "max_chars": 20}"#,
        );
        let v = parse_json(&out);
        let patch = v["patch"].as_str().expect("patch");
        assert!(patch.contains("[truncated]"));
    }

    #[test]
    fn read_patch_invalid_args_json() {
        let ctx = sample_ctx(None);
        let out = run_tool(&ctx, "read_patch", "not-json");
        let v = parse_json(&out);
        assert!(v["error"].as_str().unwrap().contains("invalid arguments"));
    }

    #[test]
    fn read_file_at_ref_no_repo_errors() {
        let ctx = sample_ctx(None);
        let out = run_tool(
            &ctx,
            "read_file_at_ref",
            r#"{"git_ref": "head", "path": "src/foo.rs"}"#,
        );
        let v = parse_json(&out);
        assert!(v["error"]
            .as_str()
            .unwrap()
            .contains("no local repository path"));
    }

    #[test]
    fn read_file_at_ref_invalid_git_ref_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = sample_ctx(Some(dir.path().to_path_buf()));
        let out = run_tool(
            &ctx,
            "read_file_at_ref",
            r#"{"git_ref": "main", "path": "src/foo.rs"}"#,
        );
        let v = parse_json(&out);
        assert!(v["error"].as_str().unwrap().contains("base\" or \"head"));
    }

    #[test]
    fn read_file_at_ref_path_traversal_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = sample_ctx(Some(dir.path().to_path_buf()));
        let out = run_tool(
            &ctx,
            "read_file_at_ref",
            r#"{"git_ref": "head", "path": "../etc/passwd"}"#,
        );
        let v = parse_json(&out);
        assert!(v["error"].as_str().unwrap().contains("invalid path"));
    }

    #[test]
    fn grep_no_repo_errors() {
        let ctx = sample_ctx(None);
        let out = run_tool(&ctx, "grep_repo", r#"{"pattern": "foo"}"#);
        let v = parse_json(&out);
        assert!(v["error"]
            .as_str()
            .unwrap()
            .contains("no local repository path for grep"));
    }

    #[test]
    fn grep_changed_finds_match() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rel = PathBuf::from("src/foo.rs");
        std::fs::create_dir_all(dir.path().join("src")).expect("mkdir");
        std::fs::write(dir.path().join(&rel), "fn needle() {}\n").expect("write");

        let ctx = sample_ctx(Some(dir.path().to_path_buf()));
        let out = run_tool(
            &ctx,
            "grep_repo",
            r#"{"pattern": "needle", "scope": "changed"}"#,
        );
        let v = parse_json(&out);
        assert_eq!(v["matches"].as_array().unwrap().len(), 1);
        assert_eq!(v["matches"][0]["text"], "fn needle() {}");
        assert_eq!(v["skipped_non_utf8_files"], 0);
    }

    #[test]
    fn grep_skips_non_utf8_file_and_counts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rel = PathBuf::from("src/foo.rs");
        std::fs::create_dir_all(dir.path().join("src")).expect("mkdir");
        std::fs::write(dir.path().join(&rel), [0xffu8, 0xfe, 0xfd]).expect("write binary");

        let ctx = sample_ctx(Some(dir.path().to_path_buf()));
        let out = run_tool(&ctx, "grep_repo", r#"{"pattern": "x", "scope": "changed"}"#);
        let v = parse_json(&out);
        assert_eq!(v["skipped_non_utf8_files"], 1);
        assert_eq!(v["matches"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn unknown_tool_errors() {
        let ctx = sample_ctx(None);
        let out = run_tool(&ctx, "unknown_tool", "{}");
        let v = parse_json(&out);
        assert!(v["error"].as_str().unwrap().contains("unknown tool"));
    }

    #[test]
    fn lookup_hunk_returns_correct_info() {
        let ctx = sample_ctx(None);
        // new_line 2 == '+    let x = 2;'
        let out = run_tool(&ctx, "lookup_hunk", r#"{"path": "src/foo.rs", "line": 2}"#);
        let v = parse_json(&out);
        assert_eq!(v["path"], "src/foo.rs");
        assert_eq!(v["hunk_index"], 0);
        assert_eq!(v["kind"], "addition");
        assert!(v["hunk_header"].as_str().unwrap().starts_with("@@"));
        assert!(v["context_lines"].is_array());
    }

    #[test]
    fn lookup_hunk_error_on_bogus_line() {
        let ctx = sample_ctx(None);
        let out = run_tool(
            &ctx,
            "lookup_hunk",
            r#"{"path": "src/foo.rs", "line": 999}"#,
        );
        let v = parse_json(&out);
        assert!(v["error"].as_str().unwrap().contains("not a commentable"));
    }

    #[test]
    fn read_hunk_context_fallback_without_repo() {
        let ctx = sample_ctx(None);
        let out = run_tool(
            &ctx,
            "read_hunk_context",
            r#"{"path": "src/foo.rs", "line": 2}"#,
        );
        let v = parse_json(&out);
        assert_eq!(v["source"], "diff_hunk");
        assert_eq!(v["center_line"], 2);
        assert!(v["lines"].is_array());
        assert!(!v["lines"].as_array().unwrap().is_empty());
    }

    #[test]
    fn read_hunk_context_falls_back_when_sha_not_in_repo() {
        // sample_ctx uses a fake head_sha, so read_file_at_commit fails in any
        // real directory and the tool should fall back to diff_hunk.
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = sample_ctx(Some(dir.path().to_path_buf()));
        let out = run_tool(
            &ctx,
            "read_hunk_context",
            r#"{"path": "src/foo.rs", "line": 2}"#,
        );
        let v = parse_json(&out);
        assert_eq!(v["source"], "diff_hunk");
        assert!(v["lines"].is_array());
    }

    #[test]
    fn read_hunk_context_error_on_bogus_line() {
        let ctx = sample_ctx(None);
        let out = run_tool(
            &ctx,
            "read_hunk_context",
            r#"{"path": "src/foo.rs", "line": 999}"#,
        );
        let v = parse_json(&out);
        assert!(v["error"].as_str().unwrap().contains("not a commentable"));
    }
}
