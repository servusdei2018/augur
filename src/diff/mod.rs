//! Parse unified diffs into files, hunks, and line mappings for tools and GitHub validation.

use anyhow::{bail, Result};
use std::collections::HashSet;
use std::ops::Range;

/// Full parsed diff with references into the original patch text.
#[derive(Debug, Clone)]
pub struct ParsedDiff {
    pub raw: String,
    pub files: Vec<FileDiff>,
}

/// One file section in a unified diff.
#[derive(Debug, Clone)]
pub struct FileDiff {
    /// Repository-relative path on the "new" side (after change).
    pub path: String,
    pub old_path: Option<String>,
    pub kind: FileChangeKind,
    /// Byte range in `ParsedDiff::raw` covering this file's diff hunk(s).
    pub range: Range<usize>,
    pub hunks: Vec<Hunk>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileChangeKind {
    Added,
    Deleted,
    Modified,
    Renamed,
}

#[derive(Debug, Clone)]
pub struct Hunk {
    pub old_start: u32,
    pub old_count: u32,
    pub new_start: u32,
    pub new_count: u32,
    /// Lines with running old/new line numbers (None where not applicable).
    pub lines: Vec<HunkLine>,
}

#[derive(Debug, Clone)]
pub struct HunkLine {
    pub kind: LineKind,
    pub old_line: Option<u32>,
    pub new_line: Option<u32>,
    /// Line text without the leading ` ` / `+` / `-` / `\`.
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind {
    Context,
    Addition,
    Removal,
    NoNewline,
}

/// Information about a specific new-side line found in the parsed diff.
#[derive(Debug, Clone)]
pub struct LineInfo {
    pub path: String,
    pub hunk_index: usize,
    /// The raw `@@ -a,b +c,d @@` header string for the containing hunk.
    pub hunk_header: String,
    pub new_line: u32,
    pub old_line: Option<u32>,
    pub kind: LineKind,
    /// Up to ±5 surrounding diff lines (with `+`/`-`/` ` prefix), centred on this line.
    pub context_lines: Vec<String>,
}

/// A radius-bounded slice of a single hunk centred on a new-side line.
#[derive(Debug, Clone)]
pub struct HunkContext {
    pub path: String,
    pub hunk_index: usize,
    pub hunk_header: String,
    /// `(new_line, prefix + text)` — `new_line` is `None` for removal lines.
    pub lines: Vec<(Option<u32>, String)>,
    /// Smallest new-side line number in the slice (ignoring removals).
    pub new_start: u32,
    /// Largest new-side line number in the slice (ignoring removals).
    pub new_end: u32,
}

/// Parse a unified diff (local `git diff` or GitHub `application/vnd.github.diff`).
pub fn parse_unified_diff(raw: &str) -> Result<ParsedDiff> {
    let raw_owned = raw.to_string();
    let files = parse_files(&raw_owned)?;
    Ok(ParsedDiff {
        raw: raw_owned,
        files,
    })
}

fn parse_files(raw: &str) -> Result<Vec<FileDiff>> {
    let lines: Vec<&str> = raw.lines().collect();
    let mut i = 0;
    let mut files = Vec::new();

    while i < lines.len() {
        let line = lines[i];
        if line.starts_with("diff --git ") {
            let start = line_offset(raw, i);
            let (file, next_i) = parse_one_file(&lines, i)?;
            let end = if next_i < lines.len() {
                line_offset(raw, next_i)
            } else {
                raw.len()
            };
            let mut file = file;
            file.range = start..end;
            files.push(file);
            i = next_i;
        } else {
            i += 1;
        }
    }

    Ok(files)
}

fn line_offset(full: &str, line_idx: usize) -> usize {
    let mut pos = 0;
    for (idx, line) in full.lines().enumerate() {
        if idx == line_idx {
            return pos;
        }
        pos += line.len() + 1;
    }
    full.len()
}

fn parse_one_file(lines: &[&str], start: usize) -> Result<(FileDiff, usize)> {
    let mut i = start;
    let git_line = lines[i];
    i += 1;

    let (old_git, new_git) = parse_diff_git_line(git_line)?;

    // Skip extended headers until ---/+++
    let mut old_path = None::<String>;
    let mut new_path = None::<String>;

    while i < lines.len() {
        let l = lines[i];
        if l.starts_with("--- ") {
            old_path = Some(normalize_diff_path(l.strip_prefix("--- ").unwrap_or("")));
            i += 1;
            break;
        }
        if l.starts_with("diff --git ") {
            break;
        }
        i += 1;
    }

    if i >= lines.len() {
        bail!("invalid diff header: missing +++ after ---");
    }

    if lines.get(i).map(|s| s.starts_with("+++ ")).unwrap_or(false) {
        new_path = Some(normalize_diff_path(
            lines[i].strip_prefix("+++ ").unwrap_or(""),
        ));
        i += 1;
    }

    let old_p = old_path
        .clone()
        .or_else(|| old_git.clone())
        .unwrap_or_default();
    let new_p = new_path
        .clone()
        .or_else(|| new_git.clone())
        .unwrap_or_default();

    let path = if !new_p.is_empty() && new_p != "/dev/null" {
        new_p.clone()
    } else {
        old_p.clone()
    };

    let kind = classify_kind(&old_p, &new_p);

    let mut hunks = Vec::new();
    while i < lines.len() {
        let l = lines[i];
        if l.starts_with("diff --git ") {
            break;
        }
        if l.starts_with("@@ ") {
            let (hunk, ni) = parse_hunk(lines, i)?;
            hunks.push(hunk);
            i = ni;
        } else {
            i += 1;
        }
    }

    Ok((
        FileDiff {
            path,
            old_path: old_git
                .or(Some(old_p))
                .filter(|s| !s.is_empty() && s != "/dev/null"),
            kind,
            range: 0..0,
            hunks,
        },
        i,
    ))
}

fn parse_diff_git_line(line: &str) -> Result<(Option<String>, Option<String>)> {
    let rest = line
        .strip_prefix("diff --git ")
        .ok_or_else(|| anyhow::anyhow!("invalid diff header: {line}"))?;
    let parts: Vec<&str> = rest.split_whitespace().collect();
    if parts.len() >= 2 {
        let a = strip_ab_prefix(parts[0]);
        let b = strip_ab_prefix(parts[1]);
        return Ok((Some(a), Some(b)));
    }
    bail!("invalid diff header: {line}")
}

fn strip_ab_prefix(s: &str) -> String {
    s.strip_prefix("a/")
        .or_else(|| s.strip_prefix("b/"))
        .unwrap_or(s)
        .to_string()
}

fn normalize_diff_path(header_path: &str) -> String {
    let t = header_path.trim();
    let t = t
        .strip_prefix("a/")
        .or_else(|| t.strip_prefix("b/"))
        .unwrap_or(t);
    t.trim_matches('"').to_string()
}

fn classify_kind(old_p: &str, new_p: &str) -> FileChangeKind {
    if old_p == "/dev/null" || old_p.is_empty() {
        FileChangeKind::Added
    } else if new_p == "/dev/null" || new_p.is_empty() {
        FileChangeKind::Deleted
    } else if old_p != new_p {
        FileChangeKind::Renamed
    } else {
        FileChangeKind::Modified
    }
}

fn parse_hunk(lines: &[&str], start: usize) -> Result<(Hunk, usize)> {
    let header = lines[start];
    let rest = header
        .strip_prefix("@@ ")
        .and_then(|s| s.split(" @@").next())
        .ok_or_else(|| anyhow::anyhow!("invalid diff header: {header}"))?;

    // -old_start,old_count +new_start,new_count
    let mut parts = rest.split_whitespace();
    let minus = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("invalid diff header: {header}"))?;
    let plus = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("invalid diff header: {header}"))?;

    let (old_start, old_count) = parse_hunk_range(minus.trim_start_matches('-'))?;
    let (new_start, new_count) = parse_hunk_range(plus.trim_start_matches('+'))?;

    let mut hunk_lines = Vec::new();
    let mut i = start + 1;
    let mut old_line = old_start;
    let mut new_line = new_start;

    while i < lines.len() {
        let l = lines[i];
        if l.starts_with("@@ ") || l.starts_with("diff --git ") {
            break;
        }
        if l.starts_with('\\') {
            hunk_lines.push(HunkLine {
                kind: LineKind::NoNewline,
                old_line: None,
                new_line: None,
                text: l.to_string(),
            });
            i += 1;
            continue;
        }
        if l.is_empty() {
            // Some patches have empty lines; treat as context if we can't classify
            i += 1;
            continue;
        }
        let first = l.as_bytes()[0];
        let text = if l.len() > 1 {
            l[1..].to_string()
        } else {
            String::new()
        };

        match first {
            b' ' => {
                hunk_lines.push(HunkLine {
                    kind: LineKind::Context,
                    old_line: Some(old_line),
                    new_line: Some(new_line),
                    text,
                });
                old_line += 1;
                new_line += 1;
            }
            b'+' => {
                hunk_lines.push(HunkLine {
                    kind: LineKind::Addition,
                    old_line: None,
                    new_line: Some(new_line),
                    text,
                });
                new_line += 1;
            }
            b'-' => {
                hunk_lines.push(HunkLine {
                    kind: LineKind::Removal,
                    old_line: Some(old_line),
                    new_line: None,
                    text,
                });
                old_line += 1;
            }
            _ => {
                // e.g. malformed; skip
                i += 1;
                continue;
            }
        }
        i += 1;
    }

    Ok((
        Hunk {
            old_start,
            old_count,
            new_start,
            new_count,
            lines: hunk_lines,
        },
        i,
    ))
}

fn parse_hunk_range(s: &str) -> Result<(u32, u32)> {
    let mut it = s.split(',');
    let start: u32 = it
        .next()
        .ok_or_else(|| anyhow::anyhow!("invalid diff header: {s}"))?
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid diff header: {s}"))?;
    let count: u32 = it.next().map(|c| c.parse().unwrap_or(1)).unwrap_or(1);
    Ok((start, count))
}

impl ParsedDiff {
    /// Slice of the original diff text for one file (by new path).
    pub fn file_patch(&self, path: &str) -> Option<&str> {
        self.files
            .iter()
            .find(|f| f.path == path)
            .map(|f| self.raw.get(f.range.clone()).unwrap_or(""))
    }

    /// Collect (path, new_line) pairs that exist on the right-hand side and are valid for inline review.
    pub fn commentable_new_lines(&self) -> Vec<(String, u32)> {
        let mut out = Vec::new();
        for f in &self.files {
            if f.kind == FileChangeKind::Deleted {
                continue;
            }
            for h in &f.hunks {
                for ln in &h.lines {
                    match ln.kind {
                        LineKind::Context | LineKind::Addition => {
                            if let Some(n) = ln.new_line {
                                out.push((f.path.clone(), n));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        out
    }

    /// Set of `(path, new_line)` pairs valid for inline review on the right-hand side.
    pub fn commentable_line_set(&self) -> HashSet<(String, u32)> {
        self.commentable_new_lines().into_iter().collect()
    }

    /// True if `(path, line)` appears as a new-side line in this diff (approximate GitHub RIGHT line).
    pub fn is_commentable_line(&self, path: &str, line: u32) -> bool {
        self.commentable_new_lines()
            .iter()
            .any(|(p, l)| p == path && *l == line)
    }

    /// Total added + removed lines (rough size metric).
    pub fn line_delta_estimate(&self) -> i64 {
        let mut add = 0i64;
        let mut del = 0i64;
        for f in &self.files {
            for h in &f.hunks {
                for ln in &h.lines {
                    match ln.kind {
                        LineKind::Addition => add += 1,
                        LineKind::Removal => del += 1,
                        _ => {}
                    }
                }
            }
        }
        add + del
    }

    /// Look up the hunk containing a specific new-side line number.
    ///
    /// Returns `None` if the path is unknown or `new_line` is not on the
    /// commentable (Context or Addition) new side of any hunk.
    pub fn lookup_line(&self, path: &str, new_line: u32) -> Option<LineInfo> {
        let file = self.files.iter().find(|f| f.path == path)?;
        for (hunk_index, hunk) in file.hunks.iter().enumerate() {
            for (line_pos, ln) in hunk.lines.iter().enumerate() {
                if matches!(ln.kind, LineKind::Context | LineKind::Addition)
                    && ln.new_line == Some(new_line)
                {
                    let hunk_header = hunk_header_str(hunk);
                    const RADIUS: usize = 5;
                    let start = line_pos.saturating_sub(RADIUS);
                    let end = (line_pos + RADIUS + 1).min(hunk.lines.len());
                    let context_lines = hunk.lines[start..end]
                        .iter()
                        .map(|l| {
                            let prefix = line_prefix(l.kind);
                            format!("{}{}", prefix, l.text)
                        })
                        .collect();
                    return Some(LineInfo {
                        path: path.to_string(),
                        hunk_index,
                        hunk_header,
                        new_line,
                        old_line: ln.old_line,
                        kind: ln.kind,
                        context_lines,
                    });
                }
            }
        }
        None
    }

    /// Return a radius-bounded slice of a hunk centred on a new-side line.
    ///
    /// Returns `None` if `path`, `hunk_index`, or `center_new_line` is not found.
    pub fn hunk_context(
        &self,
        path: &str,
        hunk_index: usize,
        center_new_line: u32,
        radius: usize,
    ) -> Option<HunkContext> {
        let file = self.files.iter().find(|f| f.path == path)?;
        let hunk = file.hunks.get(hunk_index)?;
        let hunk_header = hunk_header_str(hunk);
        let center_pos = hunk
            .lines
            .iter()
            .position(|l| l.new_line == Some(center_new_line))?;
        let start = center_pos.saturating_sub(radius);
        let end = (center_pos + radius + 1).min(hunk.lines.len());
        let slice = &hunk.lines[start..end];
        let lines: Vec<(Option<u32>, String)> = slice
            .iter()
            .map(|l| (l.new_line, format!("{}{}", line_prefix(l.kind), l.text)))
            .collect();
        let new_start = slice
            .iter()
            .find_map(|l| l.new_line)
            .unwrap_or(center_new_line);
        let new_end = slice
            .iter()
            .rev()
            .find_map(|l| l.new_line)
            .unwrap_or(center_new_line);
        Some(HunkContext {
            path: path.to_string(),
            hunk_index,
            hunk_header,
            lines,
            new_start,
            new_end,
        })
    }
}

/// Format a hunk header string from a `Hunk`.
fn hunk_header_str(h: &Hunk) -> String {
    format!(
        "@@ -{},{} +{},{} @@",
        h.old_start, h.old_count, h.new_start, h.new_count
    )
}

/// Single-character prefix for a diff line.
fn line_prefix(kind: LineKind) -> char {
    match kind {
        LineKind::Context => ' ',
        LineKind::Addition => '+',
        LineKind::Removal => '-',
        LineKind::NoNewline => '\\',
    }
}

/// Truncate by whole files until `max_chars` of patch text is included. Returns (patch, truncated).
pub fn truncate_diff_by_files(raw: &str, max_chars: usize) -> (String, bool) {
    let Ok(parsed) = parse_unified_diff(raw) else {
        if raw.len() <= max_chars {
            return (raw.to_string(), false);
        }
        let mut s = raw.chars().take(max_chars).collect::<String>();
        s.push_str(
            "\n\n[... diff truncated by augur (parse failed); review may be incomplete ...]\n",
        );
        return (s, true);
    };

    if parsed.raw.len() <= max_chars {
        return (parsed.raw.clone(), false);
    }

    let mut out = String::new();
    for f in &parsed.files {
        let chunk = parsed.raw.get(f.range.clone()).unwrap_or("");
        if out.is_empty() {
            if chunk.len() <= max_chars {
                out.push_str(chunk);
            } else {
                out.push_str(chunk);
                break;
            }
        } else if out.len() + chunk.len() <= max_chars {
            out.push_str(chunk);
        } else {
            break;
        }
    }

    if out.is_empty() && parsed.raw.len() > max_chars {
        let mut s = raw.chars().take(max_chars).collect::<String>();
        s.push_str("\n\n[... diff truncated by augur (no structured files); review may be incomplete ...]\n");
        return (s, true);
    }

    if out.len() < parsed.raw.len() {
        out.push_str(
            "\n[... diff truncated by augur at file boundary; review may be incomplete ...]\n",
        );
        (out, true)
    } else {
        (out, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"diff --git a/src/foo.rs b/src/foo.rs
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

    #[test]
    fn parses_one_file_and_hunk() {
        let p = parse_unified_diff(SAMPLE).unwrap();
        assert_eq!(p.files.len(), 1);
        assert_eq!(p.files[0].path, "src/foo.rs");
        assert_eq!(p.files[0].hunks.len(), 1);
        let h = &p.files[0].hunks[0];
        assert_eq!(h.old_start, 1);
        assert_eq!(h.new_start, 1);
        let kinds: Vec<_> = h.lines.iter().map(|l| l.kind).collect();
        assert!(kinds.contains(&LineKind::Context));
        assert!(kinds.contains(&LineKind::Removal));
        assert!(kinds.contains(&LineKind::Addition));
    }

    #[test]
    fn commentable_lines_include_context_and_additions() {
        let p = parse_unified_diff(SAMPLE).unwrap();
        let cl = p.commentable_new_lines();
        assert!(cl.iter().any(|(_, l)| *l == 1));
        assert!(cl.iter().any(|(_, l)| *l == 3));
        assert!(p.is_commentable_line("src/foo.rs", 3));
    }

    #[test]
    fn truncate_by_files_keeps_whole_first_file() {
        let huge = "x".repeat(50_000);
        let mut diff = String::from(SAMPLE);
        diff.push_str("\ndiff --git a/huge b/huge\n--- a/huge\n+++ b/huge\n@@ -0,0 +1,1 @@\n+");
        diff.push_str(&huge);

        let (t, trunc) = truncate_diff_by_files(&diff, 800);
        assert!(trunc);
        assert!(t.contains("src/foo.rs"));
        assert!(!t.contains(&huge[..100]));
    }

    #[test]
    fn lookup_line_finds_addition() {
        let p = parse_unified_diff(SAMPLE).unwrap();
        // new_line 2 is '+    let x = 2;'
        let info = p.lookup_line("src/foo.rs", 2).expect("line 2 is an addition");
        assert_eq!(info.hunk_index, 0);
        assert_eq!(info.kind, LineKind::Addition);
        assert_eq!(info.new_line, 2);
        assert!(info.hunk_header.starts_with("@@"));
        assert!(!info.context_lines.is_empty());
    }

    #[test]
    fn lookup_line_finds_context() {
        let p = parse_unified_diff(SAMPLE).unwrap();
        // new_line 1 is ' fn main() {'
        let info = p.lookup_line("src/foo.rs", 1).expect("line 1 is context");
        assert_eq!(info.kind, LineKind::Context);
    }

    #[test]
    fn lookup_line_returns_none_for_unknown_line() {
        let p = parse_unified_diff(SAMPLE).unwrap();
        assert!(p.lookup_line("src/foo.rs", 999).is_none());
        assert!(p.lookup_line("nonexistent.rs", 1).is_none());
    }

    #[test]
    fn hunk_context_clips_to_hunk_boundary() {
        let p = parse_unified_diff(SAMPLE).unwrap();
        // Radius larger than hunk should return all hunk lines
        let ctx = p
            .hunk_context("src/foo.rs", 0, 2, 100)
            .expect("has context");
        let total_hunk_lines = p.files[0].hunks[0].lines.len();
        assert_eq!(ctx.lines.len(), total_hunk_lines);
    }

    #[test]
    fn hunk_context_respects_small_radius() {
        let p = parse_unified_diff(SAMPLE).unwrap();
        // radius=1 → at most 3 lines (pos-1, pos, pos+1)
        let ctx = p
            .hunk_context("src/foo.rs", 0, 2, 1)
            .expect("has context");
        assert!(ctx.lines.len() <= 3);
    }

    #[test]
    fn hunk_context_none_for_bad_inputs() {
        let p = parse_unified_diff(SAMPLE).unwrap();
        assert!(p.hunk_context("nonexistent.rs", 0, 1, 5).is_none());
        assert!(p.hunk_context("src/foo.rs", 99, 1, 5).is_none());
        assert!(p.hunk_context("src/foo.rs", 0, 999, 5).is_none());
    }
}
