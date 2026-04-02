//! Local git diffs using `git2` (same semantics as `git diff base..head`).

use std::path::Path;

use anyhow::{Context, Result};
use git2::{DiffFormat, DiffLine, Repository};

/// Produce a unified patch for `git diff base..head` (merge-base to head).
pub fn diff_range(repo_path: &Path, base_ref: &str, head_ref: &str) -> Result<String> {
    let repo = Repository::open(repo_path)
        .with_context(|| format!("failed to open git repository at {}", repo_path.display()))?;

    let base = repo
        .revparse_single(base_ref)
        .with_context(|| format!("failed to resolve base ref `{base_ref}`"))?
        .peel_to_commit()
        .with_context(|| format!("`{base_ref}` does not point to a commit"))?;

    let head = repo
        .revparse_single(head_ref)
        .with_context(|| format!("failed to resolve head ref `{head_ref}`"))?
        .peel_to_commit()
        .with_context(|| format!("`{head_ref}` does not point to a commit"))?;

    let merge_base = repo.merge_base(base.id(), head.id()).with_context(|| {
        format!("could not compute merge base for `{base_ref}` and `{head_ref}`")
    })?;

    let ancestor = repo
        .find_commit(merge_base)
        .context("merge-base commit missing")?;

    let old_tree = ancestor.tree().context("merge-base tree")?;
    let new_tree = head.tree().context("head tree")?;

    let diff = repo.diff_tree_to_tree(Some(&old_tree), Some(&new_tree), None)?;

    let mut out = Vec::new();
    diff.print(DiffFormat::Patch, |_delta, _hunk, line: DiffLine<'_>| {
        out.extend_from_slice(line.content());
        true
    })
    .context("failed to format diff")?;

    String::from_utf8(out).context("diff was not valid UTF-8")
}
