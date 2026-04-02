//! Read blobs and resolve refs inside a local repository.

use std::path::Path;

use anyhow::{Context, Result};
use git2::Repository;

/// Resolve a ref-ish string to a commit OID hex string.
pub fn resolve_to_commit(repo_path: &Path, refish: &str) -> Result<String> {
    let repo = Repository::open(repo_path)
        .with_context(|| format!("failed to open git repository at {}", repo_path.display()))?;
    let obj = repo
        .revparse_single(refish)
        .with_context(|| format!("failed to resolve `{refish}`"))?;
    let commit = obj
        .peel_to_commit()
        .with_context(|| format!("`{refish}` does not point to a commit"))?;
    Ok(commit.id().to_string())
}

/// Read a UTF-8 file at `path` from the tree at `commit_sha` (hex OID).
pub fn read_file_at_commit(repo_path: &Path, commit_sha: &str, path: &str) -> Result<String> {
    let repo = Repository::open(repo_path)
        .with_context(|| format!("failed to open git repository at {}", repo_path.display()))?;
    let oid = git2::Oid::from_str(commit_sha).context("invalid commit SHA")?;
    let commit = repo.find_commit(oid).context("commit not found")?;
    let tree = commit.tree().context("commit tree")?;

    let entry = tree
        .get_path(Path::new(path))
        .with_context(|| format!("path not found in tree: {path}"))?;
    let obj = entry.to_object(&repo).context("object load")?;
    let blob = obj.into_blob().map_err(|_| anyhow::anyhow!("not a blob"))?;
    let data = blob.content();
    String::from_utf8(data.to_vec())
        .map_err(|e| anyhow::anyhow!("{path}: not valid UTF-8 (binary or non-UTF-8 text): {e}"))
}
