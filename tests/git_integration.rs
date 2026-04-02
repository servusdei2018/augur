//! Integration tests for git-backed helpers using a temporary repository.
//!
//! Repositories are created under `target/augur-git-test/` (inside the workspace) so `git init`
//! can create `.git` in environments that disallow temp dirs outside the project.

use std::path::{Path, PathBuf};

use augur::git::{diff_range, read_file_at_commit, resolve_to_commit};
use git2::{Repository, Signature};

fn temp_repo_dir() -> tempfile::TempDir {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("augur-git-test");
    std::fs::create_dir_all(&base).expect("mkdir target/augur-git-test");
    tempfile::Builder::new()
        .prefix("repo-")
        .tempdir_in(base)
        .expect("tempdir in target/")
}

#[test]
fn diff_range_and_read_file_across_two_commits() {
    let dir = temp_repo_dir();
    let repo = Repository::init(dir.path()).expect("init");
    let sig = Signature::now("Test", "test@example.com").expect("sig");

    std::fs::write(dir.path().join("hello.txt"), "one\n").expect("write");
    let mut index = repo.index().expect("index");
    index.add_path(Path::new("hello.txt")).expect("add");
    let tree_id = index.write_tree().expect("tree");
    let tree = repo.find_tree(tree_id).expect("find tree");
    let c1 = repo
        .commit(Some("HEAD"), &sig, &sig, "first", &tree, &[])
        .expect("commit1");

    std::fs::write(dir.path().join("hello.txt"), "two\n").expect("write2");
    let mut index = repo.index().expect("index2");
    index.add_path(Path::new("hello.txt")).expect("add2");
    let tree_id = index.write_tree().expect("tree2");
    let tree = repo.find_tree(tree_id).expect("find tree2");
    let c1_commit = repo.find_commit(c1).expect("c1");
    repo.commit(Some("HEAD"), &sig, &sig, "second", &tree, &[&c1_commit])
        .expect("commit2");

    let diff = diff_range(dir.path(), "HEAD~1", "HEAD").expect("diff_range");
    assert!(diff.contains("hello.txt"), "diff should name file: {diff}");
    assert!(
        diff.contains("+two") || diff.contains("two"),
        "diff should show new content: {diff}"
    );

    let head_sha = resolve_to_commit(dir.path(), "HEAD").expect("resolve");
    let content = read_file_at_commit(dir.path(), &head_sha, "hello.txt").expect("read");
    assert_eq!(content, "two\n");
}
