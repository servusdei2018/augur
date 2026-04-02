//! Local git operations: diffs and reading objects at specific commits.

mod diff;
pub mod repo;

pub use diff::diff_range;
pub use repo::{read_file_at_commit, resolve_to_commit};
