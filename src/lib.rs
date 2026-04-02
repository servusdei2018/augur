//! Augur: unified code review for local git diffs and GitHub PRs.

pub mod agent;
pub mod cli;
pub mod diff;
pub mod git;
pub mod github;
pub mod llm;
pub mod review;
pub mod tools;

pub use cli::{Augur, Commands, LlmCli, ReviewArgs, ReviewRunOpts, ReviewTarget};
