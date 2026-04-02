//! Command-line interface for `augur`.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Code review for local git branches and GitHub pull requests using an OpenAI-compatible API.
///
/// Environment (see README for details):
/// - `OPENAI_API_KEY` — required for LLM calls
/// - `OPENAI_API_BASE` — optional; default OpenAI API if unset
/// - `OPENAI_MODEL` — optional model id
/// - `GITHUB_TOKEN` or `GH_TOKEN` — required for `review pr` (read PR + post review)
/// - `GITHUB_API_URL` or `GITHUB_HOST` — optional; see README for GitHub Enterprise
#[derive(Parser, Debug)]
#[command(name = "augur", version, about, long_about = LONG_ABOUT)]
pub struct Augur {
    #[command(subcommand)]
    pub command: Commands,
}

const LONG_ABOUT: &str = r#"Augur generates code reviews from unified diffs using an OpenAI-compatible HTTP API.

Examples:
  augur review local --base main --head feature/foo
  augur review pr octo-org hello-world 42
  augur review pr octo-org hello-world 42 --dry-run
  augur review local --base main --head feature --single-shot

Environment:
  OPENAI_API_KEY     API key for the LLM provider (required).
  OPENAI_API_BASE    Base URL for OpenAI-compatible APIs (optional).
  OPENAI_MODEL       Model id (optional; provider default if unset).
  GITHUB_TOKEN       GitHub personal access token for `review pr` (or GH_TOKEN).
  GITHUB_API_URL     GitHub REST API base (optional; GitHub Enterprise: full URL, e.g. …/api/v3).
  GITHUB_HOST        GitHub Enterprise hostname only if GITHUB_API_URL unset (API at https://HOST/api/v3).
"#;

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Generate a review from a diff
    Review(ReviewArgs),
}

#[derive(Parser, Debug)]
pub struct ReviewArgs {
    #[command(flatten)]
    pub llm: LlmCli,

    #[command(flatten)]
    pub run: ReviewRunOpts,

    #[command(subcommand)]
    pub target: ReviewTarget,
}

/// Review mode and agent budgets.
#[derive(Parser, Debug, Clone)]
pub struct ReviewRunOpts {
    /// Send one LLM request with a file-wise truncated diff (no tool calls).
    #[arg(long)]
    pub single_shot: bool,

    #[arg(long, default_value_t = 24)]
    pub max_rounds: u32,

    #[arg(long, default_value_t = 48)]
    pub max_tool_calls: u32,

    /// Soft cap on unified diff size for single-shot mode and parsing (may exceed slightly so
    /// at least one whole file remains reviewable when the first file is huge).
    #[arg(long, default_value_t = 120_000)]
    pub max_diff_chars: usize,

    /// Max characters returned by `read_patch` per call.
    #[arg(long, default_value_t = 32_000)]
    pub max_patch_chars: usize,

    /// Max lines returned by `read_file_at_ref` per call.
    #[arg(long, default_value_t = 400)]
    pub max_file_lines: usize,

    /// Max grep matches per call.
    #[arg(long, default_value_t = 80)]
    pub max_grep_matches: usize,

    /// Print JSON with `markdown` and `findings` instead of human-readable output.
    #[arg(long)]
    pub json: bool,
}

/// Shared LLM-related flags (env + CLI overrides).
#[derive(Parser, Debug, Clone)]
pub struct LlmCli {
    /// Model id (overrides OPENAI_MODEL).
    #[arg(long, env = "OPENAI_MODEL")]
    pub model: Option<String>,

    /// API base URL, e.g. https://api.openai.com/v1 (overrides OPENAI_API_BASE).
    #[arg(long, env = "OPENAI_API_BASE")]
    pub api_base: Option<String>,

    /// API key (overrides OPENAI_API_KEY).
    #[arg(long, env = "OPENAI_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum ReviewTarget {
    /// Review changes between two refs (git `base..head` range semantics).
    Local {
        /// Base ref (branch, tag, or commit).
        #[arg(long)]
        base: String,

        /// Head ref (branch, tag, or commit).
        #[arg(long)]
        head: String,

        /// Path to the git repository (default: current directory).
        #[arg(long)]
        repo: Option<PathBuf>,
    },
    /// Fetch a GitHub PR, review it, and post a pull request review with optional inline comments.
    Pr {
        /// Repository owner (user or organization).
        owner: String,

        /// Repository name.
        repo: String,

        /// Pull request number.
        number: u64,

        /// Run the LLM and print the review, but do not POST to GitHub.
        #[arg(long)]
        dry_run: bool,

        /// Local clone of the repository (enables read_file_at_ref and grep tools for the agent).
        #[arg(long)]
        repo_path: Option<PathBuf>,

        /// Override the review action posted to GitHub.
        /// Accepted values: `approve`, `request-changes`, `comment` (case-insensitive).
        /// When omitted the agent's recommendation is used; falls back to `comment`.
        #[arg(long, value_name = "ACTION")]
        review_action: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::{Augur, Commands, ReviewTarget};
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn parse_review_local_defaults() {
        let augur = Augur::try_parse_from([
            "augur",
            "review",
            "local",
            "--base",
            "main",
            "--head",
            "feature/foo",
        ])
        .expect("parse");
        match augur.command {
            Commands::Review(args) => {
                assert!(!args.run.single_shot);
                assert_eq!(args.run.max_diff_chars, 120_000);
                match args.target {
                    ReviewTarget::Local { base, head, repo } => {
                        assert_eq!(base, "main");
                        assert_eq!(head, "feature/foo");
                        assert!(repo.is_none());
                    }
                    _ => panic!("expected local"),
                }
            }
        }
    }

    #[test]
    fn parse_review_pr_with_flags() {
        let augur = Augur::try_parse_from([
            "augur",
            "review",
            "pr",
            "octo-org",
            "hello-world",
            "42",
            "--dry-run",
            "--repo-path",
            "/tmp/checkout",
        ])
        .expect("parse");
        match augur.command {
            Commands::Review(args) => match args.target {
                ReviewTarget::Pr {
                    owner,
                    repo,
                    number,
                    dry_run,
                    repo_path,
                    review_action: _,
                } => {
                    assert_eq!(owner, "octo-org");
                    assert_eq!(repo, "hello-world");
                    assert_eq!(number, 42);
                    assert!(dry_run);
                    assert_eq!(repo_path, Some(PathBuf::from("/tmp/checkout")));
                }
                _ => panic!("expected pr"),
            },
        }
    }

    #[test]
    fn parse_review_local_single_shot_and_budgets() {
        // Shared `ReviewRunOpts` flags belong on `review` before the target subcommand.
        let augur = Augur::try_parse_from([
            "augur",
            "review",
            "--single-shot",
            "--max-diff-chars",
            "50000",
            "--max-rounds",
            "8",
            "local",
            "--base",
            "main",
            "--head",
            "HEAD",
        ])
        .expect("parse");
        match augur.command {
            Commands::Review(args) => {
                assert!(args.run.single_shot);
                assert_eq!(args.run.max_diff_chars, 50000);
                assert_eq!(args.run.max_rounds, 8);
            }
        }
    }
}
