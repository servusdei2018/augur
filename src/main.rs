use anyhow::{Context, Result};
use augur::agent::{findings_to_markdown, run_review_agent, AgentConfig, ReviewOutput};
use augur::cli::{Augur, Commands, ReviewArgs, ReviewRunOpts, ReviewTarget};
use augur::diff::parse_unified_diff;
use augur::git;
use augur::git::repo::resolve_to_commit;
use augur::github::{self, PrReviewRequest, ReviewAction};
use augur::llm::LlmConfig;
use augur::review::{
    changed_files_summary, maybe_truncate_diff, system_prompt, user_prompt_local,
    user_prompt_manifest_local, user_prompt_manifest_pr, user_prompt_pr,
};
use augur::tools::ToolContext;
use clap::Parser;

struct ReviewPrArgs {
    owner: String,
    repo_name: String,
    number: u64,
    dry_run: bool,
    repo_path: Option<std::path::PathBuf>,
    cli_action: Option<String>,
}

fn build_agent_config(run: &ReviewRunOpts) -> AgentConfig {
    AgentConfig {
        max_rounds: run.max_rounds,
        max_tool_calls: run.max_tool_calls,
        max_tool_output_chars: 128_000,
        max_patch_chars: run.max_patch_chars,
        max_file_lines: run.max_file_lines,
        max_grep_matches: run.max_grep_matches,
        max_context_tool_results: run.max_context_tool_results,
        max_context_chars: run.max_context_chars,
    }
}

fn print_review_output(out: &ReviewOutput, json: bool) -> Result<()> {
    if json {
        let v = serde_json::json!({
            "markdown": out.markdown,
            "findings": out.findings,
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    print!("{}", out.markdown);
    print!("{}", findings_to_markdown(&out.findings));
    if !out.markdown.ends_with('\n') {
        println!();
    }
    Ok(())
}

async fn review_local(
    cfg: &LlmConfig,
    args: &ReviewArgs,
    base: String,
    head: String,
    repo: std::path::PathBuf,
) -> Result<()> {
    let run = &args.run;
    let diff = git::diff_range(&repo, &base, &head)?;
    let (diff, _) = maybe_truncate_diff(&diff, run.max_diff_chars);
    let parsed = parse_unified_diff(&diff).context("parse unified diff")?;
    let base_sha = resolve_to_commit(&repo, &base)?;
    let head_sha = resolve_to_commit(&repo, &head)?;

    if run.single_shot {
        let user = user_prompt_local(&diff, &base, &head);
        let text = cfg
            .complete(system_prompt(), &user)
            .await
            .context("LLM completion failed")?;
        if run.json {
            let v = serde_json::json!({ "markdown": text, "findings": [] });
            println!("{}", serde_json::to_string_pretty(&v)?);
        } else {
            print!("{text}");
            if !text.ends_with('\n') {
                println!();
            }
        }
        return Ok(());
    }

    let tool_ctx = ToolContext {
        repo_root: Some(repo.clone()),
        base_sha,
        head_sha,
        parsed,
        max_patch_chars: run.max_patch_chars,
        max_file_lines: run.max_file_lines,
        max_grep_matches: run.max_grep_matches,
    };

    let summary = changed_files_summary(&tool_ctx.parsed);
    let manifest = user_prompt_manifest_local(
        &base,
        &head,
        &tool_ctx.base_sha,
        &tool_ctx.head_sha,
        &summary,
    );
    let agent_cfg = build_agent_config(run);
    let out = run_review_agent(cfg, &agent_cfg, tool_ctx, &manifest)
        .await
        .context("agent review failed")?;
    print_review_output(&out, run.json)
}

async fn review_pr(cfg: &LlmConfig, args: &ReviewArgs, pr: ReviewPrArgs) -> Result<()> {
    let ReviewPrArgs {
        owner,
        repo_name,
        number,
        dry_run,
        repo_path,
        cli_action,
    } = pr;
    let run = &args.run;
    let octo = github::octocrab_from_env()?;
    let info = github::fetch_pr_info(&octo, &owner, &repo_name, number).await?;
    let diff = github::fetch_pr_diff(&owner, &repo_name, number).await?;
    let (diff, _) = maybe_truncate_diff(&diff, run.max_diff_chars);
    let parsed = parse_unified_diff(&diff).context("parse unified diff")?;

    let local_repo = repo_path.clone().or_else(|| std::env::current_dir().ok());

    if run.single_shot {
        let user = user_prompt_pr(
            &diff,
            &owner,
            &repo_name,
            number,
            &info.title,
            &info.user_login,
        );
        let text = cfg
            .complete(system_prompt(), &user)
            .await
            .context("LLM completion failed")?;
        if dry_run {
            tracing::info!("dry-run: not posting review to GitHub");
        } else {
            // For single-shot mode, no agent action is available; honour the CLI flag or default.
            let action = cli_action
                .as_deref()
                .and_then(ReviewAction::from_str_loose)
                .unwrap_or_default();
            github::post_pr_review(
                &octo,
                &owner,
                &repo_name,
                number,
                PrReviewRequest {
                    body: &text,
                    head_sha: &info.head_sha,
                    inline_comments: &[],
                    action,
                },
            )
            .await?;
            eprintln!("Posted pull request review to {owner}/{repo_name}#{number}.");
        }
        if run.json {
            let v = serde_json::json!({ "markdown": text, "findings": [] });
            println!("{}", serde_json::to_string_pretty(&v)?);
        } else {
            print!("{text}");
            if !text.ends_with('\n') {
                println!();
            }
        }
        return Ok(());
    }

    if repo_path.is_none() {
        tracing::warn!(
            "PR agent mode: no --repo-path; read_file_at_ref and grep_repo use the current directory. \
             Pass --repo-path /path/to/clone for a known-good local checkout matching this PR."
        );
    }

    let tool_ctx = ToolContext {
        repo_root: local_repo,
        base_sha: info.base_sha.clone(),
        head_sha: info.head_sha.clone(),
        parsed,
        max_patch_chars: run.max_patch_chars,
        max_file_lines: run.max_file_lines,
        max_grep_matches: run.max_grep_matches,
    };

    let summary = changed_files_summary(&tool_ctx.parsed);
    let manifest = user_prompt_manifest_pr(
        &owner,
        &repo_name,
        number,
        &info.title,
        &info.user_login,
        &info.base_sha,
        &info.head_sha,
        &summary,
    );
    let agent_cfg = build_agent_config(run);
    let out = run_review_agent(cfg, &agent_cfg, tool_ctx, &manifest)
        .await
        .context("agent review failed")?;

    if dry_run {
        tracing::info!("dry-run: not posting review to GitHub");
        print_review_output(&out, run.json)?;
        return Ok(());
    }

    let inline: Vec<(String, u32, String)> = out
        .findings
        .iter()
        .map(|f| (f.path.clone(), f.line, f.body.clone()))
        .collect();

    // Action resolution: CLI flag → agent suggestion → default (Comment).
    let action = cli_action
        .as_deref()
        .and_then(ReviewAction::from_str_loose)
        .or_else(|| {
            out.suggested_action
                .as_deref()
                .and_then(ReviewAction::from_str_loose)
        })
        .unwrap_or_default();

    tracing::info!("Posting review with action: {}", action.as_api_str());

    github::post_pr_review(
        &octo,
        &owner,
        &repo_name,
        number,
        PrReviewRequest {
            body: &out.markdown,
            head_sha: &info.head_sha,
            inline_comments: &inline,
            action,
        },
    )
    .await?;
    eprintln!("Posted pull request review to {owner}/{repo_name}#{number}.");

    if run.json {
        print_review_output(&out, true)?;
    } else {
        print_review_output(&out, false)?;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let augur = Augur::parse();

    match augur.command {
        Commands::Review(args) => {
            let cfg = LlmConfig::from_cli(
                args.llm.api_key.clone(),
                args.llm.api_base.clone(),
                args.llm.model.clone(),
            )?;

            match args.target {
                ReviewTarget::Local {
                    ref base,
                    ref head,
                    ref repo,
                } => {
                    let cwd = std::env::current_dir().context("current directory")?;
                    let repo_path = repo.as_ref().unwrap_or(&cwd).clone();
                    review_local(&cfg, &args, base.clone(), head.clone(), repo_path).await?;
                }
                ReviewTarget::Pr {
                    ref owner,
                    ref repo,
                    number,
                    dry_run,
                    ref repo_path,
                    ref review_action,
                } => {
                    review_pr(
                        &cfg,
                        &args,
                        ReviewPrArgs {
                            owner: owner.clone(),
                            repo_name: repo.clone(),
                            number,
                            dry_run,
                            repo_path: repo_path.clone(),
                            cli_action: review_action.clone(),
                        },
                    )
                    .await?;
                }
            }
        }
    }

    Ok(())
}
