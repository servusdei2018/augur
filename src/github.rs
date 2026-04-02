//! GitHub: PR metadata, raw diff, and pull request reviews via `octocrab` + authenticated HTTP.

use anyhow::{Context, Result};
use octocrab::Octocrab;
use serde::Serialize;

/// The type of formal review to post to GitHub.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ReviewAction {
    /// Leave only a comment; does not change the PR status.
    #[default]
    Comment,
    /// Formally approve the PR.
    Approve,
    /// Request changes; blocks merging (for repos that enforce reviews).
    RequestChanges,
}

impl ReviewAction {
    /// GitHub API string for the `event` field.
    pub fn as_api_str(&self) -> &'static str {
        match self {
            ReviewAction::Comment => "COMMENT",
            ReviewAction::Approve => "APPROVE",
            ReviewAction::RequestChanges => "REQUEST_CHANGES",
        }
    }

    /// Parse from a case-insensitive string (CLI / agent output).
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().replace('-', "_").as_str() {
            "COMMENT" => Some(ReviewAction::Comment),
            "APPROVE" => Some(ReviewAction::Approve),
            "REQUEST_CHANGES" => Some(ReviewAction::RequestChanges),
            _ => None,
        }
    }
}

const DIFF_ACCEPT: &str = "application/vnd.github.diff";

/// GitHub REST API base URL for HTTP requests and `octocrab`.
///
/// Precedence: `GITHUB_API_URL` (full base, e.g. `https://api.github.com` or
/// `https://github.example.com/api/v3`), else if `GITHUB_HOST` is set then
/// `https://{GITHUB_HOST}/api/v3` (typical GitHub Enterprise Server), else
/// `https://api.github.com`.
pub fn github_rest_api_base() -> String {
    github_rest_api_base_from_vars(
        std::env::var("GITHUB_API_URL"),
        std::env::var("GITHUB_HOST"),
    )
}

fn github_rest_api_base_from_vars(
    api_url: Result<String, std::env::VarError>,
    host: Result<String, std::env::VarError>,
) -> String {
    if let Ok(url) = api_url {
        let s = url.trim().trim_end_matches('/').to_string();
        if !s.is_empty() {
            return s;
        }
    }
    if let Ok(host) = host {
        let h = host.trim().trim_matches('/');
        if !h.is_empty() {
            return format!("https://{h}/api/v3");
        }
    }
    "https://api.github.com".to_string()
}

/// Build an Octocrab client using `GITHUB_TOKEN` or `GH_TOKEN` and the same API base as
/// [`fetch_pr_diff`](fetch_pr_diff).
pub fn octocrab_from_env() -> Result<Octocrab> {
    let token = std::env::var("GITHUB_TOKEN")
        .or_else(|_| std::env::var("GH_TOKEN"))
        .context("GITHUB_TOKEN or GH_TOKEN is required for GitHub operations")?;

    let base = github_rest_api_base();
    Octocrab::builder()
        .personal_token(token)
        .base_uri(base)
        .context("invalid GITHUB_API_URL / GITHUB_HOST")?
        .build()
        .context("failed to build GitHub client")
}

/// Metadata for prompt context and posting reviews.
#[derive(Debug, Clone)]
pub struct PullRequestInfo {
    pub title: String,
    pub user_login: String,
    pub head_sha: String,
    pub base_sha: String,
}

/// Fetch PR title, author, and base/head SHAs.
pub async fn fetch_pr_info(
    octo: &Octocrab,
    owner: &str,
    repo: &str,
    number: u64,
) -> Result<PullRequestInfo> {
    let pr = octo
        .pulls(owner, repo)
        .get(number)
        .await
        .with_context(|| format!("failed to fetch PR {owner}/{repo}#{number}"))?;

    Ok(PullRequestInfo {
        title: pr.title.unwrap_or_default(),
        user_login: pr.user.map(|u| u.login).unwrap_or_default(),
        head_sha: pr.head.sha,
        base_sha: pr.base.sha,
    })
}

/// Raw unified diff for the PR (same as `application/vnd.github.diff`).
pub async fn fetch_pr_diff(owner: &str, repo: &str, number: u64) -> Result<String> {
    let base = github_rest_api_base();
    let token = std::env::var("GITHUB_TOKEN")
        .or_else(|_| std::env::var("GH_TOKEN"))
        .context("GITHUB_TOKEN or GH_TOKEN missing")?;
    fetch_pr_diff_with_token(&base, &token, owner, repo, number).await
}

async fn fetch_pr_diff_with_token(
    api_base: &str,
    token: &str,
    owner: &str,
    repo: &str,
    number: u64,
) -> Result<String> {
    let url = format!("{api_base}/repos/{owner}/{repo}/pulls/{number}");

    let client = reqwest::Client::builder()
        .user_agent("augur-cli")
        .build()
        .context("reqwest client")?;

    let resp = client
        .get(&url)
        .header(reqwest::header::ACCEPT, DIFF_ACCEPT)
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {}", token))
        .send()
        .await
        .with_context(|| format!("failed to GET diff for {owner}/{repo}#{number}"))?;

    let status = resp.status();
    let body = resp
        .text()
        .await
        .with_context(|| format!("failed to read diff body (status {status})"))?;

    if !status.is_success() {
        anyhow::bail!("GitHub API error {status}: {body}");
    }

    Ok(body)
}

#[derive(Debug, Serialize)]
struct CreateReviewBody {
    body: String,
    event: String,
    commit_id: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    comments: Vec<ReviewCommentPayload>,
}

#[derive(Debug, Serialize)]
struct ReviewCommentPayload {
    path: String,
    body: String,
    line: u32,
    side: String,
}

/// Parameters for [`post_pr_review`].
pub struct PrReviewRequest<'a> {
    /// Markdown body for the top-level review comment.
    pub body: &'a str,
    /// The commit SHA at the PR head (required by the GitHub API).
    pub head_sha: &'a str,
    /// Per-line inline comments: `(file path, new-side line number, comment body)`.
    pub inline_comments: &'a [(String, u32, String)],
    /// The type of review to submit.
    pub action: ReviewAction,
}

/// Post a pull request review with Markdown body and optional per-line comments (`RIGHT` side).
///
/// `req.action` controls the review type posted to GitHub:
/// - [`ReviewAction::Comment`] — a plain review comment (no approval / block)
/// - [`ReviewAction::Approve`] — formally approves the PR
/// - [`ReviewAction::RequestChanges`] — requests changes (blocks merging where enforced)
pub async fn post_pr_review(
    octo: &Octocrab,
    owner: &str,
    repo: &str,
    number: u64,
    req: PrReviewRequest<'_>,
) -> Result<()> {
    let route = format!("/repos/{}/{}/pulls/{}/reviews", owner, repo, number);

    let comments: Vec<ReviewCommentPayload> = req
        .inline_comments
        .iter()
        .map(|(path, line, text)| ReviewCommentPayload {
            path: path.clone(),
            body: text.clone(),
            line: *line,
            side: "RIGHT".to_string(),
        })
        .collect();

    let payload = CreateReviewBody {
        body: req.body.to_string(),
        event: req.action.as_api_str().to_string(),
        commit_id: req.head_sha.to_string(),
        comments,
    };

    let _: serde_json::Value = octo
        .post(route, Some(&payload))
        .await
        .with_context(|| format!("failed to create review on {owner}/{repo}#{number}"))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::env::VarError;

    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn github_rest_api_base_defaults() {
        assert_eq!(
            github_rest_api_base_from_vars(Err(VarError::NotPresent), Err(VarError::NotPresent)),
            "https://api.github.com"
        );
    }

    #[test]
    fn github_rest_api_url_preempts_host() {
        assert_eq!(
            github_rest_api_base_from_vars(
                Ok("https://api.corp.example/v3/".into()),
                Ok("ignored".into()),
            ),
            "https://api.corp.example/v3"
        );
    }

    #[test]
    fn github_rest_api_host_when_url_unset() {
        assert_eq!(
            github_rest_api_base_from_vars(Err(VarError::NotPresent), Ok("git.ghe.test".into())),
            "https://git.ghe.test/api/v3"
        );
    }

    #[test]
    fn github_rest_api_whitespace_only_url_falls_through_to_host() {
        assert_eq!(
            github_rest_api_base_from_vars(Ok("  ".into()), Ok("git.ghe.test".into())),
            "https://git.ghe.test/api/v3"
        );
    }

    #[tokio::test]
    async fn fetch_pr_diff_hits_mock_server() {
        let srv = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/repos/acme/demo/pulls/7"))
            .and(header("accept", DIFF_ACCEPT))
            .and(header("authorization", "Bearer testtoken"))
            .respond_with(ResponseTemplate::new(200).set_body_string("diff --git a/x b/x"))
            .mount(&srv)
            .await;

        let body = fetch_pr_diff_with_token(&srv.uri(), "testtoken", "acme", "demo", 7)
            .await
            .expect("diff fetch");

        assert!(body.contains("diff --git"));
    }

    #[tokio::test]
    async fn post_pr_review_posts_expected_json() {
        let srv = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/repos/acme/demo/pulls/7/reviews"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&srv)
            .await;

        let octo = Octocrab::builder()
            .personal_token("testtoken")
            .base_uri(srv.uri())
            .expect("base")
            .build()
            .expect("octo");

        post_pr_review(
            &octo,
            "acme",
            "demo",
            7,
            PrReviewRequest {
                body: "## LGTM",
                head_sha: "abc123",
                inline_comments: &[("src/lib.rs".to_string(), 10, "nit".to_string())],
                action: ReviewAction::Comment,
            },
        )
        .await
        .expect("post review");
    }

    #[test]
    fn review_action_api_strings() {
        assert_eq!(ReviewAction::Comment.as_api_str(), "COMMENT");
        assert_eq!(ReviewAction::Approve.as_api_str(), "APPROVE");
        assert_eq!(ReviewAction::RequestChanges.as_api_str(), "REQUEST_CHANGES");
    }

    #[test]
    fn review_action_from_str_loose() {
        assert_eq!(
            ReviewAction::from_str_loose("approve"),
            Some(ReviewAction::Approve)
        );
        assert_eq!(
            ReviewAction::from_str_loose("REQUEST_CHANGES"),
            Some(ReviewAction::RequestChanges)
        );
        assert_eq!(
            ReviewAction::from_str_loose("request-changes"),
            Some(ReviewAction::RequestChanges)
        );
        assert_eq!(
            ReviewAction::from_str_loose("comment"),
            Some(ReviewAction::Comment)
        );
        assert_eq!(ReviewAction::from_str_loose("bogus"), None);
    }
}
