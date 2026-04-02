# augur

CLI code review for local git branches and GitHub pull requests, using any OpenAI-compatible HTTP API.

## Requirements

- Rust toolchain
- For `review pr`: a GitHub personal access token with permission to read the PR and create reviews

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes (for LLM) | API key for your provider. |
| `OPENAI_API_BASE` | No | Base URL for OpenAI-compatible APIs (e.g. `https://api.openai.com/v1`). Omit for default OpenAI. |
| `OPENAI_MODEL` | No | Model id; defaults to `gpt-4o-mini` if unset. |
| `GITHUB_TOKEN` or `GH_TOKEN` | Yes for `review pr` | Token used for GitHub API (same idea as the `gh` CLI). |
| `GITHUB_API_URL` | No | REST API root URL. Default is `https://api.github.com`. For **GitHub Enterprise Server**, set the full API base (often `https://<hostname>/api/v3`). |
| `GITHUB_HOST` | No | **GitHub Enterprise Server** hostname only if `GITHUB_API_URL` is unset; augur uses `https://<GITHUB_HOST>/api/v3`. Do not set this for github.com—leave unset or set `GITHUB_API_URL` explicitly. |

### GitHub token scopes

- **Private repositories:** `repo`
- **Public repositories only:** `public_repo` is often enough for read + review on public PRs; use `repo` if you hit permission errors.

## Usage

Local review (`git diff` semantics match `git diff <base>..<head>`: changes on `head` since the merge-base with `base`):

```bash
augur review local --base main --head feature/foo
augur review local --base main --head feature/foo --repo /path/to/repo
```

Pull request (fetches the PR diff, runs the LLM, posts a **Pull Request Review** with event `COMMENT`):

```bash
export GITHUB_TOKEN=ghp_...
export OPENAI_API_KEY=sk-...
augur review pr octo-org hello-world 42
```

**Agent mode (default)** uses tools to inspect the patch; `read_file_at_ref` and `grep_repo` need a local clone that matches the PR. If you do not pass `--repo-path`, augur uses the **current directory** (see log warning); pass an explicit clone when in doubt:

```bash
augur review pr octo-org hello-world 42 --repo-path /path/to/hello-world
```

**Diff size:** `--max-diff-chars` is a soft limit; augur may keep a large first file so the review is never empty on an oversized patch.

Dry-run (compute the review and print it, do **not** post to GitHub):

```bash
augur review pr octo-org hello-world 42 --dry-run
```

Override LLM settings per run:

```bash
augur review local --base main --head HEAD --model gpt-4o --api-base https://api.example.com/v1 --api-key "$KEY"
```

## Logging

Set `RUST_LOG` for [`tracing`](https://crates.io/crates/tracing), e.g. `RUST_LOG=debug`.

## License

Augur is distributed under the MIT License. Refer to the `LICENSE` file for details.