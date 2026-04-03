# Integrating Augur into GitHub Actions

This guide explains how to use **augur** within your CI/CD pipelines to automatically review Pull Requests.

## Overview

Augur can be easily integrated into GitHub Actions by downloading the prebuilt binaries directly from the latest release. This avoid the overhead of installing the Rust toolchain and building from source on every run.

## Download Links

Augur provides prebuilt binaries for several platforms. You can download the **latest** version using these links:

| Platform | Architecture | Latest Download URL |
|----------|--------------|---------------------|
| **Linux** | x86_64 (musl) | `https://github.com/servusdei2018/augur/releases/latest/download/augur-x86_64-unknown-linux-musl.tar.gz` |
| **macOS** | x86_64 | `https://github.com/servusdei2018/augur/releases/latest/download/augur-x86_64-apple-darwin.tar.gz` |
| **macOS** | Apple Silicon | `https://github.com/servusdei2018/augur/releases/latest/download/augur-aarch64-apple-darwin.tar.gz` |
| **Windows**| x86_64 | `https://github.com/servusdei2018/augur/releases/latest/download/augur-x86_64-pc-windows-msvc.zip` |

> [!NOTE]
> For a **specific version** (e.g., `v0.1.0`), the URL format changes slightly. You must omit `latest` and specify the tag in the path:
> `https://github.com/servusdei2018/augur/releases/download/${VERSION}/augur-${TARGET}.${EXT}`
>
> **Example**: `https://github.com/servusdei2018/augur/releases/download/v0.1.0/augur-aarch64-apple-darwin.tar.gz`

## GitHub Action Example

Add the following workflow to `.github/workflows/augur-review.yml` in your repository.

> [!IMPORTANT]
> Ensure you have `OPENAI_API_KEY` stored in your repository secrets.
> For pull request reviews, the `GITHUB_TOKEN` provided by Actions is usually sufficient, but ensure it has `pull-requests: write` permissions.

```yaml
name: Augur Code Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: read
  pull-requests: write

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v6
        with:
          fetch-depth: 0 # Required for augur to see history

      - name: Download Augur
        run: |
          curl -L "https://github.com/servusdei2018/augur/releases/latest/download/augur-x86_64-unknown-linux-musl.tar.gz" | tar -xz
          chmod +x augur
          sudo mv augur /usr/local/bin/

      - name: Run Augur Review
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          REPO_OWNER=$(echo $GITHUB_REPOSITORY | cut -d'/' -f1)
          REPO_NAME=$(echo $GITHUB_REPOSITORY | cut -d'/' -f2)
          PR_NUMBER=${{ github.event.pull_request.number }}
          
          augur review pr "$REPO_OWNER" "$REPO_NAME" "$PR_NUMBER"
```

## Security Considerations

- **Secrets**: Never hardcode your `OPENAI_API_KEY`. Use GitHub Secrets.
- **Workflow Triggers**: If you are using `pull_request_target` to allow reviews on forks, be extremely careful about what code you execute, as `pull_request_target` runs in the context of the base repository.

## Customization

You can pass additional flags to `augur` to customize the review:

```bash
augur review pr $OWNER $REPO $PR \
  --model "arcee-ai/trinity-large-thinking" \
  --max-diff-chars 50000 \
  --api-base "https://api.kilo.ai/api/gateway"
```

For more details on CLI options, see the [README.md](README.md).
