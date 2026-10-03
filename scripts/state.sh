#!/usr/bin/env bash
# Persists pipeline state (data cache, model, predictions log) on the `state`
# branch between GitHub Actions runs.
#
#   scripts/state.sh pull   # clone the branch into ./state (or start empty)
#   scripts/state.sh push   # save ./state back as a single squashed commit
#
# The branch is force-pushed as one commit each time, so the weekly model
# pickle and parquet caches never pile up in git history. predictions.csv
# itself holds the full history.
set -euo pipefail

BRANCH="${STATE_BRANCH:-state}"
DIR="${STATE_DIR:-state}"
REMOTE="${STATE_REMOTE:-https://x-access-token:${GH_TOKEN:-}@github.com/${GITHUB_REPOSITORY:-}.git}"

case "${1:-}" in
  pull)
    if git ls-remote --exit-code --heads "$REMOTE" "$BRANCH" >/dev/null 2>&1; then
      git clone --quiet --depth 1 --branch "$BRANCH" "$REMOTE" "$DIR"
    else
      echo "No '$BRANCH' branch yet; starting with empty state"
      mkdir -p "$DIR"
      git -C "$DIR" init --quiet -b "$BRANCH"
      git -C "$DIR" remote add origin "$REMOTE"
    fi
    mkdir -p "$DIR/data" "$DIR/models"
    ;;
  push)
    cd "$DIR"
    git config user.name "github-actions[bot]"
    git config user.email "41898283+github-actions[bot]@users.noreply.github.com"
    git checkout --quiet --orphan snapshot
    git add -A
    if git diff --cached --quiet; then
      echo "State unchanged"
      exit 0
    fi
    git commit --quiet -m "State snapshot $(date -u +%Y-%m-%dT%H:%MZ)"
    git push --quiet --force origin "snapshot:$BRANCH"
    echo "State saved to '$BRANCH'"
    ;;
  *)
    echo "usage: $0 pull|push" >&2
    exit 2
    ;;
esac
