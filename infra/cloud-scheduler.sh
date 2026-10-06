#!/usr/bin/env bash
# Starts the GitHub Actions workflow from Google Cloud Scheduler, which fires on
# time. GitHub's own cron is best-effort and has run 3-4 hours late; its
# schedule stays in daily.yml as a fallback (runs are idempotent, so extra
# runs add nothing).
#
# Uses 5 jobs. The first 3 per billing account are free; the 2 closing-odds
# jobs cost $0.10/month each. Delete the old nhl-predict / nhl-resolve /
# nhl-retrain jobs first so they don't count against the free 3.
#
# Usage (e.g. in Cloud Shell):
#   GITHUB_TOKEN=github_pat_... ./infra/cloud-scheduler.sh
#
# GITHUB_TOKEN: a fine-grained personal access token limited to the
# nhl-betting-model repository, with "Actions: Read and write" permission only.
set -euo pipefail

PROJECT="${PROJECT:-josh-crowhurt-personal-bq}"
LOCATION="${LOCATION:-us-central1}"
REPO="${REPO:-joshcrowhurst/nhl-betting-model}"
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN to a fine-grained token with Actions read/write on $REPO}"

URI="https://api.github.com/repos/${REPO}/actions/workflows/daily.yml/dispatches"
HEADERS="Accept=application/vnd.github+json,X-GitHub-Api-Version=2022-11-28,Content-Type=application/json,Authorization=Bearer ${GITHUB_TOKEN}"

gcloud services enable cloudscheduler.googleapis.com --project="$PROJECT"

# name | cron (America/New_York, so it follows daylight saving) | workflow tasks
JOBS=(
  "nhl-gh-daily|13 10 * * *|resolve,predict"
  "nhl-gh-daily-backup|13 13 * * *|resolve,predict"
  "nhl-gh-retrain|29 4 * * 1|resolve,retrain"
  "nhl-gh-close|40 18 * * *|close"
  "nhl-gh-close-late|40 21 * * *|close"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r name schedule tasks <<<"$job"
  body="{\"ref\":\"main\",\"inputs\":{\"tasks\":\"${tasks}\"}}"
  args=(--location="$LOCATION" --project="$PROJECT" --schedule="$schedule"
        --time-zone="America/New_York" --uri="$URI" --http-method=POST
        --message-body="$body" --attempt-deadline=60s
        --max-retry-attempts=3 --min-backoff=120s)
  if gcloud scheduler jobs describe "$name" --location="$LOCATION" --project="$PROJECT" >/dev/null 2>&1; then
    gcloud scheduler jobs update http "$name" "${args[@]}" --update-headers="$HEADERS" >/dev/null
    echo "Updated $name ($schedule ET): $tasks"
  else
    gcloud scheduler jobs create http "$name" "${args[@]}" --headers="$HEADERS" >/dev/null
    echo "Created $name ($schedule ET): $tasks"
  fi
done

echo
echo "Test it now (starts a real run; safe, runs are idempotent):"
echo "  gcloud scheduler jobs run nhl-gh-daily --location=$LOCATION --project=$PROJECT"
