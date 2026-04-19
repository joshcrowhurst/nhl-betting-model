#!/usr/bin/env bash
# One-shot GCP infrastructure provisioning for the NHL betting model.
# Run once from your local machine with gcloud authenticated.
#
# Usage:
#   chmod +x infra/setup.sh
#   ./infra/setup.sh
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project josh-crowhurt-personal-bq

set -euo pipefail

PROJECT_ID="josh-crowhurt-personal-bq"
REGION="us-central1"
SERVICE_NAME="nhl-model"
INSTANCE_NAME="nhl-db"
INSTANCE_CONNECTION_NAME="${PROJECT_ID}:${REGION}:${INSTANCE_NAME}"
DB_USER="nhl"
DB_NAME="nhl_betting"
GCS_BUCKET="nhl-betting-model"
REPO_NAME="nhl-model"   # Cloud Source Repos mirror OR GitHub trigger name

echo "==> Setting project"
gcloud config set project "$PROJECT_ID"

# ── Enable required APIs ──────────────────────────────────────────────────────
echo "==> Enabling APIs..."
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com \
  --project="$PROJECT_ID"

# ── GCS Bucket ────────────────────────────────────────────────────────────────
echo "==> Creating GCS bucket gs://${GCS_BUCKET}..."
gcloud storage buckets create "gs://${GCS_BUCKET}" \
  --location="$REGION" \
  --project="$PROJECT_ID" \
  --uniform-bucket-level-access 2>/dev/null || echo "  (bucket already exists)"

# ── Cloud SQL ─────────────────────────────────────────────────────────────────
echo "==> Creating Cloud SQL instance (this takes ~5 minutes)..."
gcloud sql instances create "$INSTANCE_NAME" \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region="$REGION" \
  --storage-size=10GB \
  --storage-auto-increase \
  --no-assign-ip \
  --project="$PROJECT_ID" 2>/dev/null || echo "  (instance already exists)"

echo "==> Creating database..."
gcloud sql databases create "$DB_NAME" \
  --instance="$INSTANCE_NAME" \
  --project="$PROJECT_ID" 2>/dev/null || echo "  (database already exists)"

echo "==> Creating DB user..."
DB_PASS=$(openssl rand -base64 24 | tr -d '=+/')
gcloud sql users create "$DB_USER" \
  --instance="$INSTANCE_NAME" \
  --password="$DB_PASS" \
  --project="$PROJECT_ID" 2>/dev/null || echo "  (user already exists — not updating password)"

# ── Secret Manager ────────────────────────────────────────────────────────────
echo "==> Creating secrets in Secret Manager..."

_create_secret() {
  local name="$1"
  local value="$2"
  if ! gcloud secrets describe "$name" --project="$PROJECT_ID" &>/dev/null; then
    echo -n "$value" | gcloud secrets create "$name" \
      --data-file=- \
      --replication-policy=automatic \
      --project="$PROJECT_ID"
    echo "  Created secret: $name"
  else
    echo "  (secret $name already exists — update manually if needed)"
  fi
}

_create_secret "nhl-db-pass" "$DB_PASS"

# Prompt for secrets that can't be auto-generated
echo ""
echo "  !! You need to provide the following secrets manually."
echo "  !! Run these commands after this script completes:"
echo ""
echo "  echo -n '<YOUR_ODDS_API_KEY>' | gcloud secrets create nhl-odds-api-key --data-file=- --replication-policy=automatic"
echo "  echo -n '<YOUR_SENDGRID_KEY>' | gcloud secrets create nhl-sendgrid-key --data-file=- --replication-policy=automatic"
JOB_SECRET=$(openssl rand -hex 20)
_create_secret "nhl-job-secret" "$JOB_SECRET"
echo ""
echo "  Job secret (auto-generated): $JOB_SECRET"
echo "  Save this — Cloud Scheduler will send it as X-Job-Secret header."
echo ""

# ── Cloud Build trigger (GitHub) ──────────────────────────────────────────────
echo "==> Cloud Build GitHub trigger must be created manually in the Console:"
echo "  https://console.cloud.google.com/cloud-build/triggers?project=${PROJECT_ID}"
echo "  Connect your GitHub repo → trigger on push to 'main' → use cloudbuild.yaml"

# ── Cloud Run service account permissions ─────────────────────────────────────
echo "==> Granting Cloud Run SA access to Cloud SQL and GCS..."
SA_EMAIL="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"-compute@developer.gserviceaccount.com

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/cloudsql.client" --quiet

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin" --quiet

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor" --quiet

# ── Cloud Scheduler jobs ──────────────────────────────────────────────────────
# NOTE: CLOUD_RUN_URL must be filled in after first deploy.
# Run `gcloud run services describe nhl-model --region=us-central1 --format='value(status.url)'`
# then replace the placeholder below and re-run this block.

CLOUD_RUN_URL="${CLOUD_RUN_URL:-https://nhl-model-PLACEHOLDER.run.app}"
SCHEDULER_SA="${SA_EMAIL}"

echo "==> Creating Cloud Scheduler jobs..."
echo "    (Using Cloud Run URL: ${CLOUD_RUN_URL})"
echo "    Update CLOUD_RUN_URL env var and re-run if this is a placeholder."

_create_scheduler() {
  local name="$1"
  local schedule="$2"
  local path="$3"
  gcloud scheduler jobs describe "$name" --location="$REGION" --project="$PROJECT_ID" &>/dev/null && {
    echo "  (scheduler job $name already exists)"
    return
  }
  gcloud scheduler jobs create http "$name" \
    --location="$REGION" \
    --schedule="$schedule" \
    --time-zone="America/New_York" \
    --uri="${CLOUD_RUN_URL}${path}" \
    --http-method=POST \
    --headers="X-Job-Secret=${JOB_SECRET},Content-Type=application/json" \
    --message-body="{}" \
    --attempt-deadline=1800s \
    --project="$PROJECT_ID"
  echo "  Created: $name ($schedule ET)"
}

# Predict: 11:00 AM ET daily (before most puck drops at ~7 PM ET)
_create_scheduler "nhl-predict" "0 11 * * *" "/jobs/predict"

# Resolve: 2:00 AM ET daily (all games finished by then)
_create_scheduler "nhl-resolve" "0 2 * * *" "/jobs/resolve"

# Retrain: 3:00 AM ET every Monday
_create_scheduler "nhl-retrain" "0 3 * * 1" "/jobs/retrain"

echo ""
echo "✅  Infrastructure setup complete."
echo ""
echo "Next steps:"
echo "  1. Add secrets for nhl-odds-api-key and nhl-sendgrid-key (commands shown above)"
echo "  2. Connect GitHub repo in Cloud Build console and push to 'main' to deploy"
echo "  3. After first deploy, get the Cloud Run URL:"
echo "     gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)'"
echo "  4. Update Cloud Scheduler job URIs with the real URL (or set CLOUD_RUN_URL and re-run)"
echo "  5. Upload your trained model to GCS:"
echo "     gsutil cp models_saved/moneyline_latest.pkl gs://${GCS_BUCKET}/models/moneyline_latest.pkl"
