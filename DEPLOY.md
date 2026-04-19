# Deployment Runbook

## One-time setup

### 1. Provision GCP infrastructure
```bash
chmod +x infra/setup.sh
./infra/setup.sh
```

This creates:
- Cloud SQL (PostgreSQL) instance `nhl-db` in us-central1
- GCS bucket `nhl-betting-model` for model storage
- Secret Manager secrets (DB password auto-generated; API keys prompted)
- Cloud Run IAM bindings
- Cloud Scheduler jobs (predict 11 AM ET, resolve 2 AM ET, retrain Mon 3 AM ET)

### 2. Add API key secrets manually
```bash
echo -n 'b59d20ac809a9b87b148f453e1023b91' | \
  gcloud secrets create nhl-odds-api-key --data-file=- --replication-policy=automatic

echo -n '<YOUR_SENDGRID_KEY>' | \
  gcloud secrets create nhl-sendgrid-key --data-file=- --replication-policy=automatic
```

### 3. Connect GitHub repo → Cloud Build
1. Go to https://console.cloud.google.com/cloud-build/triggers?project=josh-crowhurt-personal-bq
2. Click "Connect Repository" → select your GitHub repo
3. Create trigger: push to `main` → use `cloudbuild.yaml`

### 4. First deploy
Push to `main` — Cloud Build will build and deploy automatically.

Get your Cloud Run URL:
```bash
gcloud run services describe nhl-model --region=us-central1 --format='value(status.url)'
```

### 5. Update Cloud Scheduler job URIs
After getting the real URL, update each scheduler job in the GCP console
(Cloud Scheduler → nhl-predict / nhl-resolve / nhl-retrain → Edit → update URI).

Or run:
```bash
URL="https://your-real-url.run.app"
for job in nhl-predict nhl-resolve nhl-retrain; do
  path=$([ "$job" = "nhl-predict" ] && echo "/jobs/predict" || \
         [ "$job" = "nhl-resolve" ] && echo "/jobs/resolve" || echo "/jobs/retrain")
  gcloud scheduler jobs update http "$job" \
    --location=us-central1 \
    --uri="${URL}${path}"
done
```

### 6. Upload trained model to GCS
```bash
python run.py train --seasons 5
gsutil cp models_saved/moneyline_latest.pkl gs://nhl-betting-model/models/moneyline_latest.pkl
```

### 7. Deploy the React frontend

```bash
cd frontend

# Install dependencies (requires Node 18+)
npm install

# Set the real API URL
echo "REACT_APP_API_URL=https://your-real-url.run.app" > .env.production

# Build
npm run build

# Deploy to Firebase Hosting
npm install -g firebase-tools
firebase login
firebase use --add   # select project josh-crowhurt-personal-bq
firebase deploy --only hosting
```

---

## Daily operations (fully automated once deployed)

| Time (ET) | Job | What it does |
|-----------|-----|-------------|
| 11:00 AM  | predict | Fetches today's games + odds, runs model, emails you |
| 2:00 AM   | resolve | Fills in game results, updates performance cache |
| Mon 3 AM  | retrain | Retrains on 6 seasons of data, saves new model to GCS |

---

## Local development

```bash
# Set env vars
export DATABASE_URL=postgresql://nhl:password@localhost:5432/nhl_betting
export ODDS_API_KEY=b59d20ac809a9b87b148f453e1023b91
export SENDGRID_API_KEY=<your_key>

# Run API server
uvicorn src.api.main:app --reload --port 8080

# Run frontend dev server (in another terminal)
cd frontend && npm start
```

---

## Manually triggering jobs

```bash
SECRET=$(gcloud secrets versions access latest --secret=nhl-job-secret)
URL=$(gcloud run services describe nhl-model --region=us-central1 --format='value(status.url)')

curl -X POST "${URL}/jobs/predict" -H "X-Job-Secret: ${SECRET}" -H "Content-Type: application/json"
curl -X POST "${URL}/jobs/resolve" -H "X-Job-Secret: ${SECRET}" -H "Content-Type: application/json"
curl -X POST "${URL}/jobs/retrain" -H "X-Job-Secret: ${SECRET}" -H "Content-Type: application/json"
```
