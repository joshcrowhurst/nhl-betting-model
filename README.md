# NHL betting model

An XGBoost model that predicts NHL moneyline winners. GitHub Actions runs it
every morning, emails the day's picks, and publishes a dashboard on GitHub Pages.
It costs nothing to run.

## How it runs

`.github/workflows/daily.yml` runs on a schedule:

| When (ET, approx.) | What |
|---|---|
| ~10:15 AM daily | Record last night's results, predict today's games, email the picks |
| ~1:45 PM daily | Catch-up run in case the morning run started late or failed. It only predicts games not yet predicted, so you won't get a duplicate email |
| ~4:30 AM Mondays | Retrain the model on the last 6 seasons |

Every run also rebuilds the dashboard. The pipeline keeps its state on the
`state` branch: cached NHL data, the trained model, `predictions.csv` and
`runs.jsonl`. Don't delete that branch.
The branch is a single commit that gets replaced on each run, so it doesn't grow.

The first run has no cache yet. It downloads about 6 seasons of box scores and
trains the model, which takes about 45 minutes. Later runs take a few minutes.

## Setup (one time)

1. **Merge this branch to `main`.** GitHub only runs scheduled workflows from the default branch.
2. **Turn on Pages:** Settings → Pages → Source: **GitHub Actions**.
3. **Add repository secrets** under Settings → Secrets and variables → Actions:
   | Secret | Value |
   |---|---|
   | `GMAIL_USER` | your Gmail address |
   | `GMAIL_APP_PASSWORD` | a [Google app password](https://myaccount.google.com/apppasswords) (needs 2-step verification) |
   | `EMAIL_TO` | optional; defaults to `GMAIL_USER` |
   | `ODDS_API_KEY` | optional; from the-odds-api.com. Without it you get picks and probabilities but no odds or value bets. The free tier (500 requests a month) is enough: each run uses at most 1 |
4. **Do the first run by hand:** Actions → *Daily NHL picks* → Run workflow.
   Set `tasks` to `resolve,retrain,predict` and tick *force_email*.

The dashboard will be at `https://<your-user>.github.io/nhl-betting-model/`.

## Running locally

```bash
pip install -r requirements.txt
python run.py daily --no-email          # resolve + predict into ./data
python run.py site --out _site          # build the dashboard
cd _site && python -m http.server       # view it at http://localhost:8000
python run.py backtest                  # walk-forward backtest
```

## Shutting down the old Google Cloud setup

The old version ran on Cloud Run with an always-on Cloud SQL database, which
cost about $9–10 a month. Delete the old resources once this is live:

```bash
P=josh-crowhurt-personal-bq
gcloud sql instances delete nhl-db --project=$P                     # the main cost
gcloud run services delete nhl-model --region=us-central1 --project=$P
for j in nhl-predict nhl-resolve nhl-retrain; do
  gcloud scheduler jobs delete $j --location=us-central1 --project=$P --quiet
done
gcloud builds triggers list --project=$P                          # delete the GitHub push trigger
gcloud artifacts docker images list gcr.io/$P/nhl-model --project=$P   # delete old images
gcloud storage rm -r gs://nhl-betting-model --project=$P
for s in nhl-db-pass nhl-odds-api-key SENDGRID_API_KEY nhl-job-secret; do
  gcloud secrets delete $s --project=$P --quiet
done
```

If you used the old Firebase Hosting site, run `firebase hosting:disable` to
take it down.
