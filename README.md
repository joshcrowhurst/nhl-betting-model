# NHL betting model

An XGBoost model that predicts NHL moneyline winners. GitHub Actions runs it
every morning, emails the day's picks, and publishes a dashboard on GitHub Pages.
It costs nothing to run.

## How it runs

`.github/workflows/daily.yml` runs on a schedule:

| When (ET, approx.) | What |
|---|---|
| ~10:15 AM daily | Record last night's results, predict today's games, email the picks |
| ~11:45 AM and ~2:15 PM daily | Backup runs, because GitHub sometimes starts scheduled runs hours late or skips them. They only predict games not yet predicted, so you won't get a duplicate email |
| ~4:30 AM Mondays | Retrain the model on the last 6 seasons |

Every pick shows the starting goalies and a two- or three-sentence explanation.
- **Starting goalies:** taken from [Daily Faceoff](https://www.dailyfaceoff.com/starting-goalies)
  when it lists them (confirmed or likely). Otherwise the team's usual starter
  is used, shown as *projected*. They're shown for context only. A backtest
  with the starter's own save % and starting share as model features showed no
  gain over the team-level goalie form the model already uses, so they're left out.
- **Explanations:** built from the model's own per-feature contributions
  (XGBoost SHAP values), so they show what actually drove each probability.
  No external service is involved.

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

## Reliable start times (Google Cloud Scheduler)

GitHub's built-in schedule is best-effort and has started runs 3–4 hours late.
`infra/cloud-scheduler.sh` sets up five Cloud Scheduler jobs that start the
workflow on time via GitHub's API. The first three are free; the two
closing-odds jobs cost $0.10/month each. The GitHub schedule in `daily.yml`
stays as a fallback; extra runs are harmless. If you already ran the script,
run it again to add the two new jobs.

| Job | When (ET, follows daylight saving) | Tasks |
|---|---|---|
| `nhl-gh-daily` | 10:13 AM daily | resolve, predict, email |
| `nhl-gh-daily-backup` | 1:13 PM daily | catch-up |
| `nhl-gh-retrain` | 4:29 AM Mondays | resolve, retrain |
| `nhl-gh-close` | 6:40 PM daily | closing odds for games not yet started |
| `nhl-gh-close-late` | 9:40 PM daily | closing odds for West Coast games |

Setup:
1. Delete the old scheduler jobs first (see the shutdown section below), so they
   don't count against the free allowance.
2. Create a GitHub **fine-grained** personal access token: GitHub → Settings →
   Developer settings → Fine-grained tokens → Generate. Under *Repository access*,
   choose only `nhl-betting-model`. Under *Permissions → Actions*, choose
   **Read and write**. Leave everything else as no access.
3. In [Cloud Shell](https://shell.cloud.google.com), with this repo cloned:
   ```bash
   GITHUB_TOKEN=github_pat_... ./infra/cloud-scheduler.sh
   gcloud scheduler jobs run nhl-gh-daily --location=us-central1   # test: starts a run
   ```
   The test should show a new *Daily NHL picks* run in the Actions tab within a minute.

The token is stored in the job's request headers, which only people with access
to the GCP project can read. If the token expires, re-run the script with a new
one; it updates the existing jobs.

## Odds (The Odds API)

With the `ODDS_API_KEY` secret set, each prediction run fetches live odds (one
request, about 30 credits a month), and every pick gets:
- **Consensus odds and value bets:** the median price across US bookmakers, the
  margin-free market probability, and expected value for each side. A value bet is
  the side with positive expected value. If a market blend has been deployed (see
  below), the blended probability is used instead of the model's own.
- **Kelly stake:** a suggested stake as a % of bankroll for each value bet, using a
  quarter of the Kelly amount and capped at 3%.
- **Price shopping:** flags a bookmaker whose best price beats the consensus fair
  price by 2% or more, whatever the model says.
- **Closing-line value:** the `close` runs (6:40 and 9:40 PM ET) record the last
  odds before each game starts. The dashboard shows whether value bets beat the
  closing price on average, which is the quickest sign of a real edge.

Historical odds need a paid plan (~$30 for a month of 20K credits). Each
historical snapshot costs 10 credits and is cached in the `state` branch. Run these
from Actions → *Daily NHL picks* → Run workflow:

| `tasks` | What it does | Credits (approx.) |
|---|---|---|
| `backfill` | Adds odds as of pick time, plus closing odds, to predictions made without them | 10 per distinct pick time + 10 per distinct start time |
| `market` | Walk-forward backtest from 2023-24 joined to each morning's odds: model vs market vs blend, value-bet and price-shop results. Deploys the blend if it beats the model out of sample | ~2,000 per season, ~7,000 the first time |

`max_credits` caps what one run may spend (default 12,000). Every run stops
fetching when 300 credits remain, so the daily live odds keep working.

## Running locally

```bash
pip install -r requirements.txt
python run.py daily --no-email          # resolve + predict into ./data
python run.py site --out _site          # build the dashboard
cd _site && python -m http.server       # view it at http://localhost:8000
python run.py backtest                  # walk-forward backtest
python run.py compare-goalie            # backtest: current model vs +starting goalie vs previous settings
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

## Backtests on GitHub

`.github/workflows/backtest.yml` runs on pushes to `claude/**` branches that
touch the model. It backtests the current model against variants (with the
starting-goalie features, and with the previous deeper XGBoost settings) on real
NHL data and posts a results table in the run summary. It also logs what
the live data sources currently return, so the parsers can be checked.
