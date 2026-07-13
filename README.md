# App Store Review Analyzer

REST API that collects App Store reviews for a given app, computes rating
metrics, runs sentiment analysis, and produces an HTML report with
actionable, evidence-backed insights.

**Stack:** FastAPI · PostgreSQL · SQLAlchemy/Alembic · Gemini (sentiment +
insights, with local fallbacks) · matplotlib (charts).

## Contents

- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Run with Docker (recommended)](#run-with-docker-recommended)
  - [Run without Docker](#run-without-docker)
  - [Environment variables](#environment-variables)
- [API at a glance](#api-at-a-glance)
- [Design review](#design-review)
  - [1. Review collection — custom RSS client](#1-review-collection--custom-rss-client)
  - [2. Sentiment analysis — evaluated on a golden dataset](#2-sentiment-analysis--evaluated-on-a-golden-dataset)
  - [3. Report structure](#3-report-structure)
- [Future scalability](#future-scalability)
- [Tests](#tests)

## Setup

### Prerequisites

- **Docker + Docker Compose** (easiest path), or
- **Python 3.12+ with [`uv`](https://docs.astral.sh/uv/)** and a reachable
  PostgreSQL instance if running without Docker.
- Optional: a free **Gemini API key** from
  <https://aistudio.google.com/apikey>. Without it the app still works — it
  falls back to local NLP (see [Environment variables](#environment-variables)).

### Run with Docker (recommended)

```bash
cp .env.example .env            # defaults work out of the box
docker compose up -d --build    # starts Postgres, runs migrations, starts the API
curl http://localhost:8000/docs # interactive API docs — should return 200
```

That's it. Migrations (`alembic upgrade head`) run automatically on
container start. Useful commands:

```bash
docker compose logs -f app   # tail app logs
docker compose down          # stop (keeps the DB volume)
docker compose down -v       # stop and wipe the DB
```

> **WSL2 note:** if `docker` says "permission denied", enable your distro in
> Docker Desktop → Settings → Resources → **WSL Integration**, then open a
> new terminal.

### Run without Docker

Requires Postgres reachable at the `DATABASE_URL` in `.env`:

```bash
cp .env.example .env             # edit DATABASE_URL if needed
uv sync
uv run alembic upgrade head
uv run fastapi dev app/main.py
```

### Environment variables

All configuration lives in `.env` (see `.env.example` for full comments):

| Variable | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | yes (non-Docker) | Postgres connection string. Ignored under Docker Compose — the app container builds it from the `POSTGRES_*` values. |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | yes (Docker) | Credentials for the Postgres container and the app's connection to it. Defaults work as-is. |
| `GEMINI_API_KEY` | no | Enables Gemini-powered sentiment analysis and actionable-insight generation. **If unset, the app automatically falls back** to VADER (sentiment) and TF-IDF keyword themes (insights) — less accurate, but fully local. |
| `GEMINI_MODEL_NAME` | no | Sentiment model, default `gemini-flash-lite-latest`. |
| `GEMINI_INSIGHTS_MODEL_NAME` | no | Insight-generation model, default `gemini-3.1-flash-lite`. |

## API at a glance

Everything is keyed by `sample_id` — the id returned when you collect
reviews. The API is deliberately synchronous: collecting ~100 reviews plus
NLP takes a few seconds, not enough to justify job queues or polling.

| Method | Path | Returns |
|---|---|---|
| POST | `/api/v1/reviews` | Collects up to `sample_size` reviews for `{app_id, country_code}` and persists them. Responds `201` with the full sample. |
| GET | `/api/v1/reviews/{sample_id}` | Raw persisted reviews. |
| GET | `/api/v1/reviews/{sample_id}/download?format=json\|csv` | Same data as a downloadable file. |
| GET | `/api/v1/metrics/{sample_id}` | Average rating + rating distribution. |
| GET | `/api/v1/insights/{sample_id}` | Sentiment distribution, negative-review keywords, actionable insights. |
| GET | `/api/v1/reports/{sample_id}` | Self-contained HTML report with charts. |

Quick start:

```bash
curl -X POST http://localhost:8000/api/v1/reviews \
  -H "Content-Type: application/json" \
  -d '{"app_id": 284882215, "country_code": "us", "sample_size": 100}'
# -> {"id": "<sample_id>", ...}

curl http://localhost:8000/api/v1/reports/<sample_id> -o report.html
```

## Design review

### 1. Review collection — custom RSS client

We fetch reviews from Apple's (undocumented) customer-reviews RSS feed with
a **custom client** (`app/reviews/appstore/`) instead of an off-the-shelf
scraping library. The existing open-source options were ruled out on
inspection: the maintained ones had their **last release ~2 years ago**, are
**synchronous only** (this is an async FastAPI service), and have **no
reliable error handling** for the feed's real-world failure modes — Apple
returns `200` with a missing `entry` key for nonexistent apps, plain-text
`400`s for bad country codes, and undocumented IP-based `403`/`429`
throttling with no `Retry-After` headers.

The custom client:

- validates the app id first via the official iTunes Lookup API (so "app
  doesn't exist" is a clean `404`, not an empty scrape);
- paginates the RSS feed (hard-capped by Apple at 500 reviews/app) with
  retry + exponential backoff + jitter on `403`/`429`/`5xx`, and
  self-throttles between pages;
- maps failures to three domain exceptions → `404` (app not found), `400`
  (invalid country), `503` (App Store unavailable), each logged with request
  context.

Feed research notes (live-tested formats, error codes, rate-limit behavior)
are in [`docs/APPSTORE_RSS_RESEARCH.md`](docs/APPSTORE_RSS_RESEARCH.md).

### 2. Sentiment analysis — evaluated on a golden dataset

Rather than picking an NLP library on faith, we **built a golden dataset of
1,500 real reviews** (3 apps from different categories × 500 reviews, so the
class mix isn't skewed by one app's user base), **labeled it with Claude and
then manually reviewed the labels**, and scored the candidate approaches
against it:

| Method | Accuracy | Macro F1 | Recall: positive | Recall: neutral | Recall: negative |
|---|---|---|---|---|---|
| **Gemini (`gemini-flash-lite-latest`)** | **92.3%** | **0.87** | 97.2% | 63.0% | 98.1% |
| Rating-derived (naive baseline) | 83.5% | 0.69 | 98.9% | 18.9% | 87.1% |
| VADER | 69.5% | 0.55 | 89.9% | 11.9% | 59.8% |
| TextBlob | 60.9% | 0.49 | 84.7% | 30.8% | 29.7% |

We ranked by **macro F1, not accuracy**: the set is 56% positive, so a
classifier that always says "positive" would look decent on accuracy while
never catching an unhappy user — the reviews this product exists to surface.

**Result: Gemini won decisively**, especially on negative recall (98% vs.
VADER's 60% and TextBlob's 30%), so it's the production classifier. VADER
remains the automatic fallback when no API key is set or Gemini is
unavailable. Full per-method confusion matrices, example disagreements, and
caveats: [`docs/SENTIMENT_ANALYSIS_RESULTS.md`](docs/SENTIMENT_ANALYSIS_RESULTS.md).
The whole evaluation is reproducible via `scripts/sentiment_eval/`.

Sentiment is classified from the review **text only** — never the star
rating — which lets the insights endpoint surface rating/sentiment
disagreements (e.g. complaints hidden behind 4–5★ ratings).

### 3. Report structure

`GET /api/v1/reports/{sample_id}` renders a single self-contained HTML page
(Jinja2 template, charts embedded as base64 PNGs — no static files):

1. **Stat tiles** — sample size, average rating, sentiment split.
2. **Charts** — rating distribution and sentiment distribution (matplotlib).
3. **Executive summary** — a short prose overview of the sample.
4. **Negative-review keywords** — TF-IDF (unigrams + bigrams) fitted on the
   whole sample but scored on rating ≤ 2 reviews, so it surfaces terms
   *concentrated in complaints* rather than terms common everywhere.
5. **Actionable insights** — one card per complaint theme. A second Gemini
   structured-output call groups complaint reviews into up to five themes,
   each with a problem summary and a concrete recommended fix. The LLM only
   drafts text and cites review ids; **the application verifies every cited
   id, drops non-recurring themes, and computes all counts itself** — an
   ungrounded response is discarded. Each card lists *every* supporting
   review (stars, title, date, version, excerpt), so any claim in the report
   can be checked against its raw evidence.

The same data is available as JSON via `/api/v1/metrics/{sample_id}` and
`/api/v1/insights/{sample_id}` (which also states, via
`actionable_insights_source`, whether Gemini or the local fallback produced
the insights).

## Future scalability

The endpoints are synchronous **by choice** for this test task — it kept the
focus on the analysis part. For real production usage, the first things I
would refactor:

1. **Persist computed results.** The DB currently stores only reviews;
   metrics and insights should be stored per sample too, so repeated
   `/insights` and `/reports` calls return the stored result instead of
   recomputing (and re-spending Gemini quota) every time.
2. **Make collection and insight generation async.** These already take
   several seconds, and future improvements will only make them longer.
   Move them to a task queue (e.g. Celery) with the standard process
   pattern:
   - `POST /collect` → `202` + `process_id`
   - `GET /status/{process_id}` → `status` + `result_id | null`
   - `GET /result/{result_id}` → the result
3. **Rate limiting** on the review-collection and report endpoints — both
   are expensive (Apple's IP-based throttling on one side, LLM quota on the
   other), so they're the first candidates for abuse.

### Toward continuous monitoring

For extending functionality, the natural next step is a **lifetime review
monitoring system** instead of one-off samples: reviews live in the DB
permanently, the history is backfilled once via a paid third-party review
scraper service (Apple's RSS feed only exposes the latest 500 reviews per
app), and from then on our own RSS client collects new reviews
incrementally on a cron job in near real time.

### LLM cost, and how to cut it if it matters

At current Gemini 3.1 Flash-Lite pricing ($0.25 / 1M input tokens, $1.50 /
1M output, July 2026), with the average review at ~140 characters (~35
tokens of text, ~60 with the JSON envelope, measured on the sample data),
**analyzing 1,000 reviews costs about $0.05**:

| Call | Requests | Input tokens | Output tokens | Cost |
|---|---|---|---|---|
| Sentiment (batches of 150) | 7 | ~65k | ~15k (label per review) | ~$0.04 |
| Actionable insights (complaint subset, ~40%) | 1 | ~25k | ~3k (5 themes + evidence ids) | ~$0.01 |

That's negligible for on-demand samples, but at continuous-monitoring scale
(many apps × new reviews daily) it adds up. If cost becomes a problem, the
optimization is a **VADER pre-filter**: when the star rating and VADER
agree, the sentiment is obvious and doesn't need an LLM — 1–2★ + VADER
negative is negative, 4–5★ + VADER positive is positive, 3★ + VADER
neutral is neutral. Only the remaining disagreement cases get
double-checked with a Gemini call. On the golden dataset, VADER and the
rating-derived label agree ~75% of the time, so this cuts sentiment LLM
volume roughly **4×** while spending the LLM budget exactly where lexicon
methods are weakest.

## Tests

```bash
docker compose up -d db
docker compose exec db psql -U postgres -c "CREATE DATABASE reviews_test;"  # once
uv run pytest
```

The suite runs route-level tests against a real `reviews_test` Postgres
database; `tests/conftest.py` creates the schema per session and rolls each
test back. Note: `db`'s port isn't published by default — add
`ports: ["5432:5432"]` to the `db` service in `docker-compose.yml` (or use an
already-exposed Postgres) so pytest on the host can reach it.
