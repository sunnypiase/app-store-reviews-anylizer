# App Store Review Analyzer

FastAPI service for collecting and analyzing App Store reviews.

## Running with Docker (recommended)

The stack is Postgres + the API, run via `docker compose`. Migrations run
automatically on container start.

### 1. Get `docker` working in this WSL distro

This repo was developed in WSL2 with Docker Desktop on Windows. `docker` only
works inside a WSL distro once **WSL Integration** is turned on for that
distro:

1. Open **Docker Desktop** → **Settings** → **Resources** → **WSL
   Integration**.
2. Enable the toggle for your distro (e.g. "Ubuntu").
3. **Apply & Restart**.
4. Open a **new** terminal window (enabling integration adds you to the
   `docker` group; already-open shells won't pick that up).

After that, plain `docker` / `docker compose` work natively — verify with
`docker version` (should show both a `Client` and `Server` block, no
"permission denied").

### 2. Configure environment

```bash
cp .env.example .env
```

Defaults in `.env.example` work out of the box for local Docker use — no
edits needed unless you want different credentials.

### 3. Build and start

```bash
docker compose up -d --build
```

This builds the app image, starts Postgres (waits for its healthcheck), then
starts the app — which runs `alembic upgrade head` before launching uvicorn.

### 4. Verify it's up

```bash
docker compose ps                    # both services should be Up (db: healthy)
docker compose logs app --tail=50    # look for the uvicorn startup banner
curl http://localhost:8000/docs      # interactive API docs, should return 200
```

### 5. Common operations

```bash
docker compose logs -f app                       # tail app logs
docker compose exec app alembic current           # check applied migration
docker compose exec db psql -U postgres -d reviews  # open a DB shell
docker compose restart app                         # restart just the app
docker compose down                                 # stop (keeps DB volume)
docker compose down -v                              # stop and wipe the DB volume
```

To generate a new migration after changing a model in `app/*/models.py`, run
it against a running `db` (autogenerate needs a live connection to diff
against):

```bash
docker compose up -d db
docker compose run --rm app alembic revision --autogenerate -m "describe the change"
```

Review the generated file in `alembic/versions/` before applying it — it's
written into the repo via the bind between the container and your working
tree only if you mount the source (this project's `Dockerfile` bakes the
source in at build time, so for iterating on migrations locally it's easier
to run Alembic directly on the host — see below).

## Running locally without Docker

Requires a Postgres instance reachable at the URL in `.env`
(`DATABASE_URL`), and [`uv`](https://docs.astral.sh/uv/).

```bash
cp .env.example .env      # edit DATABASE_URL if not using the default
uv sync
uv run alembic upgrade head
uv run fastapi dev app/main.py
```

Generate a new migration the same way, directly on the host:

```bash
uv run alembic revision --autogenerate -m "describe the change"
```

## Running tests

```bash
uv run pytest
```

The whole suite needs a dedicated `reviews_test` Postgres database — a
session-scoped fixture in `tests/conftest.py` (re)creates the schema once
per test run, and route-level tests exercise real queries against it:

```bash
# db's port isn't published by default (see docker-compose.yml) — expose it
# temporarily, same as for local alembic autogenerate:
#   ports:
#     - "5432:5432"
docker compose up -d db
docker compose exec db psql -U postgres -c "CREATE DATABASE reviews_test;"  # once
uv run pytest
```

`tests/conftest.py` creates/drops the `reviews_test` schema once per test
session from the SQLAlchemy models directly (no Alembic needed for tests)
and wraps each test in a rolled-back transaction.

## Approach & design decisions

The API is fully synchronous, on purpose: collecting up to a few hundred
reviews (a handful of paginated HTTP calls) and running lightweight NLP over
them takes low single-digit seconds, not long enough to justify a job queue,
background worker, or polling. An earlier iteration of this project used a
Postgres-backed job/worker pipeline for exactly that reason and it turned
out to be more machinery than the problem needed — this version replaces it
with plain request/response endpoints.

Every resource is keyed off `sample_id`, the id of the `ReviewSample`
produced by a collect call:

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/reviews` | Fetch up to `sample_size` reviews for `{app_id, country_code}` from the App Store and persist them. Returns the full sample immediately (**201**) — no polling. |
| GET | `/api/v1/reviews/{sample_id}` | The raw persisted sample (all reviews). |
| GET | `/api/v1/reviews/{sample_id}/download?format=json\|csv` | The same data as a file attachment. |
| GET | `/api/v1/metrics/{sample_id}` | Average rating + rating distribution. Computed on the fly, nothing stored. |
| GET | `/api/v1/insights/{sample_id}` | Sentiment distribution, rating/sentiment disagreements, negative-review keywords, actionable insights. Computed on the fly. |
| GET | `/api/v1/reports/{sample_id}` | An HTML report combining the above, with embedded charts. |

A collect call creates a fresh sample every time rather than deduplicating
on `(app_id, country_code)` — reviews change over time, and there's no
job state to reuse, so simplicity wins over avoiding a duplicate fetch.

**Data collection** (`app/reviews/appstore/`) validates the app exists via
Apple's iTunes Lookup API, then paginates the (undocumented) customer
reviews RSS feed, retrying 403/429/5xx with backoff+jitter and
self-throttling between pages — Apple's rate limiting is IP-based and
undocumented, so this is a deliberately conservative default. Three
domain exceptions (`AppNotFoundError`, `InvalidCountryCodeError`,
`AppStoreUnavailableError`) are mapped to `404`/`400`/`503` by exception
handlers in `app/main.py`, each logged with request context before
responding — see `app/reviews/appstore/http.py`, `lookup_client.py`, and
`reviews_client.py` for the retry/logging details.

**Insights** (`app/insights/service.py`) uses two standard, lightweight NLP
techniques rather than a heavyweight model, since a per-request model
download/inference round trip isn't worth it for ≤100 short reviews:
- **Sentiment**: `vaderSentiment`, a rule-based analyzer tuned for short
  informal text (its original use case is social media posts, which reviews
  closely resemble). Each review's compound score is bucketed into
  positive/neutral/negative using VADER's standard thresholds.
- **Negative-review keywords**: reviews with a rating ≤ 2 are "negative
  reviews." TF-IDF (`scikit-learn`, unigrams + bigrams) is fit against the
  *whole* sample so ubiquitous terms score low, then evaluated only on the
  negative subset, so the ranking reflects terms concentrated in complaints
  rather than just common words. Actionable insights are then built by
  taking the top distinct-themed phrases (bigrams preferred over the
  unigrams they contain) and reporting evidence count + example reviews for
  each — a deliberately simple, rule-based step; no LLM call is in scope
  here.

**Reports** (`app/reports/`) render a Jinja2 HTML template with rating and
sentiment distribution charts generated by `matplotlib` and embedded as
base64 PNGs — no static file serving or temp files needed.

## Sample report

`docs/sample_report/` contains a real run against the Facebook app
(`app_id=284882215`, `us` store, 100 reviews) produced end-to-end from this
codebase: `facebook_report.html` (open it directly in a browser),
`facebook_metrics.json`, `facebook_insights.json`, and the raw
`facebook_reviews.json`. Reproduce it (or run it against any other app) with:

```bash
curl -X POST http://localhost:8000/api/v1/reviews \
  -H "Content-Type: application/json" \
  -d '{"app_id": 284882215, "country_code": "us", "sample_size": 100}'
# -> {"id": "<sample_id>", ...}

curl http://localhost:8000/api/v1/metrics/<sample_id>
curl http://localhost:8000/api/v1/insights/<sample_id>
curl http://localhost:8000/api/v1/reports/<sample_id> -o report.html
```

**Known limitations:**
- Apple's App Store rate limits are undocumented and IP-based.
  Self-throttling and retry-with-backoff reduce, but don't eliminate,
  403/429s under bursty load — a collect call can occasionally take several
  seconds or fail with `503` under load.
- TF-IDF keyword extraction is inherently noisy on short review text
  (a handful of words each); it surfaces genuinely useful themes but isn't
  as precise as a purpose-built keyphrase model would be.
- No auth/rate limiting on the API itself — out of scope for this MVP.
