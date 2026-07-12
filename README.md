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

## Sentiment analysis evaluation (`scripts/sentiment_eval/`)

Standalone research scripts (not part of the running API) that collect a
golden set of real reviews, manually label their sentiment, then compare a
"default" lexicon-based NLP approach (VADER, TextBlob) against a Gemini LLM
call. Results are written to `docs/SENTIMENT_ANALYSIS_RESULTS.md`.

### 1. Install the eval dependencies

```bash
uv sync --group eval
```

### 2. Get a free Gemini API key

1. Go to <https://aistudio.google.com/apikey> and create a free-tier API key.
2. Add it to `.env` at the repo root:
   ```
   GEMINI_API_KEY=your-key-here
   ```
   (Optional: `GEMINI_MODEL_NAME` to override the default
   `gemini-flash-lite-latest` -- chosen because its free tier is a
   15-requests/minute cap rather than the 20-requests/*day* cap on
   `gemini-flash-latest`, which isn't enough to score a 1500-review set.)

### 3. Run the pipeline, in order

```bash
# 1. Fetch 500 real reviews each for 3 apps via the App Store collector
#    (edit the APPS list in collect_multi.py to change which apps/how many)
uv run python -m scripts.sentiment_eval.collect_multi

# 1b. Or fetch reviews for a single app instead
uv run python -m scripts.sentiment_eval.collect_reviews --app-id 1459969523 --country us --limit 100

# 2. Regenerate the manually labeled golden dataset (labels are hardcoded in
#    the script -- see its docstring for the labeling methodology)
uv run python -m scripts.sentiment_eval.build_golden_dataset

# 3. Score with VADER + TextBlob (no network, no API key needed)
uv run python -m scripts.sentiment_eval.baseline_lexicon

# 4. Score with Gemini (needs GEMINI_API_KEY from step 2 above; batches of
#    150 reviews fired concurrently, resumable if a run gets interrupted)
uv run python -m scripts.sentiment_eval.gemini_sentiment

# 5. Compare all approaches against the golden labels -- produces a
#    per-label correctness matrix (precision/recall/F1 + confusion matrix)
#    and a macro-F1 ranking, since raw accuracy is misleading on an
#    imbalanced class distribution
uv run python -m scripts.sentiment_eval.evaluate
```

Each script's output lands in `scripts/sentiment_eval/data/`. Step 5 always
produces `docs/SENTIMENT_ANALYSIS_RESULTS.md`; if step 4 hasn't been run yet,
the Gemini section is left marked as pending rather than failing.
