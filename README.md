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
