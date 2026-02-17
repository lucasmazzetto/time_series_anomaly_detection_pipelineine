# Anomaly Detection API Architecture

## Scope

This document describes the current project structure and runtime flow for:
- `POST /fit/{series_id}`
- `POST /predict/{series_id}`
- `GET /healthcheck`
- `GET /plot`

## Project Structure

```text
app/
  api/                # FastAPI route handlers
  middleware/         # Request latency middleware
  services/           # Application use-cases
  core/               # Model/trainer abstractions and implementation
  schemas/            # Request/response/domain validation models
  repositories/       # Storage interface + local filesystem adapter
  database/           # ORM entities and query helpers
  db.py               # Engine/session configuration
  main.py             # App bootstrap and route registration
docs/
  architecture.md
  training-sequence.mmd
  fit-sequence.mmd
  healthcheck-sequence.mmd
  plot-view-flow.mmd
migrations/
  versions/           # Alembic schema migrations
tests/                # API, service, schema, core and repository tests
data/                 # Persisted artifacts (models/data)
```

## Layer Responsibilities

- API layer (`app/api/*.py`)
  - Validates path/query/body input with Pydantic/FastAPI
  - Instantiates service objects and returns typed responses
- Service layer (`app/services/*.py`)
  - Contains use-case logic and exception mapping to HTTP status codes
- Core layer (`app/core/*.py`)
  - Model contract (`Model`), trainer contract (`Trainer`), and `SimpleModel`
- Schema layer (`app/schemas/*.py`)
  - Input validation and conversion (`TrainData -> TimeSeries`, `PredictData -> DataPoint`)
- Persistence layer
  - Database metadata: `app/database/*.py`
  - Redis latency cache access: `app/database/latency.py`
  - Artifact storage: `app/repositories/local_storage.py`
  - Session lifecycle: `app/db.py`
- Middleware (`app/middleware/latency.py`)
  - Tracks latency for successful `/fit/*` and `/predict/*` requests
  - Stores raw latency samples in Redis (bounded history via list trim)

## Runtime Components

- App bootstrap: `app/main.py`
  - Registers routers: train, predict, healthcheck, plot
  - Attaches middleware: `track_request_latency`
- Services:
  - `TrainService` orchestrates training + metadata + artifact writes
  - `PredictService` resolves metadata/artifact and performs prediction
  - `HealthCheckService` reads Redis raw latencies and computes avg/P95
  - `PlotService` resolves training-data metadata/artifact and renders Plotly HTML
- Database entities:
  - `AnomalyDetectionRecord` in `anomaly_detection_models`
  - `SeriesVersionRecord` in `series_versions`
  - `LatencyRecord` in Redis lists (`train_latencies`, `predict_latencies`)

## Endpoint Flows

### `POST /fit/{series_id}`

1. Route validates `series_id` and `TrainData`.
2. Service converts to `TimeSeries` and runs training preflight validation.
3. Trainer calls model fit, then returns model state.
4. Metadata row is created and version is assigned atomically when needed.
5. Model state and training data are persisted as JSON artifacts.
6. Metadata paths are updated, transaction is committed, and `TrainResponse` is returned.

### `POST /predict/{series_id}`

1. Route validates `series_id`, `PredictData`, and query `version`.
2. Version is normalized (`0`, `1`, `v1`, `V1` supported).
3. Service resolves latest/specific model metadata.
4. Service loads model artifact, restores model state, and predicts anomaly.
5. `PredictResponse` returns anomaly flag and resolved model version.

### `GET /healthcheck`

1. Reads raw latency values from Redis lists (`train_latencies`, `predict_latencies`).
2. Computes average and P95 in `HealthCheckService`.
3. Counts trained series from `series_versions`.
4. Returns latency metrics and trained-series counter.

### `GET /plot`

1. Route validates `series_id` and query `version` (supports `0`, `1`, `v1`, `V1`).
2. Service resolves latest/specific training-data metadata (`data_path`).
3. Service loads persisted training data from local storage.
4. Service renders a Plotly bar chart and returns HTML.

## Training Sequence Diagram

The same diagram is also available at `docs/training-sequence.mmd`.
An equivalent copy focused on `/fit` naming is available at `docs/fit-sequence.mmd`.

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant MW as Latency Middleware
    participant Route as FastAPI /fit/{series_id}
    participant Service as TrainService
    participant Trainer as AnomalyDetectionTrainer
    participant Model as SimpleModel
    participant Record as AnomalyDetectionRecord
    participant Version as SeriesVersionRecord
    participant DB as PostgreSQL
    participant Cache as LatencyRecord
    participant Redis as Redis
    participant Store as LocalStorage
    participant FS as Local Filesystem

    Client->>MW: POST /fit/{series_id}
    MW->>Route: call_next(request)
    Route->>Route: Validate SeriesId + TrainData
    Route->>Service: train(series_id, payload)
    Service->>Service: _to_time_series + validate_for_training()

    Service->>Trainer: train(time_series)
    Trainer->>Model: fit(data, callback)
    Trainer->>Model: save()
    Model-->>Trainer: ModelState
    Trainer-->>Service: ModelState

    Service->>Record: build(series_id, version=None, paths=None)
    Service->>Record: save(session, record)
    Record->>Version: next_version(session, series_id)
    Version->>DB: INSERT .. ON CONFLICT .. RETURNING
    DB-->>Version: assigned version
    Version-->>Record: version
    Record->>DB: INSERT + FLUSH
    DB-->>Record: metadata persisted

    Service->>Store: save_state(series_id, version, state)
    Store->>FS: write model JSON
    FS-->>Store: model_path

    Service->>Store: save_data(series_id, version, time_series)
    Store->>FS: write training data JSON
    FS-->>Store: data_path

    Service->>Record: update(model_path, data_path)
    Service->>Record: commit()
    Record->>DB: COMMIT

    Service-->>Route: TrainResponse
    Route-->>MW: 200 response
    MW->>Cache: push_latency(train, elapsed_ms)
    Cache->>Redis: RPUSH + LTRIM
    MW-->>Client: 200 response
```

## Validation Rules

- `SeriesId`: non-empty, trimmed, regex `[A-Za-z0-9._-]+`, rejects `..`
- `TrainData`: non-negative integer timestamps, finite numeric values, same length arrays
- `TimeSeries`: at least 2 points, strictly increasing timestamps
- `PredictData`: timestamp is non-empty digits-only string, value is finite numeric
- `Version`: accepts digits with optional `v`/`V` prefix

## Fit Sequence Diagram

The same diagram is also available at `docs/fit-sequence.mmd`.

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant MW as Latency Middleware
    participant Route as FastAPI /fit/{series_id}
    participant Service as TrainService
    participant Trainer as AnomalyDetectionTrainer
    participant Model as SimpleModel
    participant Record as AnomalyDetectionRecord
    participant Version as SeriesVersionRecord
    participant DB as PostgreSQL
    participant Cache as LatencyRecord
    participant Redis as Redis
    participant Store as LocalStorage
    participant FS as Local Filesystem

    Client->>MW: POST /fit/{series_id}
    MW->>Route: call_next(request)
    Route->>Route: Validate SeriesId + TrainData
    Route->>Service: train(series_id, payload)
    Service->>Service: _to_time_series + validate_for_training()

    Service->>Trainer: train(time_series)
    Trainer->>Model: fit(data, callback)
    Trainer->>Model: save()
    Model-->>Trainer: ModelState
    Trainer-->>Service: ModelState

    Service->>Record: build(series_id, version=None, paths=None)
    Service->>Record: save(session, record)
    Record->>Version: next_version(session, series_id)
    Version->>DB: INSERT .. ON CONFLICT .. RETURNING
    DB-->>Version: assigned version
    Version-->>Record: version
    Record->>DB: INSERT + FLUSH
    DB-->>Record: metadata persisted

    Service->>Store: save_state(series_id, version, state)
    Store->>FS: write model JSON
    FS-->>Store: model_path

    Service->>Store: save_data(series_id, version, time_series)
    Store->>FS: write training data JSON
    FS-->>Store: data_path

    Service->>Record: update(model_path, data_path)
    Service->>Record: commit()
    Record->>DB: COMMIT

    Service-->>Route: TrainResponse
    Route-->>MW: 200 response
    MW->>Cache: push_latency(train, elapsed_ms)
    Cache->>Redis: RPUSH + LTRIM
    MW-->>Client: 200 response
```

## Healthcheck Sequence Diagram

The same diagram is also available at `docs/healthcheck-sequence.mmd`.

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant MW as Latency Middleware
    participant Route as FastAPI /healthcheck
    participant Service as HealthCheckService
    participant Cache as LatencyRecord
    participant Redis as Redis
    participant Version as SeriesVersionRecord
    participant DB as PostgreSQL

    Client->>MW: GET /healthcheck
    MW->>Route: call_next(request)
    Route->>Service: healthcheck(session)
    Service->>Cache: get_latencies(train)
    Cache->>Redis: LRANGE train_latencies
    Redis-->>Cache: train latency list
    Cache-->>Service: train latencies
    Service->>Cache: get_latencies(predict)
    Cache->>Redis: LRANGE predict_latencies
    Redis-->>Cache: predict latency list
    Cache-->>Service: predict latencies
    Service->>Service: compute avg + p95
    Service->>Version: count_series(session)
    Version->>DB: SELECT COUNT(series_id)
    DB-->>Version: total trained series
    Version-->>Service: series_count
    Service-->>Route: HealthCheckResponse
    Route-->>MW: 200 response
    MW-->>Client: 200 response
```

## Plot View Flow Diagram

The same diagram is also available at `docs/plot-view-flow.mmd`.

```mermaid
sequenceDiagram
    autonumber
    actor Browser as Client/Browser
    participant MW as Latency Middleware
    participant Route as FastAPI /plot
    participant Service as PlotService
    participant Record as AnomalyDetectionRecord
    participant DB as PostgreSQL
    participant Store as LocalStorage
    participant FS as Local Filesystem
    participant Plotly as Plotly Renderer

    Browser->>MW: GET /plot?series_id=...&version=vN
    MW->>Route: call_next(request)
    Route->>Route: Validate SeriesId + Version
    Route->>Service: render_training_data(series_id, version)
    Service->>Record: get_last_training_data() or get_training_data()
    Record->>DB: SELECT metadata row
    DB-->>Record: data_path + version
    Record-->>Service: metadata dict
    Service->>Store: load_data(data_path)
    Store->>FS: read training data JSON
    FS-->>Store: TimeSeries payload
    Store-->>Service: TimeSeries
    Service->>Plotly: px.bar(...) + to_html(...)
    Plotly-->>Service: chart HTML
    Service-->>Route: full HTML document
    Route-->>MW: 200 text/html
    MW-->>Browser: 200 text/html
```

## Persistence and Versioning

- `anomaly_detection_models`
  - primary key: `(series_id, version)`
  - stores `model_path`, `data_path`, `created_at`, `updated_at`
- `series_versions`
  - primary key: `series_id`
  - stores `last_version`
- Version increment strategy:
  - PostgreSQL upsert with `RETURNING` to ensure atomic version allocation

## Artifact Storage

- Model state path pattern:
  - `./data/models/<series_id>/<series_id>_model_v<version>.json`
- Training data path pattern:
  - `./data/data/<series_id>/<series_id>_data_v<version>.json`
- Folder resolution precedence:
  - params dict keys
  - environment variables
  - fallback defaults

## Configuration Defaults

- `DATABASE_URL`: `postgresql+psycopg2://postgres:postgres@db:5432/postgres`
- `MIN_TRAINING_DATA_POINTS`: `3`
- `REDIS_URL`: `redis://redis:6379/0`
- `LATENCY_HISTORY_LIMIT`: `10`
- Storage fallback folders:
  - model state: `./data/models`
  - training data: `./data/data`

## Error Behavior

- Training:
  - validation/preflight errors -> HTTP `422`
  - unexpected runtime errors -> HTTP `500`
  - session rollback before re-raising failures
- Prediction:
  - invalid inputs -> HTTP `422` or `400` (service defensive checks)
  - missing metadata/artifact -> HTTP `404`
  - missing `model_path` or unexpected errors -> HTTP `500`
- Latency cache/healthcheck:
  - middleware Redis write failures are logged and ignored (request still succeeds)
  - healthcheck Redis read failures return zeroed latency metrics
