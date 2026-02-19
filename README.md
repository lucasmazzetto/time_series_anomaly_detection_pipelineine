# Time Series Anomaly Detection Pipeline

This project provides a simple API for univariate time-series anomaly detection with support for multiple series, model versioning, artifact persistence, and predictions, designed as a clean and testable pipeline.

Main frameworks and tools used in this project:

> - [FastAPI](https://fastapi.tiangolo.com/): HTTP API routes and request/response handling
> - [Pydantic](https://docs.pydantic.dev/): schema validation and data parsing
> - [SQLAlchemy](https://www.sqlalchemy.org/): ORM and database access layer
> - [PostgreSQL](https://www.postgresql.org/): metadata and model version persistence
> - [Redis](https://redis.io/): latency telemetry storage
> - [NumPy](https://numpy.org/): baseline anomaly model statistics
> - [Plotly](https://plotly.com/python/): `/plot` HTML visualization of training data
> - [Alembic](https://alembic.sqlalchemy.org/): database schema migrations

## 🛠️ Setup

1. Clone the repository:

```bash
git clone git@github.com:lucasmazzetto/time_series_anomaly_detection_pipelineine.git
```

2. Create `.env` from `.env.example`:

```bash
cd time_series_anomaly_detection_pipelineine
mv .env.example .env
```

3. Edit `.env` as needed.  
   If you only want to run locally for testing purposes, defaults are enough.

## 🐳 Docker

Build and start all containers:

```bash
docker compose up --build
```

Stop containers:

```bash
docker compose down
```

> [!NOTE]
> This `docker-compose.yml` is designed for local development: it builds
> the API image from `Dockerfile`, pulls PostgreSQL and Redis images, and
> wires all services for local execution. If PostgreSQL and Redis are hosted
> in distributed or managed servers, you can split or adapt the compose file
> and point the API to external services via `.env`.

## 🚀 API

Main API flow:

1. Train a series with `POST /fit/{series_id}`
2. Predict anomalies with `POST /predict/{series_id}` (optional query string: `?version=<number>`; default is latest)
3. Check service and metrics with `GET /healthcheck`
4. Visualize data with `GET /plot?series_id=<id>&version=<number>` (query string required; open this endpoint in a browser to see the rendered chart)

### Example usage

#### Train:

```bash
curl -X POST http://localhost:8000/fit/sensor_01 \
  -H "Content-Type: application/json" \
  -d '{"timestamps":[1705000000,1705000001,1705000002,1705000003],"values":[10.0,11.2,10.4,10.8]}'
```

#### Predict (latest model):

```bash
curl -X POST http://localhost:8000/predict/sensor_01 \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"1705000010","value":13.0}'
```

#### Predict (specific version):

```bash
curl -X POST "http://localhost:8000/predict/sensor_01?version=v1" \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"1705000011","value":10.3}'
```

#### Healthcheck:

```bash
curl http://localhost:8000/healthcheck
```

## 📚 Interactive docs:

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## 🏛️ Architecture

Detailed architecture and flow documentation:

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Training sequence: [docs/training-sequence.mmd](docs/training-sequence.mmd)
- Fit sequence: [docs/fit-sequence.mmd](docs/fit-sequence.mmd)
- Healthcheck sequence: [docs/healthcheck-sequence.mmd](docs/healthcheck-sequence.mmd)
- Plot flow: [docs/plot-view-flow.mmd](docs/plot-view-flow.mmd)

## 🧪 Testing

Run all tests inside the API container:

```bash
docker compose exec api pytest -q
```

## ⏱️ Benchmark

This project includes a stress benchmark jupyter notebook.

**Summary usage:**

1. Start the stack (`docker compose up --build`)
2. Create and activate a virtual environment on your host machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install notebook dependencies on your host machine:

```bash
pip install -r benchmark/requirements.txt
```

4. Open the notebook on your host machine (outside Docker)
5. Set `BASE_URL` (usually `http://localhost:8000`)
6. Run all cells

Full instructions at: [benchmark/README.md](benchmark/README.md)

## ⚙️ Configuration

Default environment variables from `.env.example`:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg2://postgres:postgres@db:5432/postgres` | SQLAlchemy database connection |
| `DATABASE_HOST` | `db` | PostgreSQL host (if not using explicit `DATABASE_URL`) |
| `DATABASE_PORT` | `5432` | PostgreSQL port |
| `DATABASE_NAME` | `postgres` | PostgreSQL database name |
| `DATABASE_USER` | `postgres` | PostgreSQL username |
| `DATABASE_PASSWORD` | `postgres` | PostgreSQL password |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL |
| `LATENCY_HISTORY_LIMIT` | `100` | Max latency samples stored per route type |
| `MIN_TRAINING_DATA_POINTS` | `3` | Minimum points required for training |
| `MODEL_STATE_FOLDER` | `./data/models` | Folder for persisted model states |
| `MODEL_FOLDER` | `./data/models` | Backward-compatible alias for model folder |
| `TRAINING_DATA_FOLDER` | `./data/data` | Folder for persisted training datasets |
| `DATA_FOLDER` | `./data/data` | Backward-compatible alias for data folder |

## 📈 Scalability

To scale this project, run multiple stateless API instances behind a reverse proxy/load balancer (for example NGINX, Traefik, or a cloud LB) and distribute traffic across replicas. Keep persistence services shared: connect all API instances to central PostgreSQL (metadata/model versions) and Redis (telemetry/cache), ideally with managed/high-availability setups.

Artifacts and training data must also be shared across instances. One option is mounting `MODEL_STATE_FOLDER` and `TRAINING_DATA_FOLDER` on a Network File System (NFS) so every replica reads/writes the same storage. Another option is implementing a bucket-backed storage provider (for example AWS S3) that follows the existing storage interface, removing dependency on local disk and improving elasticity.

For production operation, also consider:

- running Alembic migrations as a one-time job per deployment
- autoscaling API replicas based on CPU/latency
- setting health/readiness checks at the load balancer

## 🏁 Roadmap

The next improvements required to make this project more robust are documented in the roadmap and are currently on the radar:

- [docs/ROADMAP.md](docs/ROADMAP.md)
