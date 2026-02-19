# 🏁 Roadmap

This roadmap lists the next improvements required to make the project more robust and production-ready.

## ⬆️ High

- **API authentication with tokens**: use token-based authentication (for example, JWT or opaque API tokens) and enforce protected access for `fit`, `predict`, and operational endpoints.

- **Run migrations outside API startup and scale API workers**: execute Alembic migrations as a one-time deployment step and run the API with multiple workers behind a load balancer.

- **Request-size and payload guards for training**: enforce max payload size and max points per training request to protect memory and response time under heavy input.

## ➡️ Medium

- **Asynchronous training with a queue**: move training to background workers (for example, Celery/RQ + Redis broker) so `fit` can return quickly with a `job_id`, and expose job-status endpoints.

- **Training job progress endpoint**: add an endpoint that returns training status and completion percentage, especially for long-running training jobs.

- **Pagination strategy for large payloads**: improve endpoints that may process large datasets with pagination, chunking, or streaming to avoid request freezing and timeouts.

- **Integration tests with real PostgreSQL and Redis**: complement mocked unit tests with containerized integration tests to catch connection, migration, and runtime regressions.

- **Logs and observability**: add logs and more operational metrics and traces for better production monitoring and incident response.

- **Security middleware hardening**: configure CORS, trusted hosts, and security headers based on deployment environment.

## ⬇️ Low

- **Prediction caching for hot models and versions**: cache the most used model versions in memory or Redis to avoid repeatedly loading model parameters from disk or object storage.

- **Configurable database pool tuning**: move DB pool size/timeout/recycle settings to environment variables for per-environment tuning.
