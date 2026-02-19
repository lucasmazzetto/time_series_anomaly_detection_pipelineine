import os


def _get_first_env(keys: tuple[str, ...], default: str) -> str:
    """@brief Return the first non-empty environment variable from a key list.

    @param keys Candidate environment variable names in lookup order.
    @param default Fallback value when all keys are unset/empty.
    @return Resolved environment value.
    """
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


def get_database_host() -> str:
    """@brief Return PostgreSQL host used by the API.

    @return Database host from `DATABASE_HOST` (default `db`).
    """
    return os.getenv("DATABASE_HOST", "db")


def get_database_port() -> int:
    """@brief Return PostgreSQL port used by the API.

    @return Database port from `DATABASE_PORT` (default `5432`).
    """
    return int(os.getenv("DATABASE_PORT", "5432"))


def get_database_name() -> str:
    """@brief Return PostgreSQL database name used by the API.

    @return Database name from `DATABASE_NAME` (default `postgres`).
    """
    return os.getenv("DATABASE_NAME", "postgres")


def get_database_user() -> str:
    """@brief Return PostgreSQL username used by the API.

    @return Database user from `DATABASE_USER` (default `postgres`).
    """
    return os.getenv("DATABASE_USER", "postgres")


def get_database_password() -> str:
    """@brief Return PostgreSQL password used by the API.

    @return Database password from `DATABASE_PASSWORD` (default `postgres`).
    """
    return os.getenv("DATABASE_PASSWORD", "postgres")


def get_database_url() -> str:
    """@brief Return SQLAlchemy database URL for PostgreSQL.

    @description Uses `DATABASE_URL` when provided; otherwise composes URL from
    host/port/name/user/password environment getters.

    @return SQLAlchemy-compatible PostgreSQL URL.
    """
    explicit_url = os.getenv("DATABASE_URL")
    if explicit_url:
        return explicit_url

    return (
        "postgresql+psycopg2://"
        f"{get_database_user()}:{get_database_password()}"
        f"@{get_database_host()}:{get_database_port()}/{get_database_name()}"
    )


def get_latency_history_limit() -> int:
    """@brief Return max number of latency samples retained in Redis lists.

    @return Integer history limit from `LATENCY_HISTORY_LIMIT` (default `100`).
    """
    return int(os.getenv("LATENCY_HISTORY_LIMIT", "100"))


def get_redis_url() -> str:
    """@brief Return Redis connection URL used by application components.

    @return Redis URL from `REDIS_URL` (default `redis://redis:6379/0`).
    """
    return os.getenv("REDIS_URL", "redis://redis:6379/0")


def get_min_training_data_points() -> int:
    """@brief Return minimum number of points required for training.

    @return Integer from `MIN_TRAINING_DATA_POINTS` (default `3`).
    """
    return int(os.getenv("MIN_TRAINING_DATA_POINTS", "3"))


def get_model_state_folder() -> str:
    """@brief Return folder path used to persist model state artifacts.

    @return First value from `MODEL_STATE_FOLDER`/`MODEL_FOLDER`,
    otherwise `./data/models`.
    """
    return _get_first_env(("MODEL_STATE_FOLDER", "MODEL_FOLDER"), "./data/models")


def get_training_data_folder() -> str:
    """@brief Return folder path used to persist training data artifacts.

    @return First value from `TRAINING_DATA_FOLDER`/`DATA_FOLDER`,
    otherwise `./data/data`.
    """
    return _get_first_env(("TRAINING_DATA_FOLDER", "DATA_FOLDER"), "./data/data")


def get_storage_backend() -> str:
    """@brief Return configured artifact storage backend.

    @return Backend identifier from `STORAGE_BACKEND` (`local` or `s3`).
    @throws ValueError If backend is not supported.
    """
    backend = os.getenv("STORAGE_BACKEND", "local").strip().lower()
    if backend not in {"local", "s3"}:
        raise ValueError("STORAGE_BACKEND must be either 'local' or 's3'.")
    return backend


def get_aws_s3_bucket() -> str:
    """@brief Return S3 bucket name for artifact persistence.

    @return Non-empty bucket name from `AWS_S3_BUCKET`.
    @throws ValueError If bucket is unset or empty.
    """
    bucket = os.getenv("AWS_S3_BUCKET", "").strip()
    if not bucket:
        raise ValueError("AWS_S3_BUCKET must be set when STORAGE_BACKEND is 's3'.")
    return bucket


def get_aws_s3_region() -> str:
    """@brief Return AWS region used to create S3 client connections.

    @return Region from `AWS_S3_REGION` (default `us-east-1`).
    """
    return os.getenv("AWS_S3_REGION", "us-east-1").strip()


def get_aws_s3_prefix() -> str:
    """@brief Return optional key prefix used for S3 object paths.

    @return Prefix from `AWS_S3_PREFIX` (default empty string).
    """
    return os.getenv("AWS_S3_PREFIX", "").strip()


def get_aws_s3_endpoint_url() -> str | None:
    """@brief Return optional custom endpoint URL for S3-compatible APIs.

    @return Endpoint URL from `AWS_S3_ENDPOINT_URL`, or None when unset.
    """
    value = os.getenv("AWS_S3_ENDPOINT_URL", "").strip()
    return value or None


def get_aws_access_key_id() -> str | None:
    """@brief Return optional static AWS access key id.

    @return Access key id from `AWS_ACCESS_KEY_ID`, or None when unset.
    """
    value = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    return value or None


def get_aws_secret_access_key() -> str | None:
    """@brief Return optional static AWS secret access key.

    @return Secret access key from `AWS_SECRET_ACCESS_KEY`, or None when unset.
    """
    value = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    return value or None


def get_aws_session_token() -> str | None:
    """@brief Return optional AWS session token.

    @return Session token from `AWS_SESSION_TOKEN`, or None when unset.
    """
    value = os.getenv("AWS_SESSION_TOKEN", "").strip()
    return value or None
