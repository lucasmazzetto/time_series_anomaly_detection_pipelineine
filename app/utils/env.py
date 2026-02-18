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

    @return Integer history limit from `LATENCY_HISTORY_LIMIT` (default `500`).
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
