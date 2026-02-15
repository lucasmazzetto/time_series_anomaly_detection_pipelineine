from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from data.tables import Base, AnomalyDetectionModel
from utils.params import load_params


class Database(ABC):
    @abstractmethod
    def add_training(self, series_id: str) -> int:
        pass

    @abstractmethod
    def save_training_data(self, series_id: str, version: int, data_path: str) -> None:
        pass

    @abstractmethod
    def save_model_metrics(
        self, series_id: str, version: int, model_path: str, metrics: Dict[str, Any]
    ) -> None:
        pass


class SQLDatabase(Database):
    def __init__(self, url: str, echo: bool = False) -> None:
        self.engine = create_engine(url, echo=echo, future=True)
        self.Session = sessionmaker(
            bind=self.engine, class_=Session, expire_on_commit=False, future=True
        )
        self.migrate()

    def migrate(self) -> None:
        Base.metadata.create_all(self.engine)
        self._run_sql_migrations()

    def _run_sql_migrations(self) -> None:
        migrations_path = Path(__file__).resolve().parent / "migrations"
        if not migrations_path.exists():
            return

        with self.engine.begin() as conn:
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    filename TEXT PRIMARY KEY,
                    applied_at TIMESTAMP WITH TIME ZONE NOT NULL
                )
                """
            )
            applied = {
                row[0]
                for row in conn.execute(text("SELECT filename FROM schema_migrations"))
            }
            for migration in sorted(migrations_path.glob("*.sql")):
                if migration.name in applied:
                    continue
                sql = migration.read_text(encoding="utf-8")
                statements = _split_sql_statements(sql)
                for statement in statements:
                    normalized = statement.strip().lower()
                    if not normalized:
                        continue
                    if _is_comment_only(statement):
                        continue
                    if normalized.startswith("begin") or normalized.startswith("commit"):
                        continue
                    conn.exec_driver_sql(statement)
                conn.execute(
                    text(
                        "INSERT INTO schema_migrations (filename, applied_at) "
                        "VALUES (:filename, :applied_at)"
                    ),
                    {"filename": migration.name, "applied_at": datetime.now(timezone.utc)},
                )

    def _next_version(self, session: Session, series_id: str) -> int:
        stmt = select(func.max(AnomalyDetectionModel.version)).where(
            AnomalyDetectionModel.series_id == series_id
        )
        current = session.execute(stmt).scalar_one_or_none()
        return int(current or 0) + 1

    def _get_or_create_record(
        self, session: Session, series_id: str, version: int
    ) -> AnomalyDetectionModel:
        record = session.get(AnomalyDetectionModel, (series_id, version))
        if record is None:
            now = datetime.now(timezone.utc)
            record = AnomalyDetectionModel(
                series_id=series_id, version=version, created_at=now, updated_at=now
            )
            session.add(record)
        return record

    def add_training(self, series_id: str) -> int:
        with self.Session() as session:
            version = self._next_version(session, series_id)
            now = datetime.now(timezone.utc)
            record = AnomalyDetectionModel(series_id=series_id, version=version, created_at=now, updated_at=now)
            session.add(record)
            session.commit()
            return version

    def save_training_data(self, series_id: str, version: int, data_path: str) -> None:
        with self.Session() as session:
            record = self._get_or_create_record(session, series_id, version)
            record.data_path = data_path
            record.updated_at = datetime.now(timezone.utc)
            session.commit()

    def save_model_metrics(
        self, series_id: str, version: int, model_path: str, metrics: Dict[str, Any]
    ) -> None:
        with self.Session() as session:
            record = self._get_or_create_record(session, series_id, version)
            record.model_path = model_path
            record.updated_at = datetime.now(timezone.utc)
            session.commit()


def _build_sqlite_url(params: Dict[str, Any]) -> str:
    url = params.get("database_url")
    if isinstance(url, str) and url.startswith("sqlite"):
        return url
    path = Path(params.get("database_path", "data/training.db")).resolve()
    return f"sqlite:///{path}"


def database_from_params(params: Dict[str, Any] | None = None) -> SQLDatabase:
    params = params or load_params()
    provider = str(params.get("database", "SQLite")).strip().lower()
    echo = bool(params.get("database_echo", False))

    if provider in {"sqlite", "sqlite3"}:
        url = _build_sqlite_url(params)
        return SQLDatabase(url=url, echo=echo)

    if provider in {"postgresql", "postgres"}:
        url = params.get("database_url")
        if not url:
            raise ValueError("database_url must be set for PostgreSQL.")
        return SQLDatabase(url=url, echo=echo)

    raise ValueError(f"Unsupported database provider: {provider}")


def _split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    prev_char = ""
    for ch in sql:
        if ch == "'" and not in_double and prev_char != "\\":
            in_single = not in_single
        elif ch == '"' and not in_single and prev_char != "\\":
            in_double = not in_double

        if ch == ";" and not in_single and not in_double:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
        else:
            current.append(ch)
        prev_char = ch

    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def _is_comment_only(statement: str) -> bool:
    for line in statement.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("--"):
            continue
        return False
    return True
