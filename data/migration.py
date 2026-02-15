from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

from data.database import database_from_params


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return cleaned or "migration"


def _next_migration_number(migrations_path: Path) -> int:
    numbers = []
    for migration in migrations_path.glob("*.sql"):
        match = re.match(r"(\d+)_", migration.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def create_migration(name: str | None) -> Path:
    migrations_path = Path(__file__).resolve().parent / "migrations"
    migrations_path.mkdir(parents=True, exist_ok=True)
    number = _next_migration_number(migrations_path)
    slug = _slugify(name or "migration")
    filename = f"{number:03d}_{slug}.sql"
    path = migrations_path / filename
    timestamp = datetime.now(timezone.utc).isoformat()
    contents = (
        f"-- Migration: {slug}\n"
        f"-- Created at: {timestamp}\n"
        "-- Write your SQL statements below.\n"
    )
    path.write_text(contents, encoding="utf-8")
    return path


def run_migrations() -> None:
    database = database_from_params()
    database.migrate()
