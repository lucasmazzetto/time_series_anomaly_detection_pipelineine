from __future__ import annotations

import argparse

from fastapi import FastAPI
from api import train

from data.migration import create_migration, run_migrations

app = FastAPI()

app.include_router(train.router)

@app.get("/")
async def root():
    return {"message": "Time Series Anomaly Detection API"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time Series Anomaly Detection utilities.")
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Run SQL migrations and create the database if needed.",
    )
    parser.add_argument(
        "--makemigrations",
        nargs="?",
        const="migration",
        help="Create a new SQL migration file (optionally named).",
    )
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    if args.makemigrations:
        path = create_migration(args.makemigrations)
        print(f"Created migration: {path}")
        return 0
    if args.migrate:
        run_migrations()
        print("Migrations applied.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
