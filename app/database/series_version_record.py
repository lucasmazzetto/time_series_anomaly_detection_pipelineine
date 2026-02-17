from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db import Base


class SeriesVersionRecord(Base):
    __tablename__ = "series_versions"

    series_id = Column(String, primary_key=True)
    last_version = Column(Integer, nullable=False)

    @staticmethod
    def next_version(session: Session, series_id: str) -> int:
        """
        @brief Atomically get the next version for a series.

        @description Uses PostgreSQL `INSERT ... ON CONFLICT ... DO UPDATE ...
        RETURNING` to avoid race conditions while incrementing per-series
        counters under concurrency.

        @param session Active SQLAlchemy session for the transaction.
        @param series_id The series identifier whose version should advance.
        @return The next version number for the series.
        """
        stmt = insert(SeriesVersionRecord).values(
            series_id=series_id,
            last_version=1,
        )
        
        stmt = stmt.on_conflict_do_update(
            index_elements=[SeriesVersionRecord.series_id],
            set_={"last_version": SeriesVersionRecord.last_version + 1},
        ).returning(SeriesVersionRecord.last_version)

        return int(session.execute(stmt).scalar_one())
