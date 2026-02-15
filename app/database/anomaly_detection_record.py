from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy.orm import object_session

from sqlalchemy import Column, DateTime, Integer, String

from app.db import Base
from app.database.series_version_record import SeriesVersionRecord


class AnomalyDetectionRecord(Base):
    __tablename__ = "anomaly_detection_models"

    series_id = Column(String, primary_key=True)
    version = Column(Integer, primary_key=True)
    model_path = Column(String, nullable=True)
    data_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    def touch(self) -> None:
        """
        @brief Update the in-memory `updated_at` timestamp.

        @return None
        """
        self.updated_at = datetime.now(timezone.utc)

    @staticmethod
    def _timestamp() -> datetime:
        """
        @brief Provide a UTC timestamp for record creation/update.

        @return A timezone-aware UTC datetime.
        """
        return datetime.now(timezone.utc)

    @classmethod
    def build(
        cls,
        series_id: str,
        version: int | None = None,
        model_path: str | None = None,
        data_path: str | None = None,
    ) -> "AnomalyDetectionRecord":
        """
        @brief Build a new record with consistent timestamps.

        @param series_id The series identifier for the model.
        @param version The version to persist; if None, it will be assigned on save.
        @param model_path The persisted model path, if already known.
        @param data_path The persisted data path, if already known.
        @return A new AnomalyDetectionRecord instance (not yet persisted).
        """
        now = cls._timestamp()
        return cls(
            series_id=series_id,
            version=version,
            model_path=model_path,
            data_path=data_path,
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def save(session: Session | AsyncSession, model: "AnomalyDetectionRecord") -> int:
        """
        @brief Persist a record and return its version.

        @description If the version is missing, it is assigned atomically via
        `SeriesVersionRecord` to avoid race conditions under concurrent inserts.
        
        @param session Active SQLAlchemy session for the transaction.
        @param model The record to be saved.
        @return The version stored for this record.
        """
        if model.version is None:
            model.version = SeriesVersionRecord.next_version(
                session, model.series_id
            )
        session.add(model)
        session.commit()
        session.refresh(model)
        return int(model.version)

    def update(self, *, model_path: str | None, data_path: str | None) -> None:
        """
        @brief Update storage paths and persist changes.

        @param model_path New model path to store.
        @param data_path New data path to store.
        @return None
        """
        self.model_path = model_path
        self.data_path = data_path
        self.touch()
        
        session = object_session(self)

        if session is None:
            raise RuntimeError("AnomalyDetectionRecord is not attached to a session")
        
        session.add(self)
        session.commit()
