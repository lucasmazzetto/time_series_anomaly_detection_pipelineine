from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from sqlalchemy import Column, DateTime, Integer, String

from app.db import Base


class AnomalyDetectionRecord(Base):
    __tablename__ = "anomaly_detection_models"

    series_id = Column(String, primary_key=True)
    version = Column(Integer, primary_key=True)
    model_path = Column(String, nullable=True)
    data_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    @staticmethod
    def _latest_timestamp() -> datetime:
        return datetime.now(timezone.utc)

    @classmethod
    def build(
        cls,
        series_id: str,
        version: int,
        model_path: str,
        data_path: str,
    ) -> "AnomalyDetectionRecord":
        now = cls._latest_timestamp()
        return cls(
            series_id=series_id,
            version=version,
            model_path=model_path,
            data_path=data_path,
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def next_version_from(latest: int | None) -> int:
        if latest is None:
            return 1
        return int(latest) + 1

    @classmethod
    def get_latest_version(
        cls, session: Session | AsyncSession, series_id: str
    ) -> int | None:
        stmt = select(func.max(cls.version)).where(cls.series_id == series_id)
        return session.execute(stmt).scalar()

    @staticmethod
    def save(session: Session | AsyncSession, model: "AnomalyDetectionRecord") -> None:
        session.add(model)
        session.commit()
