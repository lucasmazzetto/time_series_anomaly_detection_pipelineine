from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class AnomalyDetectionModel(Base):
    __tablename__ = "anomaly_detection_models"

    series_id = Column(String, primary_key=True)
    version = Column(Integer, primary_key=True)
    model_path = Column(String, nullable=True)
    data_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
