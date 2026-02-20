from fastapi import HTTPException, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.trainer import Trainer
from app.database.anomaly_detection import AnomalyDetectionRecord
from app.storage.storage import Storage
from app.schemas.time_series import TimeSeries
from app.schemas.train_data import TrainData
from app.schemas.train_response import TrainResponse
from app.utils.error import validation_error_details, value_error_details


class TrainService:
    def __init__(self, session: Session, trainer: Trainer, storage: Storage) -> None:
        """@brief Initialize training service with persistence dependencies.

        @param session Active database session.
        @param trainer Trainer implementation for model fitting.
        @param storage Storage backend for artifacts.
        """
        self._session = session
        self.trainer = trainer
        self.storage = storage

    @staticmethod
    def _to_time_series(payload: TrainData | TimeSeries) -> TimeSeries:
        """@brief Convert incoming training payload to validated TimeSeries.

        @description Converts API payloads into domain models and always
        enforces training preflight validation rules.

        @param payload Training payload from API schema or domain schema.
        @return Validated TimeSeries ready for model training.
        """
        if isinstance(payload, TrainData):
            # API payload still needs conversion from parallel arrays to domain model
            return payload.to_time_series()

        # We are always enforcing training preflight rules here
        return payload.validate_for_training()

    def train(self, series_id: str, payload: TrainData | TimeSeries) -> TrainResponse:
        """@brief Train a model and persist its record and artifacts.

        @description Trains the model, persists model metadata in the database,
        writes model/data artifacts to storage, and commits the record paths.

        @param series_id Identifier of the series to train.
        @param payload Training data payload (raw API model or TimeSeries).
        @return Training response payload describing the outcome.
        @throws HTTPException HTTP 422 for payload validation/preflight errors.
        @throws HTTPException HTTP 500 for unexpected runtime failures.
        """
        version = None
        model_path = None
        data_path = None

        try:
            time_series = self._to_time_series(payload)
            state = self.trainer.train(time_series)

            model_record = AnomalyDetectionRecord.build(
                series_id=series_id,
                version=version,
                model_path=model_path,
                data_path=data_path,
            )

            version = AnomalyDetectionRecord.save(self._session, model_record)

            model_path = self.storage.save_state(series_id, version, state)
            data_path = self.storage.save_data(series_id, version, time_series)

            model_record.update(model_path=model_path, data_path=data_path)
            model_record.commit()

            return TrainResponse(
                series_id=series_id,
                version=str(version),
                points_used=len(time_series.data),
            )

        # Preserve native Pydantic validation payloads for client-side field mapping
        except ValidationError as exc:
            self._session.rollback()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=validation_error_details(exc),
            ) from exc
        # Domain/value preflight errors are returned as a generic 422 message
        except ValueError as exc:
            self._session.rollback()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=value_error_details(exc),
            ) from exc
        # Re-raise expected HTTP failures after transaction rollback
        except HTTPException:
            self._session.rollback()
            raise
        # Collapse unexpected runtime failures into a stable 500 response
        except Exception as exc:
            self._session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected error while training model.",
            ) from exc
