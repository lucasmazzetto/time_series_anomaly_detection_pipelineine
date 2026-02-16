from fastapi import HTTPException, status
from app.core.trainer import Trainer
from app.repositories.storage import Storage

from sqlalchemy.orm import Session

from app.database.anomaly_detection_record import AnomalyDetectionRecord
from app.schemas import DataPoint, PredictData, TimeSeries, TrainData
from app.schemas import PredictResponse


class AnomalyDetectionTrainingService:
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

        @param payload Training payload from API schema or domain schema.
        @return Validated TimeSeries ready for model training.
        @throws HTTPException If payload fails preflight validation.
        """
        try:
            if isinstance(payload, TrainData):
                return payload.to_time_series()
            return payload.validate_for_training()
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=[{"loc": ["body"], "msg": str(exc), "type": "value_error"}],
            ) from exc

    def train(self, series_id: str, payload: TrainData | TimeSeries) -> bool:
        """@brief Train a model and persist its record and artifacts.

        @param series_id Identifier of the series to train.
        @param payload Training data payload (raw API model or TimeSeries).
        @return True if training and persistence succeeded, otherwise False.
        """
        time_series = self._to_time_series(payload)

        version = None
        model_path = None
        data_path = None
    
        try:
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

            return True
        
        except Exception:
            
            self._session.rollback()

            return False


class AnomalyDetectionPredictionService:
    def __init__(self, session: Session, model: object, storage: Storage) -> None:
        """@brief Initialize prediction service with dependencies.

        @param session Active database session.
        @param model Model instance used for predictions.
        @param storage Storage backend for artifact access.
        """
        self._session = session
        self.model = model
        self.storage = storage

    @staticmethod
    def _validate_predict_inputs(series_id: str, version: int) -> None:
        """@brief Validate user inputs required by prediction.

        @param series_id Identifier of the series to predict for.
        @param version Model version identifier to use.
        @return None.
        """
        if not series_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="series_id must be a non-empty string.",
            )

        if version < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="version must be greater than or equal to 0.",
            )

    def _get_model_data(self, series_id: str, version: int) -> dict[str, object]:
        """@brief Retrieve model metadata for prediction.

        @param series_id Identifier of the series to predict for.
        @param version Model version identifier to use (0 means latest).
        @return Serialized model metadata dictionary.
        """
        try:
            # If `version == 0` means "use latest model version"
            if version == 0:
                return AnomalyDetectionRecord.get_last_model(
                    self._session, series_id
                )

            return AnomalyDetectionRecord.get_model_version(
                self._session, series_id, version
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

    @staticmethod
    def _to_data_point(payload: PredictData | DataPoint) -> DataPoint:
        """@brief Convert prediction payload into validated DataPoint."""
        try:
            if isinstance(payload, PredictData):
                return payload.to_data_point()
            return payload
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=[{"loc": ["body"], "msg": str(exc), "type": "value_error"}],
            ) from exc

    def predict(
        self, series_id: str, version: int, payload: PredictData | DataPoint
    ) -> PredictResponse:
        """@brief Predict anomaly status for a single data point.

        @param series_id Identifier of the series to predict for.
        @param version Model version identifier to use.
        @param payload Input prediction payload (raw API model or DataPoint).
        @return Prediction response containing anomaly flag and resolved version.
        """
        data_point = self._to_data_point(payload)
        self._validate_predict_inputs(series_id, version)
        model_data = self._get_model_data(series_id, version)

        model_path = model_data.get("model_path")

        if model_path is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    f"Model path is missing for series_id '{series_id}' "
                    f"and version '{model_data['version']}'."
                ),
            )

        try:
            state = self.storage.load_state(model_path)
            self.model.load(state)
            prediction = bool(self.model.predict(data_point))
        except FileNotFoundError as exc:
            # Artifact path exists in DB but the file is missing on disk
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model artifact was not found at path '{model_path}'.",
            ) from exc
        except HTTPException:
            # Preserve explicit HTTP errors raised by lower layers
            raise
        except Exception as exc:
            # Map any unexpected runtime failure to a generic server error
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected error while predicting anomaly.",
            ) from exc

        return PredictResponse(
            anomaly=prediction,
            model_version=str(model_data["version"]),
        )
