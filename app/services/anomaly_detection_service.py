from app.core.trainer import Trainer
from app.repositories.storage import Storage

from sqlalchemy.orm import Session

from app.database.anomaly_detection_record import AnomalyDetectionRecord
from app.core.schema import DataPoint, TimeSeries


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

    def train(self, series_id: str, payload: TimeSeries) -> bool:
        """@brief Train a model and persist its record and artifacts.

        @param series_id Identifier of the series to train.
        @param payload Training data payload.
        @return True if training and persistence succeeded, otherwise False.
        """
        version = None
        model_path = None
        data_path = None
    
        try:
            state = self.trainer.train(payload)

            model_record = AnomalyDetectionRecord.build(
                series_id=series_id,
                version=version,
                model_path=model_path,
                data_path=data_path,
            )

            version = AnomalyDetectionRecord.save(self._session, model_record)

            model_path = self.storage.save_state(series_id, version, state)
            data_path = self.storage.save_data(series_id, version, payload)

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

    def predict(self, series_id: str, version: int, payload: DataPoint) -> int:
        """@brief Predict anomaly status for a single data point.

        @param series_id Identifier of the series to predict for.
        @param version Model version identifier to use.
        @param payload Input data point for prediction.
        @return Resolved model version used for prediction.
        """
        if version == 0:
            model_data = AnomalyDetectionRecord.get_last_model(
                self._session, series_id
            )
        else:
            model_data = AnomalyDetectionRecord.get_model_version(
                self._session, series_id, version
            )

        model_path = model_data.get("model_path")

        if model_path is None:
            raise ValueError(
                f"Model path is missing for series_id '{series_id}' "
                f"and version '{model_data['version']}'."
            )

        state = self.storage.load_state(model_path)
        
        self.model.load(state)

        prediction = self.model.predict(payload)

        return prediction
