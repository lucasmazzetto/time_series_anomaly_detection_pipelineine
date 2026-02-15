from app.core.trainer import Trainer
from app.repositories.storage import Storage

from sqlalchemy.orm import Session

from app.database.anomaly_detection_record import AnomalyDetectionRecord
from app.core.schema import TimeSeries


class AnomalyDetectionTrainingService:
    def __init__(self, session: Session, trainer: Trainer, storage: Storage) -> None:
        self._session = session
        self.trainer = trainer
        self.storage = storage

    def train(self, series_id: str, payload: TimeSeries) -> bool:
        version = None
        model_path = None
        data_path = None
    
        try:
            state = self.trainer.train(payload)

            model = AnomalyDetectionRecord.build(
                series_id=series_id,
                version=version,
                model_path=model_path,
                data_path=data_path,
            )

            version = AnomalyDetectionRecord.save(self._session, model)

            model_path = self.storage.save_state(series_id, version, state)
            data_path = self.storage.save_data(series_id, version, payload)

            model.update(model_path=model_path, data_path=data_path)

            return True
        
        except Exception:
            
            self._session.rollback()

            return False
