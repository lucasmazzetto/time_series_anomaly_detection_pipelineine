from app.core.trainer import AnomalyDetectionTrainer
from app.core.model import SimpleModel
from app.repositories.local_storage import LocalStorage

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.database.anomaly_detection_record import AnomalyDetectionRecord
from app.core.schema import TimeSeries


class BaseAnomalyDetectionService:
    def __init__(self, session: Session | AsyncSession) -> None:
        self._session = session
    

class AnomalyDetectionService(BaseAnomalyDetectionService):
    def __init__(self, session: Session) -> None:
        super().__init__(session)
        self.trainer = AnomalyDetectionTrainer(model=SimpleModel)
        self.storage = LocalStorage()

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
