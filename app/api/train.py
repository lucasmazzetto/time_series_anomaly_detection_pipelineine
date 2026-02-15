from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.schema import TrainData, TrainResponse
from app.db import get_session
from app.services.anomaly_detection_service import AnomalyDetectionTrainingService

from app.core.trainer import AnomalyDetectionTrainer
from app.core.model import SimpleModel
from app.repositories.storage import LocalStorage

router = APIRouter(tags=["train"])

@router.post("/fit/{series_id}", response_model=TrainResponse)
def train(
    series_id: str, payload: TrainData, session: Session = Depends(get_session)
) -> TrainResponse:
    time_series = payload.to_time_series()

    service = AnomalyDetectionTrainingService(
        session=session, 
        trainer=AnomalyDetectionTrainer(model=SimpleModel()), 
        storage=LocalStorage()
    )
    
    success = service.train(series_id, time_series)

    message = "Training failed."

    if success:
        message = "Training successfully started."

    return TrainResponse(
        series_id=series_id, 
        message=message, 
        success=success
    )
