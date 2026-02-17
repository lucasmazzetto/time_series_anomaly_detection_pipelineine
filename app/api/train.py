from typing import Annotated

from fastapi import APIRouter, Depends, Path
from sqlalchemy.orm import Session

from app.db import get_session
from app.schemas.series_id import SeriesId
from app.schemas.train_data import TrainData
from app.schemas.train_response import TrainResponse
from app.services.train import TrainService

from app.core.trainer import AnomalyDetectionTrainer
from app.core.simple_model import SimpleModel
from app.repositories.local_storage import LocalStorage

router = APIRouter(tags=["train"])

@router.post("/fit/{series_id}", response_model=TrainResponse)
def train(
    series_id: Annotated[SeriesId, Path()],
    payload: TrainData,
    session: Session = Depends(get_session),
) -> TrainResponse:
    """@brief Start training for a series and persist its artifacts.

    @param series_id Identifier of the series to train.
    @param payload Training payload containing timestamps and values.
    @param session Active database session for persistence.
    @return Training response payload describing the outcome.
    """
    service = TrainService(
        session=session, 
        trainer=AnomalyDetectionTrainer(model=SimpleModel()), 
        storage=LocalStorage()
    )
    
    return service.train(series_id, payload)
