from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.model import SimpleModel
from app.db import get_session
from app.repositories.storage import LocalStorage
from app.schemas import PredictData, PredictVersion
from app.services.anomaly_detection_service import AnomalyDetectionPredictionService
from app.schemas import PredictResponse

router = APIRouter(tags=["Prediction"])


@router.post("/predict/{series_id}", response_model=PredictResponse)
def predict(
    series_id: str,
    payload: PredictData,
    version: Annotated[PredictVersion, Query()] = PredictVersion(version="0"),
    session: Session = Depends(get_session),
) -> PredictResponse:
    """@brief Predict anomaly status for a single data point.

    @description Validates inputs, resolves model version, loads state, and predicts.

    @param series_id Identifier of the series to predict for.
    @param payload Prediction payload containing timestamp and value.
    @param version Optional model version query object; accepts values like 1, v1, or V1.
    @param session Active database session for future model lookup.
    @return Prediction response containing anomaly flag and model version.
    """
    service = AnomalyDetectionPredictionService(
        session=session,
        model=SimpleModel(),
        storage=LocalStorage(),
    )

    version_int = version.to_int()
    return service.predict(series_id, version_int, payload)
