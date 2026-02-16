from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.schema import PredictData, PredictResponse, PredictVersion
from app.core.model import SimpleModel
from app.db import get_session
from app.repositories.storage import LocalStorage
from app.services.anomaly_detection_service import AnomalyDetectionPredictionService

router = APIRouter(tags=["Prediction"])


@router.post("/predict/{series_id}", response_model=PredictResponse)
def predict(
    series_id: str,
    payload: PredictData,
    version: str = Query(default="0"),
    session: Session = Depends(get_session),
) -> PredictResponse:
    """@brief Predict anomaly status for a single data point.

    @description Prediction is currently a placeholder pending model loading.

    @param series_id Identifier of the series to predict for.
    @param payload Prediction payload containing timestamp and value.
    @param version Optional model version; accepts values like 1, v1, or V1.
    @param session Active database session for future model lookup.
    @return Prediction response containing anomaly flag and model version.
    """
    service = AnomalyDetectionPredictionService(
        session=session,
        model=SimpleModel(),
        storage=LocalStorage(),
    )

    sanitized_version = PredictVersion(version=version)
    version_int = sanitized_version.to_int()

    data_point = payload.to_data_point()
    anomaly = service.predict(series_id, version_int, data_point)

    return PredictResponse(anomaly=anomaly, model_version=version_int)
