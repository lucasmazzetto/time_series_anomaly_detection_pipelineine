from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.orm import Session

from app.core.simple_model import SimpleModel
from app.db import get_session
from app.schemas.predict_data import PredictData
from app.schemas.predict_response import PredictResponse
from app.schemas.predict_version import Version
from app.schemas.series_id import SeriesId
from app.services.predict import PredictService
from app.storage.storage import Storage
from app.utils.storage import get_storage

router = APIRouter(tags=["Prediction"])


@router.post("/predict/{series_id}", response_model=PredictResponse)
def predict(series_id: Annotated[SeriesId, Path()],
            payload: PredictData, 
            version: Annotated[Version, Query()] = Version(version="0"),
            storage: Storage = Depends(get_storage),
            session: Session = Depends(get_session)) -> PredictResponse:
    """@brief Predict anomaly status for a single data point.

    @description Validates inputs, resolves model version, loads state, and predicts.

    @param series_id Identifier of the series to predict for.
    @param payload Prediction payload containing timestamp and value.
    @param version Optional model version query object; accepts values like 1, v1, or V1.
    @param storage Storage adapter resolved from the configured backend.
    @param session Active database session for future model lookup.
    @return Prediction response containing anomaly flag and model version.
    """
    service = PredictService(
        session=session,
        model=SimpleModel(),
        storage=storage,
    )

    version_int = version.to_int()
    return service.predict(series_id, version_int, payload)
