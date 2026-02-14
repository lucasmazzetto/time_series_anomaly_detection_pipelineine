from fastapi import APIRouter
from schemas.training import TrainData, TrainResponse

router = APIRouter()

@router.post(
    "/fit/{series_id}",
    response_model=TrainResponse,
    tags=["Training"],
    operation_id="training_fit"
)
async def fit(series_id: str, data: TrainData):
    """@brief Initiates training for the specified time series.

    @param series_id Identifier of the time series to train.
    @param data Payload describing training data and settings.
    @return TrainResponse indicating that training started.
    """

    # TODO: send the received training request to the training manager.

    return TrainResponse(
        series_id=series_id, 
        message="Model training started", 
        success=True
    )
