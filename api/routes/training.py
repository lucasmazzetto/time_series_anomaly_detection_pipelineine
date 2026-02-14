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
    return TrainResponse(
        series_id=series_id, 
        message="Model training started", 
        success=True
    )
