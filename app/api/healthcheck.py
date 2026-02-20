from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_session
from app.schemas.health_check_response import HealthCheckResponse
from app.services.healthcheck import HealthCheckService

router = APIRouter(tags=["Health Check"])


@router.get("/healthcheck", response_model=HealthCheckResponse)
def healthcheck(session: Session = Depends(get_session)) -> HealthCheckResponse:
    """@brief Return API health metrics for training and inference.

    @param session Active SQLAlchemy session used to query trained series count.
    @return HealthCheckResponse with series count and latency metrics.
    """
    service = HealthCheckService(session=session)
    return service.healthcheck()
