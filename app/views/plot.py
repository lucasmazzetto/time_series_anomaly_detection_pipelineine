from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from app.schemas.predict_version import Version
from app.schemas.series_id import SeriesId
from app.services.plot import PlotService

router = APIRouter(tags=["View"])


@router.get("/plot", response_class=HTMLResponse)
def plot(series_id: Annotated[SeriesId, Query()],
         version: Annotated[Version, Query()] = Version(version="0")) -> HTMLResponse:
    """@brief Render the plot view for a series/version."""
    service = PlotService()
    html = service.render_training_data(series_id, version.to_int())
    return HTMLResponse(content=html)
