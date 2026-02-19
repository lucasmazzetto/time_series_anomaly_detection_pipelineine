from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse

from app.schemas.predict_version import Version
from app.schemas.series_id import SeriesId
from app.services.plot import PlotService
from app.storage.storage import Storage
from app.utils.storage import get_storage

router = APIRouter(tags=["View"])


@router.get("/plot", response_class=HTMLResponse)
def plot(series_id: Annotated[SeriesId, Query()],
         version: Annotated[Version, Query()] = Version(version="0"),
         storage: Storage = Depends(get_storage)) -> HTMLResponse:
    """@brief Render the plot view for a series/version.

    @param series_id Identifier of the series to render.
    @param version Optional model version query object (`0` resolves latest).
    @param storage Storage adapter resolved from the configured backend.
    @return HTMLResponse containing the rendered Plotly page.
    """
    service = PlotService(storage=storage)
    html = service.render_training_data(series_id, version.to_int())
    return HTMLResponse(content=html)
