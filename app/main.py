from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from app.api.healthcheck import router as healthcheck_router
from app.api.predict import router as predict_router
from app.api.train import router as train_router
from app.middleware.latency import track_request_latency
from app.views.plot import router as plot_router

app = FastAPI(title="Time Series Anomaly Detection API")
app.middleware("http")(track_request_latency)
app.include_router(train_router)
app.include_router(predict_router)
app.include_router(healthcheck_router)
app.include_router(plot_router)


@app.exception_handler(SQLAlchemyTimeoutError)
async def handle_db_pool_timeout(
    _request: Request, _exc: SQLAlchemyTimeoutError
) -> JSONResponse:
    """@brief Return a graceful 503 when database pool capacity is exhausted.

    @param _request Incoming request associated with the failure.
    @param _exc SQLAlchemy timeout raised while acquiring a DB connection.
    @return JSONResponse with HTTP 503 and retry-friendly error detail.
    """
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "Database connection pool exhausted. Please retry shortly."
        },
    )
