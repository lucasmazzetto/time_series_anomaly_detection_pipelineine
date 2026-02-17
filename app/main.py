from fastapi import FastAPI

from app.api.healthcheck import router as healthcheck_router
from app.api.predict import router as predict_router
from app.api.train import router as train_router
from app.middleware.latency import track_request_latency

app = FastAPI(title="Anomaly Detection API")
app.middleware("http")(track_request_latency)
app.include_router(train_router)
app.include_router(predict_router)
app.include_router(healthcheck_router)
