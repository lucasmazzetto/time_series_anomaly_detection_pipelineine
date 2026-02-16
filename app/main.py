from fastapi import FastAPI

from app.api.predict import router as predict_router
from app.api.train import router as train_router

app = FastAPI(title="Anomaly Detection API")
app.include_router(train_router)
app.include_router(predict_router)
