from fastapi import FastAPI

from app.api.train import router as train_router

app = FastAPI(title="Anomaly Detection API")
app.include_router(train_router)
