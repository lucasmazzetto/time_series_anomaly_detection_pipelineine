from fastapi import FastAPI
from api.routes import training

app = FastAPI()

app.include_router(training.router)

@app.get("/")
async def root():
    return {"message": "Time Series Anomaly Detection API"}
