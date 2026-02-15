from fastapi import FastAPI
from api import train

app = FastAPI()

app.include_router(train.router)

@app.get("/")
async def root():
    return {"message": "Time Series Anomaly Detection API"}
