from pydantic import BaseModel


class PredictResponse(BaseModel):
    anomaly: bool
    model_version: str
