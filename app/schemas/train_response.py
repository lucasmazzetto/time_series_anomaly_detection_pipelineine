from pydantic import BaseModel


class TrainResponse(BaseModel):
    series_id: str
    version: str
    points_used: int
