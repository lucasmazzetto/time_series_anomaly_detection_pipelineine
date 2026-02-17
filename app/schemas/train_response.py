from pydantic import BaseModel


class TrainResponse(BaseModel):
    
    series_id: str
    message: str
    success: bool
