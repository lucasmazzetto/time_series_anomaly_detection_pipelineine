from pydantic import BaseModel


class PredictResponse(BaseModel):
    """@brief Structured response outputed after a prediction attempt.

    @var anomaly: Flag indicating whether the point is anomalous.
    @var model_version: Version identifier of the model used for prediction.
    """

    anomaly: bool
    model_version: str
