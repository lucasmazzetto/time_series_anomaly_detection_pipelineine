from pydantic import BaseModel


class TrainResponse(BaseModel):
    """@brief Structured response outputed after a training attempt.

    @var series_id: Identifier of the series that was targeted.
    @var message: Summary of the training outcome.
    @var success: Flag telling if training succeeded.
    """

    series_id: str
    message: str
    success: bool
