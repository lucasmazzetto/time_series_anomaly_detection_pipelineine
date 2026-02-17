from math import isfinite

from pydantic import BaseModel, Field, field_validator

from app.schemas.data_point import DataPoint


class PredictData(BaseModel):
    
    timestamp: str = Field(
        ...,
        description="Timestamp value in unix timestamp string format",
    )
    value: float = Field(...)

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, timestamp: str) -> str:
        """@brief Validate timestamp input for type and value constraints.

        @param timestamp Unix timestamp string to validate.
        @return Validated timestamp string.
        """
        if isinstance(timestamp, bool) or not isinstance(timestamp, str):
            raise ValueError("Timestamp must be provided as a string.")
        if not timestamp.strip():
            raise ValueError("Timestamp must be a non-empty string.")
        if not timestamp.isdigit():
            raise ValueError("Timestamp must contain only digits.")
        return timestamp

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float) -> float:
        """@brief Validate value input for type and finite constraints.

        @param value Numeric value to validate.
        @return Validated numeric value.
        """
        if value is None:
            raise ValueError("Value cannot be None, NaN, or infinite.")
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError("Value must be a float or int.")
        if not isfinite(value):
            raise ValueError("Value cannot be None, NaN, or infinite.")
        return float(value)

    def to_data_point(self) -> DataPoint:
        """@brief Convert input payload into a DataPoint model.

        @return DataPoint instance with parsed timestamp and value.
        """
        return DataPoint(timestamp=int(self.timestamp), value=float(self.value))
