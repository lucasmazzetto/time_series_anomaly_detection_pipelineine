from typing import Any

from pydantic import BaseModel, Field


class ModelState(BaseModel):
    
    model: str = Field(..., description="Model identifier.")
    parameters: dict[str, Any] = Field(
        ..., description="Serializable model parameters."
    )
    metrics: dict[str, Any] | None = Field(
        default=None, description="Optional training metrics."
    )
