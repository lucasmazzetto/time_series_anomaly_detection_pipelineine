import re

from pydantic import BaseModel, Field, field_validator


class PredictVersion(BaseModel):
    
    version: str = Field(
        default="0",
        description="Model version in formats like 1, v1, or V1.",
    )

    @field_validator("version")
    @classmethod
    def sanitize_version(cls, version: str) -> str:
        """@brief Sanitize version input keeping only numeric characters.

        @param version Raw version value received from query string.
        @return Sanitized numeric version string.
        """
        if isinstance(version, bool):
            raise ValueError("Version must be a string or integer-like value.")

        # Keep only ASCII digits to neutralize querystring noise/injection payloads.
        value = str(version).strip()
        digits_only = re.sub(r"[^0-9]", "", value)
        if not digits_only:
            raise ValueError("Version must contain at least one digit.")

        return digits_only

    def to_int(self) -> int:
        """@brief Convert sanitized version string into integer.

        @return Integer model version.
        """
        return int(self.version)
