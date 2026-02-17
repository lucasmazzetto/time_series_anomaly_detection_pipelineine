from fastapi import HTTPException, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.database.anomaly_detection_record import AnomalyDetectionRecord
from app.repositories.storage import Storage
from app.schemas.data_point import DataPoint
from app.schemas.predict_data import PredictData
from app.schemas.predict_response import PredictResponse


class PredictService:
    def __init__(self, session: Session, model: object, storage: Storage) -> None:
        """@brief Initialize prediction service with dependencies.

        @param session Active database session.
        @param model Model instance used for predictions.
        @param storage Storage backend for artifact access.
        """
        self._session = session
        self.model = model
        self.storage = storage

    @staticmethod
    def _validate_predict_inputs(series_id: str, version: int) -> None:
        """@brief Validate user inputs required by prediction.

        @param series_id Identifier of the series to predict for.
        @param version Model version identifier to use.
        @return None.
        @throws HTTPException If inputs are invalid (HTTP 400).
        """
        if not series_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="series_id must be a non-empty string.",
            )

        if version < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="version must be greater than or equal to 0.",
            )

    def _get_model_data(self, series_id: str, version: int) -> dict[str, object]:
        """@brief Retrieve model metadata for prediction.

        @description Resolves latest model metadata when `version == 0`,
        otherwise resolves the explicitly requested version.

        @param series_id Identifier of the series to predict for.
        @param version Model version identifier to use (0 means latest).
        @return Serialized model metadata dictionary.
        @throws HTTPException If model metadata is missing (HTTP 404).
        """
        try:
            if version == 0:
                return AnomalyDetectionRecord.get_last_model(self._session, series_id)

            return AnomalyDetectionRecord.get_model_version(
                self._session, series_id, version
            )
        # Convert missing-model metadata into a resource-not-found API response
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

    @staticmethod
    def _to_data_point(payload: PredictData | DataPoint) -> DataPoint:
        """@brief Convert prediction payload into validated DataPoint.

        @param payload Input payload from API schema or domain schema.
        @return DataPoint ready for model inference.
        """
        if isinstance(payload, PredictData):
            # API payload needs conversion (timestamp str -> int) into domain model
            return payload.to_data_point()

        # Already a DataPoint: Pydantic validation already happened at creation time
        return payload

    def predict(
        self, series_id: str, version: int, payload: PredictData | DataPoint
    ) -> PredictResponse:
        """@brief Predict anomaly status for a single data point.

        @description Validates payload and inputs, resolves model metadata,
        loads persisted model state, and executes inference.

        @param series_id Identifier of the series to predict for.
        @param version Model version identifier to use (0 means latest).
        @param payload Input prediction payload (raw API model or DataPoint).
        @return Prediction response containing anomaly flag and resolved version.
        @throws HTTPException HTTP 400 for invalid inputs.
        @throws HTTPException HTTP 404 when metadata or artifact is not found.
        @throws HTTPException HTTP 422 for payload validation/conversion errors.
        @throws HTTPException HTTP 500 for missing model_path or unexpected failures.
        """
        try:
            data_point = self._to_data_point(payload)
        # Preserve native Pydantic validation payloads for client-side field mapping
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=exc.errors(),
            ) from exc
        # Domain/value conversion errors are returned as a generic 422 message
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

        self._validate_predict_inputs(series_id, version)
        model_data = self._get_model_data(series_id, version)

        model_path = model_data.get("model_path")
        if model_path is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    f"Model path is missing for series_id '{series_id}' "
                    f"and version '{model_data['version']}'."
                ),
            )

        try:
            state = self.storage.load_state(model_path)
            self.model.load(state)
            prediction = bool(self.model.predict(data_point))
        # Missing artifact on disk is a client-visible not-found condition
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model artifact was not found at path '{model_path}'.",
            ) from exc
        # Re-raise expected HTTP failures from downstream operations
        except HTTPException:
            raise
        # Collapse unexpected runtime failures into a stable 500 response
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected error while predicting anomaly.",
            ) from exc

        return PredictResponse(
            anomaly=prediction,
            model_version=str(model_data["version"]),
        )
