import plotly.express as px
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from app.db import SessionLocal
from app.database.anomaly_detection import AnomalyDetectionRecord
from app.repositories.local_storage import LocalStorage
from app.repositories.storage import Storage
from app.schemas.time_series import TimeSeries


class PlotService:
    def __init__(
        self, session: Session | None = None, storage: Storage | None = None
    ) -> None:
        """@brief Initialize plotting service dependencies.

        @param session Active SQLAlchemy session. When omitted, a local
        session is created and automatically closed after rendering.
        @param storage Storage backend used to load persisted training data.
        """
        self._owns_session = session is None
        self._session = session or SessionLocal()
        self.storage = storage or LocalStorage()

    @staticmethod
    def _validate_plot_inputs(series_id: str, version: int) -> None:
        """@brief Validate input arguments required by plotting.

        @param series_id Identifier of the series to render.
        @param version Requested model/training-data version (0 means latest).
        @return None.
        @throws HTTPException HTTP 400 when inputs are invalid.
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

    def _get_training_data(self, series_id: str, version: int) -> dict[str, object]:
        """@brief Resolve training-data metadata for latest or explicit version.

        @param series_id Identifier of the series to render.
        @param version Requested version (0 resolves latest).
        @return Serialized metadata row from `anomaly_detection_models`.
        @throws HTTPException HTTP 404 when metadata is not found.
        """
        try:
            if version == 0:
                return AnomalyDetectionRecord.get_last_training_data(
                    self._session, series_id
                )

            return AnomalyDetectionRecord.get_training_data(
                self._session, series_id, version
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

    def _load_training_data(self, data_path: str) -> TimeSeries:
        """@brief Load persisted training data from storage.

        @param data_path Filesystem path stored in model metadata.
        @return Validated `TimeSeries` payload read from disk.
        @throws HTTPException HTTP 404 when artifact file is missing.
        @throws HTTPException HTTP 500 for unexpected storage errors.
        """
        try:
            return self.storage.load_data(data_path)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training data artifact was not found at path '{data_path}'.",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected error while loading training data.",
            ) from exc

    @staticmethod
    def render_series(series_id: str, version: int, payload: TimeSeries) -> str:
        """@brief Render a Plotly bar chart from a `TimeSeries` payload.

        @param series_id Identifier used in chart title.
        @param version Resolved version used in chart title.
        @param payload Training data points to be visualized.
        @return Full HTML document containing the rendered chart.
        """
        timestamps = [point.timestamp for point in payload.data]
        
        date_time = [
            datetime.fromtimestamp(point.timestamp, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )
            for point in payload.data
        ]

        values = [point.value for point in payload.data]

        figure = px.bar(
            x=date_time,
            y=values,
            labels={"x": "Date and Time (UTC)", "y": "Value"},
            title=f"Training data for {series_id} (v{version})",
        )

        figure.update_traces(
            customdata=timestamps,
            hovertemplate=(
                "Date and Time (UTC): %{x}<br>"
                "Timestamp: %{customdata}<extra></extra><br>"
                "Value: %{y}<br>"
            ),
        )
        figure.update_layout(template="plotly_white")

        return figure.to_html(full_html=True, include_plotlyjs="cdn")

    def render_training_data(self, series_id: str, version: int) -> str:
        """@brief Orchestrate metadata lookup, data loading, and HTML rendering.

        @param series_id Identifier of the series to render.
        @param version Requested version (0 resolves latest).
        @return Full HTML document containing the rendered chart.
        @throws HTTPException HTTP 400 for invalid input values.
        @throws HTTPException HTTP 404 when metadata/artifact is missing.
        @throws HTTPException HTTP 500 when `data_path` is absent in metadata.
        """
        try:
            self._validate_plot_inputs(series_id, version)
            training_data = self._get_training_data(series_id, version)

            data_path = training_data.get("data_path")
            
            if data_path is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(
                        f"Training data path is missing for series_id '{series_id}' "
                        f"and version '{training_data['version']}'."
                    ),
                )

            payload = self._load_training_data(data_path)
            return self.render_series(series_id, int(training_data["version"]), payload)
        finally:
            if self._owns_session:
                self._session.close()
