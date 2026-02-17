import plotly.express as px


class PlotService:
    def render_training_data(self, series_id: str, version: int) -> str:
        """@brief Generate a simple hardcoded training-data chart."""
        timestamps = ["1700000000", "1700000001", "1700000002", "1700000003"]
        values = [10.0, 12.5, 9.0, 11.3]

        figure = px.bar(
            x=timestamps,
            y=values,
            labels={"x": "timestamp", "y": "value"},
            title=f"Training data for {series_id} (v{version})",
        )
        figure.update_layout(template="plotly_white")

        return figure.to_html(full_html=True, include_plotlyjs="cdn")
