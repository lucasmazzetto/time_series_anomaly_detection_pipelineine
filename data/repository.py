from __future__ import annotations
import pickle
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict


class Storage(ABC):
    @abstractmethod
    def save_metrics(self, series_id: str, metrics: Dict[str, Any], version: int) -> Path:
        pass

    @abstractmethod
    def load_metrics(self, series_id: str) -> Dict[str, Any] | None:
        pass

    @abstractmethod
    def save_training_data(self, series_id: str, data: Any, version: int) -> Path:
        pass


class FileStorage(Storage):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe_series_id(self, series_id: str) -> str:
        series_key = str(series_id).strip()
        if not series_key:
            raise ValueError("series_id must be a non-empty string.")
        return series_key.replace("/", "_").replace("\\", "_")

    def _metrics_path(self, series_id: str, version: int) -> Path:
        series_key = self._safe_series_id(series_id)
        return self.root / f"{series_key}_v{version}_metrics.pkl"

    def _training_data_path(self, series_id: str, version: int) -> Path:
        series_key = self._safe_series_id(series_id)
        return self.root / f"{series_key}_v{version}_training_data.pkl"

    def save_metrics(self, series_id: str, metrics: Dict[str, Any], version: int) -> Path:
        path = self._metrics_path(series_id, version)
        with path.open("wb") as handle:
            pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def load_metrics(self, series_id: str) -> Dict[str, Any] | None:
        series_key = self._safe_series_id(series_id)
        candidates = list(self.root.glob(f"{series_key}_v*_metrics.pkl"))
        if candidates:
            def _version(path: Path) -> int:
                stem = path.stem
                marker = "_v"
                start = stem.find(marker)
                if start == -1:
                    return 0
                start += len(marker)
                end = stem.find("_metrics", start)
                if end == -1:
                    return 0
                try:
                    return int(stem[start:end])
                except ValueError:
                    return 0

            path = max(candidates, key=_version)
        else:
            path = self.root / f"{series_key}_metrics.pkl"
            if not path.exists():
                return None
        with path.open("rb") as handle:
            return pickle.load(handle)

    def save_training_data(self, series_id: str, data: Any, version: int) -> Path:
        path = self._training_data_path(series_id, version)
        with path.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path
