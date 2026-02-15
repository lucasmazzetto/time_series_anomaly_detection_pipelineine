from pathlib import Path
from typing import Any, Dict
from functools import lru_cache
import yaml


@lru_cache(maxsize=1)
def load_params() -> Dict[str, Any]:
    """@brief Load parameters from the repository configuration.

    @returns A dictionary of values parsed from `config/params.yaml`.
    @raises FileNotFoundError if `config/params.yaml` is missing.
    @raises ValueError if the YAML document is invalid.
    """
    package_root = Path(__file__).resolve().parents[1]
    candidates = (
        Path("config/params.yaml").resolve(),
        package_root / "config/params.yaml",
    )
    params_path = next((path for path in candidates if path.exists()), None)

    if params_path is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Parameters file not found. Searched: {searched}")

    with params_path.open("r", encoding="utf-8") as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Parameters file is not a valid YAML document: {params_path}") from exc

    return params
