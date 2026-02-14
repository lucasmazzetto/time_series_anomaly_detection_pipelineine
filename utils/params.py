from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache
import yaml


@lru_cache(maxsize=1)
def load_params() -> Dict[str, Any]:
    """@brief Load parameters from the repository configuration.

    @returns A dictionary of values parsed from `config/params.yaml`.
    @raises FileNotFoundError if `config/params.yaml` is missing.
    @raises ValueError if the YAML document is invalid.
    """
    params_path = Path("config/params.yaml").resolve()

    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")

    with params_path.open("r", encoding="utf-8") as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Parameters file is not a valid YAML document: {params_path}") from exc

    return params
