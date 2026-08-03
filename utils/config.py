import os
from pathlib import Path
from typing import Any

import yaml

def load_config(config_path: str | Path) -> dict[str, Any]:
                                                                         
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")

    _apply_env_overrides(config)
    return config

def _apply_env_overrides(config: dict[str, Any], prefix: str = "FND") -> None:
                                                                              
    for key, value in os.environ.items():
        if not key.startswith(f"{prefix}_"):
            continue

        parts = key[len(prefix) + 1 :].lower().split("__")
        target = config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]

        target[parts[-1]] = _coerce_env_value(value)

def _coerce_env_value(value: str) -> str | int | float | bool:
                                                                      
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered.isdigit() or (lowered.startswith("-") and lowered[1:].isdigit()):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value

def get_project_root() -> Path:
                                               
    return Path(__file__).resolve().parent.parent
