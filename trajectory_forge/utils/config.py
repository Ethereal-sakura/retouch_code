"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load a YAML config file, returning an empty dict if the file does not exist."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}
