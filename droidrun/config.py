"""Project-wide configuration loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for earlier interpreters
    import tomli as tomllib  # type: ignore

logger = logging.getLogger("droidrun.config")

_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_PATH: Optional[Path] = None


def _config_search_paths() -> list[Path]:
    cwd = Path.cwd()
    home = Path.home()
    return [
        cwd / "droidrun.toml",
        cwd / "droidrun_config.toml",
        home / ".droidrun" / "config.toml",
        home / ".droidrun.toml",
    ]


def find_config_file() -> Optional[Path]:
    """Return the first config file found in the supported search locations"""

    for path in _config_search_paths():
        if path.is_file():
            return path
    return None


def load_config(force_reload: bool = False) -> Dict[str, Any]:
    """Load the global configuration as a dictionary"""

    global _CONFIG_CACHE, _CONFIG_PATH

    if not force_reload and _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = find_config_file()
    _CONFIG_PATH = path

    if not path:
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE

    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
            if not isinstance(data, dict):
                logger.warning("Config file %s did not contain a TOML table.", path)
                data = {}
            _CONFIG_CACHE = data
    except Exception as exc:  # pragma: no cover - log and continue with defaults
        logger.warning("Failed to read config file %s: %s", path, exc)
        _CONFIG_CACHE = {}

    return _CONFIG_CACHE


def get_config_path() -> Optional[Path]:
    """Path used for configuration, if any"""

    if _CONFIG_PATH is None:
        load_config()
    return _CONFIG_PATH


def get_memory_config() -> Dict[str, Any]:
    """Return the memory configuration section, defaulting to an empty mapping"""

    config = load_config()
    memory_section = config.get("memory", {})
    if isinstance(memory_section, dict):
        return memory_section
    logger.warning("Config file section 'memory' must be a table; ignoring invalid value")
    return {}


__all__ = [
    "find_config_file",
    "get_config_path",
    "get_memory_config",
    "load_config",
]
