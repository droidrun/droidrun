from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping

PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_\.\-]+)\}\}")


logger = logging.getLogger("droidrun")


class CredentialManager:
    """Load and manage secrets, resolve placeholders, and redact outputs.

    Supports:
    - Environment variables (DROIDRUN_CREDENTIALS_*, DROIDRUN_CREDENTIALS_JSON)
    - JSON file at ~/.droidrun/credentials.json or a provided path
    - Profiles via top-level object keys, selected by name
    - Deep placeholder resolution in arbitrary Python data structures
    - Redaction of known secret values from text/logs
    """

    def __init__(
        self,
        *,
        credentials: Mapping[str, str] | None = None,
        file_path: str | os.PathLike[str] | None = None,
        profile: str | None = None,
    ) -> None:
        self._raw: Dict[str, Any] = {}
        self._secrets: Dict[str, str] = {}

        # Load from file first (lowest precedence)
        file_data = self._load_from_file(file_path)
        if isinstance(file_data, dict):
            self._raw = file_data

        # Load from env JSON (overrides file)
        env_json_data = self._load_from_env_json()
        if isinstance(env_json_data, dict):
            # merge
            self._raw.update(env_json_data)

        # Choose profile view
        if self._looks_profiled(self._raw):
            active = profile or os.getenv("DROIDRUN_CREDENTIALS_PROFILE", "default")
            selected = self._raw.get(active) or {}
            logger.debug(
                "CredentialManager: using profile '%s' (keys: %s)",
                active,
                list(selected.keys()),
            )
        else:
            selected = self._raw
            logger.debug("CredentialManager: using flat secret map (no profiles detected)")

        # Overlay flat env vars (highest precedence)
        flat_env = self._load_from_env_flat()
        selected = {**selected, **flat_env}

        # Overlay explicit credentials argument if given
        if credentials:
            selected.update(dict(credentials))

        # Normalize keys to UPPER_SNAKE_CASE
        normalize_keys: Dict[str, str] = {}
        for k, v in selected.items():
            if v is None:
                continue
            if not isinstance(v, (str, int, float, bool)):
                # Only scalar secrets are supported; JSON can store nested
                # structures but we only keep scalars here.
                continue
            key = str(k).upper()
            normalize_keys[key] = str(v)

        self._secrets = normalize_keys
        logger.debug(
            "CredentialManager: initialized secrets (keys only) -> %s",
            sorted(self._secrets.keys()),
        )
        # Precompute redaction pairs sorted by descending length to avoid partial overlaps
        self._redaction_pairs = sorted(
            ((value, f"{{{{{key}}}}}") for key, value in self._secrets.items()),
            key=lambda p: len(p[0]),
            reverse=True,
        )

    # ---------------------------- Loading helpers ----------------------------
    def _default_file_path(self) -> Path:
        return Path.home() / ".droidrun" / "credentials.json"

    def _load_from_file(self, file_path: str | os.PathLike[str] | None) -> Dict[str, Any]:
        path = Path(file_path) if file_path else self._default_file_path()
        if not path.exists():
            logger.debug(f"CredentialManager: credentials file not found at {path}")
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                logger.debug(
                    "CredentialManager: loaded credentials file %s (top-level keys: %s)",
                    path,
                    list(data.keys()),
                )
                return data
        except Exception as e:
            logger.debug(f"CredentialManager: failed to load credentials file {path}: {e}")
            return {}

    def _load_from_env_json(self) -> Dict[str, Any]:
        raw = os.getenv("DROIDRUN_CREDENTIALS_JSON")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            logger.debug(
                "CredentialManager: loaded secrets from DROIDRUN_CREDENTIALS_JSON "
                "(top-level keys: %s)",
                list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )
            return data
        except Exception as e:
            logger.debug("CredentialManager: invalid DROIDRUN_CREDENTIALS_JSON: %s", e)
            return {}

    def _load_from_env_flat(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        prefix = "DROIDRUN_CREDENTIALS_"
        for k, v in os.environ.items():
            if not k.startswith(prefix):
                continue
            key = k[len(prefix) :].upper()
            if key in ("JSON", "FILE", "PROFILE"):
                continue
            out[key] = v
        if out:
            logger.debug(
                "CredentialManager: loaded %d flat env secrets (keys only)", len(out)
            )
        return out

    def _looks_profiled(self, data: Dict[str, Any]) -> bool:
        # Heuristic: if any value is a dict of scalars, treat as profiles
        return any(isinstance(v, dict) for v in data.values())

    # ------------------------------ Public API --------------------------------
    def get(self, name: str, default: str | None = None) -> str | None:
        return self._secrets.get(name.upper(), default)

    def known_placeholders(self) -> Dict[str, str]:
        return {f"{{{{{k}}}}}": v for k, v in self._secrets.items()}

    def resolve_placeholders(self, obj: Any) -> Any:
        """Deep-resolve placeholders in strings, lists, and dicts."""
        if isinstance(obj, str):
            return self._resolve_in_string(obj)
        if isinstance(obj, list):
            return [self.resolve_placeholders(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(self.resolve_placeholders(x) for x in obj)
        if isinstance(obj, dict):
            return {k: self.resolve_placeholders(v) for k, v in obj.items()}
        return obj

    def _resolve_in_string(self, s: str) -> str:
        def _replace(m: re.Match[str]) -> str:
            key = m.group(1).upper()
            return self._secrets.get(key, m.group(0))

        return PLACEHOLDER_RE.sub(_replace, s)

    def redact(self, text: str) -> str:
        """Replace known secret values with their placeholders."""
        if not text:
            return text
        redacted = text
        for value, placeholder in self._redaction_pairs:
            if not value:
                continue
            redacted = redacted.replace(value, placeholder)
        return redacted

    def redact_obj(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, list):
            return [self.redact_obj(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(self.redact_obj(x) for x in obj)
        if isinstance(obj, dict):
            return {k: self.redact_obj(v) for k, v in obj.items()}
        return obj
