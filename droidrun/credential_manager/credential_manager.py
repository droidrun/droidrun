"""
Credential Manager for secure secret storage and retrieval.
Supports both file-based (YAML) and in-memory (dict) modes.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from droidrun.config_manager.path_resolver import PathResolver

logger = logging.getLogger("droidrun")


class CredentialNotFoundError(KeyError):
    """Raised when a secret ID is not found."""

    pass


class CredentialManager:
    """
    Manages credential storage and retrieval.

    Supports two modes:
    1. File-based: Load from YAML file (credentials.yaml)
    2. In-memory: Load from dict passed programmatically

    Examples:
        # File mode
        cm = CredentialManager(credentials_path="credentials.yaml")

        # In-memory mode
        cm = CredentialManager(credentials_dict={"MY_PASSWORD": "secret123"})

        # Usage
        secret = cm.get_credential("MY_PASSWORD")
        all_ids = cm.list_available_secrets()
    """

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        credentials_dict: Optional[dict] = None,
    ):
        """
        Initialize credential manager.

        Args:
            credentials_path: Path to credentials.yaml (resolved via PathResolver)
            credentials_dict: Direct credentials dict (in-memory mode)

        Raises:
            ValueError: If both or neither arguments are provided
        """
        if credentials_path and credentials_dict:
            raise ValueError(
                "Provide either credentials_path OR credentials_dict, not both"
            )

        if credentials_dict is not None:
            # In-memory mode (direct dict)
            self.path: Optional[Path] = None
            self.mode = "in-memory"
            self.secrets = self._load_from_dict(credentials_dict)
            logger.info(
                f"✅ Loaded {len(self.secrets)} secrets from in-memory dict"
            )
        elif credentials_path:
            # File-based mode (YAML)
            self.path = PathResolver.resolve(credentials_path, must_exist=True)
            self.mode = "file"
            self.secrets = self._load_from_file()
            logger.info(f"✅ Loaded {len(self.secrets)} secrets from {self.path}")
        else:
            raise ValueError(
                "Must provide either credentials_path or credentials_dict"
            )

    def _load_from_file(self) -> Dict[str, str]:
        """
        Load credentials from YAML file.

        File format:
            secrets:
              MY_PASSWORD:
                value: "secret123"
                enabled: true
              SIMPLE_KEY: "simple_value"  # Auto-enabled

        Returns:
            Dict of enabled secrets {secret_id: secret_value}
        """
        with open(self.path, "r") as f:
            data = yaml.safe_load(f)

        if not data or "secrets" not in data:
            logger.warning(f"No 'secrets' section found in {self.path}")
            return {}

        secrets = {}
        for secret_id, secret_data in data["secrets"].items():
            # Support both dict and string format
            if isinstance(secret_data, dict):
                enabled = secret_data.get("enabled", True)
                value = secret_data.get("value", "")
            else:
                # String format - auto-enabled
                enabled = True
                value = secret_data

            if enabled and value:
                secrets[secret_id] = value
                logger.debug(f"Loaded secret: {secret_id}")
            else:
                logger.debug(
                    f"Skipped secret: {secret_id} (enabled={enabled}, has_value={bool(value)})"
                )

        return secrets

    def _load_from_dict(self, credentials_dict: dict) -> Dict[str, str]:
        """
        Load credentials from in-memory dict.

        Args:
            credentials_dict: Simple dict like {"MY_PASSWORD": "secret123", ...}

        Returns:
            Validated secrets dict
        """
        secrets = {}
        for secret_id, secret_value in credentials_dict.items():
            if isinstance(secret_value, str) and secret_value:
                secrets[secret_id] = secret_value
            else:
                logger.warning(
                    f"Skipped invalid secret: {secret_id} (type={type(secret_value)})"
                )

        return secrets

    def get_credential(self, secret_id: str) -> str:
        """
        Get secret value by ID.

        Args:
            secret_id: Secret identifier

        Returns:
            Secret value (never logged)

        Raises:
            CredentialNotFoundError: If secret ID not found
        """
        # ALWAYS log access (without value)
        logger.info(f"🔑 Accessing secret: '{secret_id}'")

        if secret_id not in self.secrets:
            available = list(self.secrets.keys())
            raise CredentialNotFoundError(
                f"Secret '{secret_id}' not found. Available: {available}"
            )

        return self.secrets[secret_id]

    def list_available_secrets(self) -> List[str]:
        """
        List available secret IDs (not values).

        Returns:
            List of secret IDs
        """
        return list(self.secrets.keys())

    def has_credential(self, secret_id: str) -> bool:
        """
        Check if secret ID exists.

        Args:
            secret_id: Secret identifier

        Returns:
            True if secret exists, False otherwise
        """
        return secret_id in self.secrets

    def __repr__(self) -> str:
        """String representation."""
        count = len(self.secrets)
        if self.mode == "file":
            return f"<CredentialManager mode=file path={self.path} secrets={count}>"
        else:
            return f"<CredentialManager mode=in-memory secrets={count}>"
