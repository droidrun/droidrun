"""
Credential Manager - Secure credential management for DroidRun agents.
"""

import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger("droidrun-credential-manager")


class CredentialManager:
    """
    A secure credential manager that loads credentials from environment variables
    and .env files without exposing them in prompts or logs.
    """

    def __init__(self, env_file_path: Optional[str] = None):
        """
        Initialize the CredentialManager.

        Args:
            env_file_path: Optional path to .env file. If None, looks for .env in current directory.
        """
        self.credentials: Dict[str, str] = {}
        self.env_file_path = env_file_path or ".env"
        self._load_credentials()

    def _load_credentials(self) -> None:
        """
        Load credentials from environment variables and .env file.
        """
        # Load from .env file if it exists
        env_path = Path(self.env_file_path)
        if env_path.exists():
            try:
                if DOTENV_AVAILABLE:
                    # Use python-dotenv for better .env file parsing
                    load_dotenv(env_path, override=False)
                    logger.debug(f"Loaded .env file using python-dotenv: {env_path}")
                else:
                    # Fallback to manual parsing
                    with open(env_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            # Skip empty lines and comments
                            if not line or line.startswith('#'):
                                if '=' in line:
                                    key, value = line.split('=', 1)
                                    key = key.strip()
                                    value = value.strip()
                                    # Remove quotes if present
                                    if value.startswith('"') and value.endswith('"'):
                                        value = value[1:-1]
                                    elif value.startswith("'") and value.endswith("'"):
                                        value = value[1:-1]
                                    self.credentials[key] = value
                    logger.debug(f"Loaded credentials from {env_path} (manual parsing)")
            except Exception as e:
                logger.warning(f"Failed to load .env file {env_path}: {e}")

        # Load from environment variables (these take precedence over .env file)
        for key, value in os.environ.items():
            if key.startswith(('CRED_', 'USER_', 'PASS_', 'API_', 'TOKEN_')):
                self.credentials[key] = value

        logger.debug(f"Loaded {len(self.credentials)} credentials")

    def get_credential(self, key: str) -> Optional[str]:
        """
        Get a credential by key.

        Args:
            key: The credential key

        Returns:
            The credential value or None if not found
        """
        return self.credentials.get(key)

    def get_credentials(self, keys: list[str]) -> Dict[str, Optional[str]]:
        """
        Get multiple credentials by their keys.

        Args:
            keys: List of credential keys

        Returns:
            Dictionary mapping keys to their values (None if not found)
        """
        return {key: self.credentials.get(key) for key in keys}

    def resolve_placeholders(self, text: str) -> str:
        """
        Resolve credential placeholders in text (e.g., {{USER_NAME}} -> actual username).

        Args:
            text: Text containing placeholders like {{USER_NAME}}

        Returns:
            Text with placeholders resolved to actual credential values
        """
        if not text:
            return text

        result = text
        for key, value in self.credentials.items():
            placeholder = f"{{{{{key}}}}}"
            if placeholder in result:
                result = result.replace(placeholder, value)
                logger.debug(f"Resolved placeholder {placeholder}")

        return result

    def has_credential(self, key: str) -> bool:
        """
        Check if a credential exists.

        Args:
            key: The credential key

        Returns:
            True if credential exists, False otherwise
        """
        return key in self.credentials

    def list_credentials(self) -> list[str]:
        """
        List all available credential keys (without values for security).

        Returns:
            List of credential keys
        """
        return list(self.credentials.keys())

    def reload_credentials(self) -> None:
        """
        Reload credentials from environment variables and .env file.
        """
        self.credentials.clear()
        self._load_credentials()
        logger.info("Credentials reloaded")

    def add_credential(self, key: str, value: str) -> None:
        """
        Add or update a credential in memory (not persisted).

        Args:
            key: The credential key
            value: The credential value
        """
        self.credentials[key] = value
        logger.debug(f"Added credential: {key}")

    def remove_credential(self, key: str) -> bool:
        """
        Remove a credential from memory.

        Args:
            key: The credential key

        Returns:
            True if credential was removed, False if not found
        """
        if key in self.credentials:
            del self.credentials[key]
            logger.debug(f"Removed credential: {key}")
            return True
        return False

    def get_credential_info(self) -> Dict[str, Any]:
        """
        Get information about available credentials (without exposing values).

        Returns:
            Dictionary with credential information
        """
        return {
            "total_credentials": len(self.credentials),
            "credential_keys": list(self.credentials.keys()),
            "env_file_path": self.env_file_path,
            "env_file_exists": Path(self.env_file_path).exists()
        }


# Global credential manager instance
_credential_manager: Optional[CredentialManager] = None


def get_credential_manager(env_file_path: Optional[str] = None) -> CredentialManager:
    """
    Get the global credential manager instance.

    Args:
        env_file_path: Optional path to .env file for initialization

    Returns:
        The global CredentialManager instance
    """
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager(env_file_path)
    return _credential_manager


def reset_credential_manager() -> None:
    """
    Reset the global credential manager instance.
    """
    global _credential_manager
    _credential_manager = None
