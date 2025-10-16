"""
Credential loading utilities for DroidAgent.
Handles both config-based and direct dict credentials.
"""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from droidrun.config_manager.config_manager import CredentialsConfig
    from droidrun.credential_manager.credential_manager import CredentialManager

logger = logging.getLogger("droidrun")


def load_credential_manager(
    credentials: "CredentialsConfig | dict | None",
) -> Optional["CredentialManager"]:
    """
    Load credential manager from either CredentialsConfig or direct dict.

    Args:
        credentials: Either CredentialsConfig object, dict of credentials, or None

    Returns:
        CredentialManager instance or None if no credentials provided

    Examples:
        # Option 1: CredentialsConfig (from config file)
        credential_manager = load_credential_manager(config.credentials)

        # Option 2: Direct dict
        credential_manager = load_credential_manager({"MY_PASSWORD": "secret123"})

        # Option 3: None (no credentials)
        credential_manager = load_credential_manager(None)
    """
    from droidrun.credential_manager.credential_manager import CredentialManager

    if credentials is None:
        logger.debug("No credentials provided")
        return None

    # Option 1: Direct dict (in-memory mode)
    if isinstance(credentials, dict):
        if not credentials:
            logger.debug("NO credentials provided")
            return None

        logger.info(f"Loading credentials from in-memory dict ({len(credentials)} keys)")
        try:
            return CredentialManager(credentials_dict=credentials)
        except Exception as e:
            logger.error(f"Failed to load credentials from dict: {e}")
            return None

    # Option 2: CredentialsConfig (file mode)
    # Check if it's a CredentialsConfig-like object (duck typing)
    if hasattr(credentials, "enabled") and hasattr(credentials, "file_path"):
        if not credentials.enabled:
            logger.debug("Credentials disabled in config")
            return None

        logger.info(
            f"Loading credentials from config file: {credentials.file_path}"
        )
        try:
            return CredentialManager(credentials_path=credentials.file_path)
        except FileNotFoundError:
            logger.warning(
                f"Credentials file not found: {credentials.file_path}"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to load credentials from file: {e}")
            return None

    # Unknown type
    logger.warning(
        f"Unknown credentials type: {type(credentials)}. "
        "Expected CredentialsConfig, dict, or None"
    )
    return None
