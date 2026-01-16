"""External agent loader - dynamic imports."""
import importlib
import logging
from typing import Any, Callable, Dict, Optional, TypedDict

logger = logging.getLogger("droidrun")


class ExternalAgentModule(TypedDict):
    """Type for loaded external agent module."""

    run: Callable
    config: Dict[str, Any]


def load_agent(name: str) -> Optional[ExternalAgentModule]:
    """
    Dynamically load an external agent by name.

    Args:
        name: Agent module name (e.g., "mai_ui", "autoglm")

    Returns:
        Dict with 'run' function and 'config' defaults, or None if failed.
    """
    try:
        module = importlib.import_module(f"droidrun.agent.external.{name}")

        if not hasattr(module, "run"):
            logger.error(f"External agent '{name}' missing run() function")
            return None

        return {
            "run": module.run,
            "config": getattr(module, "DEFAULT_CONFIG", {}),
        }

    except ImportError as e:
        logger.error(f"Failed to load external agent '{name}': {e}")
        return None
