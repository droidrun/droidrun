from posthog import Posthog
from pathlib import Path
from uuid import uuid4
import os
import logging
from .events import TelemetryEvent

logger = logging.getLogger("droidrun-telemetry")
droidrun_logger = logging.getLogger("droidrun")

PROJECT_API_KEY = "phc_XyD3HKIsetZeRkmnfaBughs8fXWYArSUFc30C0HmRiO"
HOST = "https://eu.i.posthog.com"
USER_ID_PATH = Path.home() / ".droidrun" / "user_id"
RUN_ID = str(uuid4())

TELEMETRY_ENABLED_MESSAGE = "🕵️  Anonymized telemetry enabled. See https://docs.droidrun.ai/v3/guides/telemetry for more information."
TELEMETRY_DISABLED_MESSAGE = "🛑 Anonymized telemetry disabled. Consider setting the DROIDRUN_TELEMETRY_ENABLED environment variable to 'true' to enable telemetry and help us improve DroidRun."

posthog = Posthog(
    project_api_key=PROJECT_API_KEY,
    host=HOST,
    disable_geoip=False,
)

# Optional redactor that can be set by the CLI to remove secrets
_redactor = None  # callable or None


def set_redactor(redactor):
    """Set a redactor callable applied to messages and properties.

    The redactor is expected to accept either a string or a nested
    Python structure and return the redacted version.
    """
    global _redactor
    _redactor = redactor


def is_telemetry_enabled():
    telemetry_enabled = os.environ.get("DROIDRUN_TELEMETRY_ENABLED", "true")
    enabled = telemetry_enabled.lower() in ["true", "1", "yes", "y"]
    logger.debug(f"Telemetry enabled: {enabled}")
    return enabled


def print_telemetry_message():
    if is_telemetry_enabled():
        msg = TELEMETRY_ENABLED_MESSAGE
        if _redactor:
            msg = _redactor(msg)
        droidrun_logger.info(msg)

    else:
        msg = TELEMETRY_DISABLED_MESSAGE
        if _redactor:
            msg = _redactor(msg)
        droidrun_logger.info(msg)


# Print telemetry message on import
print_telemetry_message()


def get_user_id() -> str:
    try:
        if not USER_ID_PATH.exists():
            USER_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
            USER_ID_PATH.touch()
            USER_ID_PATH.write_text(str(uuid4()))
        uid = USER_ID_PATH.read_text()
        dbg = f"User ID: {uid}"
        if _redactor:
            dbg = _redactor(dbg)
        logger.debug(dbg)
        return uid
    except Exception as e:
        logger.error(f"Error getting user ID: {e}")
        return "unknown"


def capture(event: TelemetryEvent, user_id: str | None = None):
    try:
        if not is_telemetry_enabled():
            logger.debug(f"Telemetry disabled, skipping capture of {event}")
            return
        event_name = type(event).__name__
        event_data = event.model_dump()
        properties = {
            "run_id": RUN_ID,
            **event_data,
        }

        # Redact properties
        if _redactor:
            properties = _redactor(properties)

        posthog.capture(event_name, distinct_id=user_id or get_user_id(), properties=properties)
        dbg = f"Captured event: {event_name} with properties: {event}"
        if _redactor:
            dbg = _redactor(dbg)
        logger.debug(dbg)
    except Exception as e:
        logger.error(f"Error capturing event: {e}")


def flush():
    try:
        if not is_telemetry_enabled():
            logger.debug(f"Telemetry disabled, skipping flush")
            return
        posthog.flush()
        logger.debug(f"Flushed telemetry data")
    except Exception as e:
        logger.error(f"Error flushing telemetry data: {e}")
