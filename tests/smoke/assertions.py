"""Smoke test assertions."""

import re
import logging

logger = logging.getLogger("smoke")


class AssertionError(Exception):
    pass


def assert_result_success(result) -> None:
    """Assert the agent reported success."""
    if not result.success:
        raise AssertionError(
            f"Agent reported failure: {result.reason}"
        )


def assert_structured_output(result) -> None:
    """Assert structured output contains a valid Android version."""
    output = result.structured_output
    if output is None:
        raise AssertionError("No structured output returned")

    version = getattr(output, "android_version", None)
    if version is None:
        raise AssertionError(
            f"structured_output missing 'android_version' field: {output}"
        )

    if not re.match(r"^\d+", str(version)):
        raise AssertionError(
            f"android_version doesn't look like a version: '{version}'"
        )


def assert_type_secret_called(tool_events: list) -> None:
    """Assert type_secret was called and succeeded."""
    for event in tool_events:
        if event.tool_name == "type_secret" and event.success:
            return

    names = [e.tool_name for e in tool_events]
    raise AssertionError(
        f"type_secret not found or failed in tool events. Tools called: {names}"
    )


def assert_package_name(ui_state, expected_substring: str) -> None:
    """Assert the device's current package name contains the expected substring."""
    pkg = ui_state.phone_state.package_name or ""
    if expected_substring.lower() not in pkg.lower():
        raise AssertionError(
            f"Expected package containing '{expected_substring}', got '{pkg}'"
        )


ASSERTION_MAP = {
    "result_success": lambda ctx: assert_result_success(ctx["result"]),
    "structured_output": lambda ctx: assert_structured_output(ctx["result"]),
    "type_secret_called": lambda ctx: assert_type_secret_called(ctx["tool_events"]),
    "package_name": lambda ctx: assert_package_name(
        ctx["ui_state"], ctx["expected_package"]
    ),
}


def run_assertions(assertion_names: list[str], context: dict) -> list[str]:
    """Run named assertions and return list of failure messages."""
    failures = []
    for name in assertion_names:
        fn = ASSERTION_MAP.get(name)
        if fn is None:
            failures.append(f"Unknown assertion: {name}")
            continue
        try:
            fn(context)
            logger.info(f"  PASS: {name}")
        except (AssertionError, Exception) as e:
            logger.error(f"  FAIL: {name} — {e}")
            failures.append(f"{name}: {e}")
    return failures
