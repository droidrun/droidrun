#!/usr/bin/env python3
"""Smoke test runner for droidrun.

Provisions a cloud device (or uses an existing one), runs 4 agent configurations
sequentially, asserts expected outcomes, generates trajectory GIFs, and writes a summary.

Usage:
    # Auto-provision a temporary device:
    MOBILERUN_API_KEY=xxx python tests/smoke/run.py --output-dir=artifacts

    # Use an existing device (skips provisioning/termination):
    MOBILERUN_API_KEY=xxx python tests/smoke/run.py --device-id=UUID --output-dir=artifacts
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from mobilerun import AsyncMobilerun

from tests.smoke.agent_runner import run_agent
from tests.smoke.assertions import run_assertions
from tests.smoke.config import SMOKE_TESTS
from tests.smoke.device import (
    get_ui_state,
    press_home,
    provision_device,
    terminate_device,
    wait_for_ready,
)
from tests.smoke.gif import create_gif

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke")


def _langfuse_session_url(host: str, session_id: str) -> str:
    """Build a Langfuse session URL."""
    host = host.rstrip("/")
    return f"{host}/sessions/{session_id}"


def write_summary(output_dir: Path, results: list[dict], langfuse_host: str) -> None:
    """Write summary.md with pass/fail status, GIF links, and trace links."""
    lines = ["# Smoke Test Results\n"]

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    lines.append(f"**{passed}/{total} passed**\n")
    lines.append("| Test | Mode | Vision | Status | Time | Details |")
    lines.append("|------|------|--------|--------|------|---------|")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        mode = "reasoning" if r["reasoning"] else "fast"
        vision = "on" if r["vision"] else "off"
        elapsed = f"{r['elapsed']:.0f}s"
        details = r.get("error", "") or ", ".join(r.get("failures", []))
        if not details:
            details = r.get("reason", "")
        details = details.replace("\n", " ").replace("|", "\\|")[:100]
        lines.append(f"| {r['name']} | {mode} | {vision} | {status} | {elapsed} | {details} |")

    # Langfuse traces
    has_traces = any(r.get("langfuse_session_id") for r in results)
    if has_traces:
        lines.append("")
        lines.append("## Langfuse Traces\n")
        for r in results:
            sid = r.get("langfuse_session_id")
            if sid:
                url = _langfuse_session_url(langfuse_host, sid)
                lines.append(f"- **{r['name']}**: [{sid[:8]}...]({url})")
            else:
                lines.append(f"- **{r['name']}**: _no trace_")

    # Trajectory files
    lines.append("")
    lines.append("## Trajectories\n")
    for r in results:
        traj_dir = f"trajectories/{r['name']}"
        lines.append(f"- **{r['name']}**: [`{traj_dir}/`]({traj_dir}/)")

    # GIFs
    lines.append("")
    lines.append("## Trajectory GIFs\n")
    for r in results:
        gif_name = f"{r['name']}.gif"
        if r["has_gif"]:
            lines.append(f"### {r['name']}\n")
            lines.append(f"![{r['name']}]({gif_name})\n")
        else:
            lines.append(f"### {r['name']}\n")
            lines.append("_No screenshots captured._\n")

    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(lines))
    logger.info(f"Summary written to {summary_path}")


async def main(output_dir: Path, device_id_arg: str | None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("MOBILERUN_API_KEY")
    if not api_key:
        logger.error("MOBILERUN_API_KEY env var is required")
        return 1

    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not google_key:
        logger.error("GOOGLE_API_KEY or GEMINI_API_KEY env var is required")
        return 1

    base_url = os.environ.get("MOBILERUN_BASE_URL", "https://api.mobilerun.ai/v1")
    langfuse_host = os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

    client = AsyncMobilerun(api_key=api_key, base_url=base_url)
    device_id = device_id_arg
    provisioned = False

    try:
        if device_id:
            logger.info(f"Using existing device: {device_id}")
        else:
            device = await provision_device(client)
            device_id = device.id
            provisioned = True
            await wait_for_ready(client, device_id, timeout=120)

        results = []

        for test_config in SMOKE_TESTS:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Running: {test_config.name}")
            logger.info(f"  mode={'reasoning' if test_config.reasoning else 'fast'} vision={test_config.vision}")
            logger.info(f"  task: {test_config.task}")
            logger.info(f"{'=' * 60}")

            # Reset device to home screen
            await press_home(client, device_id)

            # Trajectory dir per test
            trajectory_dir = str(output_dir / "trajectories" / test_config.name)

            t0 = time.monotonic()
            run_result = await run_agent(
                test_config,
                device_id,
                api_key,
                base_url,
                trajectory_dir=trajectory_dir,
                langfuse_host=langfuse_host,
            )
            elapsed = time.monotonic() - t0

            # Build assertion context
            assertion_ctx = {
                "result": run_result.result,
                "tool_events": run_result.tool_events,
                "expected_package": test_config.expected_package,
            }

            # Get UI state for package name assertion
            if "package_name" in test_config.assertions:
                try:
                    assertion_ctx["ui_state"] = await get_ui_state(client, device_id)
                except Exception as e:
                    logger.warning(f"Failed to get UI state: {e}")
                    assertion_ctx["ui_state"] = None

            # Run assertions
            if run_result.error:
                failures = [f"agent_error: {run_result.error}"]
            else:
                failures = run_assertions(test_config.assertions, assertion_ctx)

            passed = len(failures) == 0

            # Generate GIF
            gif_path = create_gif(
                run_result.screenshots,
                output_dir / f"{test_config.name}.gif",
            )

            # Log result
            status = "PASS" if passed else "FAIL"
            logger.info(f"\n  Result: {status} ({elapsed:.0f}s, {len(run_result.screenshots)} screenshots)")
            if run_result.langfuse_session_id:
                logger.info(f"  Langfuse session: {run_result.langfuse_session_id}")
            if failures:
                for f in failures:
                    logger.error(f"    {f}")

            results.append({
                "name": test_config.name,
                "passed": passed,
                "elapsed": elapsed,
                "reasoning": test_config.reasoning,
                "vision": test_config.vision,
                "failures": failures,
                "error": run_result.error,
                "reason": run_result.result.reason if run_result.result else "",
                "has_gif": gif_path is not None,
                "langfuse_session_id": run_result.langfuse_session_id,
            })

        # Summary
        write_summary(output_dir, results, langfuse_host)

        logger.info(f"\n{'=' * 60}")
        logger.info("SUMMARY")
        logger.info(f"{'=' * 60}")
        all_passed = all(r["passed"] for r in results)
        for r in results:
            icon = "PASS" if r["passed"] else "FAIL"
            logger.info(f"  [{icon}] {r['name']} ({r['elapsed']:.0f}s)")

        total_passed = sum(1 for r in results if r["passed"])
        logger.info(f"\n  {total_passed}/{len(results)} passed")

        return 0 if all_passed else 1

    finally:
        if provisioned and device_id:
            await terminate_device(client, device_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run droidrun smoke tests")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for GIFs and summary (default: artifacts)",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default=None,
        help="Use an existing device ID instead of provisioning a new one",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(main(args.output_dir, args.device_id))
    sys.exit(exit_code)
