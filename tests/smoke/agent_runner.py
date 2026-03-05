"""Run DroidAgent from source against a cloud device."""

import logging
import os
from dataclasses import dataclass, field
from uuid import uuid4

from pydantic import BaseModel, Field

from droidrun import DroidAgent, DroidrunConfig, load_llm
from droidrun.agent.common.events import ScreenshotEvent, ToolExecutionEvent
from droidrun.agent.droid.events import ResultEvent
from droidrun.tools.driver.cloud import CloudDriver

from tests.smoke.config import SmokeTestConfig

logger = logging.getLogger("smoke")

LLM_MODEL = "gemini-3.1-flash-lite-preview"


class AndroidVersion(BaseModel):
    """Structured output model for extracting Android version."""

    android_version: str = Field(description="The Android version number (e.g. '14', '15')")


@dataclass
class RunResult:
    result: ResultEvent | None = None
    screenshots: list[bytes] = field(default_factory=list)
    tool_events: list[ToolExecutionEvent] = field(default_factory=list)
    error: str | None = None
    langfuse_session_id: str | None = None


async def run_agent(
    test_config: SmokeTestConfig,
    device_id: str,
    api_key: str,
    base_url: str,
    trajectory_dir: str | None = None,
    langfuse_host: str | None = None,
) -> RunResult:
    """Run a single smoke test agent and collect results."""
    run_result = RunResult()

    # Ensure screenshots are emitted even for non-vision runs
    os.environ["DROIDRUN_STREAM_SCREENSHOTS"] = "1"

    try:
        driver = CloudDriver(
            device_id=device_id,
            api_key=api_key,
            base_url=base_url,
        )

        config = DroidrunConfig()
        config.agent.reasoning = test_config.reasoning
        config.agent.max_steps = test_config.max_steps
        config.agent.streaming = False
        config.agent.fast_agent.vision = test_config.vision
        config.agent.manager.vision = test_config.vision
        config.agent.executor.vision = test_config.vision
        config.telemetry.enabled = False

        # Trajectory writer
        if trajectory_dir:
            config.logging.save_trajectory = "all"
            config.logging.trajectory_path = trajectory_dir
            config.logging.trajectory_gifs = True
        else:
            config.logging.save_trajectory = "none"

        # Langfuse tracing
        langfuse_secret = os.environ.get("LANGFUSE_SECRET_KEY", "")
        langfuse_public = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        if langfuse_secret and langfuse_public:
            session_id = str(uuid4())
            run_result.langfuse_session_id = session_id
            config.tracing.enabled = True
            config.tracing.provider = "langfuse"
            config.tracing.langfuse_secret_key = langfuse_secret
            config.tracing.langfuse_public_key = langfuse_public
            config.tracing.langfuse_host = langfuse_host or os.environ.get(
                "LANGFUSE_HOST", "https://us.cloud.langfuse.com"
            )
            config.tracing.langfuse_session_id = session_id
            config.tracing.langfuse_user_id = "smoke-test"
        else:
            config.tracing.enabled = False

        llm = load_llm("GoogleGenAI", model=LLM_MODEL)

        credentials = None
        if test_config.credentials:
            credentials = {"test-account": "smoketest123"}

        output_model = None
        if test_config.output_schema:
            output_model = AndroidVersion

        agent = DroidAgent(
            goal=test_config.task,
            config=config,
            llms=llm,
            driver=driver,
            credentials=credentials,
            output_model=output_model,
            timeout=300,
        )

        handler = agent.run()
        async for event in handler.stream_events():
            if isinstance(event, ScreenshotEvent):
                run_result.screenshots.append(event.screenshot)
            elif isinstance(event, ToolExecutionEvent):
                run_result.tool_events.append(event)

        run_result.result = await handler

    except Exception as e:
        logger.error(f"Agent run failed: {e}")
        run_result.error = str(e)

    return run_result
