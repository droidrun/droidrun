"""
DroidRun CLI - Command line interface for controlling Android devices through LLM agents.
"""

import asyncio
import logging
import os
import sys
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import wraps
from typing import List, Dict, Any, Optional

import click
import importlib.metadata
import tomllib
from pathlib import Path
from async_adbutils import adb
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from droidrun import ResultEvent, DroidAgent
from droidrun.cli.logs import LogHandler
from droidrun.config_manager import DroidrunConfig
from droidrun.macro.cli import macro_cli
from droidrun.portal import (
    PORTAL_PACKAGE_NAME,
    download_portal_apk,
    enable_portal_accessibility,
    ping_portal,
    ping_portal_content,
    ping_portal_tcp,
)
from droidrun.telemetry import print_telemetry_message
from droidrun.agent.utils.llm_picker import load_llm
import json


@dataclass
class TestRunResult:
    """
    Result from a DroidRun test execution.

    This is returned by run_command_with_result() and provides full details
    about the test run including status, reasoning, and step-by-step observations.

    Attributes:
        status: "passed" or "failed" based on whether the goal was achieved
        reasoning: List of reasoning steps/observations during execution
        final_reason: The final reason/answer from the agent
        steps_taken: Number of steps executed
        action_history: List of actions taken during execution
        summary_history: List of summaries for each step
        success_rate: Percentage of successful actions (0.0 to 1.0)
        error: Error message if the run failed with an exception
    """
    status: str  # "passed" or "failed"
    reasoning: List[str] = field(default_factory=list)
    final_reason: str = ""
    steps_taken: int = 0
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    summary_history: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "reasoning": self.reasoning,
            "final_reason": self.final_reason,
            "steps_taken": self.steps_taken,
            "action_history": self.action_history,
            "summary_history": self.summary_history,
            "success_rate": self.success_rate,
            "error": self.error,
        }

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

console = Console()


def configure_logging(goal: str, debug: bool, rich_text: bool = True):
    logger = logging.getLogger("droidrun")
    logger.handlers = []

    handler = LogHandler(goal, rich_text=rich_text)
    handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s %(message)s", "%H:%M:%S")
        if debug
        else logging.Formatter("%(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)

    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

    return handler


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


async def get_portal_version(device_obj) -> str | None:
    try:
        version_output = await device_obj.shell(
            "content query --uri content://com.droidrun.portal/version"
        )

        if "result=" in version_output:
            json_str = version_output.split("result=", 1)[1].strip()
            version_data = json.loads(json_str)

            if version_data.get("status") == "success":
                return version_data.get("data")
        return None
    except Exception:
        return None


async def run_command(
    command: str,
    config_path: str | None = None,
    device: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    steps: int | None = None,
    base_url: str | None = None,
    api_base: str | None = None,
    vision: bool | None = None,
    manager_vision: bool | None = None,
    executor_vision: bool | None = None,
    codeact_vision: bool | None = None,
    reasoning: bool | None = None,
    stream: bool | None = None,
    tracing: bool | None = None,
    debug: bool | None = None,
    tcp: bool | None = None,
    save_trajectory: str | None = None,
    ios: bool = False,
    temperature: float | None = None,
    product_id: str | None = None,
    test_run_id: str | None = None,
    tcue_id: str | None = None,
    gcp_bucket: str | None = None,
    keep_local: bool = False,
    rich_text: bool | None = None,
    **kwargs,
) -> TestRunResult:
    """
    Run a command on your Android device and return full test results.

    This is the Python API hook for external systems (like Nova) to call DroidRun
    and get detailed results including status, reasoning, and step-by-step observations.

    Args:
        command: The natural language command/goal to execute
        config_path: Optional path to a YAML config file
        device: Device serial number or IP address
        provider: LLM provider (OpenAI, Anthropic, etc.)
        model: LLM model name
        steps: Maximum number of steps
        base_url: Base URL for API
        api_base: Base URL for OpenAI-like APIs
        vision: Enable vision for all agents
        manager_vision: Enable vision for manager agent
        executor_vision: Enable vision for executor agent
        codeact_vision: Enable vision for codeact agent
        reasoning: Enable planning with reasoning
        stream: Stream LLM responses
        tracing: Enable tracing
        debug: Enable debug logging
        tcp: Use TCP communication
        save_trajectory: Trajectory saving level ("none", "step", "action")
        ios: Run on iOS device
        temperature: LLM temperature
        product_id: Product ID for GCP logging
        test_run_id: Test run ID for GCP logging
        tcue_id: Test case under execution ID for GCP logging
        gcp_bucket: GCP bucket name
        keep_local: Keep local trajectory files after GCP upload
        rich_text: Enable rich text console output (uses config default if not specified)
        **kwargs: Additional arguments passed to LLM

    Returns:
        TestRunResult: Full test results including:
            - status: "passed" or "failed"
            - reasoning: List of step-by-step observations
            - final_reason: The agent's final answer/reason
            - steps_taken: Number of steps executed
            - action_history: List of actions taken
            - summary_history: List of step summaries
            - success_rate: Percentage of successful actions
            - error: Error message if execution failed

    Example:
        >>> import asyncio
        >>> from droidrun.cli.main import run_command, TestRunResult
        >>>
        >>> result = asyncio.run(run_command(
        ...     command="Open Settings and navigate to WiFi",
        ...     device="emulator-5554",
        ...     reasoning=True,
        ... ))
        >>> print(f"Status: {result.status}")
        >>> print(f"Reasoning: {result.reasoning}")
    """
    # Load config: use provided file or defaults
    if config_path:
        config = DroidrunConfig.from_yaml(config_path)
    else:
        config = DroidrunConfig()

    # Print cloud link in a box
    if config.logging.rich_text:
        cloud_text = Text()
        cloud_text.append("✨ Try DroidRun Cloud: ", style="bold cyan")
        cloud_text.append(
            "https://cloud.droidrun.ai/sign-in", style="bold blue underline"
        )
        cloud_panel = Panel(
            cloud_text,
            border_style="cyan",
            padding=(0, 1),
        )
        console.print(cloud_panel)
    else:
        console.print("\n✨ Try DroidRun Cloud: https://cloud.droidrun.ai/sign-in\n")

    # Initialize logging first (use config default if debug not specified)
    debug_mode = debug if debug is not None else config.logging.debug
    log_handler = configure_logging(command, debug_mode, config.logging.rich_text)
    logger = logging.getLogger("droidrun")

    log_handler.update_step("Initializing...")

    with log_handler.render():
        try:
            logger.info(f"🚀 Starting: {command}")
            if config_path:
                logger.info(f"📄 Config: {config_path}")
            else:
                logger.info("📄 Config: defaults")

            print_telemetry_message()

            # ================================================================
            # STEP 1: Apply CLI overrides via direct mutation
            # ================================================================

            # Vision overrides
            if vision is not None:
                # --vision flag overrides all agents
                config.agent.manager.vision = vision
                config.agent.executor.vision = vision
                config.agent.codeact.vision = vision
                logger.debug(f"CLI override: vision={vision} (all agents)")
            else:
                # Apply individual agent vision overrides
                if manager_vision is not None:
                    config.agent.manager.vision = manager_vision
                if executor_vision is not None:
                    config.agent.executor.vision = executor_vision
                if codeact_vision is not None:
                    config.agent.codeact.vision = codeact_vision

            # Agent overrides
            if steps is not None:
                config.agent.max_steps = steps
            if reasoning is not None:
                config.agent.reasoning = reasoning
            if stream is not None:
                config.agent.streaming = stream

            # Device overrides
            if device is not None:
                config.device.serial = device
            if tcp is not None:
                config.device.use_tcp = tcp

            # Logging overrides
            if debug is not None:
                config.logging.debug = debug
            if save_trajectory is not None:
                config.logging.save_trajectory = save_trajectory

            # GCP logging overrides
            if product_id and test_run_id and tcue_id:
                config.logging.gcp.enabled = True
                config.logging.gcp.product_id = product_id
                config.logging.gcp.test_run_id = test_run_id
                config.logging.gcp.tcue_id = tcue_id
                config.logging.gcp.keep_local = keep_local
                if gcp_bucket:
                    config.logging.gcp.bucket_name = gcp_bucket
                else:
                    config.logging.gcp.bucket_name = "nova_assets"  # Default bucket
                # Auto-enable trajectory saving for GCP upload
                if config.logging.save_trajectory == "none":
                    config.logging.save_trajectory = "action"
                logger.debug(
                    f"CLI override: GCP logging enabled -> "
                    f"{config.logging.gcp.bucket_name}/{product_id}/{test_run_id}/{tcue_id}"
                    f" (keep_local={keep_local})"
                )

            # Tracing overrides
            if tracing is not None:
                config.tracing.enabled = tracing

            # Platform overrides
            if ios:
                config.device.platform = "ios"

            # ================================================================
            # STEP 2: Initialize DroidAgent with config
            # ================================================================

            log_handler.update_step("Initializing DroidAgent...")

            mode = (
                "planning with reasoning"
                if config.agent.reasoning
                else "direct execution"
            )
            logger.info(f"🤖 Agent mode: {mode}")
            logger.info(
                f"👁️  Vision settings: Manager={config.agent.manager.vision}, "
                f"Executor={config.agent.executor.vision}, CodeAct={config.agent.codeact.vision}"
            )

            if config.tracing.enabled:
                logger.info("🔍 Tracing enabled")

            # Build DroidAgent kwargs for LLM loading
            droid_agent_kwargs = {"runtype": "cli"}
            llm = None

            if provider or model:
                assert (
                    provider and model
                ), "Either both provider and model must be provided or none of them"
                llm_kwargs = {}
                if temperature is not None:
                    llm_kwargs["temperature"] = temperature
                if base_url is not None:
                    llm_kwargs["base_url"] = base_url
                if api_base is not None:
                    llm_kwargs["api_base"] = api_base
                llm = load_llm(provider, model=model, **llm_kwargs, **kwargs)
            else:
                if temperature is not None:
                    droid_agent_kwargs["temperature"] = temperature
                if base_url is not None:
                    droid_agent_kwargs["base_url"] = base_url
                if api_base is not None:
                    droid_agent_kwargs["api_base"] = api_base

            if not ios:
                try:
                    device_obj = await adb.device(config.device.serial)
                    if device_obj:
                        portal_version = await get_portal_version(device_obj)

                        if not portal_version or portal_version < "0.4.1":
                            logger.warning(
                                f"⚠️  Portal version {portal_version} is outdated"
                            )
                            console.print(
                                f"\n[yellow]Portal version {portal_version} < 0.4.1. Running setup...[/]\n"
                            )

                            await _setup_portal(
                                path=None, device=config.device.serial, debug=debug_mode
                            )

                    else:
                        logger.debug("Could not get portal version, skipping check")
                except Exception as e:
                    logger.warning(f"Version check failed: {e}")

            droid_agent = DroidAgent(
                goal=command,
                llms=llm,
                config=config,
                timeout=1000,
                **droid_agent_kwargs,
            )

            # ================================================================
            # STEP 3: Run agent and collect results
            # ================================================================

            logger.debug("▶️  Starting agent execution...")
            logger.debug("Press Ctrl+C to stop")
            log_handler.update_step("Running agent...")

            try:
                handler = droid_agent.run()

                async for event in handler.stream_events():
                    log_handler.handle_event(event)

                result: ResultEvent = await handler

                # Extract data from shared state for TestRunResult
                shared_state = droid_agent.shared_state
                action_outcomes = shared_state.action_outcomes
                success_rate = (
                    sum(action_outcomes) / len(action_outcomes)
                    if action_outcomes
                    else 0.0
                )

                # Build reasoning list from summary history
                full_reasoning = []
                for i, summary in enumerate(shared_state.summary_history):
                    step_status = "PASS" if i < len(action_outcomes) and action_outcomes[i] else "FAIL"
                    full_reasoning.append(f"Step {i+1} [{step_status}]: {summary}")

                # Add final reason as last reasoning step if available
                if result.reason:
                    full_reasoning.append(f"Final: {result.reason}")

                return TestRunResult(
                    status="passed" if result.success else "failed",
                    reasoning=full_reasoning,
                    final_reason=result.reason,
                    steps_taken=result.steps,
                    action_history=list(shared_state.action_history),
                    summary_history=list(shared_state.summary_history),
                    success_rate=success_rate,
                    error=None,
                )

            except KeyboardInterrupt:
                log_handler.is_completed = True
                log_handler.is_success = False
                log_handler.current_step = "Stopped by user"
                logger.info("⏹️ Stopped by user")
                return TestRunResult(
                    status="failed",
                    reasoning=["Execution stopped by user"],
                    final_reason="Stopped by user",
                    error="KeyboardInterrupt",
                )

            except Exception as e:
                log_handler.is_completed = True
                log_handler.is_success = False
                log_handler.current_step = f"Error: {e}"
                logger.error(f"💥 Error: {e}")
                if config.logging.debug:
                    import traceback
                    logger.debug(traceback.format_exc())
                return TestRunResult(
                    status="failed",
                    reasoning=[f"Execution error: {str(e)}"],
                    final_reason=str(e),
                    error=str(e),
                )

        except Exception as e:
            log_handler.current_step = f"Error: {e}"
            logger.error(f"💥 Setup error: {e}")
            debug_mode = debug if debug is not None else config.logging.debug
            if debug_mode:
                import traceback
                logger.debug(traceback.format_exc())
            return TestRunResult(
                status="failed",
                reasoning=[f"Setup error: {str(e)}"],
                final_reason=str(e),
                error=str(e),
            )


class DroidRunCLI(click.Group):
    def parse_args(self, ctx, args):
        # If the first arg is not an option and not a known command, treat as 'run'
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args.insert(0, "run")

        return super().parse_args(ctx, args)


def _print_version(ctx, param, value):
    """Click callback to print version and exit early when --version is passed."""
    if not value or ctx.resilient_parsing:
        return
    version = None
    try:
        version = importlib.metadata.version("droidrun")
        # print("debug: step 1")
    except Exception:
        pass

    if not version:
        try:
            from droidrun import __version__ as pkg_version

            version = pkg_version
            # print("debug: step 2")
        except Exception:
            pass

    if not version:
        try:
            repo_root = Path(__file__).resolve().parents[2]
            pyproject = repo_root / "pyproject.toml"
            if pyproject.exists():
                with pyproject.open("rb") as f:
                    data = tomllib.load(f)
                    version = data.get("project", {}).get("version")
            # print("debug: step 3")
        except Exception:
            version = None

    if not version:
        version = "unknown"
    click.echo(f"v{version}")
    ctx.exit()


@click.group(cls=DroidRunCLI)
@click.option(
    "--version",
    is_flag=True,
    callback=_print_version,
    expose_value=False,
    is_eager=True,
    help="Show droidrun version and exit",
)
def cli():
    """DroidRun - Control your Android device through LLM agents."""
    pass


@cli.command()
@click.argument("command", type=str)
@click.option("--config", "-c", help="Path to custom config file", default=None)
@click.option("--device", "-d", help="Device serial number or IP address", default=None)
@click.option(
    "--provider",
    "-p",
    help="LLM provider (OpenAI, Ollama, Anthropic, GoogleGenAI, DeepSeek)",
    default=None,
)
@click.option(
    "--model",
    "-m",
    help="LLM model name",
    default=None,
)
@click.option("--temperature", type=float, help="Temperature for LLM", default=None)
@click.option("--steps", type=int, help="Maximum number of steps", default=None)
@click.option(
    "--base_url",
    "-u",
    help="Base URL for API (e.g., OpenRouter or Ollama)",
    default=None,
)
@click.option(
    "--api_base",
    help="Base URL for API (e.g., OpenAI or OpenAI-Like)",
    default=None,
)
@click.option(
    "--vision/--no-vision",
    default=None,
    help="Enable vision capabilites by using screenshots for all agents.",
)
@click.option(
    "--reasoning/--no-reasoning", default=None, help="Enable planning with reasoning"
)
@click.option(
    "--stream/--no-stream",
    default=None,
    help="Stream LLM responses to console in real-time",
)
@click.option(
    "--tracing/--no-tracing", default=None, help="Enable Arize Phoenix tracing"
)
@click.option("--debug/--no-debug", default=None, help="Enable verbose debug logging")
@click.option(
    "--tcp/--no-tcp",
    default=None,
    help="Use TCP communication for device control",
)
@click.option(
    "--save-trajectory",
    type=click.Choice(["none", "step", "action"]),
    help="Trajectory saving level: none (no saving), step (save per step), action (save per action)",
    default=None,
)
@click.option("--ios", type=bool, default=None, help="Run on iOS device")
@click.option(
    "--product-id",
    help="Product ID for GCP logging (enables GCP upload when set with --test-run-id and --tcue-id)",
    default=None,
)
@click.option(
    "--test-run-id",
    help="Test run ID for GCP logging",
    default=None,
)
@click.option(
    "--tcue-id",
    help="Test case under execution ID for GCP logging",
    default=None,
)
@click.option(
    "--gcp-bucket",
    help="GCP bucket name for trajectory uploads (default: nova_assets)",
    default=None,
)
@click.option(
    "--keep-local",
    is_flag=True,
    default=False,
    help="Keep local trajectory files after GCP upload (for debugging)",
)
@coro
async def run(
    command: str,
    config: str | None,
    device: str | None,
    provider: str | None,
    model: str | None,
    steps: int | None,
    base_url: str | None,
    api_base: str | None,
    temperature: float | None,
    vision: bool | None,
    reasoning: bool | None,
    stream: bool | None,
    tracing: bool | None,
    debug: bool | None,
    tcp: bool | None,
    save_trajectory: str | None,
    ios: bool | None,
    product_id: str | None,
    test_run_id: str | None,
    tcue_id: str | None,
    gcp_bucket: str | None,
    keep_local: bool,
):
    """Run a command on your Android device using natural language."""

    try:
        result = await run_command(
            command=command,
            config_path=config,
            device=device,
            provider=provider,
            model=model,
            steps=steps,
            base_url=base_url,
            api_base=api_base,
            vision=vision,
            reasoning=reasoning,
            stream=stream,
            tracing=tracing,
            debug=debug,
            tcp=tcp,
            temperature=temperature,
            save_trajectory=save_trajectory,
            ios=ios if ios is not None else False,
            product_id=product_id,
            test_run_id=test_run_id,
            tcue_id=tcue_id,
            gcp_bucket=gcp_bucket,
            keep_local=keep_local,
        )
    finally:
        # Disable DroidRun keyboard after execution
        # Note: Port forwards are managed automatically and persist until device disconnect
        try:
            if not (ios if ios is not None else False):
                device_obj = await adb.device(device)
                if device_obj:
                    await device_obj.shell(
                        "ime disable com.droidrun.portal/.input.DroidrunKeyboardIME"
                    )
        except Exception:
            click.echo("Failed to disable DroidRun keyboard")

    # Exit with appropriate code based on test result status
    sys.exit(0 if result.status == "passed" else 1)


@cli.command()
@coro
async def devices():
    """List connected Android devices."""
    try:
        devices = await adb.list()
        if not devices:
            console.print("[yellow]No devices connected.[/]")
            return

        console.print(f"[green]Found {len(devices)} connected device(s):[/]")
        for device in devices:
            console.print(f"  • [bold]{device.serial}[/]")
    except Exception as e:
        console.print(f"[red]Error listing devices: {e}[/]")


@cli.command()
@click.argument("serial")
@coro
async def connect(serial: str):
    """Connect to a device over TCP/IP."""
    try:
        device = await adb.connect(serial)
        if device.count("already connected"):
            console.print(f"[green]Successfully connected to {serial}[/]")
        else:
            console.print(f"[red]Failed to connect to {serial}: {device}[/]")
    except Exception as e:
        console.print(f"[red]Error connecting to device: {e}[/]")


@cli.command()
@click.argument("serial")
@coro
async def disconnect(serial: str):
    """Disconnect from a device."""
    try:
        success = await adb.disconnect(serial, raise_error=True)
        if success:
            console.print(f"[green]Successfully disconnected from {serial}[/]")
        else:
            console.print(f"[yellow]Device {serial} was not connected[/]")
    except Exception as e:
        console.print(f"[red]Error disconnecting from device: {e}[/]")


async def _setup_portal(path: str | None, device: str | None, debug: bool):
    """Internal async function to install and enable the DroidRun Portal on a device."""
    try:
        if not device:
            devices = await adb.list()
            if not devices:
                console.print("[yellow]No devices connected.[/]")
                return

            device = devices[0].serial
            console.print(f"[blue]Using device:[/] {device}")

        device_obj = await adb.device(device)
        if not device_obj:
            console.print(
                f"[bold red]Error:[/] Could not get device object for {device}"
            )
            return

        if not path:
            console.print("[bold blue]Downloading DroidRun Portal APK...[/]")
            apk_context = download_portal_apk(debug)
        else:
            console.print(f"[bold blue]Using provided APK:[/] {path}")
            apk_context = nullcontext(path)

        with apk_context as apk_path:
            if not os.path.exists(apk_path):
                console.print(f"[bold red]Error:[/] APK file not found at {apk_path}")
                return

            console.print(f"[bold blue]Step 1/2: Installing APK:[/] {apk_path}")
            try:
                await device_obj.install(
                    apk_path, uninstall=True, flags=["-g"], silent=not debug
                )
            except Exception as e:
                console.print(f"[bold red]Installation failed:[/] {e}")
                return

            console.print("[bold green]Installation successful![/]")

            console.print("[bold blue]Step 2/2: Enabling accessibility service[/]")

            try:
                await enable_portal_accessibility(device_obj)

                console.print("[green]Accessibility service enabled successfully![/]")
                console.print(
                    "\n[bold green]Setup complete![/] The DroidRun Portal is now installed and ready to use."
                )

            except Exception as e:
                console.print(
                    f"[yellow]Could not automatically enable accessibility service: {e}[/]"
                )
                console.print(
                    "[yellow]Opening accessibility settings for manual configuration...[/]"
                )

                await device_obj.shell(
                    "am start -a android.settings.ACCESSIBILITY_SETTINGS"
                )

                console.print(
                    "\n[yellow]Please complete the following steps on your device:[/]"
                )
                console.print(
                    f"1. Find [bold]{PORTAL_PACKAGE_NAME}[/] in the accessibility services list"
                )
                console.print("2. Tap on the service name")
                console.print(
                    "3. Toggle the switch to [bold]ON[/] to enable the service"
                )
                console.print("4. Accept any permission dialogs that appear")

                console.print(
                    "\n[bold green]APK installation complete![/] Please manually enable the accessibility service using the steps above."
                )

    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")

        if debug:
            import traceback

            traceback.print_exc()


@cli.command()
@click.option("--device", "-d", help="Device serial number or IP address", default=None)
@click.option(
    "--path",
    help="Path to the Droidrun Portal APK to install on the device. If not provided, the latest portal apk version will be downloaded and installed.",
    default=None,
)
@click.option(
    "--debug", is_flag=True, help="Enable verbose debug logging", default=False
)
@coro
async def setup(path: str | None, device: str | None, debug: bool):
    """Install and enable the DroidRun Portal on a device."""
    await _setup_portal(path, device, debug)


@cli.command()
@click.option("--device", "-d", help="Device serial number or IP address", default=None)
@click.option(
    "--tcp/--no-tcp",
    default=None,
    help="Use TCP communication for device control",
)
@click.option("--debug/--no-debug", default=None, help="Enable verbose debug logging")
@coro
async def ping(device: str | None, tcp: bool | None, debug: bool | None):
    """Ping a device to check if it is ready and accessible."""
    # Handle None defaults
    debug_mode = debug if debug is not None else False
    use_tcp_mode = tcp if tcp is not None else False

    try:
        device_obj = await adb.device(device)
        if not device_obj:
            console.print(f"[bold red]Error:[/] Could not find device {device}")
            return

        await ping_portal(device_obj, debug_mode)

        if use_tcp_mode:
            await ping_portal_tcp(device_obj, debug_mode)
        else:
            await ping_portal_content(device_obj, debug_mode)

        console.print(
            "[bold green]Portal is installed and accessible. You're good to go![/]"
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        if debug_mode:
            import traceback

            traceback.print_exc()


# Add macro commands as a subgroup
cli.add_command(macro_cli, name="macro")


async def test(
    command: str,
    config_path: str | None = None,
    device: str | None = None,
    steps: int | None = None,
    vision: bool | None = None,
    reasoning: bool | None = None,
    tracing: bool | None = None,
    debug: bool | None = None,
    use_tcp: bool | None = None,
    save_trajectory: str | None = None,
    temperature: float | None = None,
    ios: bool = False,
):
    # Load config: use provided file or defaults
    if config_path:
        config = DroidrunConfig.from_yaml(config_path)
    else:
        config = DroidrunConfig()

    # Initialize logging first (use config default if debug not specified)
    debug_mode = debug if debug is not None else config.logging.debug
    log_handler = configure_logging(command, debug_mode, config.logging.rich_text)
    logger = logging.getLogger("droidrun")

    log_handler.update_step("Initializing...")

    with log_handler.render():
        try:
            logger.info(f"🚀 Starting: {command}")
            if config_path:
                logger.info(f"📄 Config: {config_path}")
            else:
                logger.info("📄 Config: defaults")

            print_telemetry_message()

            # ================================================================
            # STEP 1: Apply CLI overrides via direct mutation
            # ================================================================

            # Vision overrides
            if vision is not None:
                # --vision flag overrides all agents
                config.agent.manager.vision = vision
                config.agent.executor.vision = vision
                config.agent.codeact.vision = vision
                logger.debug(f"CLI override: vision={vision} (all agents)")

            # Agent overrides
            if steps is not None:
                config.agent.max_steps = steps
            if reasoning is not None:
                config.agent.reasoning = reasoning

            # Device overrides
            if device is not None:
                config.device.serial = device
            if use_tcp is not None:
                config.device.use_tcp = use_tcp

            # Logging overrides
            if debug is not None:
                config.logging.debug = debug
            if save_trajectory is not None:
                config.logging.save_trajectory = save_trajectory

            # Tracing overrides
            if tracing is not None:
                config.tracing.enabled = tracing

            # Platform overrides
            if ios:
                config.device.platform = "ios"

            # ================================================================
            # STEP 2: Initialize DroidAgent with config
            # ================================================================

            log_handler.update_step("Initializing DroidAgent...")

            mode = (
                "planning with reasoning"
                if config.agent.reasoning
                else "direct execution"
            )
            logger.info(f"🤖 Agent mode: {mode}")
            logger.info(
                f"👁️  Vision settings: Manager={config.agent.manager.vision}, "
                f"Executor={config.agent.executor.vision}, CodeAct={config.agent.codeact.vision}"
            )

            if config.tracing.enabled:
                logger.info("🔍 Tracing enabled")

            # Build DroidAgent kwargs for LLM loading
            droid_agent_kwargs = {}
            if temperature is not None:
                droid_agent_kwargs["temperature"] = temperature

            droid_agent = DroidAgent(
                goal=command,
                config=config,
                timeout=1000,
                **droid_agent_kwargs,
            )

            # ================================================================
            # STEP 3: Run agent
            # ================================================================

            logger.debug("▶️  Starting agent execution...")
            logger.debug("Press Ctrl+C to stop")
            log_handler.update_step("Running agent...")

            try:
                handler = droid_agent.run()

                async for event in handler.stream_events():
                    log_handler.handle_event(event)
                result = await handler  # noqa: F841

            except KeyboardInterrupt:
                log_handler.is_completed = True
                log_handler.is_success = False
                log_handler.current_step = "Stopped by user"
                logger.info("⏹️ Stopped by user")

            except Exception as e:
                log_handler.is_completed = True
                log_handler.is_success = False
                log_handler.current_step = f"Error: {e}"
                logger.error(f"💥 Error: {e}")
                if config.logging.debug:
                    import traceback

                    logger.debug(traceback.format_exc())

        except Exception as e:
            log_handler.current_step = f"Error: {e}"
            logger.error(f"💥 Setup error: {e}")
            debug_mode = debug if debug is not None else config.logging.debug
            if debug_mode:
                import traceback

                logger.debug(traceback.format_exc())


if __name__ == "__main__":
    command = "open youtube and play a song by shakira"
    command = "use open_app to open the settings and search for the battery and enter the first result"
    device = None
    provider = "GoogleGenAI"
    model = "models/gemini-2.5-flash"
    temperature = 0
    api_key = os.getenv("GOOGLE_API_KEY")
    steps = 15
    vision = True
    reasoning = False
    tracing = True
    debug = True
    use_tcp = False
    base_url = None
    api_base = None
    ios = False
    save_trajectory = "none"
    asyncio.run(run_command(command, device="emulator-5556", reasoning=False))
