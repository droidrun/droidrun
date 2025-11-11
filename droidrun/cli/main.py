"""
DroidRun CLI - Command line interface for controlling Android devices through LLM agents.
"""

import asyncio
import logging
import os
import sys
import warnings
from contextlib import nullcontext
from functools import wraps

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
    check_portal_accessibility,
    download_portal_apk,
    enable_portal_accessibility,
    ping_portal,
    ping_portal_content,
    ping_portal_tcp,
)
from droidrun.telemetry import print_telemetry_message
from droidrun.config_manager.path_resolver import PathResolver
from droidrun.agent.utils.llm_picker import load_llm

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

console = Console()

# Ensure config.yaml exists in package dir
package_config_path = PathResolver.get_project_root() / "config.yaml"
if not package_config_path.exists():
    # Config not found, copy config_example.yaml to package dir
    import shutil

    example_path = PathResolver.get_project_root() / "config_example.yaml"
    shutil.copy2(example_path, package_config_path)
    console.print(f"[blue]Created config.yaml in package at: {package_config_path}[/]")
else:
    console.print(f"[blue]Using config from package: {package_config_path}[/]")


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


async def check_portal_setup(
    device_serial: str | None,
    use_tcp: bool,
    rich_text: bool,
    debug_mode: bool,
    ios: bool = False,
) -> tuple[bool, str | None]:
    """
    Check if DroidRun Portal is installed and accessibility service is enabled.

    Args:
        device_serial: Device serial number or None for default device
        use_tcp: Whether TCP mode is being used
        rich_text: Whether to use rich text formatting
        debug_mode: Whether debug mode is enabled
        ios: Whether running on iOS

    Returns:
        Tuple of (is_setup_ok, warning_message)
    """
    # Skip check for iOS
    if ios:
        return True, None

    try:
        device_obj = await adb.device(device_serial)
        if not device_obj:
            warning = "Could not connect to device"
            return False, warning

        # Check if portal is installed
        try:
            packages = await device_obj.list_packages()
            if PORTAL_PACKAGE_NAME not in packages:
                warning = (
                    "DroidRun Portal is not installed on the device. "
                    "Run 'droidrun setup' to install it."
                )
                return False, warning
        except Exception as e:
            if debug_mode:
                warning = f"Failed to check portal installation: {e}"
            else:
                warning = "Failed to check portal installation"
            return False, warning

        # Check if accessibility service is enabled
        try:
            if not await check_portal_accessibility(device_obj, debug=debug_mode):
                warning = (
                    "DroidRun Portal accessibility service is not enabled. "
                    "Run 'droidrun setup' or enable it manually in Settings > Accessibility."
                )
                return False, warning
        except Exception as e:
            if debug_mode:
                warning = f"Failed to check accessibility service: {e}"
            else:
                warning = "Failed to check accessibility service"
            return False, warning

        # Check communication mode
        try:
            if use_tcp:
                # Test TCP connection
                from droidrun.tools import AdbTools

                tools = AdbTools(serial=device_obj.serial, use_tcp=True)
                await tools.connect()
                await tools.disconnect()
            else:
                # Test content provider
                state = await device_obj.shell(
                    "content query --uri content://com.droidrun.portal/state"
                )
                if "Row: 0 result=" not in state:
                    warning = "DroidRun Portal is not responding via content provider"
                    return False, warning
        except Exception as e:
            if debug_mode:
                warning = f"Failed to verify portal communication: {e}"
            else:
                warning = "Failed to verify portal communication"
            return False, warning

        return True, None

    except Exception as e:
        if debug_mode:
            warning = f"Error checking portal setup: {e}"
        else:
            warning = "Error checking portal setup"
        return False, warning


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
    tracing: bool | None = None,
    debug: bool | None = None,
    tcp: bool | None = None,
    save_trajectory: str | None = None,
    ios: bool = False,
    allow_drag: bool | None = None,
    temperature: float | None = None,
    **kwargs,
) -> bool:
    """Run a command on your Android device using natural language.

    Returns:
        bool: True if the task completed successfully, False otherwise.
    """
    # Load config and apply CLI overrides via direct mutation
    config_path = config_path or "config.yaml"
    config = DroidrunConfig.from_yaml(config_path)

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

            # Device overrides
            if device is not None:
                config.device.serial = device
            if tcp is not None:
                config.device.use_tcp = tcp

            # Tools overrides
            if allow_drag is not None:
                config.tools.allow_drag = allow_drag

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
            # STEP 1.5: Check Portal Setup (Android only)
            # ================================================================

            log_handler.update_step("Checking device setup...")

            # Determine device serial and TCP mode from config
            device_serial = device if device is not None else config.device.serial
            use_tcp = tcp if tcp is not None else config.device.use_tcp

            setup_ok, warning_msg = await check_portal_setup(
                device_serial=device_serial,
                use_tcp=use_tcp,
                rich_text=config.logging.rich_text,
                debug_mode=debug_mode,
                ios=ios,
            )

            if not setup_ok and warning_msg:
                # Display warning based on rich_text setting
                if config.logging.rich_text:
                    warning_panel = Panel(
                        f"⚠️  {warning_msg}",
                        border_style="yellow",
                        padding=(0, 1),
                    )
                    console.print(warning_panel)
                else:
                    console.print(f"\n⚠️  WARNING: {warning_msg}\n")

                logger.warning(warning_msg)
                logger.warning("Continuing anyway, but execution may fail...")

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

            droid_agent = DroidAgent(
                goal=command,
                llms=llm,
                config=config,
                timeout=1000,
                **droid_agent_kwargs,
            )

            # ================================================================
            # STEP 3: Run agent
            # ================================================================

            logger.info("▶️  Starting agent execution...")
            logger.info("Press Ctrl+C to stop")
            log_handler.update_step("Running agent...")

            try:
                handler = droid_agent.run()

                async for event in handler.stream_events():
                    log_handler.handle_event(event)
                result: ResultEvent = await handler
                return result.success

            except KeyboardInterrupt:
                log_handler.is_completed = True
                log_handler.is_success = False
                log_handler.current_step = "Stopped by user"
                logger.info("⏹️ Stopped by user")
                return False

            except Exception as e:
                log_handler.is_completed = True
                log_handler.is_success = False
                log_handler.current_step = f"Error: {e}"
                logger.error(f"💥 Error: {e}")
                if config.logging.debug:
                    import traceback

                    logger.debug(traceback.format_exc())
                return False

        except Exception as e:
            log_handler.current_step = f"Error: {e}"
            logger.error(f"💥 Setup error: {e}")
            debug_mode = debug if debug is not None else config.logging.debug
            if debug_mode:
                import traceback

                logger.debug(traceback.format_exc())
            return False


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
    tracing: bool | None,
    debug: bool | None,
    tcp: bool | None,
    save_trajectory: str | None,
    ios: bool | None,
):
    """Run a command on your Android device using natural language."""

    try:
        success = await run_command(
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
            tracing=tracing,
            debug=debug,
            tcp=tcp,
            temperature=temperature,
            save_trajectory=save_trajectory,
            ios=ios if ios is not None else False,
        )
    finally:
        # Disable DroidRun keyboard after execution
        # Note: Port forwards are managed automatically and persist until device disconnect
        try:
            if not (ios if ios is not None else False):
                device_obj = await adb.device(device)
                if device_obj:
                    await device_obj.shell(
                        "ime disable com.droidrun.portal/.DroidrunKeyboardIME"
                    )
        except Exception:
            click.echo("Failed to disable DroidRun keyboard")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


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
    device: str | None = None,
    steps: int | None = None,
    vision: bool | None = None,
    reasoning: bool | None = None,
    tracing: bool | None = None,
    debug: bool | None = None,
    use_tcp: bool | None = None,
    save_trajectory: str | None = None,
    allow_drag: bool | None = None,
    temperature: float | None = None,
    ios: bool = False,
):
    config = DroidrunConfig.from_yaml("config.yaml")

    # Initialize logging first (use config default if debug not specified)
    debug_mode = debug if debug is not None else config.logging.debug
    log_handler = configure_logging(command, debug_mode, config.logging.rich_text)
    logger = logging.getLogger("droidrun")

    log_handler.update_step("Initializing...")

    with log_handler.render():
        try:
            logger.info(f"🚀 Starting: {command}")

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

            # Tools overrides
            if allow_drag is not None:
                config.tools.allow_drag = allow_drag

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

            logger.info("▶️  Starting agent execution...")
            logger.info("Press Ctrl+C to stop")
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
    allow_drag = False
    asyncio.run(run_command(command, device="emulator-5556", reasoning=False))
