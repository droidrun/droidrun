"""
DroidRun CLI - Command line interface for controlling Android devices through LLM agents.
"""

import asyncio
import click
import os
import logging
import time
import sys
import contextlib
import warnings
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.spinner import Spinner
from rich.align import Align
from droidrun.tools import DeviceManager, Tools, load_tools
from droidrun.agent.droid import DroidAgent
from droidrun.agent.utils.llm_picker import load_llm
from droidrun.cli.event_handler import EventHandler
from functools import wraps

# Suppress all warnings
warnings.filterwarnings("ignore")

console = Console()
device_manager = DeviceManager()

current_step = "Initializing..."
spinner = Spinner("dots")

def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

def create_layout():
    """Create a layout with logs at top and status at bottom"""
    layout = Layout()
    layout.split(
        Layout(name="logs"),
        Layout(name="goal", size=3),
        Layout(name="status", size=3)
    )
    return layout

def update_layout(layout, log_list, step_message, current_time, goal=None, completed=False, success=None):
    """Update the layout with current logs and step information"""
    from rich.text import Text
    import shutil
    
    # Cache terminal size to avoid frequent recalculation
    try:
        terminal_height = shutil.get_terminal_size().lines
    except:
        terminal_height = 24  # fallback
    
    # Reserve space for panels and borders (more conservative estimate)
    other_components_height = 10  # goal panel + status panel + borders + padding
    available_log_lines = max(8, terminal_height - other_components_height)
    
    # Only show recent logs, but ensure we don't flicker
    visible_logs = log_list[-available_log_lines:] if len(log_list) > available_log_lines else log_list
    
    # Ensure we always have some content to prevent panel collapse
    if not visible_logs:
        visible_logs = ["Initializing..."]
    
    log_content = "\n".join(visible_logs)
    
    layout["logs"].update(Panel(
        log_content,
        title=f"Activity Log ({len(log_list)} entries)", 
        border_style="blue",
        title_align="left",
        padding=(0, 1),
        height=available_log_lines + 2
    ))
    
    if goal:
        goal_text = Text(goal, style="bold")
        layout["goal"].update(Panel(
            goal_text,
            title="Goal", 
            border_style="magenta",
            title_align="left",
            padding=(0, 1),
            height=3
        ))
    
    step_display = Text()
    
    if completed:
        if success:
            step_display.append("✓ ", style="bold green")
            panel_title = "Completed"
            panel_style = "green"
        else:
            step_display.append("✗ ", style="bold red")  
            panel_title = "Failed"
            panel_style = "red"
    else:
        step_display.append("⚡ ", style="bold yellow")
        panel_title = "Status"
        panel_style = "yellow"
    
    step_display.append(step_message)
    
    layout["status"].update(Panel(
        step_display, 
        title=panel_title, 
        border_style=panel_style,
        title_align="left",
        padding=(0, 1),
        height=3
    ))

@coro
async def run_command(command: str, device: str | None, provider: str, model: str, steps: int, api_base: str, reasoning: bool, tracing: bool, debug: bool, save_trajectory: bool = False, trajectory_dir: str = None, **kwargs):
    """Run a command on your Android device using natural language."""
    original_stderr = sys.stderr
    
    try:
        configure_logging(debug)
        
        global current_step
        current_step = "Initializing..."
        logs = []
        
        layout = create_layout()
        
        @contextlib.contextmanager
        def suppress_stderr():
            if not debug:
                class NullDevice:
                    def write(self, s): pass
                    def flush(self): pass
                sys.stderr = NullDevice()
            try:
                yield
            finally:
                if not debug:
                    sys.stderr = original_stderr
        
        with suppress_stderr(), Live(layout, refresh_per_second=4, console=console) as live:
            update_count = 0
            
            def update_display():
                nonlocal update_count
                update_count += 1
                
                if update_count % 2 != 0 and not event_handler.is_completed:
                    return
                    
                current_time = time.time()
                update_layout(
                    layout, 
                    logs, 
                    event_handler.current_step if 'event_handler' in locals() else current_step, 
                    current_time, 
                    goal=command, 
                    completed=event_handler.is_completed if 'event_handler' in locals() else False,
                    success=event_handler.is_success if 'event_handler' in locals() else None
                )
            
            event_handler = EventHandler(logs, debug=debug)
            
            try:
                logs.append(f"🚀 Starting: {command}")
                update_display()
                
                if not kwargs.get("temperature"):
                    kwargs["temperature"] = 0
                    
                current_step = "Setting up tools..."
                update_display()
                
                # Device setup
                if device is None:
                    logs.append("🔍 Finding connected device...")
                    update_display()
                    device_manager = DeviceManager()
                    devices = await device_manager.list_devices()
                    if not devices:
                        raise ValueError("No connected devices found.")
                    device = devices[0].serial
                    logs.append(f"📱 Using device: {device}")
                else:
                    logs.append(f"📱 Using device: {device}")

                update_display()

                # LLM setup
                current_step = "Initializing LLM..."
                event_handler.current_step = current_step
                update_display()
                llm = load_llm(provider_name=provider, model=model, api_base=api_base, **kwargs)
                logs.append(f"🧠 LLM ready: {provider}/{model}")

                # Agent setup
                current_step = "Initializing DroidAgent..."
                event_handler.current_step = current_step
                update_display()
                
                mode = "planning with reasoning" if reasoning else "direct execution"
                logs.append(f"🤖 Agent mode: {mode}")
                
                if tracing:
                    logs.append("🔍 Tracing enabled")
                
                droid_agent = DroidAgent(
                    goal=command,
                    llm=llm,
                    max_steps=steps,
                    timeout=1000,
                    max_retries=3,
                    reasoning=reasoning,
                    enable_tracing=tracing,
                    debug=debug,
                    device_serial=device
                )

                logs.append("▶️ Starting agent execution...")
                logs.append("Press Ctrl+C to stop")
                current_step = "Running agent..."
                event_handler.current_step = current_step
                update_display()

                try:
                    handler = droid_agent.run()
                    
                    async for event in handler.stream_events():
                        event_handler.handle_event(event)
                        update_display()
                    
                    result = await handler
                    
                    await asyncio.sleep(0.5)
                    update_display()
                    await asyncio.sleep(1.5)

                except KeyboardInterrupt:
                    logs.append("⏹️ Stopped by user")
                    event_handler.current_step = "Stopped by user"
                    event_handler.is_completed = True
                    event_handler.is_success = False
                    update_display()
                    
                except Exception as e:
                    logs.append(f"💥 Error: {e}")
                    event_handler.current_step = f"Error: {e}"
                    event_handler.is_completed = True
                    event_handler.is_success = False
                    if debug:
                        import traceback
                        logs.append(traceback.format_exc())
                    update_display()

                # Final pause
                await asyncio.sleep(1)

            except Exception as e:
                logs.append(f"💥 Setup error: {e}")
                event_handler.current_step = f"Error: {e}"
                if debug:
                    import traceback
                    logs.append(traceback.format_exc())
                
                update_display()
                await asyncio.sleep(1.5)
    
    finally:
        # Always restore original stderr
        sys.stderr = original_stderr

def configure_logging(debug: bool):
    """Configure logging verbosity based on debug flag."""
    warnings.filterwarnings("ignore")
    
    logging.getLogger().setLevel(logging.CRITICAL + 1)
    logging.getLogger().disabled = True
    
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.handlers = []
        logger.propagate = False
        logger.setLevel(logging.CRITICAL + 1)
    
    droidrun_logger = logging.getLogger("droidrun")
    droidrun_logger.disabled = True
    droidrun_logger.handlers = []
    droidrun_logger.propagate = False
    droidrun_logger.setLevel(logging.CRITICAL + 1)
    
    for logger_name in ["adb", "android", "asyncio", "urllib3", "requests", "httpx", "httpcore"]:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.handlers = []
        logger.propagate = False
        logger.setLevel(logging.CRITICAL + 1)
    
    logging.getLogger().handlers = []


class DroidRunCLI(click.Group):
    def parse_args(self, ctx, args):
        if args and not args[0].startswith('-') and args[0] not in self.commands:
            args.insert(0, 'run')
        return super().parse_args(ctx, args)

@click.group(cls=DroidRunCLI)
def cli():
    """DroidRun - Control your Android device through LLM agents."""
    pass

@cli.command()
@click.argument('command', type=str)
@click.option('--device', '-d', help='Device serial number or IP address', default=None)
@click.option('--provider', '-p', help='LLM provider (openai, ollama, anthropic, gemini, deepseek)', default='Gemini')
@click.option('--model', '-m', help='LLM model name', default="models/gemini-2.5-pro-preview-05-06")
@click.option('--temperature', type=float, help='Temperature for LLM', default=0.2)
@click.option('--steps', type=int, help='Maximum number of steps', default=15)
@click.option('--api_base', '-u', help='Base URL for API (e.g., OpenRouter or Ollama)', default=None)
@click.option('--reasoning/--no-reasoning', is_flag=True, help='Enable/disable planning with reasoning', default=False)
@click.option('--tracing', is_flag=True, help='Enable Arize Phoenix tracing', default=False)
@click.option('--debug', is_flag=True, help='Enable verbose debug logging', default=False)
@click.option('--save-trajectory', is_flag=True, help='Save agent trajectory to file', default=False)
@click.option('--trajectory-dir', help='Directory to save trajectory (default: "trajectories")', default="trajectories")
def run(command: str, device: str | None, provider: str, model: str, steps: int, api_base: str, temperature: float, reasoning: bool, tracing: bool, debug: bool, save_trajectory: bool, trajectory_dir: str):
    """Run a command on your Android device using natural language."""
    # Call our standalone function
    return run_command(command, device, provider, model, steps, api_base, reasoning, tracing, debug, temperature=temperature, save_trajectory=save_trajectory, trajectory_dir=trajectory_dir)

@cli.command()
@coro
async def devices():
    """List connected Android devices."""
    try:
        devices = await device_manager.list_devices()
        if not devices:
            console.print("[yellow]No devices connected.[/]")
            return

        console.print(f"[green]Found {len(devices)} connected device(s):[/]")
        for device in devices:
            console.print(f"  • [bold]{device.serial}[/]")
    except Exception as e:
        console.print(f"[red]Error listing devices: {e}[/]")

@cli.command()
@click.argument('ip_address')
@click.option('--port', '-p', default=5555, help='ADB port (default: 5555)')
@coro
async def connect(ip_address: str, port: int):
    """Connect to a device over TCP/IP."""
    try:
        device = await device_manager.connect(ip_address, port)
        if device:
            console.print(f"[green]Successfully connected to {ip_address}:{port}[/]")
        else:
            console.print(f"[red]Failed to connect to {ip_address}:{port}[/]")
    except Exception as e:
        console.print(f"[red]Error connecting to device: {e}[/]")

@cli.command()
@click.argument('serial')
@coro
async def disconnect(serial: str):
    """Disconnect from a device."""
    try:
        success = await device_manager.disconnect(serial)
        if success:
            console.print(f"[green]Successfully disconnected from {serial}[/]")
        else:
            console.print(f"[yellow]Device {serial} was not connected[/]")
    except Exception as e:
        console.print(f"[red]Error disconnecting from device: {e}[/]")

@cli.command()
@click.option('--path', required=True, help='Path to the APK file to install')
@click.option('--device', '-d', help='Device serial number or IP address', default=None)
@coro
async def setup(path: str, device: str | None):
    """Install an APK file and enable it as an accessibility service."""
    try:
        if not os.path.exists(path):
            console.print(f"[bold red]Error:[/] APK file not found at {path}")
            return
            
        if not device:
            devices = await device_manager.list_devices()
            if not devices:
                console.print("[yellow]No devices connected.[/]")
                return
            
            device = devices[0].serial
            console.print(f"[blue]Using device:[/] {device}")
        
        device_obj = await device_manager.get_device(device)
        if not device_obj:
            console.print(f"[bold red]Error:[/] Could not get device object for {device}")
            return
        tools = Tools(serial=device)
        console.print(f"[bold blue]Step 1/2: Installing APK:[/] {path}")
        result = await tools.install_app(path, False, True)
        
        if "Error" in result:
            console.print(f"[bold red]Installation failed:[/] {result}")
            return
        else:
            console.print(f"[bold green]Installation successful![/]")
        
        console.print(f"[bold blue]Step 2/2: Enabling accessibility service[/]")
        
        package = "com.droidrun.portal"
        
        try:
            await device_obj._adb.shell(device, "settings put secure enabled_accessibility_services com.droidrun.portal/com.droidrun.portal.DroidrunPortalService")
            
            await device_obj._adb.shell(device, "settings put secure accessibility_enabled 1")
            
            console.print("[green]Accessibility service enabled successfully![/]")
            console.print("\n[bold green]Setup complete![/] The DroidRun Portal is now installed and ready to use.")
            
        except Exception as e:
            console.print(f"[yellow]Could not automatically enable accessibility service: {e}[/]")
            console.print("[yellow]Opening accessibility settings for manual configuration...[/]")
            
            await device_obj._adb.shell(device, "am start -a android.settings.ACCESSIBILITY_SETTINGS")
            
            console.print("\n[yellow]Please complete the following steps on your device:[/]")
            console.print(f"1. Find [bold]{package}[/] in the accessibility services list")
            console.print("2. Tap on the service name")
            console.print("3. Toggle the switch to [bold]ON[/] to enable the service")
            console.print("4. Accept any permission dialogs that appear")
            
            console.print("\n[bold green]APK installation complete![/] Please manually enable the accessibility service using the steps above.")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    command = "Open the settings app"
    device = None
    provider = "GoogleGenAI"
    model = "models/gemini-2.5-flash-preview-05-20"
    temperature = 0
    api_key = os.getenv("GEMINI_API_KEY")
    steps = 15
    reasoning = True
    tracing = True
    debug = True
    base_url = None
    run_command(command=command, device=device, provider=provider, model=model, steps=steps, temperature=temperature, reasoning=reasoning, tracing=tracing, debug=debug, api_base=base_url, api_key=api_key)