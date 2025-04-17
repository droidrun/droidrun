"""
DroidRun CLI - Command line interface for controlling Android devices through LLM agents.
"""

import asyncio
import click
import os
from rich.console import Console
from rich import print as rprint
from droidrun.tools import DeviceManager
from droidrun.agent import run_agent
from functools import wraps

# Import the install_app function directly for the setup command
from droidrun.tools.actions import install_app

console = Console()
device_manager = DeviceManager()

def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

# Define the run command as a standalone function to be used as both a command and default
@coro
async def run_command(command: str, device: str | None, provider: str, model: str, steps: int, vision: bool):
    """Run a command on your Android device using natural language."""
    console.print(f"[bold blue]Executing command:[/] {command}")
    
    # Auto-detect Gemini if model starts with "gemini-"
    if model and model.startswith("gemini-"):
        provider = "gemini"
    
    # Print vision status
    if vision:
        console.print("[blue]Vision capabilities are enabled.[/]")
    else:
        console.print("[blue]Vision capabilities are disabled.[/]")
    
    # Get API keys from environment variables
    api_key = None
    if provider.lower() == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            console.print("[bold red]Error:[/] OPENAI_API_KEY environment variable not set")
            return
        if not model:
            model = "gpt-4o-mini"
    elif provider.lower() == 'anthropic':
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            console.print("[bold red]Error:[/] ANTHROPIC_API_KEY environment variable not set")
            return
        if not model:
            model = "claude-3-sonnet-20240229"
    elif provider.lower() == 'gemini':
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            console.print("[bold red]Error:[/] GEMINI_API_KEY environment variable not set")
            return
        if not model:
            model = "gemini-2.0-flash"
    else:
        console.print(f"[bold red]Error:[/] Unsupported provider: {provider}")
        return
    
    try:
        # Try to find a device if none specified
        if not device:
            devices = await device_manager.list_devices()
            if not devices:
                console.print("[yellow]No devices connected.[/]")
                return
            
            device = devices[0].serial
            console.print(f"[blue]Using device:[/] {device}")
        
        # Set the device serial in the environment variable
        os.environ["DROIDRUN_DEVICE_SERIAL"] = device
        console.print(f"[blue]Set DROIDRUN_DEVICE_SERIAL to:[/] {device}")
        
        # Run the agent
        console.print("[bold blue]Running ReAct agent...[/]")
        console.print("[yellow]Press Ctrl+C to stop execution[/]")
        
        try:
            steps = await run_agent(
                task=command,
                device_serial=device,  # Still pass for backward compatibility
                llm_provider=provider,
                model_name=model,
                api_key=api_key,
                vision=vision
            )
            
            # Final message
            console.print(f"[bold green]Execution completed with {len(steps)} steps[/]")
        except ValueError as e:
            if "does not support vision" in str(e):
                console.print(f"[bold red]Vision Error:[/] {e}")
                console.print("[yellow]Please specify a vision-capable model with the --model flag.[/]")
                console.print("[blue]Recommended models:[/]")
                console.print("  - OpenAI: gpt-4o or gpt-4-vision")
                console.print("  - Anthropic: claude-3-opus-20240229 or claude-3-sonnet-20240229")
                console.print("  - Gemini: gemini-pro-vision")
                return
            else:
                raise  # Re-raise other ValueError exceptions
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")

# Custom Click multi-command class to handle both subcommands and default behavior
class DroidRunCLI(click.Group):
    def parse_args(self, ctx, args):
        # Check if the first argument might be a task rather than a command
        if args and not args[0].startswith('-') and args[0] not in self.commands:
            # Insert the 'run' command before the first argument if it's not a known command
            args.insert(0, 'run')
        return super().parse_args(ctx, args)

@click.group(cls=DroidRunCLI)
def cli():
    """DroidRun - Control your Android device through LLM agents."""
    pass

@cli.command()
@click.argument('command', type=str)
@click.option('--device', '-d', help='Device serial number or IP address', default=None)
@click.option('--provider', '-p', help='LLM provider (openai, anthropic, or gemini)', default='openai')
@click.option('--model', '-m', help='LLM model name', default=None)
@click.option('--steps', type=int, help='Maximum number of steps', default=15)
@click.option('--vision', is_flag=True, help='Enable vision capabilities')
def run(command: str, device: str | None, provider: str, model: str, steps: int, vision: bool):
    """Run a command on your Android device using natural language."""
    # Call our standalone function
    return run_command(command, device, provider, model, steps, vision)

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
    """Install the DroidRun Portal APK file onto the device."""
    try:
        # Check if APK file exists
        if not os.path.exists(path):
            console.print(f"[bold red]Error:[/] APK file not found at {path}")
            return
            
        # Try to find a device if none specified
        if not device:
            devices = await device_manager.list_devices()
            if not devices:
                console.print("[yellow]No devices connected.[/]")
                return
            
            device = devices[0].serial
            console.print(f"[blue]Using device:[/] {device}")
        
        # Set the device serial in the environment variable
        os.environ["DROIDRUN_DEVICE_SERIAL"] = device
        console.print(f"[blue]Set DROIDRUN_DEVICE_SERIAL to:[/] {device}")
        
        # Install the APK file
        console.print(f"[bold blue]Installing APK:[/] {path} onto {device}")
        result = await install_app(path, False, True, device)
        
        if "Error" in result:
            console.print(f"[bold red]Installation failed:[/] {result}")
            return
        else:
            console.print(f"[bold green]Installation successful![/]")
            # Directly call the start function to enable the service
            console.print("\n[bold blue]Proceeding to enable the accessibility service...[/]")
            await start(device=device)

    except Exception as e:
        console.print(f"[bold red]Error during setup:[/] {e}")
        import traceback
        traceback.print_exc()

@cli.command()
@click.option('--device', '-d', help='Device serial number or IP address', default=None)
@coro
async def start(device: str | None):
    """Enable the DroidRun Portal accessibility service"""
    service_component = "com.droidrun.portal/com.droidrun.portal.DroidrunPortalService"
    package = "com.droidrun.portal" # For messages

    try:
        # Try to find a device if none specified
        if not device:
            devices = await device_manager.list_devices()
            if not devices:
                console.print("[yellow]No devices connected.[/]")
                return
            
            device = devices[0].serial
            console.print(f"[blue]Using device:[/] {device}")

        # Get a device object for ADB commands
        device_obj = await device_manager.get_device(device)
        if not device_obj:
            console.print(f"[bold red]Error:[/] Could not get device object for {device}")
            return
        
        console.print(f"[bold blue]Attempting to enable DroidRun service on {device}...[/]")
        
        try:
            # Get current enabled services
            current_services_str = await device_obj._adb.shell(device, "settings get secure enabled_accessibility_services")
            current_services_str = current_services_str.strip() if current_services_str else ""
            
            # Check if already enabled
            if service_component in current_services_str.split(':'):
                console.print(f"[green]DroidRun service ({service_component}) is already enabled on {device}.[/]")
                # Ensure accessibility is globally enabled too
                await device_obj._adb.shell(device, "settings put secure accessibility_enabled 1")
                console.print("[green]Confirmed global accessibility setting is enabled.[/]")
                return

            # Add our service to the list
            if current_services_str:
                new_services_str = f"{current_services_str}:{service_component}"
            else:
                new_services_str = service_component

            # Use the exact command to enable the specific service
            await device_obj._adb.shell(device, f"settings put secure enabled_accessibility_services '{new_services_str}'")
            
            # Also enable accessibility services globally
            await device_obj._adb.shell(device, "settings put secure accessibility_enabled 1")
            
            console.print("[green]Accessibility service enabled successfully![/]")
            console.print("\n[bold green]DroidRun Portal is now active.[/]")
            
        except Exception as e:
            console.print(f"[yellow]Could not automatically enable accessibility service: {e}[/]")
            console.print("[yellow]Opening accessibility settings for manual configuration...[/]")
            
            # Fallback: Open the accessibility settings page
            await device_obj._adb.shell(device, "am start -a android.settings.ACCESSIBILITY_SETTINGS")
            
            console.print("\n[yellow]Please complete the following steps on your device:[/]")
            console.print(f"1. Find [bold]{package}[/] in the accessibility services list (may be under 'Downloaded apps')")
            console.print("2. Tap on the service name")
            console.print("3. Toggle the switch to [bold]ON[/] to enable the service")
            console.print("4. Accept any permission dialogs that appear")
            
            console.print("\n[yellow]Please manually enable the accessibility service using the steps above.")
            
    except Exception as e:
        console.print(f"[bold red]Error starting service:[/] {e}")
        import traceback
        traceback.print_exc()

@cli.command()
@click.option('--device', '-d', help='Device serial number or IP address', default=None)
@coro
async def stop(device: str | None):
    """Disable the DroidRun Portal accessibility service."""
    service_component = "com.droidrun.portal/com.droidrun.portal.DroidrunPortalService"
    
    try:
        # Try to find a device if none specified
        if not device:
            devices = await device_manager.list_devices()
            if not devices:
                console.print("[yellow]No devices connected.[/]")
                return
            
            device = devices[0].serial
            console.print(f"[blue]Using device:[/] {device}")
        
        # Get a device object for ADB commands
        device_obj = await device_manager.get_device(device)
        if not device_obj:
            console.print(f"[bold red]Error:[/] Could not get device object for {device}")
            return

        console.print(f"[bold blue]Attempting to disable DroidRun service on {device}...[/]")

        # Get the current list of enabled accessibility services
        current_services_str = await device_obj._adb.shell(device, "settings get secure enabled_accessibility_services")
        current_services_str = current_services_str.strip() if current_services_str else ""

        if not current_services_str or service_component not in current_services_str:
            console.print(f"[yellow]DroidRun service ({service_component}) is not currently enabled on {device}.[/]")
            return

        # Parse the list, remove our service, and join back
        services_list = [s for s in current_services_str.split(':') if s and s != service_component]
        new_services_str = ':'.join(services_list)

        # Update the setting
        if new_services_str:
            await device_obj._adb.shell(device, f"settings put secure enabled_accessibility_services '{new_services_str}'")
            console.print(f"[green]Successfully disabled DroidRun service by removing it from the enabled list.[/]")
        else:
            # If our service was the only one, clear the setting and potentially disable accessibility globally
            # Clearing the setting is safer than setting accessibility_enabled to 0, as the user might have enabled it for other reasons.
            await device_obj._adb.shell(device, "settings delete secure enabled_accessibility_services")
            # Optional: Disable global accessibility if needed, but might affect other things.
            # await device_obj._adb.shell(device, "settings put secure accessibility_enabled 0")
            console.print(f"[green]Successfully disabled DroidRun service. No other accessibility services were enabled.[/]")

    except Exception as e:
        console.print(f"[bold red]Error stopping service:[/] {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    cli() 