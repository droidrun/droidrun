"""
UI Actions - Core UI interaction tools for Android device control.
"""

import os
import re
import json
import time
import tempfile
import asyncio
import aiofiles
import contextlib
from typing import Optional, Dict, Tuple, List, Any
from droidrun.adb import Device, DeviceManager

# Global variable to store clickable elements for index-based tapping
CLICKABLE_ELEMENTS_CACHE = []

# Default device serial will be read from environment variable
def get_device_serial() -> str:
    """Get the device serial from environment variable.
    
    Returns:
        Device serial from environment or None
    """
    return os.environ.get("DROIDRUN_DEVICE_SERIAL", "")

async def get_device() -> Optional[Device]:
    """Get the device instance using the serial from environment variable.
    
    Returns:
        Device instance or None if not found
    """
    serial = get_device_serial()
    if not serial:
        raise ValueError("DROIDRUN_DEVICE_SERIAL environment variable not set")
        
    device_manager = DeviceManager()
    device = await device_manager.get_device(serial)
    if not device:
        raise ValueError(f"Device {serial} not found")
    
    return device

def parse_package_list(output: str) -> List[Dict[str, str]]:
    """Parse the output of 'pm list packages -f' command.

    Args:
        output: Raw command output from 'pm list packages -f'

    Returns:
        List of dictionaries containing package info with 'package' and 'path' keys
    """
    apps = []
    for line in output.splitlines():
        if line.startswith("package:"):
            # Format is: "package:/path/to/base.apk=com.package.name"
            path_and_pkg = line[8:]  # Strip "package:"
            if "=" in path_and_pkg:
                path, package = path_and_pkg.rsplit("=", 1)
                apps.append({"package": package.strip(), "path": path.strip()})
    return apps

async def get_clickables(serial: Optional[str] = None) -> str:
    """
    Get all clickable UI elements from the device using the custom TopViewService.
    
    This function interacts with the TopViewService app installed on the device
    to capture UI elements. The service writes UI data to a JSON file on the device,
    which is then pulled to the host. If no elements are found initially, it will
    retry for up to 30 seconds.
    
    Args:
        serial: Optional device serial number
    
    Returns:
        JSON string containing UI elements extracted from the device screen
    """
    try:
        # Get the device
        if serial:
            from droidrun.adb import DeviceManager
            device_manager = DeviceManager()
            device = await device_manager.get_device(serial)
            if not device:
                raise ValueError(f"Device {serial} not found")
        else:
            device = await get_device()
        
        # Create a temporary file for the JSON
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            local_path = temp.name
        
        try:
            # Set retry parameters
            max_total_time = 30  # Maximum total time to try in seconds
            retry_interval = 1.0  # Time between retries in seconds
            start_total_time = asyncio.get_event_loop().time()
            
            while True:
                # Check if we've exceeded total time
                current_time = asyncio.get_event_loop().time()
                if current_time - start_total_time > max_total_time:
                    raise ValueError(f"Failed to get UI elements after {max_total_time} seconds of retries")
                
                # Clear logcat to make it easier to find our output
                await device._adb.shell(device._serial, "logcat -c")
                
                # Trigger the custom service via broadcast to get only interactive elements
                await device._adb.shell(device._serial, "am broadcast -a com.droidrun.portal.GET_ELEMENTS")
                
                # Poll for the JSON file path
                start_time = asyncio.get_event_loop().time()
                max_wait_time = 10  # Maximum wait time in seconds
                poll_interval = 0.2  # Check every 200ms
                
                device_path = None
                while asyncio.get_event_loop().time() - start_time < max_wait_time:
                    # Check logcat for the file path
                    logcat_output = await device._adb.shell(device._serial, "logcat -d | grep \"DROIDRUN_FILE\" | grep \"JSON data written to\" | tail -1")
                    
                    # Parse the file path if present
                    match = re.search(r"JSON data written to: (.*)", logcat_output)
                    if match:
                        device_path = match.group(1).strip()
                        break
                        
                    # Wait before polling again
                    await asyncio.sleep(poll_interval)
                
                # Check if we found the file path
                if not device_path:
                    await asyncio.sleep(retry_interval)
                    continue
                    
                # Pull the JSON file from the device
                await device._adb.pull_file(device._serial, device_path, local_path)
                
                # Read the JSON file
                async with aiofiles.open(local_path, "r", encoding="utf-8") as f:
                    json_content = await f.read()
                    
                # Try to parse the JSON
                try:
                    ui_data = json.loads(json_content)
                    
                    # Filter out the "type" attribute from all elements
                    filtered_data = []
                    for element in ui_data:
                        # Create a copy of the element without the "type" attribute
                        filtered_element = {k: v for k, v in element.items() if k != "type"}
                        
                        # Also filter children if present
                        if "children" in filtered_element:
                            filtered_element["children"] = [
                                {k: v for k, v in child.items() if k != "type"}
                                for child in filtered_element["children"]
                            ]
                        
                        filtered_data.append(filtered_element)
                    
                    # If we got elements, store them and return
                    if filtered_data:
                        # Store the filtered UI data in cache
                        global CLICKABLE_ELEMENTS_CACHE
                        CLICKABLE_ELEMENTS_CACHE = filtered_data
                        
                        # Add a small sleep to ensure UI is fully loaded/processed
                        await asyncio.sleep(0.5)  # 500ms sleep
                        
                        # Convert the dictionary to a JSON string before returning
                        result = {
                            "clickable_elements": filtered_data,
                            "count": len(filtered_data),
                            "message": f"Found {len(filtered_data)} UI elements after retrying"
                        }
                        
                        return result
                    
                    # If no elements found, wait and retry
                    await asyncio.sleep(retry_interval)
                    
                except json.JSONDecodeError:
                    # If JSON parsing failed, wait and retry
                    await asyncio.sleep(retry_interval)
                    continue
            
        except Exception as e:
            # Clean up in case of error
            with contextlib.suppress(OSError):
                os.unlink(local_path)
            raise ValueError(f"Error retrieving clickable elements: {e}")
            
    except Exception as e:
        raise ValueError(f"Error getting clickable elements: {e}")

async def tap_by_index(index: int, longpress: bool = False, serial: Optional[str] = None) -> str:
    """
    Tap on a UI element by its index.
    
    This function uses the cached clickable elements from the last get_clickables call
    to find the element with the given index and tap on its center coordinates.
    
    Args:
        index: Index of the element to tap
        serial: Optional device serial (for backward compatibility)
    
    Returns:
        Result message
    """
    
    def collect_all_indices(elements):
        """Recursively collect all indices from elements and their children."""
        indices = []
        for item in elements:
            if item.get('index') is not None:
                indices.append(item.get('index'))
            # Check children if present
            children = item.get('children', [])
            indices.extend(collect_all_indices(children))
        return indices

    def find_element_by_index(elements, target_index):
        """Recursively find an element with the given index."""
        for item in elements:
            if item.get('index') == target_index:
                return item
            # Check children if present
            children = item.get('children', [])
            result = find_element_by_index(children, target_index)
            if result:
                return result
        return None
    
    try:
        # Check if we have cached elements
        if not CLICKABLE_ELEMENTS_CACHE:
            return "Error: No UI elements cached. Call get_clickables first."
        
        # Find the element with the given index (including in children)
        element = find_element_by_index(CLICKABLE_ELEMENTS_CACHE, index)
        
        if not element:
            # List available indices to help the user
            indices = sorted(collect_all_indices(CLICKABLE_ELEMENTS_CACHE))
            indices_str = ", ".join(str(idx) for idx in indices[:20])
            if len(indices) > 20:
                indices_str += f"... and {len(indices) - 20} more"
                
            return f"Error: No element found with index {index}. Available indices: {indices_str}"
        
        # Get the bounds of the element
        bounds_str = element.get('bounds')
        if not bounds_str:
            element_text = element.get('text', 'No text')
            element_type = element.get('type', 'unknown')
            element_class = element.get('className', 'Unknown class')
            return f"Error: Element with index {index} ('{element_text}', {element_class}, type: {element_type}) has no bounds and cannot be tapped"
        
        # Parse the bounds (format: "left,top,right,bottom")
        try:
            left, top, right, bottom = map(int, bounds_str.split(','))
        except ValueError:
            return f"Error: Invalid bounds format for element with index {index}: {bounds_str}"
        
        # Calculate the center of the element
        x = (left + right) // 2
        y = (top + bottom) // 2
        
        # Get the device and tap at the coordinates
        if serial:
            from droidrun.adb import DeviceManager
            device_manager = DeviceManager()
            device = await device_manager.get_device(serial)
            if not device:
                return f"Error: Device {serial} not found"
        else:
            device = await get_device()
        
        if longpress == True:
            await device._adb.shell(device._serial, f'input swipe {x} {y} {x} {y} 1000')
        else:
            await device.tap(x, y)
        
        # Add a small delay to allow UI to update
        await asyncio.sleep(2)
        
        
        # Create a descriptive response
        response_parts = []
        response_parts.append(f"Tapped element with index {index}")
        response_parts.append(f"Text: '{element.get('text', 'No text')}'")
        response_parts.append(f"Class: {element.get('className', 'Unknown class')}")
        response_parts.append(f"Type: {element.get('type', 'unknown')}")
        
        # Add information about children if present
        children = element.get('children', [])
        if children:
            child_texts = [child.get('text') for child in children if child.get('text')]
            if child_texts:
                response_parts.append(f"Contains text: {' | '.join(child_texts)}")
        
        response_parts.append(f"Coordinates: ({x}, {y})")
        
        return " | ".join(response_parts)
    except ValueError as e:
        return f"Error: {str(e)}"


# Rename the old tap function to tap_by_coordinates for backward compatibility
async def tap_by_coordinates(x: int, y: int, serial: Optional[str] = None) -> str:
    """
    Tap on the device screen at specific coordinates.
    
    Args:
        x: X coordinate
        y: Y coordinate
        serial: Optional device serial (for backward compatibility)
    """
    try:
        if serial:
            device_manager = DeviceManager()
            device = await device_manager.get_device(serial)
            if not device:
                return f"Error: Device {serial} not found"
        else:
            device = await get_device()
        
        await device.tap(x, y)
        return f"Tapped at ({x}, {y})"
    except ValueError as e:
        return f"Error: {str(e)}"

# Replace the old tap function with the new one
async def tap(index: int, longpress: bool = False, serial: Optional[str] = None) -> str:
    """
    Tap on a UI element by its index.
    
    This function uses the cached clickable elements from the last get_clickables call
    to find the element with the given index and tap on its center coordinates.
    
    Args:
        index: Index of the element to tap
        serial: Optional device serial (for backward compatibility)
    
    Returns:
        Result message
    """
    return await tap_by_index(index, longpress, serial)

async def swipe(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration_ms: int = 300,
    serial: Optional[str] = None
) -> str:
    """
    Perform a swipe gesture on the device screen.
    
    Args:
        start_x: Starting X coordinate
        start_y: Starting Y coordinate
        end_x: Ending X coordinate
        end_y: Ending Y coordinate
        duration_ms: Duration of swipe in milliseconds
        serial: Optional device serial (for backward compatibility)
    """
    try:
        if serial:
            device_manager = DeviceManager()
            device = await device_manager.get_device(serial)
            if not device:
                return f"Error: Device {serial} not found"
        else:
            device = await get_device()
        
        await device.swipe(start_x, start_y, end_x, end_y, duration_ms)
        return f"Swiped from ({start_x}, {start_y}) to ({end_x}, {end_y})"
    except ValueError as e:
        return f"Error: {str(e)}"

async def input_text(text: str, serial: Optional[str] = None) -> str:
    """
    Input text on the device.
    
    Args:
        text: Text to input. Can contain spaces and special characters.
        serial: Optional device serial (for backward compatibility)
    """
    try:
        if serial:
            device_manager = DeviceManager()
            device = await device_manager.get_device(serial)
            if not device:
                return f"Error: Device {serial} not found"
        else:
            device = await get_device()
        
        # Function to escape special characters
        def escape_text(s: str) -> str:
            # Escape special characters that need shell escaping, excluding space
            special_chars = '[]()|&;$<>\\`"\'{}#^~'  # Removed ! from special chars
            escaped = ''
            for c in s:
                if c == ' ':
                    escaped += ' '  # Just add space without escaping
                elif c == '!':
                    escaped += '\\\\!'  # Double escape for exclamation mark
                elif c in special_chars:
                    escaped += '\\' + c
                else:
                    escaped += c
            return escaped
        

        # Select all text (Ctrl+A using keycombination)
        await device._adb.shell(device._serial, 'input keycombination 113 29')
        # Delete selected text
        await device._adb.shell(device._serial, 'input keyevent KEYCODE_DEL')
        
        # Split text into smaller chunks (max 500 chars)
        chunk_size = 500
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        for chunk in chunks:
            # Escape the text chunk
            escaped_chunk = escape_text(chunk)
            
            # Try different input methods if one fails
            methods = [
                f'input text "{escaped_chunk}"',  # Standard method
                f'am broadcast -a ADB_INPUT_TEXT --es msg "{escaped_chunk}"',  # Broadcast intent method
                f'input keyboard text "{escaped_chunk}"'  # Keyboard method
            ]
            
            success = False
            last_error = None
            
            for method in methods:
                try:
                    await device._adb.shell(device._serial, method)
                    success = True
                    break
                except Exception as e:
                    last_error = str(e)
                    continue
            
            if not success:
                return f"Error: Failed to input text chunk. Last error: {last_error}"
            
            # Small delay between chunks
            await asyncio.sleep(0.1)
        
        # Hide keyboard using back button
        await device._adb.shell(device._serial, 'input keyevent KEYCODE_BACK')

        #sleep to stabilize UI after keyboard dismissing
        await asyncio.sleep(2)
        
        return f"Text input completed: {text}"
    except ValueError as e:
        return f"Error: {str(e)}"

async def press_key(keycode: int, serial: Optional[str] = None) -> str:
    """
    Press a key on the device.
    
    Common keycodes:
    - 3: HOME
    - 4: BACK
    - 24: VOLUME UP
    - 25: VOLUME DOWN
    - 26: POWER
    - 82: MENU
    
    Args:
        keycode: Android keycode to press
        serial: Optional device serial (for backward compatibility)
    """
    try:
        if serial:
            device_manager = DeviceManager()
            device = await device_manager.get_device(serial)
            if not device:
                return f"Error: Device {serial} not found"
        else:
            device = await get_device()
        
        key_names = {
            3: "HOME",
            4: "BACK",
            24: "VOLUME UP",
            25: "VOLUME DOWN",
            26: "POWER",
            82: "MENU",
        }
        key_name = key_names.get(keycode, str(keycode))
        
        await device.press_key(keycode)
        return f"Pressed key {key_name}"
    except ValueError as e:
        return f"Error: {str(e)}"

async def start_app(
    package: str,
    activity: str = "",
    serial: Optional[str] = None
) -> str:
    """
    Start an app on the device.
    
    Args:
        package: Package name (e.g., "com.android.settings")
        activity: Optional activity name
        serial: Optional device serial (for backward compatibility)
    """
    try:
        if serial:
            device_manager = DeviceManager()
            device = await device_manager.get_device(serial)
            if not device:
                return f"Error: Device {serial} not found"
        else:
            device = await get_device()
        
        result = await device.start_app(package, activity)
        return result
    except ValueError as e:
        return f"Error: {str(e)}"

async def install_app(
    apk_path: str, 
    reinstall: bool = False, 
    grant_permissions: bool = True,
    serial: Optional[str] = None
) -> str:
    """
    Install an app on the device.
    
    Args:
        apk_path: Path to the APK file
        reinstall: Whether to reinstall if app exists
        grant_permissions: Whether to grant all permissions
        serial: Optional device serial (for backward compatibility)
    """
    try:
        if serial:
            device_manager = DeviceManager()
            device = await device_manager.get_device(serial)
            if not device:
                return f"Error: Device {serial} not found"
        else:
            device = await get_device()
        
        if not os.path.exists(apk_path):
            return f"Error: APK file not found at {apk_path}"
        
        result = await device.install_app(apk_path, reinstall, grant_permissions)
        return result
    except ValueError as e:
        return f"Error: {str(e)}"

async def uninstall_app(
    package: str, 
    keep_data: bool = False,
    serial: Optional[str] = None
) -> str:
    """
    Uninstall an app from the device.
    
    Args:
        package: Package name to uninstall
        keep_data: Whether to keep app data and cache
        serial: Optional device serial (for backward compatibility)
    """
    try:
        if serial:
            device_manager = DeviceManager()
            device = await device_manager.get_device(serial)
            if not device:
                return f"Error: Device {serial} not found"
        else:
            device = await get_device()
        
        result = await device.uninstall_app(package, keep_data)
        return result
    except ValueError as e:
        return f"Error: {str(e)}"

async def take_screenshot(serial: Optional[str] = None) -> Tuple[str, bytes]:
    """
    Take a screenshot of the device.
    
    Args:
        serial: Optional device serial (for backward compatibility)
    
    Returns:
        Tuple of (local file path, screenshot data as bytes)
    """
    try:
        if serial:
            device_manager = DeviceManager()
            device = await device_manager.get_device(serial)
            if not device:
                raise ValueError(f"Device {serial} not found")
        else:
            device = await get_device()
        
        return await device.take_screenshot()
    except ValueError as e:
        raise ValueError(f"Error taking screenshot: {str(e)}")

async def list_packages(
    include_system_apps: bool = False,
    serial: Optional[str] = None
) -> Dict[str, Any]:
    """
    List installed packages on the device.
    
    Args:
        include_system_apps: Whether to include system apps (default: False)
        serial: Optional device serial (for backward compatibility)
    
    Returns:
        Dictionary containing:
        - packages: List of dictionaries with 'package' and 'path' keys
        - count: Number of packages found
        - type: Type of packages listed ("all" or "non-system")
    """
    try:
        if serial:
            device_manager = DeviceManager()
            device = await device_manager.get_device(serial)
            if not device:
                raise ValueError(f"Device {serial} not found")
        else:
            device = await get_device()
        
        # Use the direct ADB command to get packages with paths
        cmd = ["pm", "list", "packages", "-f"]
        if not include_system_apps:
            cmd.append("-3")
            
        output = await device._adb.shell(device._serial, " ".join(cmd))
        
        # Parse the package list using the function
        packages = parse_package_list(output)
        package_type = "all" if include_system_apps else "non-system"
        
        return {
            "packages": packages,
            "count": len(packages),
            "type": package_type,
            "message": f"Found {len(packages)} {package_type} packages on the device"
        }
    except ValueError as e:
        raise ValueError(f"Error listing packages: {str(e)}")

async def complete(result: str) -> str:
    """Complete the task with a result message.
    
    Args:
        result: The result message
        
    Returns:
        Success message
    """
    return f"Task completed: {result}"

async def extract(filename: Optional[str] = None, serial: Optional[str] = None) -> str:
    """Extract and save the current UI state to a JSON file.
    
    This function captures the current UI state including all UI elements
    and saves it to a JSON file for later analysis or reference.
    
    Args:
        filename: Optional filename to save the UI state (defaults to ui_state_TIMESTAMP.json)
        serial: Optional device serial number
        
    Returns:
        Path to the saved JSON file
    """
    try:
        # Generate default filename if not provided
        if not filename:
            timestamp = int(time.time())
            filename = f"ui_state_{timestamp}.json"
        
        # Ensure the filename ends with .json
        if not filename.endswith(".json"):
            filename += ".json"
            
        # Get the UI elements
        ui_elements = await get_all_elements(serial)
        
        # Save to file
        save_path = os.path.abspath(filename)
        async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(ui_elements, indent=2))
            
        return f"UI state extracted and saved to {save_path}"
        
    except Exception as e:
        return f"Error extracting UI state: {e}"

async def get_all_elements(serial: Optional[str] = None) -> Dict[str, Any]:
    """
    Get all UI elements from the device, including non-interactive elements.
    
    This function interacts with the TopViewService app installed on the device
    to capture all UI elements, even those that are not interactive. This provides
    a complete view of the UI hierarchy for analysis or debugging purposes.
    
    Args:
        serial: Optional device serial number
    
    Returns:
        Dictionary containing all UI elements extracted from the device screen
    """
    try:
        # Get the device
        if serial:
            device_manager = DeviceManager()
            device = await device_manager.get_device(serial)
            if not device:
                raise ValueError(f"Device {serial} not found")
        else:
            device = await get_device()
        
        # Create a temporary file for the JSON
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            local_path = temp.name
        
        try:
            # Clear logcat to make it easier to find our output
            await device._adb.shell(device._serial, "logcat -c")
            
            # Trigger the custom service via broadcast to get ALL elements
            await device._adb.shell(device._serial, "am broadcast -a com.droidrun.portal.GET_ALL_ELEMENTS")
            
            # Poll for the JSON file path
            start_time = asyncio.get_event_loop().time()
            max_wait_time = 10  # Maximum wait time in seconds
            poll_interval = 0.2  # Check every 200ms
            
            device_path = None
            while asyncio.get_event_loop().time() - start_time < max_wait_time:
                # Check logcat for the file path
                logcat_output = await device._adb.shell(device._serial, "logcat -d | grep \"DROIDRUN_FILE\" | grep \"JSON data written to\" | tail -1")
                
                # Parse the file path if present
                match = re.search(r"JSON data written to: (.*)", logcat_output)
                if match:
                    device_path = match.group(1).strip()
                    break
                    
                # Wait before polling again
                await asyncio.sleep(poll_interval)
            
            # Check if we found the file path
            if not device_path:
                raise ValueError(f"Failed to find the JSON file path in logcat after {max_wait_time} seconds")
                
            # Pull the JSON file from the device
            await device._adb.pull_file(device._serial, device_path, local_path)
            
            # Read the JSON file
            async with aiofiles.open(local_path, "r", encoding="utf-8") as f:
                json_content = await f.read()
                
            # Clean up the temporary file
            with contextlib.suppress(OSError):
                os.unlink(local_path)
            
            # Try to parse the JSON
            import json
            try:
                ui_data = json.loads(json_content)
                
                return {
                    "all_elements": ui_data,
                    "count": len(ui_data) if isinstance(ui_data, list) else sum(1 for _ in ui_data.get("elements", [])),
                    "message": "Retrieved all UI elements from the device screen"
                }
            except json.JSONDecodeError:
                raise ValueError("Failed to parse UI elements JSON data")
            
        except Exception as e:
            # Clean up in case of error
            with contextlib.suppress(OSError):
                os.unlink(local_path)
            raise ValueError(f"Error retrieving all UI elements: {e}")
            
    except Exception as e:
        raise ValueError(f"Error getting all UI elements: {e}")


async def get_phone_state(serial: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the current phone state including current activity and screen size in pixels.
    
    Args:
        serial: Optional device serial number
    
    Returns:
        Dictionary with current phone state information including screen dimensions
    """
    try:
        # Get the device
        if serial:
            device_manager = DeviceManager()
            device = await device_manager.get_device(serial)
            if not device:
                raise ValueError(f"Device {serial} not found")
        else:
            device = await get_device()
        
        # Get the top resumed activity
        activity_output = await device._adb.shell(device._serial, "dumpsys activity activities | grep topResumedActivity")
        
        if not activity_output:
            # Try alternative command for older Android versions
            activity_output = await device._adb.shell(device._serial, "dumpsys activity activities | grep ResumedActivity")
        
        # Get screen size using wm size command
        screen_output = await device._adb.shell(device._serial, "wm size")
        
        # Process activity information
        current_activity = "Unable to determine current activity"
        if activity_output:
            current_activity = activity_output.strip()
        
        # Process screen size information
        screen_size = {"width": 0, "height": 0}
        if screen_output:
            # Parse output like "Physical size: 1080x2400"
            match = re.search(r"Physical size: (\d+)x(\d+)", screen_output)
            if match:
                screen_size["width"] = int(match.group(1))
                screen_size["height"] = int(match.group(2))
        
        # Return combined state
        return {
            "current_activity": current_activity,
            "screen_size": screen_size,
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": f"Error getting phone state: {str(e)}"
        }