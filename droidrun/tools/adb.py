"""
UI Actions - Core UI interaction tools for Android device control.
"""

import os
import io
import json
import time
import logging
from llama_index.core.workflow import Context
from typing import Optional, Dict, Tuple, List, Any
from droidrun.tools.tools import Tools
from droidrun.agent.common.events import TapActionEvent, KeyPressActionEvent, SwipeActionEvent, InputTextActionEvent, StartAppEvent, DragActionEvent
from droidrun.tools.tools import Tools
from adbutils import adb

logger = logging.getLogger("droidrun-tools")


class AdbTools(Tools):
    """Core UI interaction tools for Android device control."""

    def __init__(self, serial: str | None = None) -> None:
        """Initialize the AdbTools instance.

        Args:
            serial: Device serial number
        """
        self._ctx = None
        self.device = adb.device(serial=serial)
        # Instance‐level cache for clickable elements (index-based tapping)
        self.clickable_elements_cache: List[Dict[str, Any]] = []
        self.last_screenshot = None
        self.reason = None
        self.success = None
        self.finished = False
        # Memory storage for remembering important information
        self.memory: List[str] = []
        # Store all screenshots with timestamps
        self.screenshots: List[Dict[str, Any]] = []
    
    def _set_context(self, ctx: Context):
        self._ctx = ctx

    def _parse_content_provider_output(
        self, raw_output: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse the raw ADB content provider output and extract JSON data.

        Args:
            raw_output (str): Raw output from ADB content query command

        Returns:
            dict: Parsed JSON data or None if parsing failed
        """
        # The ADB content query output format is: "Row: 0 result={json_data}"
        # We need to extract the JSON part after "result="
        lines = raw_output.strip().split("\n")

        for line in lines:
            line = line.strip()

            # Look for lines that contain "result=" pattern
            if "result=" in line:
                # Extract everything after "result="
                result_start = line.find("result=") + 7
                json_str = line[result_start:]

                try:
                    # Parse the JSON string
                    json_data = json.loads(json_str)
                    return json_data
                except json.JSONDecodeError:
                    continue

            # Fallback: try to parse lines that start with { or [
            elif line.startswith("{") or line.startswith("["):
                try:
                    json_data = json.loads(line)
                    return json_data
                except json.JSONDecodeError:
                    continue

        # If no valid JSON found in individual lines, try the entire output
        try:
            json_data = json.loads(raw_output.strip())
            return json_data
        except json.JSONDecodeError:
            return None

    def tap_by_index(self, index: int) -> str:
        """
        Tap on a UI element by its index.

        This function uses the cached clickable elements
        to find the element with the given index and tap on its center coordinates.

        Args:
            index: Index of the element to tap

        Returns:
            Result message
        """

        def collect_all_indices(elements):
            """Recursively collect all indices from elements and their children."""
            indices = []
            for item in elements:
                if item.get("index") is not None:
                    indices.append(item.get("index"))
                # Check children if present
                children = item.get("children", [])
                indices.extend(collect_all_indices(children))
            return indices

        def find_element_by_index(elements, target_index):
            """Recursively find an element with the given index."""
            for item in elements:
                if item.get("index") == target_index:
                    return item
                # Check children if present
                children = item.get("children", [])
                result = find_element_by_index(children, target_index)
                if result:
                    return result
            return None

        try:
            # Check if we have cached elements
            if not self.clickable_elements_cache:
                return "Error: No UI elements cached. Call get_state first."

            # Find the element with the given index (including in children)
            element = find_element_by_index(self.clickable_elements_cache, index)

            if not element:
                # List available indices to help the user
                indices = sorted(collect_all_indices(self.clickable_elements_cache))
                indices_str = ", ".join(str(idx) for idx in indices[:20])
                if len(indices) > 20:
                    indices_str += f"... and {len(indices) - 20} more"

                return f"Error: No element found with index {index}. Available indices: {indices_str}"

            # Get the bounds of the element
            bounds_str = element.get("bounds")
            if not bounds_str:
                element_text = element.get("text", "No text")
                element_type = element.get("type", "unknown")
                element_class = element.get("className", "Unknown class")
                return f"Error: Element with index {index} ('{element_text}', {element_class}, type: {element_type}) has no bounds and cannot be tapped"

            # Parse the bounds (format: "left,top,right,bottom")
            try:
                left, top, right, bottom = map(int, bounds_str.split(","))
            except ValueError:
                return f"Error: Invalid bounds format for element with index {index}: {bounds_str}"

            # Calculate the center of the element
            x = (left + right) // 2
            y = (top + bottom) // 2

            logger.debug(
                f"Tapping element with index {index} at coordinates ({x}, {y})"
            )
            # Get the device and tap at the coordinates
            self.device.click(x, y)
            logger.debug(f"Tapped element with index {index} at coordinates ({x}, {y})")

            # Emit coordinate action event for trajectory recording
            
            if self._ctx:
                element_text = element.get("text", "No text")
                element_class = element.get("className", "Unknown class")
                
                tap_event = TapActionEvent(
                    action_type="tap",
                    description=f"Tap element at index {index}: '{element_text}' ({element_class}) at coordinates ({x}, {y})",
                    x=x,
                    y=y,
                    element_index=index,
                    element_text=element_text,
                    element_bounds=bounds_str
                )
                self._ctx.write_event_to_stream(tap_event)

            # Add a small delay to allow UI to update
            time.sleep(0.5)

            # Create a descriptive response
            response_parts = []
            response_parts.append(f"Tapped element with index {index}")
            response_parts.append(f"Text: '{element.get('text', 'No text')}'")
            response_parts.append(f"Class: {element.get('className', 'Unknown class')}")
            response_parts.append(f"Type: {element.get('type', 'unknown')}")

            # Add information about children if present
            children = element.get("children", [])
            if children:
                child_texts = [
                    child.get("text") for child in children if child.get("text")
                ]
                if child_texts:
                    response_parts.append(f"Contains text: {' | '.join(child_texts)}")

            response_parts.append(f"Coordinates: ({x}, {y})")

            return " | ".join(response_parts)
        except ValueError as e:
            return f"Error: {str(e)}"

    # Rename the old tap function to tap_by_coordinates for backward compatibility
    def tap_by_coordinates(self, x: int, y: int) -> bool:
        """
        Tap on the device screen at specific coordinates.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Bool indicating success or failure
        """
        try:
            logger.debug(f"Tapping at coordinates ({x}, {y})")
            self.device.click(x, y)
            logger.debug(f"Tapped at coordinates ({x}, {y})")
            return True
        except ValueError as e:
            logger.debug(f"Error: {str(e)}")
            return False

    # Replace the old tap function with the new one
    def tap(self, index: int) -> str:
        """
        Tap on a UI element by its index.

        This function uses the cached clickable elements from the last get_clickables call
        to find the element with the given index and tap on its center coordinates.

        Args:
            index: Index of the element to tap

        Returns:
            Result message
        """
        return self.tap_by_index(index)

    def swipe(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.3
    ) -> bool:
        """
        Performs a straight-line swipe gesture on the device screen.
        To perform a hold (long press), set the start and end coordinates to the same values and increase the duration as needed.
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            duration: Duration of swipe in seconds
        Returns:
            Bool indicating success or failure
        """
        try:
            logger.debug(
                f"Swiping from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration} seconds"
            )
            self.device.swipe(start_x, start_y, end_x, end_y, duration)

            if self._ctx:
                swipe_event = SwipeActionEvent(
                    action_type="swipe",
                    description=f"Swipe from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration} seconds",
                    start_x=start_x,
                    start_y=start_y,
                    end_x=end_x,
                    end_y=end_y,
                    duration_ms=duration * 1000
                )
                self._ctx.write_event_to_stream(swipe_event)

            time.sleep(duration)
            logger.debug(
                f"Swiped from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration} seconds"
            )
            return True
        except ValueError as e:
            print(f"Error: {str(e)}")
            return False

    def drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 3
    ) -> bool:
        """
        Performs a straight-line drag and drop gesture on the device screen.
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            duration: Duration of swipe in seconds
        Returns:
            Bool indicating success or failure
        """
        try:
            logger.debug(
                f"Dragging from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration} seconds"
            )
            self.device.drag(start_x, start_y, end_x, end_y, duration)

            if self._ctx:
                drag_event = DragActionEvent(
                    action_type="drag",
                    description=f"Drag from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration} seconds",
                    start_x=start_x,
                    start_y=start_y,
                    end_x=end_x,
                    end_y=end_y,
                    duration_ms=duration * 1000
                )
                self._ctx.write_event_to_stream(drag_event)

            time.sleep(duration)
            logger.debug(
                f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration} seconds"
            )
            return True
        except ValueError as e:
            print(f"Error: {str(e)}")
            return False

    def input_text(self, text: str) -> str:
        """
        Input text on the device.
        Always make sure that the Focused Element is not None before inputting text.

        Args:
            text: Text to input. Can contain spaces, newlines, and special characters including non-ASCII.

        Returns:
            Result message
        """
        try:
            logger.debug(f"Inputting text: {text}")
            # Save the current keyboard
            original_ime = self.device.shell("settings get secure default_input_method")
            original_ime = original_ime.strip()

            # Enable the Droidrun keyboard
            self.device.shell("ime enable com.droidrun.portal/.DroidrunKeyboardIME")

            # Set the Droidrun keyboard as the default
            self.device.shell("ime set com.droidrun.portal/.DroidrunKeyboardIME")

            # Wait for keyboard to change
            time.sleep(1)

            # Encode the text to Base64
            import base64

            encoded_text = base64.b64encode(text.encode()).decode()

            cmd = f'content insert --uri "content://com.droidrun.portal/keyboard/input" --bind base64_text:s:"{encoded_text}"'
            self.device.shell(cmd)

            # Wait for text input to complete
            time.sleep(0.5)

            if self._ctx:
                input_event = InputTextActionEvent(
                    action_type="input_text",
                    description=f"Input text: '{text[:50]}{'...' if len(text) > 50 else ''}'",
                    text=text
                )
                self._ctx.write_event_to_stream(input_event)

            # Restore the original keyboard
            if original_ime and "com.droidrun.portal" not in original_ime:
                self.device.shell(f"ime set {original_ime}")

            logger.debug(
                f"Text input completed: {text[:50]}{'...' if len(text) > 50 else ''}"
            )
            return f"Text input completed: {text[:50]}{'...' if len(text) > 50 else ''}"
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error sending text input: {str(e)}"

    def back(self) -> str:
        """
        Go back on the current view.
        This presses the Android back button.
        """
        try:
            logger.debug("Pressing key BACK")
            self.device.keyevent(3)
            return f"Pressed key BACK"
        except ValueError as e:
            return f"Error: {str(e)}"

    def press_key(self, keycode: int) -> str:
        """
        Press a key on the Android device.

        Common keycodes:
        - 3: HOME
        - 4: BACK
        - 66: ENTER
        - 67: DELETE

        Args:
            keycode: Android keycode to press
        """
        try:
            key_names = {
                66: "ENTER",
                4: "BACK",
                3: "HOME",
                67: "DELETE",
            }
            key_name = key_names.get(keycode, str(keycode))

            logger.debug(f"Pressing key {key_name}")
            self.device.keyevent(keycode)
            if self._ctx:
                key_event = KeyPressActionEvent(
                    action_type="key_press",
                    description=f"Pressed key {key_name}",
                    keycode=keycode,
                    key_name=key_name
                )
                self._ctx.write_event_to_stream(key_event)
            logger.debug(f"Pressed key {key_name}")
            return f"Pressed key {key_name}"
        except ValueError as e:
            return f"Error: {str(e)}"

    def start_app(self, package: str, activity: str | None = None) -> str:
        """
        Start an app on the device.

        Args:
            package: Package name (e.g., "com.android.settings")
            activity: Optional activity name
        """
        try:
            logger.debug(f"Starting app {package} with activity {activity}")
            if not activity:
                # Find launcher activity from dumpsys
                dumpsys_output = self.device.shell(f"cmd package resolve-activity --brief {package}") 
                activity = dumpsys_output.splitlines()[1].split("/")[1]

            print(f"Activity: {activity}")

            self.device.app_start(package, activity)

            if self._ctx:
                start_app_event = StartAppEvent(
                    action_type="start_app",
                    description=f"Start app {package}",
                    package=package,
                    activity=activity
                )
                self._ctx.write_event_to_stream(start_app_event)

            logger.debug(f"App started: {package} with activity {activity}")
            return f"App started: {package} with activity {activity}"
        except Exception as e:
            return f"Error: {str(e)}"

    def install_app(
        self, apk_path: str, reinstall: bool = False, grant_permissions: bool = True
    ) -> str:
        """
        Install an app on the device.

        Args:
            apk_path: Path to the APK file
            reinstall: Whether to reinstall if app exists
            grant_permissions: Whether to grant all permissions
        """
        try:
            if not os.path.exists(apk_path):
                return f"Error: APK file not found at {apk_path}"

            logger.debug(
                f"Installing app: {apk_path} with reinstall: {reinstall} and grant_permissions: {grant_permissions}"
            )
            result = self.device.install(
                apk_path,
                nolaunch=True,
                uninstall=reinstall,
                flags=["-g"] if grant_permissions else [],
                silent=True,
            )
            logger.debug(f"Installed app: {apk_path} with result: {result}")
            return result
        except ValueError as e:
            return f"Error: {str(e)}"

    def take_screenshot(self) -> Tuple[str, bytes]:
        """
        Take a screenshot of the device.
        This function captures the current screen and adds the screenshot to context in the next message.
        Also stores the screenshot in the screenshots list with timestamp for later GIF creation.
        """
        try:
            logger.debug("Taking screenshot")
            img = self.device.screenshot()
            img_buf = io.BytesIO()
            img_format = "PNG"
            img.save(img_buf, format=img_format)
            logger.debug("Screenshot taken")

            # Store screenshot with timestamp
            self.screenshots.append(
                {
                    "timestamp": time.time(),
                    "image_data": img_buf.getvalue(),
                    "format": img_format,  # Usually 'PNG'
                }
            )
            return img_format, img_buf.getvalue()
        except ValueError as e:
            raise ValueError(f"Error taking screenshot: {str(e)}")

    def list_packages(self, include_system_apps: bool = False) -> List[str]:
        """
        List installed packages on the device.

        Args:
            include_system_apps: Whether to include system apps (default: False)

        Returns:
            List of package names
        """
        try:
            logger.debug("Listing packages")
            return self.device.list_packages(["-3"] if not include_system_apps else [])
        except ValueError as e:
            raise ValueError(f"Error listing packages: {str(e)}")

    def complete(self, success: bool, reason: str = ""):
        """
        Mark the task as finished.

        Args:
            success: Indicates if the task was successful.
            reason: Reason for failure/success
        """
        if success:
            self.success = True
            self.reason = reason or "Task completed successfully."
            self.finished = True
        else:
            self.success = False
            if not reason:
                raise ValueError("Reason for failure is required if success is False.")
            self.reason = reason
            self.finished = True

    def remember(self, information: str) -> str:
        """
        Store important information to remember for future context.

        This information will be extracted and included into your next steps to maintain context
        across interactions. Use this for critical facts, observations, or user preferences
        that should influence future decisions.

        Args:
            information: The information to remember

        Returns:
            Confirmation message
        """
        if not information or not isinstance(information, str):
            return "Error: Please provide valid information to remember."

        # Add the information to memory
        self.memory.append(information.strip())

        # Limit memory size to prevent context overflow (keep most recent items)
        max_memory_items = 10
        if len(self.memory) > max_memory_items:
            self.memory = self.memory[-max_memory_items:]

        return f"Remembered: {information}"

    def get_memory(self) -> List[str]:
        """
        Retrieve all stored memory items.

        Returns:
            List of stored memory items
        """
        return self.memory.copy()

    def get_state(self, serial: Optional[str] = None) -> Dict[str, Any]:
        """
        Get both the a11y tree and phone state in a single call using the combined /state endpoint.

        Args:
            serial: Optional device serial number

        Returns:
            Dictionary containing both 'a11y_tree' and 'phone_state' data
        """

        try:
            logger.debug("Getting state")
            adb_output = self.device.shell(
                "content query --uri content://com.droidrun.portal/state",
            )

            state_data = self._parse_content_provider_output(adb_output)

            if state_data is None:
                return {
                    "error": "Parse Error",
                    "message": "Failed to parse state data from ContentProvider response",
                }

            if isinstance(state_data, dict) and "data" in state_data:
                data_str = state_data["data"]
                try:
                    combined_data = json.loads(data_str)
                except json.JSONDecodeError:
                    return {
                        "error": "Parse Error",
                        "message": "Failed to parse JSON data from ContentProvider data field",
                    }
            else:
                return {
                    "error": "Format Error",
                    "message": f"Unexpected state data format: {type(state_data)}",
                }

            # Validate that both a11y_tree and phone_state are present
            if "a11y_tree" not in combined_data:
                return {
                    "error": "Missing Data",
                    "message": "a11y_tree not found in combined state data",
                }

            if "phone_state" not in combined_data:
                return {
                    "error": "Missing Data",
                    "message": "phone_state not found in combined state data",
                }

            # Filter out the "type" attribute from all a11y_tree elements
            elements = combined_data["a11y_tree"]
            filtered_elements = []
            for element in elements:
                # Create a copy of the element without the "type" attribute
                filtered_element = {k: v for k, v in element.items() if k != "type"}

                # Also filter children if present
                if "children" in filtered_element:
                    filtered_element["children"] = [
                        {k: v for k, v in child.items() if k != "type"}
                        for child in filtered_element["children"]
                    ]

                filtered_elements.append(filtered_element)

            self.clickable_elements_cache = filtered_elements

            return {
                "a11y_tree": filtered_elements,
                "phone_state": combined_data["phone_state"],
            }

        except Exception as e:
            return {
                "error": str(e),
                "message": f"Error getting combined state: {str(e)}",
            }


def _shell_test_cli(serial: str, command: str) -> tuple[str, float]:
    """
    Run an adb shell command using the adb CLI and measure execution time.
    Args:
        serial: Device serial number
        command: Shell command to run
    Returns:
        Tuple of (output, elapsed_time)
    """
    import time
    import subprocess

    adb_cmd = ["adb", "-s", serial, "shell", command]
    start = time.perf_counter()
    result = subprocess.run(adb_cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    output = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    return output, elapsed


def _shell_test():
    device = adb.device("emulator-5554")
    # Native Python adb client
    start = time.time()
    res = device.shell("echo 'Hello, World!'")
    end = time.time()
    print(f"[Native] Shell execution took {end - start:.3f} seconds: {res}")

    start = time.time()
    res = device.shell("content query --uri content://com.droidrun.portal/state")
    end = time.time()
    print(f"[Native] Shell execution took {end - start:.3f} seconds: phone_state")

    # CLI version
    output, elapsed = _shell_test_cli("emulator-5554", "echo 'Hello, World!'")
    print(f"[CLI] Shell execution took {elapsed:.3f} seconds: {output}")

    output, elapsed = _shell_test_cli(
        "emulator-5554", "content query --uri content://com.droidrun.portal/state"
    )
    print(f"[CLI] Shell execution took {elapsed:.3f} seconds: phone_state")


def _list_packages():
    tools = AdbTools()
    print(tools.list_packages())


def _start_app():
    tools = AdbTools()
    tools.start_app("com.android.settings", ".Settings")


if __name__ == "__main__":
    _start_app()
