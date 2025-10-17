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
from droidrun.agent.common.events import (
    InputTextActionEvent,
    KeyPressActionEvent,
    StartAppEvent,
    SwipeActionEvent,
    TapActionEvent,
    DragActionEvent,
)
from droidrun.tools.tools import Tools
from droidrun.tools.credential_manager import get_credential_manager
from adbutils import adb
import requests
import base64

logger = logging.getLogger("droidrun-tools")
PORTAL_DEFAULT_TCP_PORT = 8080


class AdbTools(Tools):
    """Core UI interaction tools for Android device control."""

    def __init__(
        self,
        serial: str | None = None,
        use_tcp: bool = False,
        remote_tcp_port: int = PORTAL_DEFAULT_TCP_PORT,
    ) -> None:
        """Initialize the AdbTools instance.

        Args:
            serial: Device serial number
            use_tcp: Whether to use TCP communication (default: False)
            tcp_port: TCP port for communication (default: 8080)
        """
        self.device = adb.device(serial=serial)
        self.use_tcp = use_tcp
        self.remote_tcp_port = remote_tcp_port
        self.tcp_forwarded = False

        self._ctx = None
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
        # Trajectory saving level
        self.save_trajectories = "none"
        
        # Initialize credential manager
        self.credential_manager = get_credential_manager()

        self.setup_keyboard()

        # Set up TCP forwarding if requested
        if self.use_tcp:
            self.setup_tcp_forward()


    def setup_tcp_forward(self) -> bool:
        """
        Set up ADB TCP port forwarding for communication with the portal app.

        Returns:
            bool: True if forwarding was set up successfully, False otherwise
        """
        try:
            logger.debug(
                f"Setting up TCP port forwarding for port tcp:{self.remote_tcp_port} on device {self.device.serial}"
            )
            # Use adb forward command to set up port forwarding
            self.local_tcp_port = self.device.forward_port(self.remote_tcp_port)
            self.tcp_base_url = f"http://localhost:{self.local_tcp_port}"
            logger.debug(
                f"TCP port forwarding set up successfully to {self.tcp_base_url}"
            )

            # Test the connection with a ping
            try:
                response = requests.get(f"{self.tcp_base_url}/ping", timeout=5)
                if response.status_code == 200:
                    logger.debug("TCP connection test successful")
                    self.tcp_forwarded = True
                    return True
                else:
                    logger.warning(
                        f"TCP connection test failed with status: {response.status_code}"
                    )
                    return False
            except requests.exceptions.RequestException as e:
                logger.warning(f"TCP connection test failed: {e}")
                return False

        except Exception as e:
            logger.error(f"Failed to set up TCP port forwarding: {e}")
            self.tcp_forwarded = False
            return False

    def teardown_tcp_forward(self) -> bool:
        """
        Remove ADB TCP port forwarding.

        Returns:
            bool: True if forwarding was removed successfully, False otherwise
        """
        try:
            if self.tcp_forwarded:
                logger.debug(
                    f"Removing TCP port forwarding for port {self.local_tcp_port}"
                )
                # remove forwarding
                cmd = f"killforward:tcp:{self.local_tcp_port}"
                logger.debug(f"Removing TCP port forwarding: {cmd}")
                c = self.device.open_transport(cmd)
                c.close()

                self.tcp_forwarded = False
                logger.debug(f"TCP port forwarding removed")
                return True
            return True
        except Exception as e:
            logger.error(f"Failed to remove TCP port forwarding: {e}")
            return False

    def setup_keyboard(self) -> bool:
        """
        Set up the DroidRun keyboard as the default input method.
        Simple setup that just switches to DroidRun keyboard without saving/restoring.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            self.device.shell("ime enable com.droidrun.portal/.DroidrunKeyboardIME")
            self.device.shell("ime set com.droidrun.portal/.DroidrunKeyboardIME")
            logger.debug("DroidRun keyboard setup completed")
            return True

        except Exception as e:
            logger.error(f"Failed to setup DroidRun keyboard: {e}")
            return False

    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, "tcp_forwarded") and self.tcp_forwarded:
            self.teardown_tcp_forward()

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

    @Tools.ui_action
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
                    element_bounds=bounds_str,
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

    @Tools.ui_action
    def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: float = 300,
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

            if self._ctx:
                swipe_event = SwipeActionEvent(
                    action_type="swipe",
                    description=f"Swipe from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration_ms} milliseconds",
                    start_x=start_x,
                    start_y=start_y,
                    end_x=end_x,
                    end_y=end_y,
                    duration_ms=duration_ms,
                )
                self._ctx.write_event_to_stream(swipe_event)

            self.device.swipe(start_x, start_y, end_x, end_y, float(duration_ms / 1000))
            time.sleep(duration_ms / 1000)
            logger.debug(
                f"Swiped from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration_ms} milliseconds"
            )
            return True
        except ValueError as e:
            print(f"Error: {str(e)}")
            return False

    @Tools.ui_action
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
                    duration=duration,
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

    @Tools.ui_action
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

            if self.use_tcp and self.tcp_forwarded:
                # Use TCP communication
                encoded_text = base64.b64encode(text.encode()).decode()

                payload = {"base64_text": encoded_text}
                response = requests.post(
                    f"{self.tcp_base_url}/keyboard/input",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )

                logger.debug(
                     f"Keyboard input TCP response: {response.status_code}, {response.text}"
                )

                if response.status_code != 200:
                    return f"Error: HTTP request failed with status {response.status_code}: {response.text}"

            else:
                # Fallback to content provider method
                # Encode the text to Base64
                encoded_text = base64.b64encode(text.encode()).decode()

                cmd = f'content insert --uri "content://com.droidrun.portal/keyboard/input" --bind base64_text:s:"{encoded_text}"'
                self.device.shell(cmd)

            if self._ctx:
                input_event = InputTextActionEvent(
                    action_type="input_text",
                    description=f"Input text: '{text[:50]}{'...' if len(text) > 50 else ''}'",
                    text=text,
                )
                self._ctx.write_event_to_stream(input_event)

            logger.debug(
                f"Text input completed: {text[:50]}{'...' if len(text) > 50 else ''}"
            )
            return f"Text input completed: {text[:50]}{'...' if len(text) > 50 else ''}"

        except requests.exceptions.RequestException as e:
            return f"Error: TCP request failed: {str(e)}"
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error sending text input: {str(e)}"

    @Tools.ui_action
    def back(self) -> str:
        """
        Go back on the current view.
        This presses the Android back button.
        """
        try:
            logger.debug("Pressing key BACK")
            self.device.keyevent(4)

            if self._ctx:
                key_event = KeyPressActionEvent(
                    action_type="key_press",
                    description=f"Pressed key BACK",
                    keycode=4,
                    key_name="BACK",
                )
                self._ctx.write_event_to_stream(key_event)

            return f"Pressed key BACK"
        except ValueError as e:
            return f"Error: {str(e)}"

    @Tools.ui_action
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

            if self._ctx:
                key_event = KeyPressActionEvent(
                    action_type="key_press",
                    description=f"Pressed key {key_name}",
                    keycode=keycode,
                    key_name=key_name,
                )
                self._ctx.write_event_to_stream(key_event)

            logger.debug(f"Pressing key {key_name}")
            self.device.keyevent(keycode)
            logger.debug(f"Pressed key {key_name}")
            return f"Pressed key {key_name}"
        except ValueError as e:
            return f"Error: {str(e)}"

    @Tools.ui_action
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
                dumpsys_output = self.device.shell(
                    f"cmd package resolve-activity --brief {package}"
                )
                activity = dumpsys_output.splitlines()[1].split("/")[1]

            if self._ctx:
                start_app_event = StartAppEvent(
                    action_type="start_app",
                    description=f"Start app {package}",
                    package=package,
                    activity=activity,
                )
                self._ctx.write_event_to_stream(start_app_event)

            print(f"Activity: {activity}")

            self.device.app_start(package, activity)
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

    def take_screenshot(self, hide_overlay: bool = True) -> Tuple[str, bytes]:
        """
        Take a screenshot of the device.
        This function captures the current screen and adds the screenshot to context in the next message.
        Also stores the screenshot in the screenshots list with timestamp for later GIF creation.
        
        Args:
            hide_overlay: Whether to hide the overlay elements during screenshot (default: True)
        """
        try:
            logger.debug("Taking screenshot")
            img_format = "PNG"
            image_bytes = None

            if self.use_tcp and self.tcp_forwarded:
                # Add hideOverlay parameter to URL
                url = f"{self.tcp_base_url}/screenshot"
                if not hide_overlay:
                    url += "?hideOverlay=false"
                
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    tcp_response = response.json()
                    
                    # Check if response has the expected format with data field
                    if tcp_response.get("status") == "success" and "data" in tcp_response:
                        # Decode base64 string to bytes
                        base64_data = tcp_response["data"]
                        image_bytes = base64.b64decode(base64_data)
                        logger.debug("Screenshot taken via TCP")
                    else:
                        # Handle error response from server
                        error_msg = tcp_response.get("error", "Unknown error")
                        raise ValueError(f"Error taking screenshot via TCP: {error_msg}")
                else:
                    raise ValueError(f"Error taking screenshot via TCP: {response.status_code}")

            else:
                # Fallback to ADB screenshot method
                img = self.device.screenshot()
                img_buf = io.BytesIO()
                img.save(img_buf, format=img_format)
                image_bytes = img_buf.getvalue()
                logger.debug("Screenshot taken via ADB")

            # Store screenshot with timestamp
            self.screenshots.append(
                {
                    "timestamp": time.time(),
                    "image_data": image_bytes,
                    "format": img_format,
                }
            )
            return img_format, image_bytes

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error taking screenshot via TCP: {str(e)}")
        except ValueError as e:
            raise ValueError(f"Error taking screenshot: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error taking screenshot: {str(e)}")

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

    @Tools.ui_action
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

            if self.use_tcp and self.tcp_forwarded:
                # Use TCP communication
                response = requests.get(f"{self.tcp_base_url}/state", timeout=10)

                if response.status_code == 200:
                    tcp_response = response.json()

                    # Check if response has the expected format
                    if isinstance(tcp_response, dict) and "data" in tcp_response:
                        data_str = tcp_response["data"]
                        try:
                            combined_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            return {
                                "error": "Parse Error",
                                "message": "Failed to parse JSON data from TCP response data field",
                            }
                    else:
                        # Fallback: assume direct JSON format
                        combined_data = tcp_response
                else:
                    return {
                        "error": "HTTP Error",
                        "message": f"HTTP request failed with status {response.status_code}",
                    }
            else:
                # Fallback to content provider method
                adb_output = self.device.shell(
                    "content query --uri content://com.droidrun.portal/state",
                )

                state_data = self._parse_content_provider_output(adb_output)

                if state_data is None:
                    return {
                        "error": "Parse Error",
                        "message": "Failed to parse state data from ContentProvider response",
                    }

                if isinstance(state_data, dict):
                    data_str = None
                    if "data" in state_data:
                        data_str = state_data["data"]
                    
                    if data_str:
                        try:
                            combined_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            return {
                                "error": "Parse Error",
                                "message": "Failed to parse JSON data from ContentProvider response",
                            }
                    else:
                        return {
                            "error": "Format Error", 
                            "message": "Neither 'data' nor 'message' field found in ContentProvider response",
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

        except requests.exceptions.RequestException as e:
            return {
                "error": "TCP Error",
                "message": f"TCP request failed: {str(e)}",
            }
        except Exception as e:
            return {
                "error": str(e),
                "message": f"Error getting combined state: {str(e)}",
            }

    def ping(self) -> Dict[str, Any]:
        """
        Test the TCP connection using the /ping endpoint.

        Returns:
            Dictionary with ping result
        """
        try:
            if self.use_tcp and self.tcp_forwarded:
                response = requests.get(f"{self.tcp_base_url}/ping", timeout=5)

                if response.status_code == 200:
                    try:
                        tcp_response = response.json() if response.content else {}
                        logger.debug(f"Ping TCP response: {tcp_response}")
                        return {
                            "status": "success",
                            "message": "Ping successful",
                            "response": tcp_response,
                        }
                    except json.JSONDecodeError:
                        return {
                            "status": "success",
                            "message": "Ping successful (non-JSON response)",
                            "response": response.text,
                        }
                else:
                    return {
                        "status": "error",
                        "message": f"Ping failed with status {response.status_code}: {response.text}",
                    }
            else:
                return {
                    "status": "error",
                    "message": "TCP communication is not enabled",
                }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Ping failed: {str(e)}",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during ping: {str(e)}",
            }

    def fill_credentials(self, username: str = None, password: str = None, **kwargs) -> str:
        """
        Fill credentials using the credential manager.
        Supports placeholder resolution like {{USER_NAME}} and {{PASSWORD}}.
        
        Args:
            username: Username or placeholder (e.g., "{{USER_NAME}}")
            password: Password or placeholder (e.g., "{{PASSWORD}}")
            **kwargs: Additional credential key-value pairs
            
        Returns:
            Result message with resolved credentials (values masked for security)
        """
        try:
            resolved_credentials = {}
            missing_credentials = []
            
            # Process username
            if username:
                resolved_username = self.credential_manager.resolve_placeholders(username)
                if resolved_username == username and username.startswith("{{") and username.endswith("}}"):
                    missing_credentials.append(username)
                else:
                    resolved_credentials["username"] = resolved_username
            
            # Process password
            if password:
                resolved_password = self.credential_manager.resolve_placeholders(password)
                if resolved_password == password and password.startswith("{{") and password.endswith("}}"):
                    missing_credentials.append(password)
                else:
                    resolved_credentials["password"] = resolved_password
            
            # Process additional credentials
            for key, value in kwargs.items():
                resolved_value = self.credential_manager.resolve_placeholders(value)
                if resolved_value == value and value.startswith("{{") and value.endswith("}}"):
                    missing_credentials.append(value)
                else:
                    resolved_credentials[key] = resolved_value
            
            # Check for missing credentials
            if missing_credentials:
                return f"Error: Missing credentials for placeholders: {', '.join(missing_credentials)}. Available credentials: {', '.join(self.credential_manager.list_credentials())}"
            
            # Create response with masked values for security
            response_parts = []
            for key, value in resolved_credentials.items():
                if key.lower() in ['password', 'pass', 'pwd', 'secret', 'token', 'key']:
                    masked_value = "*" * len(value) if value else "None"
                    response_parts.append(f"{key}: {masked_value}")
                else:
                    response_parts.append(f"{key}: {value}")
            
            logger.debug(f"Resolved {len(resolved_credentials)} credentials")
            return f"Credentials resolved successfully: {' | '.join(response_parts)}"
            
        except Exception as e:
            logger.error(f"Error filling credentials: {e}")
            return f"Error filling credentials: {str(e)}"

    def get_credential_info(self) -> str:
        """
        Get information about available credentials.
        
        Returns:
            Information about available credentials (without exposing values)
        """
        try:
            info = self.credential_manager.get_credential_info()
            
            response_parts = []
            response_parts.append(f"Total credentials: {info['total_credentials']}")
            response_parts.append(f"Environment file: {info['env_file_path']}")
            response_parts.append(f"Environment file exists: {info['env_file_exists']}")
            
            if info['credential_keys']:
                # Group credentials by type for better organization
                cred_types = {
                    'user': [k for k in info['credential_keys'] if 'user' in k.lower()],
                    'pass': [k for k in info['credential_keys'] if any(x in k.lower() for x in ['pass', 'pwd'])],
                    'api': [k for k in info['credential_keys'] if 'api' in k.lower()],
                    'token': [k for k in info['credential_keys'] if 'token' in k.lower()],
                    'other': [k for k in info['credential_keys'] if not any(x in k.lower() for x in ['user', 'pass', 'pwd', 'api', 'token'])]
                }
                
                for cred_type, keys in cred_types.items():
                    if keys:
                        response_parts.append(f"{cred_type.title()} credentials: {', '.join(keys)}")
            else:
                response_parts.append("No credentials available")
            
            return " | ".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error getting credential info: {e}")
            return f"Error getting credential info: {str(e)}"

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
