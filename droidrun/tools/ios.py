"""
UI Actions - Core UI interaction tools for iOS device control.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from droidrun.tools.tools import Tools

logger = logging.getLogger("IOS")

SYSTEM_BUNDLE_IDENTIFIERS = [
    "ai.droidrun.droidrun-ios-portal",
    "com.apple.Bridge",
    "com.apple.DocumentsApp",
    "com.apple.Fitness",
    "com.apple.Health",
    "com.apple.Maps",
    "com.apple.MobileAddressBook",
    "com.apple.MobileSMS",
    "com.apple.Passbook",
    "com.apple.Passwords",
    "com.apple.Preferences",
    "com.apple.PreviewShell",
    "com.apple.mobilecal",
    "com.apple.mobilesafari",
    "com.apple.mobileslideshow",
    "com.apple.news",
    "com.apple.reminders",
    "com.apple.shortcuts",
    "com.apple.webapp",
]


class IOSTools(Tools):
    """Core UI interaction tools for iOS device control."""

    def __init__(self, url: str, bundle_identifiers: List[str] | None = None) -> None:
        """Initialize the IOSTools instance.

        Args:
            url: iOS device URL. This is the URL of the iOS device. It is used to send requests to the iOS device.
            bundle_identifiers: List of bundle identifiers to include in the list of packages
        """

        self.clickable_elements_cache: List[Dict[str, Any]] = []
        self.url = url
        self.last_screenshot = None
        self.reason = None
        self.success = None
        self.finished = False
        self.memory: List[str] = []
        self.screenshots: List[Dict[str, Any]] = []
        self.last_tapped_rect: Optional[str] = (
            None  # Store last tapped element's rect for text input
        )
        self.bundle_identifiers = bundle_identifiers or []
        logger.info(f"iOS device URL: {url}")

    def get_state(self) -> List[Dict[str, Any]]:
        """
        Get all clickable UI elements from the iOS device using accessibility API.

        Returns:
            List of dictionaries containing UI elements extracted from the device screen
        """
        try:
            a11y_url = f"{self.url}/vision/a11y"
            response = requests.get(a11y_url)

            if response.status_code == 200:
                a11y_data = response.json()

                # Parse the iOS accessibility tree format
                elements = self._parse_ios_accessibility_tree(
                    a11y_data["accessibilityTree"]
                )

                # Cache the elements for tap_by_index usage
                self.clickable_elements_cache = elements

                return {
                    "a11y_tree": self.clickable_elements_cache,
                    "phone_state": self._get_phone_state(),
                }
            else:
                logger.error(
                    f"Failed to get accessibility data: HTTP {response.status_code}"
                )
                raise ValueError(
                    f"Failed to get accessibility data: HTTP {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error getting clickable elements: {e}")
            # raise ValueError(f"Error getting clickable elements: {e}")

    def _parse_ios_accessibility_tree(self, a11y_data: str) -> List[Dict[str, Any]]:
        """
        Parse iOS accessibility tree format into structured elements.

        Args:
            a11y_data: Raw accessibility data from iOS device

        Returns:
            List of parsed UI elements with coordinates and properties
        """
        elements = []
        lines = a11y_data.strip().split("\n")

        # Track current element index
        element_index = 0

        for line in lines:
            # Skip empty lines and header lines
            if (
                not line.strip()
                or line.startswith("Attributes:")
                or line.startswith("Element subtree:")
                or line.startswith("Path to element:")
                or line.startswith("Query chain:")
            ):
                continue

            # Parse UI elements - look for lines with coordinates
            # Format: ElementType, {{x, y}, {width, height}}, [optional properties]
            coord_match = re.search(
                r"\{\{([0-9.]+),\s*([0-9.]+)\},\s*\{([0-9.]+),\s*([0-9.]+)\}\}", line
            )

            if coord_match:
                x, y, width, height = map(float, coord_match.groups())

                # Extract element type (the text before the first comma)
                element_type_match = re.match(r"\s*(.+?),", line)
                element_type = (
                    element_type_match.group(1).strip()
                    if element_type_match
                    else "Unknown"
                )

                # Remove leading arrows and spaces
                element_type = re.sub(r"^[→\s]+", "", element_type)

                # Extract label if present
                label_match = re.search(r"label:\s*'([^']*)'", line)
                label = label_match.group(1) if label_match else ""

                # Extract identifier if present
                identifier_match = re.search(r"identifier:\s*'([^']*)'", line)
                identifier = identifier_match.group(1) if identifier_match else ""

                # Extract placeholder value if present
                placeholder_match = re.search(r"placeholderValue:\s*'([^']*)'", line)
                placeholder = placeholder_match.group(1) if placeholder_match else ""

                # Extract value if present
                value_match = re.search(r"value:\s*([^,}]+)", line)
                value = value_match.group(1).strip() if value_match else ""

                # Calculate rect string for iOS tap API (x,y,width,height format)
                rect_str = f"{x},{y},{width},{height}"

                # Create element dictionary
                element = {
                    "index": element_index,
                    "type": element_type,
                    "className": element_type,
                    "text": label or identifier or placeholder or "",
                    "label": label,
                    "identifier": identifier,
                    "placeholder": placeholder,
                    "value": value,
                    "bounds": f"{x},{y},{x+width},{y+height}",  # left,top,right,bottom format for compatibility
                    "rect": rect_str,  # x,y,width,height format for iOS API
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "center_x": x + width / 2,
                    "center_y": y + height / 2,
                }

                # Only include interactive elements (buttons, text fields, etc.)
                interactive_types = [
                    "Button",
                    "SearchField",
                    "TextField",
                    "Cell",
                    "Switch",
                    "Slider",
                    "Stepper",
                    "Picker",
                    "Link",
                ]
                if any(
                    interactive_type in element_type
                    for interactive_type in interactive_types
                ):
                    elements.append(element)
                    element_index += 1

        return elements

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

        def find_element_by_index(elements, target_index):
            """Find an element with the given index."""
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
                return "Error: No UI elements cached. Call get_clickables first."

            # Find the element with the given index
            element = find_element_by_index(self.clickable_elements_cache, index)

            if not element:
                # List available indices to help the user
                indices = [
                    elem.get("index")
                    for elem in self.clickable_elements_cache
                    if elem.get("index") is not None
                ]
                indices_str = ", ".join(str(idx) for idx in sorted(indices)[:20])
                if len(indices) > 20:
                    indices_str += f"... and {len(indices) - 20} more"

                return f"Error: No element found with index {index}. Available indices: {indices_str}"

            # Get the element coordinates
            x = element.get("x", 0)
            y = element.get("y", 0)
            width = element.get("width", 0)
            height = element.get("height", 0)

            if not all(
                [x is not None, y is not None, width is not None, height is not None]
            ):
                element_text = element.get("text", "No text")
                element_class = element.get("className", "Unknown class")
                return f"Error: Element with index {index} ('{element_text}', {element_class}) has no coordinates and cannot be tapped"

            # Format rect in iOS format: {{x,y},{w,h}}
            ios_rect = f"{{{{{x},{y}}},{{{width},{height}}}}}"

            # Store the rect for potential text input (keep in simple format for text input)
            self.last_tapped_rect = f"{x},{y},{width},{height}"

            # Make the tap request
            tap_url = f"{self.url}/gestures/tap"
            payload = {"rect": ios_rect, "count": 1, "longPress": False}

            logger.info(f"payload {payload}")

            response = requests.post(tap_url, json=payload)
            if response.status_code == 200:
                # Add a small delay to allow UI to update
                time.sleep(0.5)

                # Create a descriptive response
                response_parts = []
                response_parts.append(f"Tapped element with index {index}")
                response_parts.append(f"Text: '{element.get('text', 'No text')}'")
                response_parts.append(
                    f"Class: {element.get('className', 'Unknown class')}"
                )
                response_parts.append(f"Rect: {ios_rect}")

                return " | ".join(response_parts)
            else:
                return f"Error: Failed to tap element. HTTP {response.status_code}"

        except Exception as e:
            return f"Error: {str(e)}"

    """def tap_by_coordinates(self, x: int, y: int) -> bool:
         # Format rect in iOS format: {{x,y},{w,h}}
        width = 1
        height = 1
        ios_rect = f"{{{{{x},{y}}},{{{width},{height}}}}}"

        # Make the tap request
        tap_url = f"{self.url}/gestures/tap"
        payload = {"rect": ios_rect, "count": 1, "longPress": False}

        logger.info(f"payload {payload}")

        response = requests.post(tap_url, json=payload)
        if response.status_code == 200:
            return True
        else:
            return False"""

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
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 300
    ) -> bool:
        """
        Performs a straight-line swipe gesture on the device screen.
        To perform a hold (long press), set the start and end coordinates to the same values and increase the duration as needed.
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            duration_ms: Duration of swipe in milliseconds (not used in iOS API)
        Returns:
            Bool indicating success or failure
        """
        try:
            # Calculate swipe direction based on coordinates
            dx = end_x - start_x
            dy = end_y - start_y

            # Determine primary direction
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "down" if dy > 0 else "up"

            swipe_url = f"{self.url}/gestures/swipe"
            payload = {"x": float(start_x), "y": float(start_y), "dir": direction}

            response = requests.post(swipe_url, json=payload)
            if response.status_code == 200:
                logger.info(
                    f"Swiped from ({start_x}, {start_y}) to ({end_x}, {end_y}) direction: {direction}"
                )
                return True
            else:
                logger.error(f"Failed to swipe: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error performing swipe: {e}")
            return False

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 3000,
    ) -> bool:
        """
        Drag from the given start coordinates to the given end coordinates.
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            duration_ms: Duration of swipe in milliseconds
        Returns:
            Bool indicating success or failure
        """
        # TODO: implement this
        logger.info("Drag action FAILED! Not implemented for iOS")
        return False

    def input_text(self, text: str) -> str:
        """
        Input text on the iOS device.

        Args:
            text: Text to input. Can contain spaces, newlines, and special characters including non-ASCII.

        Returns:
            Result message
        """
        try:
            # Use the last tapped element's rect if available, otherwise use a default
            rect = self.last_tapped_rect if self.last_tapped_rect else "0,0,100,100"

            type_url = f"{self.url}/inputs/type"
            payload = {"rect": rect, "text": text}

            response = requests.post(type_url, json=payload)
            if response.status_code == 200:
                time.sleep(0.5)  # Wait for text input to complete
                return f"Text input completed: {text[:50]}{'...' if len(text) > 50 else ''}"
            else:
                return f"Error: Failed to input text. HTTP {response.status_code}"

        except Exception as e:
            return f"Error sending text input: {str(e)}"

    def back(self) -> str:
        raise NotImplementedError("Back is not yet implemented for iOS")

    def press_key(self, keycode: int) -> str:
        # TODO: refactor this. its not about physical keys but BACK, ENTER, DELETE, etc.
        """
        Press a key on the iOS device.

        iOS Key codes:
        - 0: HOME
        - 4: ACTION
        - 5: CAMERA

        Args:
            keycode: iOS keycode to press
        """
        try:
            key_names = {0: "HOME", 4: "ACTION", 5: "CAMERA"}
            key_name = key_names.get(keycode, str(keycode))

            key_url = f"{self.url}/inputs/key"
            payload = {"key": keycode}

            response = requests.post(key_url, json=payload)
            if response.status_code == 200:
                return f"Pressed key {key_name}"
            else:
                return f"Error: Failed to press key. HTTP {response.status_code}"

        except Exception as e:
            return f"Error pressing key: {str(e)}"

    def start_app(self, package: str, activity: str = "") -> str:
        """
        Start an app on the iOS device.

        Args:
            package: Bundle identifier (e.g., "com.apple.MobileSMS")
            activity: Optional activity name (not used on iOS)
        """
        try:
            launch_url = f"{self.url}/inputs/launch"
            payload = {"bundleIdentifier": package}

            response = requests.post(launch_url, json=payload)
            if response.status_code == 200:
                time.sleep(1.0)  # Wait for app to launch
                return f"Successfully launched app: {package}"
            else:
                return f"Error: Failed to launch app {package}. HTTP {response.status_code}"

        except Exception as e:
            return f"Error launching app: {str(e)}"

    def take_screenshot(self) -> Tuple[str, bytes]:
        """
        Take a screenshot of the iOS device.
        This function captures the current screen and adds the screenshot to context in the next message.
        Also stores the screenshot in the screenshots list with timestamp for later GIF creation.
        """
        try:
            screenshot_url = f"{self.url}/vision/screenshot"
            response = requests.get(screenshot_url)

            if response.status_code == 200:
                screenshot_data = response.content

                # Store screenshot with timestamp
                screenshot_info = {
                    "timestamp": time.time(),
                    "data": screenshot_data,
                }
                self.screenshots.append(screenshot_info)
                self.last_screenshot = screenshot_data

                logger.info(
                    f"Screenshot captured successfully, size: {len(screenshot_data)} bytes"
                )
                return ("PNG", screenshot_data)
            else:
                logger.error(
                    f"Failed to capture screenshot: HTTP {response.status_code}"
                )
                raise ValueError(
                    f"Failed to capture screenshot: HTTP {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            raise ValueError(f"Error taking screenshot: {str(e)}") from e

    def _get_phone_state(self) -> Dict[str, Any]:
        """
        Get the current phone state including current activity and keyboard visibility.

        Returns:
            Dictionary with current phone state information
        """
        try:
            # For iOS, we can get some state info from the accessibility API
            a11y_url = f"{self.url}/vision/state"
            response = requests.get(a11y_url)

            if response.status_code == 200:
                state_data = response.json()

                return {
                    "current_activity": state_data["activity"],
                    "keyboard_shown": state_data["keyboardShown"],
                }
            else:
                return {
                    "error": f"Failed to get device state: HTTP {response.status_code}",
                    "current_activity": "Unknown",
                    "keyboard_shown": False,
                }

        except Exception as e:
            return {"error": str(e), "message": f"Error getting phone state: {str(e)}"}

    def list_packages(self, include_system_apps: bool = True) -> List[str]:
        all_packages = set(self.bundle_identifiers)
        if include_system_apps:
            all_packages.update(SYSTEM_BUNDLE_IDENTIFIERS)
        return sorted(list(all_packages))

    def remember(self, information: str) -> str:
        """
        Store important information to remember for future context.

        This information will be included in future LLM prompts to help maintain context
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
