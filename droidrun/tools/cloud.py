import logging
from typing import Dict, List, Tuple

from llama_index.core.workflow import Context
from mobilerun import AsyncMobilerun
from typing_extensions import Any
from droidrun.tools.geometry import find_clear_point, rects_overlap

from droidrun.tools.filters import ConciseFilter, DetailedFilter, TreeFilter
from droidrun.tools.formatters import IndexedFormatter, TreeFormatter
from droidrun.tools.tools import Tools

logger = logging.getLogger("droidrun")


class MobileRunTools(Tools):
    def __init__(
        self,
        device_id: str,
        display_id: int = 0,
        api_key: str | None = None,
        base_url: str = "https://api.mobilerun.com/v1",
        tree_filter: TreeFilter = None,
        tree_formatter: TreeFormatter = None,
        vision_enabled: bool = True,
        user_id: str | None = None,
    ):
        self.device_id = device_id
        self.display_id = display_id
        if not user_id:
            self.mobilerun = AsyncMobilerun(api_key=api_key, base_url=base_url, timeout=10.0)
        else:
            self.mobilerun = AsyncMobilerun(
                api_key="x", base_url=base_url, timeout=10.0, default_headers={"X-User-ID": user_id}
            )

        # Instance‐level cache for clickable elements (index-based tapping)
        self.clickable_elements_cache: List[Dict[str, Any]] = []
        self.reason = None
        self.success = None
        self.finished = False
        # Memory storage for remembering important information
        self.memory: List[str] = []

        if tree_filter:
            self.tree_filter = tree_filter
        else:
            self.tree_filter = ConciseFilter() if vision_enabled else DetailedFilter()
            logger.debug(
                f"Selected {self.tree_filter.__class__.__name__} (vision_enabled={vision_enabled})"
            )

        self.tree_formatter = tree_formatter or IndexedFormatter()

        # Caches
        self.raw_tree_cache = None
        self.filtered_tree_cache = None

    def _set_context(self, ctx: Context):
        self._ctx = ctx

    async def get_state(self) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any]]:
        combined_data = await self.mobilerun.devices.state.ui(
            self.device_id,
            x_device_display_id=self.display_id,
        )

        required_keys = ["a11y_tree", "phone_state", "device_context"]
        missing_keys = [key for key in required_keys if key not in combined_data]
        if missing_keys:
            raise Exception(f"Missing data in state: {', '.join(missing_keys)}")

        self.raw_tree_cache = combined_data["a11y_tree"]

        self.filtered_tree_cache = self.tree_filter.filter(
            self.raw_tree_cache, combined_data["device_context"]
        )

        formatted_text, focused_text, a11y_tree, phone_state = self.tree_formatter.format(
            self.filtered_tree_cache, combined_data["phone_state"]
        )

        self.clickable_elements_cache = a11y_tree

        return (formatted_text, focused_text, a11y_tree, phone_state)

    async def get_date(self) -> str:
        try:
            res = await self.mobilerun.devices.state.time(
                self.device_id, x_device_display_id=self.display_id
            )
        except Exception as e:
            print(f"Error: {str(e)}")
            return "unknown"
        return res

    @Tools.ui_action
    async def tap_by_index(self, index: int) -> str:
        try:
            x, y = self._extract_element_coordinates_by_index(index)
            await self.mobilerun.devices.actions.tap(
                self.device_id, x=x, y=y, x_device_display_id=self.display_id
            )
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
        return True

    @Tools.ui_action
    async def tap_on_index(self, index: int) -> str:
        """Tap at the largest visible region, avoiding overlapping elements."""
        try:
            def find_element_by_index(elements, target_index):
                for item in elements:
                    if item.get("index") == target_index:
                        return item
                    result = find_element_by_index(item.get("children", []), target_index)
                    if result:
                        return result
                return None

            def collect_all_elements(elements):
                result = []
                for item in elements:
                    result.append(item)
                    result.extend(collect_all_elements(item.get("children", [])))
                return result

            if not self.clickable_elements_cache:
                raise ValueError("No UI elements cached. Call get_state first.")

            element = find_element_by_index(self.clickable_elements_cache, index)
            if not element:
                raise ValueError(f"No element found with index {index}")

            bounds_str = element.get("bounds")
            if not bounds_str:
                raise ValueError(f"Element {index} has no bounds")

            target_bounds = tuple(map(int, bounds_str.split(",")))

            all_elements = collect_all_elements(self.clickable_elements_cache)
            blockers = []
            for el in all_elements:
                el_idx = el.get("index")
                el_bounds_str = el.get("bounds")
                if el_idx is not None and el_idx > index and el_bounds_str:
                    el_bounds = tuple(map(int, el_bounds_str.split(",")))
                    if rects_overlap(target_bounds, el_bounds):
                        blockers.append(el_bounds)

            point = find_clear_point(target_bounds, blockers)
            if not point:
                raise ValueError(f"Element {index} is fully obscured by overlapping elements")

            x, y = point
            await self.mobilerun.devices.actions.tap(
                self.device_id, x=x, y=y, x_device_display_id=self.display_id
            )
            print(f"Tapped element with index {index} at coordinates ({x}, {y})")

            response_parts = []
            response_parts.append(f"Tapped element with index {index}")
            response_parts.append(f"Text: '{element.get('text', 'No text')}'")
            response_parts.append(f"Class: {element.get('className', 'Unknown class')}")
            response_parts.append(f"Coordinates: ({x}, {y})")

            return " | ".join(response_parts)
        except Exception as e:
            return f"Error: {str(e)}"

    @Tools.ui_action
    async def swipe(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 300
    ) -> bool:
        try:
            await self.mobilerun.devices.actions.swipe(
                self.device_id,
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                duration=duration_ms,
                x_device_display_id=self.display_id,
            )
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
        return True

    @Tools.ui_action
    async def drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 3000
    ) -> bool:
        print("Error: Drag is not implemented yet")
        return False

    @Tools.ui_action
    async def input_text(self, text: str, index: int = -1, clear: bool = False) -> str:
        try:
            if index != -1:
                x, y = self._extract_element_coordinates_by_index(index)
                await self.mobilerun.devices.actions.tap(
                    self.device_id, x=x, y=y, x_device_display_id=self.display_id
                )

            await self.mobilerun.devices.keyboard.write(
                self.device_id, text=text, clear=clear, x_device_display_id=self.display_id
            )
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
        return True

    @Tools.ui_action
    async def back(self) -> str:
        return "use press_key(4) to go back"

    @Tools.ui_action
    async def press_key(self, keycode: int) -> str:
        """
        Press a key on the device.

        Args:
            keycode: The keycode to press. (Android KeyEvent code)

        Returns:
            A string indicating the keycode pressed.
        """
        if keycode == 4:
            ok = await self.global_action(1)
        elif keycode == 3:
            ok = await self.global_action(2)
        else:
            ok = None

        if ok is not None:
            return f"Pressed key {keycode}" if ok else f"Error: Failed to press key {keycode}"

        try:
            await self.mobilerun.devices.keyboard.key(
                self.device_id, action=keycode, x_device_display_id=self.display_id
            )
        except Exception as e:
            print(f"Error: {str(e)}")
            return f"Error: {str(e)}"
        return f"Pressed key {keycode}"

    @Tools.ui_action
    async def global_action(self, action: int) -> bool:
        """
        Press a key on the device.

        Args:
            keycode: The keycode to press.
            possible keycodes:
                Back	1
                Home	2
                Recents	3
                Notifications	4
                QuickSettings	5
                PowerDialog	6
                ToggleSplitScreen	7
                LockScreen	8
                TakeScreenshot	9
                KeycodeHeadsetHook	10
                AccessibilityButton	11
                AccessibilityButtonChooser	12
                AccessibilityShortcut	13
                AccessibilityAllApps	14
                DismissNotificationShade	15
                DpadUp	16
                DpadDown	17
                DpadLeft	18
                DpadRight	19
                DpadCenter	20

        Returns:
            A string indicating the keycode pressed.
        """
        try:
            await self.mobilerun.devices.actions.global_(
                self.device_id, action=action, x_device_display_id=self.display_id
            )
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
        return True

    @Tools.ui_action
    async def start_app(self, package: str, activity: str = "") -> str:
        logger.info(f"Starting app {package} with activity {activity}")
        try:
            if activity == "":
                act = None
            else:
                act = activity
            await self.mobilerun.devices.apps.start(
                package, device_id=self.device_id, activity=act, x_device_display_id=self.display_id
            )
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
        return True

    async def take_screenshot(self) -> Tuple[str, bytes]:
        try:
            # Use with_raw_response to get the raw httpx.Response object
            # instead of the parsed string content
            response = await self.mobilerun.devices.state.with_raw_response.screenshot(
                self.device_id, x_device_display_id=self.display_id
            )
            # Extract the raw binary content (PNG bytes)
            data = await response.read()
            return "PNG", data
        except Exception as e:
            print(f"Error: {str(e)}")
            return "PNG", b""

    @Tools.ui_action
    async def list_packages(self, include_system_apps: bool = False) -> List[str]:
        try:
            packages = await self.mobilerun.devices.packages.list(
                device_id=self.device_id,
                include_system_packages=include_system_apps,
                x_device_display_id=self.display_id,
            )
            return packages
        except Exception as e:
            print(f"Error: {str(e)}")
            return []

    @Tools.ui_action
    async def get_apps(self, include_system: bool = True) -> List[Dict[str, Any]]:
        try:
            apps = await self.mobilerun.devices.apps.list(
                device_id=self.device_id,
                include_system_apps=include_system,
                x_device_display_id=self.display_id,
            )
            return [app.model_dump() for app in apps]
        except Exception as e:
            print(f"Error: {str(e)}")
            return []

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

    @Tools.ui_action
    async def complete(self, success: bool, reason: str = "") -> None:
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

    def _extract_element_coordinates_by_index(self, index: int) -> Tuple[int, int]:
        """
        Extract center coordinates from an element by its index.

        Args:
            index: Index of the element to find and extract coordinates from

        Returns:
            Tuple of (x, y) center coordinates

        Raises:
            ValueError: If element not found, bounds format is invalid, or missing bounds
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

        # Check if we have cached elements
        if not self.clickable_elements_cache:
            raise ValueError("No UI elements cached. Call get_state first.")

        # Find the element with the given index (including in children)
        element = find_element_by_index(self.clickable_elements_cache, index)

        if not element:
            # List available indices to help the user
            indices = sorted(collect_all_indices(self.clickable_elements_cache))
            indices_str = ", ".join(str(idx) for idx in indices[:20])
            if len(indices) > 20:
                indices_str += f"... and {len(indices) - 20} more"
            raise ValueError(
                f"No element found with index {index}. Available indices: {indices_str}"
            )

        # Get the bounds of the element
        bounds_str = element.get("bounds")
        if not bounds_str:
            element_text = element.get("text", "No text")
            element_type = element.get("type", "unknown")
            element_class = element.get("className", "Unknown class")
            raise ValueError(
                f"Element with index {index} ('{element_text}', {element_class}, type: {element_type}) has no bounds and cannot be tapped"
            )

        # Parse the bounds (format: "left,top,right,bottom")
        try:
            left, top, right, bottom = map(int, bounds_str.split(","))
        except ValueError as e:
            raise ValueError(
                f"Invalid bounds format for element with index {index}: {bounds_str}"
            ) from e

        # Calculate the center of the element
        x = (left + right) // 2
        y = (top + bottom) // 2

        return x, y
