from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import logging
from typing import Tuple, Dict, Callable, Any, Optional
from functools import wraps
import sys

# Get a logger for this module
logger = logging.getLogger(__name__)


class Tools(ABC):
    """
    Abstract base class for all tools.
    This class provides a common interface for all tools to implement.
    """

    @staticmethod
    def ui_action(func):
        """"
        Decorator to capture screenshots and UI states for actions that modify the UI.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            result = func(*args, **kwargs)
            
            # Check if save_trajectories attribute exists and is set to "action"
            if hasattr(self, 'save_trajectories') and self.save_trajectories == "action":
                frame = sys._getframe(1)
                caller_globals = frame.f_globals
                
                step_screenshots = caller_globals.get('step_screenshots')
                step_ui_states = caller_globals.get('step_ui_states')
                
                if step_screenshots is not None:
                    step_screenshots.append(self.take_screenshot()[1])
                if step_ui_states is not None:
                    step_ui_states.append(self.get_state())
            return result
        return wrapper

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the tool.
        """
        pass

    @abstractmethod
    def tap_by_index(self, index: int) -> str:
        """
        Tap the element at the given index.
        """
        pass

    #@abstractmethod
    #async def tap_by_coordinates(self, x: int, y: int) -> bool:
    #    pass

    @abstractmethod
    def swipe(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 300
    ) -> bool:
        """
        Swipe from the given start coordinates to the given end coordinates.
        """
        pass

    @abstractmethod
    def drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 3000
    ) -> bool:
        """
        Drag from the given start coordinates to the given end coordinates.
        """
        pass

    @abstractmethod
    def input_text(self, text: str) -> str:
        """
        Input the given text into a focused input field.
        """
        pass

    @abstractmethod
    def back(self) -> str:
        """
        Press the back button.
        """
        pass

    @abstractmethod
    def press_key(self, keycode: int) -> str:
        """
        Enter the given keycode.
        """
        pass

    @abstractmethod
    def start_app(self, package: str, activity: str = "") -> str:
        """
        Start the given app.
        """
        pass

    @abstractmethod
    def take_screenshot(self) -> Tuple[str, bytes]:
        """
        Take a screenshot of the device.
        """
        pass

    @abstractmethod
    def list_packages(self, include_system_apps: bool = False) -> List[str]:
        """
        List all packages on the device.
        """
        pass

    @abstractmethod
    def remember(self, information: str) -> str:
        """
        Remember the given information. This is used to store information in the tool's memory.
        """
        pass

    @abstractmethod
    def get_memory(self) -> List[str]:
        """
        Get the memory of the tool.
        """
        pass

    @abstractmethod
    def complete(self, success: bool, reason: str = "") -> None:
        """
        Complete the tool. This is used to indicate that the tool has completed its task.
        """
        pass


def describe_tools(tools: Tools, exclude_tools: Optional[List[str]] = None) -> Dict[str, Callable[..., Any]]:
    """
    Describe the tools available for the given Tools instance.

    Args:
        tools: The Tools instance to describe.
        exclude_tools: List of tool names to exclude from the description.

    Returns:
        A dictionary mapping tool names to their descriptions.
    """
    exclude_tools = exclude_tools or []

    description = {
        # UI interaction
        "swipe": tools.swipe,
        "input_text": tools.input_text,
        "press_key": tools.press_key,
        "tap_by_index": tools.tap_by_index,
        "drag": tools.drag,
        # App management
        "start_app": tools.start_app,
        "list_packages": tools.list_packages,
        # state management
        "remember": tools.remember,
        "complete": tools.complete,
    }

    # Remove excluded tools
    for tool_name in exclude_tools:
        description.pop(tool_name, None)

    return description
