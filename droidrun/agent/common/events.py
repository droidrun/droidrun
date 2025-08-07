from llama_index.core.workflow import Event


class ScreenshotEvent(Event):
    screenshot: bytes


class MacroEvent(Event):
    """Base class for coordinate-based action events"""

    action_type: str
    description: str


class TapActionEvent(MacroEvent):
    """Event for tap actions with coordinates"""

    x: int
    y: int
    element_index: int = None
    element_text: str = ""
    element_bounds: str = ""


class SwipeActionEvent(MacroEvent):
    """Event for swipe actions with coordinates"""

    start_x: int
    start_y: int
    end_x: int
    end_y: int
    """Duration of swipe in seconds"""
    duration: int


class InputTextActionEvent(MacroEvent):
    """Event for text input actions"""

    text: str


class KeyPressActionEvent(MacroEvent):
    """Event for key press actions"""

    keycode: int
    key_name: str = ""


class StartAppEvent(MacroEvent):
    """ "Event for starting an app"""

    package: str
    activity: str = ""


class DragActionEvent(MacroEvent):
    """Event for drag actions with coordinates"""

    start_x: int
    start_y: int
    end_x: int
    end_y: int
    """Duration of drag in seconds"""
    duration: int
