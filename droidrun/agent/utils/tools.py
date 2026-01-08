import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from droidrun.config_manager.config_manager import DeviceConfig, ToolsConfig
    from droidrun.tools import Tools

from droidrun.config_manager import ClickMode

from async_adbutils import adb
from droidrun.agent.common.events import WaitEvent
from droidrun.agent.oneflows.app_starter_workflow import AppStarter
from droidrun.config_manager.config_manager import ToolsConfig
from droidrun.tools import AdbTools, IOSTools
from droidrun.tools.tools import Tools


async def create_tools_from_config(
    device_config: "DeviceConfig", vision_enabled: bool = True
) -> "Tools":
    """
    Create Tools instance from DeviceConfig.

    Args:
        device_config: Device configuration
        vision_enabled: Whether vision is enabled (for filter selection)

    Returns:
        AdbTools or IOSTools based on config

    Raises:
        ValueError: If no device found or invalid platform
    """
    is_ios = device_config.platform.lower() == "ios"
    device_serial = device_config.serial

    if not is_ios:
        # Android: auto-detect if not specified
        if device_serial is None:
            devices = await adb.list()
            if not devices:
                raise ValueError("No connected Android devices found.")
            device_serial = devices[0].serial
        return AdbTools(serial=device_serial, vision_enabled=vision_enabled)
    else:
        # iOS: require explicit device URL
        if device_serial is None:
            raise ValueError("iOS device URL required in config.device.serial")
        return IOSTools(url=device_serial)


async def resolve_tools_instance(
    tools: "Tools | ToolsConfig | None",
    device_config: "DeviceConfig",
    tools_config_fallback: "ToolsConfig | None" = None,
    credential_manager=None,
    vision_enabled: bool = True,
    click_mode: Optional[ClickMode] = None,
) -> tuple["Tools", "ToolsConfig"]:
    """
    Resolve Tools instance and ToolsConfig from various input types.

    This helper allows flexible initialization:
    - Pass a Tools instance directly (custom or pre-configured)
    - Pass a ToolsConfig to create Tools from device_config
    - Pass None to use defaults

    Args:
        tools: Either a Tools instance, ToolsConfig, or None
        device_config: Device configuration for creating Tools if needed
        tools_config_fallback: Fallback ToolsConfig when tools is a Tools instance or None
        credential_manager: Optional credential manager to attach to Tools
        vision_enabled: Whether vision is enabled (default: True)
        click_mode: Click mode for formatter selection (default: INDEX)

    Returns:
        Tuple of (tools_instance, tools_config):
        - If tools is Tools instance: (tools, tools_config_fallback or default)
        - If tools is ToolsConfig: (created from device_config, tools)
        - If tools is None: (created from device_config, tools_config_fallback or default)

    Example:
        >>> # Use custom Tools instance
        >>> custom_tools = AdbTools(serial="emulator-5554")
        >>> tools_instance, tools_cfg = resolve_tools_instance(custom_tools, device_config)
        >>>
        >>> # Use ToolsConfig (current behavior)
        >>> tools_cfg = ToolsConfig(disabled_tools=["long_press"])
        >>> tools_instance, tools_cfg = resolve_tools_instance(tools_cfg, device_config)
    """
    from droidrun.tools.formatters import IndexedFormatter, CoordinateFormatter

    # Select formatter based on click_mode
    if click_mode in (ClickMode.COORDINATE, ClickMode.HYBRID):
        tree_formatter = CoordinateFormatter()
    else:
        tree_formatter = IndexedFormatter()

    # Case 1: Tools instance provided directly
    if isinstance(tools, Tools):
        tools_instance = tools
        # Update formatter if not already set appropriately
        if hasattr(tools_instance, 'tree_formatter'):
            tools_instance.tree_formatter = tree_formatter
        # Use fallback or default ToolsConfig
        tools_cfg = tools_config_fallback if tools_config_fallback else ToolsConfig()

    # Case 2: ToolsConfig provided
    elif tools is not None and isinstance(tools, ToolsConfig):
        tools_instance = await create_tools_from_config(
            device_config, vision_enabled=vision_enabled
        )
        tools_instance.tree_formatter = tree_formatter
        tools_cfg = tools

    # Case 3: None provided
    else:
        tools_instance = await create_tools_from_config(
            device_config, vision_enabled=vision_enabled
        )
        tools_instance.tree_formatter = tree_formatter
        tools_cfg = tools_config_fallback if tools_config_fallback else ToolsConfig()

    # Attach credential manager if provided
    if credential_manager:
        tools_instance.credential_manager = credential_manager

    return tools_instance, tools_cfg


async def click(index: int, *, tools: "Tools" = None, **kwargs) -> str:
    """
    Click the element with the given index.

    Args:
        index: The index of the element to click
        tools: The Tools instance (injected automatically)

    Returns:
        Result message from the tap operation
    """
    if tools is None:
        raise ValueError("tools parameter is required")
    return await tools.tap_on_index(index)


async def long_press(index: int, *, tools: "Tools" = None, **kwargs) -> bool:
    """
    Long press the element with the given index.

    Args:
        index: The index of the element to long press
        tools: The Tools instance (injected automatically)

    Returns:
        True if successful, False otherwise
    """
    if tools is None:
        raise ValueError("tools parameter is required")
    x, y = tools._extract_element_coordinates_by_index(index)
    return await tools.swipe(x, y, x, y, 1000)


async def type(
    text: str, index: int, clear: bool = False, *, tools: "Tools" = None, **kwargs
) -> str:
    """
    Type the given text into the element with the given index.

    Args:
        text: The text to type
        index: The index of the element to type into
        clear: Whether to clear existing text before typing (default: False)
        tools: The Tools instance (injected automatically)

    Returns:
        Result message from the input operation
    """
    if tools is None:
        raise ValueError("tools parameter is required")
    return await tools.input_text(text, index, clear=clear)


async def system_button(button: str, *, tools: "Tools" = None, **kwargs) -> str:
    """
    Press a system button (back, home, or enter).

    Args:
        button: The button name (case insensitive): "back", "home", or "enter"
        tools: The Tools instance (injected automatically)

    Returns:
        Result message from the key press operation
    """
    if tools is None:
        raise ValueError("tools parameter is required")

    # Map button names to keycodes (case insensitive)
    button_map = {
        "back": 4,
        "home": 3,
        "enter": 66,
    }

    button_lower = button.lower()
    if button_lower not in button_map:
        return (
            f"Error: Unknown system button '{button}'. Valid options: back, home, enter"
        )

    keycode = button_map[button_lower]
    return await tools.press_key(keycode)


async def swipe(
    coordinate: List[int],
    coordinate2: List[int],
    duration: float = 1.0,
    *,
    tools: "Tools" = None,
    **kwargs,
) -> bool:
    """
    Swipe from one coordinate to another.

    Args:
        coordinate: Starting coordinate as [x, y]
        coordinate2: Ending coordinate as [x, y]
        duration: Duration of swipe in seconds (default: 1.0)
        tools: The Tools instance (injected automatically)

    Returns:
        True if successful, False otherwise
    """
    if tools is None:
        raise ValueError("tools parameter is required")

    if not isinstance(coordinate, list) or len(coordinate) != 2:
        raise ValueError(f"coordinate must be a list of 2 integers, got: {coordinate}")
    if not isinstance(coordinate2, list) or len(coordinate2) != 2:
        raise ValueError(
            f"coordinate2 must be a list of 2 integers, got: {coordinate2}"
        )

    start_x, start_y = coordinate
    end_x, end_y = coordinate2

    # Convert seconds to milliseconds
    duration_ms = int(duration * 1000)

    return await tools.swipe(start_x, start_y, end_x, end_y, duration_ms=duration_ms)


async def open_app(text: str, *, tools: "Tools" = None, **kwargs) -> str:
    """
    Open an app by its name.

    Args:
        text: The name of the app to open
        tools: The Tools instance (injected automatically)

    Returns:
        Result message from opening the app
    """
    if tools is None:
        raise ValueError("tools parameter is required")

    # Get LLM from tools instance
    if tools.app_opener_llm is None:
        raise RuntimeError(
            "app_opener_llm not configured. "
            "provide app_opener_llm when initializing Tools."
        )

    # Create workflow instance
    workflow = AppStarter(
        tools=tools,
        llm=tools.app_opener_llm,
        timeout=60,
        stream=tools.streaming,
        verbose=False,
    )

    # Run workflow to open an app
    result = await workflow.run(app_description=text)
    await asyncio.sleep(1)
    return result


async def wait(duration: float = 1.0, *, tools: "Tools" = None, **kwargs) -> str:
    """
    Wait for a specified duration in seconds.

    Args:
        duration: Duration to wait in seconds
        tools: The Tools instance (injected automatically)

    Returns:
        Confirmation message
    """
    # Emit WaitEvent for macro recording if context available
    if tools is not None and hasattr(tools, "_ctx") and tools._ctx is not None:
        wait_event = WaitEvent(
            action_type="wait",
            description=f"Wait for {duration} seconds",
            duration=duration,
        )
        tools._ctx.write_event_to_stream(wait_event)

    await asyncio.sleep(duration)
    return f"Waited for {duration} seconds"


def remember(information: str, *, tools: "Tools" = None, **kwargs) -> str:
    """
    Remember important information for later use.

    Args:
        information: The information to remember
        tools: The Tools instance (injected automatically)

    Returns:
        Confirmation message
    """
    if tools is None:
        raise ValueError("tools parameter is required")
    return tools.remember(information)


async def complete(
    success: bool, reason: str = "", *, tools: "Tools" = None, **kwargs
) -> None:
    """
    Mark the task as complete.

    Args:
        success: Whether the task was completed successfully
        reason: Explanation for success or failure
        tools: The Tools instance (injected automatically)

    Returns:
        None
    """
    if tools is None:
        raise ValueError("tools parameter is required")
    await tools.complete(success, reason)


# =============================================================================
# COORDINATE-BASED CLICK ACTIONS
# =============================================================================


async def click_at(x: int, y: int, *, tools: "Tools" = None, **kwargs) -> str:
    """
    Click at normalized coordinates [0-1000].

    This action allows precise clicking at any position on the screen
    using normalized coordinates, independent of screen resolution.

    Args:
        x: Normalized X coordinate [0-1000], where 0 is left edge and 1000 is right edge
        y: Normalized Y coordinate [0-1000], where 0 is top edge and 1000 is bottom edge
        tools: The Tools instance (injected automatically)

    Returns:
        Result message from the tap operation
    """
    if tools is None:
        raise ValueError("tools parameter is required")
    return await tools.tap_by_normalized_coordinate(x, y)


async def click_area(
    x1: int, y1: int, x2: int, y2: int, *, tools: "Tools" = None, **kwargs
) -> str:
    """
    Click at the center of a normalized area [0-1000].

    This action calculates the center point of the given area and clicks there.
    Useful when you want to click on an element but don't need pixel-precise accuracy.

    Args:
        x1: Top-left X coordinate [0-1000]
        y1: Top-left Y coordinate [0-1000]
        x2: Bottom-right X coordinate [0-1000]
        y2: Bottom-right Y coordinate [0-1000]
        tools: The Tools instance (injected automatically)

    Returns:
        Result message from the tap operation
    """
    if tools is None:
        raise ValueError("tools parameter is required")
    return await tools.tap_normalized_area(x1, y1, x2, y2)


# =============================================================================
# ATOMIC ACTION SIGNATURES - Single source of truth for both Executor and CodeAct
# =============================================================================

ATOMIC_ACTION_SIGNATURES = {
    "click": {
        "arguments": ["index"],
        "description": 'Click the point on the screen with specified index. Usage Example: {"action": "click", "index": element_index}',
        "function": click,
    },
    "long_press": {
        "arguments": ["index"],
        "description": 'Long press on the position with specified index. Usage Example: {"action": "long_press", "index": element_index}',
        "function": long_press,
    },
    "type": {
        "arguments": ["text", "index", "clear=False"],
        "description": 'Type text into an input box or text field. Specify the element with index to focus the input field before typing. By default, text is APPENDED to existing content. Set clear=True to clear the field first (recommended for URL bars, search fields, or when replacing text). Usage Example: {"action": "type", "text": "example.com", "index": element_index, "clear": true}',
        "function": type,
    },
    "system_button": {
        "arguments": ["button"],
        "description": 'Press a system button, including back, home, and enter. Usage example: {"action": "system_button", "button": "Home"}',
        "function": system_button,
    },
    "swipe": {
        "arguments": ["coordinate", "coordinate2", "duration=1.0"],
        "description": 'Scroll from the position with coordinate to the position with coordinate2. Duration is in seconds (default: 1.0). Please make sure the start and end points of your swipe are within the swipeable area and away from the keyboard (y1 < 1400). Usage Example: {"action": "swipe", "coordinate": [x1, y1], "coordinate2": [x2, y2], "duration": 1.5}',
        "function": swipe,
    },
    "wait": {
        "arguments": ["duration"],
        "description": 'Wait for a specified duration in seconds. Useful for waiting for animations, page loads, or other time-based operations. Usage Example: {"action": "wait", "duration": 2.0}',
        "function": wait,
    },
}

# Index-based click actions (require element index from a11y tree)
INDEX_CLICK_ACTIONS = {
    "click": ATOMIC_ACTION_SIGNATURES["click"],
    "long_press": ATOMIC_ACTION_SIGNATURES["long_press"],
}

# Coordinate-based click actions (use normalized coordinates [0-1000])
COORDINATE_CLICK_ACTIONS = {
    "click_at": {
        "arguments": ["x", "y"],
        "description": 'Click at normalized coordinates [0-1000]. x=0 is left edge, x=1000 is right edge; y=0 is top, y=1000 is bottom. Use this for precise clicking based on visual position. Usage Example: {"action": "click_at", "x": 500, "y": 300}',
        "function": click_at,
    },
    "click_area": {
        "arguments": ["x1", "y1", "x2", "y2"],
        "description": 'Click at the center of a normalized area [0-1000]. Useful when targeting an element region. Usage Example: {"action": "click_area", "x1": 100, "y1": 200, "x2": 300, "y2": 400}',
        "function": click_area,
    },
}


# =============================================================================
# TOOL FILTERING
# =============================================================================


def get_click_actions_for_mode(click_mode: ClickMode) -> Dict[str, Any]:
    """
    Get click-related actions based on click mode.

    Args:
        click_mode: The click mode (INDEX, COORDINATE, or HYBRID)

    Returns:
        Dict of click action signatures appropriate for the mode
    """
    if click_mode == ClickMode.INDEX:
        result = INDEX_CLICK_ACTIONS.copy()
    elif click_mode == ClickMode.COORDINATE:
        result = COORDINATE_CLICK_ACTIONS.copy()
    elif click_mode == ClickMode.HYBRID:
        # Hybrid mode includes both index and coordinate actions
        result = INDEX_CLICK_ACTIONS.copy()
        result.update(COORDINATE_CLICK_ACTIONS)
    else:
        # Default to index mode
        result = INDEX_CLICK_ACTIONS.copy()
    return result


def filter_atomic_actions(
    disabled_tools: List[str],
    click_mode: Optional[ClickMode] = None,
) -> Dict[str, Any]:
    """
    Filter ATOMIC_ACTION_SIGNATURES by removing disabled tools and applying click mode.

    Args:
        disabled_tools: List of tool names to exclude. Empty list = all tools enabled.
        click_mode: Click mode to determine which click actions to include.
                   If None, defaults to INDEX mode (backward compatible).

    Returns:
        Filtered dict of atomic action signatures.
    """
    # Start with base actions (excluding click/long_press which are mode-dependent)
    base_actions = {
        k: v for k, v in ATOMIC_ACTION_SIGNATURES.items()
        if k not in INDEX_CLICK_ACTIONS
    }

    # Add click actions based on mode
    if click_mode is None:
        click_mode = ClickMode.INDEX
    click_actions = get_click_actions_for_mode(click_mode)

    # Combine base and click actions
    all_actions = {**base_actions, **click_actions}

    # Apply disabled tools filter
    if not disabled_tools:
        return all_actions

    return {k: v for k, v in all_actions.items() if k not in disabled_tools}


def filter_custom_tools(
    custom_tools: Dict[str, Any],
    disabled_tools: List[str],
) -> Dict[str, Any]:
    """
    Filter custom tools dict by removing disabled tools.

    Args:
        custom_tools: Dict of custom tool signatures
        disabled_tools: List of tool names to exclude. Empty list = all tools enabled.

    Returns:
        Filtered dict of custom tool signatures.
    """
    if not custom_tools:
        return {}

    if not disabled_tools:
        return custom_tools.copy()

    return {k: v for k, v in custom_tools.items() if k not in disabled_tools}


def get_atomic_tool_descriptions(
    click_mode: Optional[ClickMode] = None,
    disabled_tools: Optional[List[str]] = None,
) -> str:
    """
    Get formatted tool descriptions for CodeAct system prompt.

    Args:
        click_mode: Click mode to determine which click actions to include.
                   If None, defaults to INDEX mode.
        disabled_tools: List of tool names to exclude.

    Returns:
        Formatted string of tool descriptions for LLM prompt
    """
    actions = filter_atomic_actions(
        disabled_tools or [],
        click_mode=click_mode,
    )

    descriptions = []
    for action_name, signature in actions.items():
        args = ", ".join(signature["arguments"])
        desc = signature["description"]
        descriptions.append(f"- {action_name}({args}): {desc}")

    return "\n".join(descriptions)


def build_custom_tool_descriptions(custom_tools: dict) -> str:
    """
    Build formatted tool descriptions from custom_tools dict.

    Args:
        custom_tools: Dictionary of custom tools in ATOMIC_ACTION_SIGNATURES format
            {
                "tool_name": {
                    "arguments": ["arg1", "arg2"],
                    "description": "Tool description with usage",
                    "function": callable
                }
            }

    Returns:
        Formatted string of custom tool descriptions for LLM prompt
    """
    if not custom_tools:
        return ""

    descriptions = []
    for action_name, signature in custom_tools.items():
        args = ", ".join(signature.get("arguments", []))
        desc = signature.get("description", f"Custom action: {action_name}")
        descriptions.append(f"- {action_name}({args}): {desc}")

    return "\n".join(descriptions)


# =============================================================================
# CREDENTIAL TOOLS
# =============================================================================


async def type_secret(
    secret_id: str, index: int, *, tools: "Tools" = None, **kwargs
) -> str:
    """
    Type a secret credential into an input field without exposing the value.

    Args:
        secret_id: Secret ID from credentials store
        index: Input field element index
        tools: Tools instance (injected automatically, must have credential_manager)

    Returns:
        Sanitized result message (NEVER includes actual secret value)
    """
    logger = logging.getLogger("droidrun")

    if tools is None:
        raise ValueError("tools parameter is required")

    if not hasattr(tools, "credential_manager") or tools.credential_manager is None:
        return "Error: Credential manager not initialized. Enable credentials in config.yaml"

    try:
        # Get secret value from credential manager
        secret_value = await tools.credential_manager.resolve_key(secret_id)

        # Type using existing input_text method
        await tools.input_text(secret_value, index)

        # Return sanitized message (NEVER log/return actual secret)
        return f"Successfully typed secret '{secret_id}' into element {index}"

    except Exception as e:
        # Log error without exposing secret
        logger.error(f"Failed to type secret '{secret_id}': {e}")
        available = (
            await tools.credential_manager.get_keys()
            if tools.credential_manager
            else []
        )
        return f"Error: Secret '{secret_id}' not found. Available: {available}"


async def build_credential_tools(credential_manager) -> dict:
    """
    Build credential-related custom tools if credential manager is available.

    Args:
        credential_manager: CredentialManager instance or None

    Returns:
        Dictionary of credential tools (empty if no credentials available)
    """
    logger = logging.getLogger("droidrun")

    if credential_manager is None:
        return {}

    # Check if there are any enabled secrets
    available_secrets = await credential_manager.get_keys()
    if not available_secrets:
        logger.debug("No enabled secrets found, credential tools disabled")
        return {}

    logger.debug(f"Building credential tools with {len(available_secrets)} secrets")

    return {
        "type_secret": {
            "arguments": ["secret_id", "index"],
            "description": 'Type a secret credential from the credential store into an input field. The agent never sees the actual secret value, only the secret_id. Usage: {"action": "type_secret", "secret_id": "MY_PASSWORD", "index": 5}',
            "function": type_secret,
        },
    }


async def build_custom_tools(credential_manager=None) -> dict:
    """
    Build all custom tools (credentials + utility tools).

    This is the master function that assembles all custom tools:
    - Credential tools (type_secret) if credential manager available
    - Utility tools (open_app) always included

    Args:
        credential_manager: CredentialManager instance or None

    Returns:
        Dictionary of all custom tools
    """
    logger = logging.getLogger("droidrun")

    custom_tools = {}

    # 1. Add credential tools (if available)
    credential_tools = await build_credential_tools(credential_manager)
    custom_tools.update(credential_tools)

    if credential_tools:
        logger.debug(
            f"Built {len(credential_tools)} credential tools: {list(credential_tools.keys())}"
        )

    # 2. Add open_app as custom tool (always available)
    custom_tools["open_app"] = {
        "arguments": ["text"],
        "description": 'Open an app by name or description. Usage: {"action": "open_app", "text": "Gmail"}',
        "function": open_app,
    }

    # 3. Future: Add other custom tools here
    # custom_tools["some_other_tool"] = {...}

    return custom_tools


async def test_open_app(mock_tools, text: str) -> str:
    return await open_app(mock_tools, text)


async def _test_main():

    from llama_index.llms.google_genai import GoogleGenAI

    from droidrun.tools.adb import AdbTools

    llm = GoogleGenAI(model="gemini-2.5-pro", temperature=0.0)
    mock_tools = AdbTools(app_opener_llm=llm, text_manipulator_llm=llm)
    await mock_tools.get_state()
    print("\n=== Testing long_press ===")
    result = long_press(tools=mock_tools, index=5)
    print(f"Result: {result}")
    input("Press Enter to continue...")
    print("\n=== Testing type ===")
    result = type(tools=mock_tools, text="Hello World", index=-1)
    print(f"Result: {result}")
    input("Press Enter to continue...")

    print("\n=== Testing system_button ===")
    result = system_button(tools=mock_tools, button="back")
    print(f"Result: {result}")
    input("Press Enter to continue...")

    print("\n=== Testing swipe ===")
    result = swipe(tools=mock_tools, coordinate=[500, 0], coordinate2=[500, 1000])
    print(f"Result: {result}")
    input("Press Enter to continue...")

    print("\n=== Testing open_app ===")
    try:
        result = await test_open_app(mock_tools, "Calculator")
        print(f"Result: {result}")
        input("Press Enter to continue...")
    except Exception as e:
        print(f"Expected error (no LLM): {e}")
        input("Press Enter to continue...")

    print("\n=== All tests completed ===")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_main())
