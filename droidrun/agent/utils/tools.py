import time
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from droidrun.config_manager.config_manager import DeviceConfig, ToolsConfig
    from droidrun.tools import Tools

from droidrun.agent.oneflows.app_starter_workflow import AppStarter


def create_tools_from_config(device_config: "DeviceConfig") -> "Tools":
    """
    Create Tools instance from DeviceConfig.

    Args:
        device_config: Device configuration

    Returns:
        AdbTools, CloudAdbTools, or IOSTools based on config

    Raises:
        ValueError: If no device found or invalid platform
    """
    from adbutils import adb
    from droidrun.tools import AdbTools, IOSTools

    is_ios = device_config.platform.lower() == "ios"
    device_serial = device_config.serial

    if not is_ios:
        # Check if using cloud devices
        if device_config.use_cloud:
            from droidrun.tools.cloud_adb import CloudAdbTools

            return CloudAdbTools(device_config=device_config)

        # Android: auto-detect if not specified
        if device_serial is None:
            devices = adb.list()
            if not devices:
                raise ValueError("No connected Android devices found.")
            device_serial = devices[0].serial

        return AdbTools(
            serial=device_serial,
            use_tcp=device_config.use_tcp,
        )
    else:
        # iOS: require explicit device URL
        if device_serial is None:
            raise ValueError("iOS device URL required in config.device.serial")
        return IOSTools(url=device_serial)


def resolve_tools_instance(
    tools: "Tools | ToolsConfig | None",
    device_config: "DeviceConfig",
    tools_config_fallback: "ToolsConfig | None" = None,
    credential_manager=None,
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
        >>> tools_cfg = ToolsConfig(allow_drag=True)
        >>> tools_instance, tools_cfg = resolve_tools_instance(tools_cfg, device_config)
    """
    # Import at runtime to avoid circular imports
    from droidrun.config_manager.config_manager import ToolsConfig
    from droidrun.tools.tools import Tools

    # Case 1: Tools instance provided directly
    if isinstance(tools, Tools):
        tools_instance = tools
        # Use fallback or default ToolsConfig
        tools_cfg = tools_config_fallback if tools_config_fallback else ToolsConfig()

    # Case 2: ToolsConfig provided
    elif tools is not None and isinstance(tools, ToolsConfig):
        tools_instance = create_tools_from_config(device_config)
        tools_cfg = tools

    # Case 3: None provided
    else:
        tools_instance = create_tools_from_config(device_config)
        tools_cfg = tools_config_fallback if tools_config_fallback else ToolsConfig()

    # Attach credential manager if provided
    if credential_manager:
        tools_instance.credential_manager = credential_manager

    return tools_instance, tools_cfg


def click(tool_instance: "Tools", index: int) -> str:
    """
    Click the element with the given index.

    Args:
        tool_instance: The Tools instance
        index: The index of the element to click

    Returns:
        Result message from the tap operation
    """
    return tool_instance.tap_by_index(index)


def long_press(tool_instance: "Tools", index: int) -> bool:
    """
    Long press the element with the given index.

    Args:
        tool_instance: The Tools instance
        index: The index of the element to long press

    Returns:
        True if successful, False otherwise
    """
    x, y = tool_instance._extract_element_coordinates_by_index(index)
    return tool_instance.swipe(x, y, x, y, 1000)


def type(tool_instance: "Tools", text: str, index: int) -> str:
    """
    Type the given text into the element with the given index.

    Args:
        tool_instance: The Tools instance
        text: The text to type
        index: The index of the element to type into

    Returns:
        Result message from the input operation
    """
    return tool_instance.input_text(text, index)


def system_button(tool_instance: "Tools", button: str) -> str:
    """
    Press a system button (back, home, or enter).

    Args:
        tool_instance: The Tools instance
        button: The button name (case insensitive): "back", "home", or "enter"

    Returns:
        Result message from the key press operation
    """
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
    return tool_instance.press_key(keycode)


def swipe(
    tool_instance: "Tools", coordinate: List[int], coordinate2: List[int]
) -> bool:
    """
    Swipe from one coordinate to another.

    Args:
        tool_instance: The Tools instance
        coordinate: Starting coordinate as [x, y]
        coordinate2: Ending coordinate as [x, y]

    Returns:
        True if successful, False otherwise
    """
    if not isinstance(coordinate, list) or len(coordinate) != 2:
        raise ValueError(f"coordinate must be a list of 2 integers, got: {coordinate}")
    if not isinstance(coordinate2, list) or len(coordinate2) != 2:
        raise ValueError(
            f"coordinate2 must be a list of 2 integers, got: {coordinate2}"
        )

    start_x, start_y = coordinate
    end_x, end_y = coordinate2

    return tool_instance.swipe(start_x, start_y, end_x, end_y, duration_ms=300)


async def open_app(tool_instance: "Tools", text: str) -> str:
    """
    Open an app by its name.

    Args:
        tool_instance: The Tools instance
        text: The name of the app to open

    Returns:
        Result message from opening the app
    """
    # Get LLM from tools instance
    if tool_instance.app_opener_llm is None:
        raise RuntimeError(
            "app_opener_llm not configured. "
            "provide app_opener_llm when initializing Tools."
        )

    # Create workflow instance
    workflow = AppStarter(
        tools=tool_instance, llm=tool_instance.app_opener_llm, timeout=60, verbose=True
    )

    # Run workflow to open an app
    result = await workflow.run(app_description=text)
    time.sleep(1)
    return result


def remember(tool_instance: "Tools", information: str) -> str:
    """
    Remember important information for later use.

    Args:
        tool_instance: The Tools instance
        information: The information to remember

    Returns:
        Confirmation message
    """
    return tool_instance.remember(information)


def complete(tool_instance: "Tools", success: bool, reason: str = "") -> None:
    """
    Mark the task as complete.

    Args:
        tool_instance: The Tools instance
        success: Whether the task was completed successfully
        reason: Explanation for success or failure

    Returns:
        None
    """
    tool_instance.complete(success, reason)


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
        "arguments": ["text", "index"],
        "description": 'Type text into an input box or text field. Specify the element with index to focus the input field before typing. Usage Example: {"action": "type", "text": "the text you want to type", "index": element_index}',
        "function": type,
    },
    "system_button": {
        "arguments": ["button"],
        "description": 'Press a system button, including back, home, and enter. Usage example: {"action": "system_button", "button": "Home"}',
        "function": system_button,
    },
    "swipe": {
        "arguments": ["coordinate", "coordinate2"],
        "description": 'Scroll from the position with coordinate to the position with coordinate2. Please make sure the start and end points of your swipe are within the swipeable area and away from the keyboard (y1 < 1400). Usage Example: {"action": "swipe", "coordinate": [x1, y1], "coordinate2": [x2, y2]}',
        "function": swipe,
    },
    # "copy": {
    #     "arguments": ["text"],
    #     "description": "Copy the specified text to the clipboard. Provide the text to copy using the 'text' argument. Example: {\"action\": \"copy\", \"text\": \"the text you want to copy\"}\nAlways use copy action to copy text to clipboard."
    #     "function": copy,
    # },
    # "paste": {
    #     "arguments": ["index", "clear"],
    #     "description": "Paste clipboard text into a text box. 'index' specifies which text box to focus on and paste into. Set 'clear' to true to clear existing text before pasting. Example: {\"action\": \"paste\", \"index\": 0, \"clear\": true}\nAlways use paste action to paste text from clipboard."
    #     "function": paste,
    # },
}


def get_atomic_tool_descriptions() -> str:
    """
    Get formatted tool descriptions for CodeAct system prompt.

    Parses ATOMIC_ACTION_SIGNATURES to create formatted descriptions.

    Returns:
        Formatted string of tool descriptions for LLM prompt
    """
    descriptions = []
    for action_name, signature in ATOMIC_ACTION_SIGNATURES.items():
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


def type_secret(tool_instance: "Tools", secret_id: str, index: int) -> str:
    """
    Type a secret credential into an input field without exposing the value.

    Args:
        tool_instance: Tools instance (must have credential_manager)
        secret_id: Secret ID from credentials store
        index: Input field element index

    Returns:
        Sanitized result message (NEVER includes actual secret value)
    """
    import logging

    logger = logging.getLogger("droidrun")

    if (
        not hasattr(tool_instance, "credential_manager")
        or tool_instance.credential_manager is None
    ):
        return "Error: Credential manager not initialized. Enable credentials in config.yaml"

    try:
        # Get secret value from credential manager
        secret_value = tool_instance.credential_manager.get_credential(secret_id)

        # Type using existing input_text method
        tool_instance.input_text(secret_value, index)

        # Return sanitized message (NEVER log/return actual secret)
        return f"Successfully typed secret '{secret_id}' into element {index}"

    except Exception as e:
        # Log error without exposing secret
        logger.error(f"Failed to type secret '{secret_id}': {e}")
        available = (
            tool_instance.credential_manager.list_available_secrets()
            if tool_instance.credential_manager
            else []
        )
        return f"Error: Secret '{secret_id}' not found. Available: {available}"


def build_credential_tools(credential_manager) -> dict:
    """
    Build credential-related custom tools if credential manager is available.

    Args:
        credential_manager: CredentialManager instance or None

    Returns:
        Dictionary of credential tools (empty if no credentials available)
    """
    import logging

    logger = logging.getLogger("droidrun")

    if credential_manager is None:
        return {}

    # Check if there are any enabled secrets
    available_secrets = credential_manager.list_available_secrets()
    if not available_secrets:
        logger.debug("No enabled secrets found, credential tools disabled")
        return {}

    logger.info(f"Building credential tools with {len(available_secrets)} secrets")

    return {
        "type_secret": {
            "arguments": ["secret_id", "index"],
            "description": 'Type a secret credential from the credential store into an input field. The agent never sees the actual secret value, only the secret_id. Usage: {"action": "type_secret", "secret_id": "MY_PASSWORD", "index": 5}',
            "function": type_secret,
        },
    }


def build_custom_tools(credential_manager=None) -> dict:
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
    import logging

    logger = logging.getLogger("droidrun")

    custom_tools = {}

    # 1. Add credential tools (if available)
    credential_tools = build_credential_tools(credential_manager)
    custom_tools.update(credential_tools)

    # 2. Add open_app as custom tool (always available)
    custom_tools["open_app"] = {
        "arguments": ["text"],
        "description": 'Open an app by name or description. Usage: {"action": "open_app", "text": "Gmail"}',
        "function": open_app,
    }

    # 3. Future: Add other custom tools here
    # custom_tools["some_other_tool"] = {...}

    logger.info(f"Built {len(custom_tools)} custom tools: {list(custom_tools.keys())}")
    return custom_tools


async def test_open_app(mock_tools, text: str) -> str:
    return await open_app(mock_tools, text)


if __name__ == "__main__":
    """
    Simple test for the tool functions.
    Tests the atomic action wrapper functions.
    """
    import asyncio
    from typing import List

    from llama_index.llms.google_genai import GoogleGenAI

    from droidrun.tools.adb import AdbTools

    llm = GoogleGenAI(model="gemini-2.5-pro", temperature=0.0)
    # Create mock tools instance
    mock_tools = AdbTools(app_opener_llm=llm, text_manipulator_llm=llm)
    # print("=== Testing click ===")
    # result = click(mock_tools, 0)
    mock_tools.get_state()
    print("\n=== Testing long_press ===")
    result = long_press(mock_tools, 5)
    print(f"Result: {result}")
    input("Press Enter to continue...")
    print("\n=== Testing type ===")
    result = type(mock_tools, "Hello World", -1)
    print(f"Result: {result}")
    input("Press Enter to continue...")

    print("\n=== Testing system_button ===")
    result = system_button(mock_tools, "back")
    print(f"Result: {result}")
    input("Press Enter to continue...")

    print("\n=== Testing swipe ===")
    result = swipe(mock_tools, [500, 0], [500, 1000])
    print(f"Result: {result}")
    input("Press Enter to continue...")

    print("\n=== Testing open_app ===")
    # This one is more complex and requires real LLM setup, so just show the structure
    try:
        result = asyncio.run(test_open_app(mock_tools, "Calculator"))
        print(f"Result: {result}")
        input("Press Enter to continue...")
    except Exception as e:
        print(f"Expected error (no LLM): {e}")
        input("Press Enter to continue...")

    print("\n=== All tests completed ===")
