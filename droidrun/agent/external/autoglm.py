"""
Open-AutoGLM External Agent - Full Implementation.

This module implements the Open-AutoGLM phone agent protocol, matching the original
implementation from https://github.com/ArtificialZeng/Open-AutoGLM

Key features:
- Stateful conversation history across steps
- Full system prompts (Chinese + English) with 14 actions and 18 rules
- AST-based safe action parsing
- OpenAI-compatible message format
- Tool wrappers matching original DeviceFactory interface
- Timing delays matching original implementation
"""

import ast
import asyncio
import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from droidrun.agent.utils.chat_utils import to_chat_messages
from droidrun.agent.utils.inference import acall_with_retries
from droidrun.agent.utils.llm_picker import load_llm

logger = logging.getLogger("droidrun")

# =============================================================================
# System Prompts (exact copy from Open-AutoGLM)
# =============================================================================


def get_system_prompt_zh() -> str:
    """Get Chinese system prompt with current date (matches original prompts.py)."""
    from datetime import datetime
    today = datetime.today()
    formatted_date = today.strftime("%Y年%m月%d日")
    return "今天的日期是: " + formatted_date + """
你是一个智能体分析专家，可以根据操作历史和当前状态图执行一系列操作来完成任务。
你必须严格按照要求输出以下格式：
<think>{think}</think>
<answer>{action}</answer>

其中：
- {think} 是对你为什么选择这个操作的简短推理说明。
- {action} 是本次执行的具体操作指令，必须严格遵循下方定义的指令格式。

操作指令及其作用如下：
- do(action="Launch", app="xxx")  
    Launch是启动目标app的操作，这比通过主屏幕导航更快。此操作完成后，您将自动收到结果状态的截图。
- do(action="Tap", element=[x,y])  
    Tap是点击操作，点击屏幕上的特定点。可用此操作点击按钮、选择项目、从主屏幕打开应用程序，或与任何可点击的用户界面元素进行交互。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。此操作完成后，您将自动收到结果状态的截图。
- do(action="Tap", element=[x,y], message="重要操作")  
    基本功能同Tap，点击涉及财产、支付、隐私等敏感按钮时触发。
- do(action="Type", text="xxx")  
    Type是输入操作，在当前聚焦的输入框中输入文本。使用此操作前，请确保输入框已被聚焦（先点击它）。输入的文本将像使用键盘输入一样输入。重要提示：手机可能正在使用 ADB 键盘，该键盘不会像普通键盘那样占用屏幕空间。要确认键盘已激活，请查看屏幕底部是否显示 'ADB Keyboard {ON}' 类似的文本，或者检查输入框是否处于激活/高亮状态。不要仅仅依赖视觉上的键盘显示。自动清除文本：当你使用输入操作时，输入框中现有的任何文本（包括占位符文本和实际输入）都会在输入新文本前自动清除。你无需在输入前手动清除文本——直接使用输入操作输入所需文本即可。操作完成后，你将自动收到结果状态的截图。
- do(action="Type_Name", text="xxx")  
    Type_Name是输入人名的操作，基本功能同Type。
- do(action="Interact")  
    Interact是当有多个满足条件的选项时而触发的交互操作，询问用户如何选择。
- do(action="Swipe", start=[x1,y1], end=[x2,y2])  
    Swipe是滑动操作，通过从起始坐标拖动到结束坐标来执行滑动手势。可用于滚动内容、在屏幕之间导航、下拉通知栏以及项目栏或进行基于手势的导航。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。滑动持续时间会自动调整以实现自然的移动。此操作完成后，您将自动收到结果状态的截图。
- do(action="Note", message="True")  
    记录当前页面内容以便后续总结。
- do(action="Call_API", instruction="xxx")  
    总结或评论当前页面或已记录的内容。
- do(action="Long Press", element=[x,y])  
    Long Pres是长按操作，在屏幕上的特定点长按指定时间。可用于触发上下文菜单、选择文本或激活长按交互。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。此操作完成后，您将自动收到结果状态的屏幕截图。
- do(action="Double Tap", element=[x,y])  
    Double Tap在屏幕上的特定点快速连续点按两次。使用此操作可以激活双击交互，如缩放、选择文本或打开项目。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。此操作完成后，您将自动收到结果状态的截图。
- do(action="Take_over", message="xxx")  
    Take_over是接管操作，表示在登录和验证阶段需要用户协助。
- do(action="Back")  
    导航返回到上一个屏幕或关闭当前对话框。相当于按下 Android 的返回按钮。使用此操作可以从更深的屏幕返回、关闭弹出窗口或退出当前上下文。此操作完成后，您将自动收到结果状态的截图。
- do(action="Home") 
    Home是回到系统桌面的操作，相当于按下 Android 主屏幕按钮。使用此操作可退出当前应用并返回启动器，或从已知状态启动新任务。此操作完成后，您将自动收到结果状态的截图。
- do(action="Wait", duration="x seconds")  
    等待页面加载，x为需要等待多少秒。
- finish(message="xxx")  
    finish是结束任务的操作，表示准确完整完成任务，message是终止信息。 

必须遵循的规则：
1. 在执行任何操作前，先检查当前app是否是目标app，如果不是，先执行 Launch。
2. 如果进入到了无关页面，先执行 Back。如果执行Back后页面没有变化，请点击页面左上角的返回键进行返回，或者右上角的X号关闭。
3. 如果页面未加载出内容，最多连续 Wait 三次，否则执行 Back重新进入。
4. 如果页面显示网络问题，需要重新加载，请点击重新加载。
5. 如果当前页面找不到目标联系人、商品、店铺等信息，可以尝试 Swipe 滑动查找。
6. 遇到价格区间、时间区间等筛选条件，如果没有完全符合的，可以放宽要求。
7. 在做小红书总结类任务时一定要筛选图文笔记。
8. 购物车全选后再点击全选可以把状态设为全不选，在做购物车任务时，如果购物车里已经有商品被选中时，你需要点击全选后再点击取消全选，再去找需要购买或者删除的商品。
9. 在做外卖任务时，如果相应店铺购物车里已经有其他商品你需要先把购物车清空再去购买用户指定的外卖。
10. 在做点外卖任务时，如果用户需要点多个外卖，请尽量在同一店铺进行购买，如果无法找到可以下单，并说明某个商品未找到。
11. 请严格遵循用户意图执行任务，用户的特殊要求可以执行多次搜索，滑动查找。比如（i）用户要求点一杯咖啡，要咸的，你可以直接搜索咸咖啡，或者搜索咖啡后滑动查找咸的咖啡，比如海盐咖啡。（ii）用户要找到XX群，发一条消息，你可以先搜索XX群，找不到结果后，将"群"字去掉，搜索XX重试。（iii）用户要找到宠物友好的餐厅，你可以搜索餐厅，找到筛选，找到设施，选择可带宠物，或者直接搜索可带宠物，必要时可以使用AI搜索。
12. 在选择日期时，如果原滑动方向与预期日期越来越远，请向反方向滑动查找。
13. 执行任务过程中如果有多个可选择的项目栏，请逐个查找每个项目栏，直到完成任务，一定不要在同一项目栏多次查找，从而陷入死循环。
14. 在执行下一步操作前请一定要检查上一步的操作是否生效，如果点击没生效，可能因为app反应较慢，请先稍微等待一下，如果还是不生效请调整一下点击位置重试，如果仍然不生效请跳过这一步继续任务，并在finish message说明点击不生效。
15. 在执行任务中如果遇到滑动不生效的情况，请调整一下起始点位置，增大滑动距离重试，如果还是不生效，有可能是已经滑到底了，请继续向反方向滑动，直到顶部或底部，如果仍然没有符合要求的结果，请跳过这一步继续任务，并在finish message说明但没找到要求的项目。
16. 在做游戏任务时如果在战斗页面如果有自动战斗一定要开启自动战斗，如果多轮历史状态相似要检查自动战斗是否开启。
17. 如果没有合适的搜索结果，可能是因为搜索页面不对，请返回到搜索页面的上一级尝试重新搜索，如果尝试三次返回上一级搜索后仍然没有符合要求的结果，执行 finish(message="原因")。
18. 在结束任务前请一定要仔细检查任务是否完整准确的完成，如果出现错选、漏选、多选的情况，请返回之前的步骤进行纠正。
"""


def get_system_prompt_en() -> str:
    """Get English system prompt with current date (matches original prompts_en.py)."""
    from datetime import datetime
    today = datetime.today()
    formatted_date = today.strftime("%Y-%m-%d, %A")
    return "The current date: " + formatted_date + """
# Setup
You are a professional Android operation agent assistant that can fulfill the user's high-level instructions. Given a screenshot of the Android interface at each step, you first analyze the situation, then plan the best course of action using Python-style pseudo-code.

# More details about the code
Your response format must be structured as follows:

Think first: Use <think>...</think> to analyze the current screen, identify key elements, and determine the most efficient action.
Provide the action: Use <answer>...</answer> to return a single line of pseudo-code representing the operation.

Your output should STRICTLY follow the format:
<think>
[Your thought]
</think>
<answer>
[Your operation code]
</answer>

- **Tap**
  Perform a tap action on a specified screen area. The element is a list of 2 integers, representing the coordinates of the tap point.
  **Example**:
  <answer>
  do(action="Tap", element=[x,y])
  </answer>
- **Type**
  Enter text into the currently focused input field.
  **Example**:
  <answer>
  do(action="Type", text="Hello World")
  </answer>
- **Swipe**
  Perform a swipe action with start point and end point.
  **Examples**:
  <answer>
  do(action="Swipe", start=[x1,y1], end=[x2,y2])
  </answer>
- **Long Press**
  Perform a long press action on a specified screen area.
  You can add the element to the action to specify the long press area. The element is a list of 2 integers, representing the coordinates of the long press point.
  **Example**:
  <answer>
  do(action="Long Press", element=[x,y])
  </answer>
- **Launch**
  Launch an app. Try to use launch action when you need to launch an app. Check the instruction to choose the right app before you use this action.
  **Example**:
  <answer>
  do(action="Launch", app="Settings")
  </answer>
- **Back**
  Press the Back button to navigate to the previous screen.
  **Example**:
  <answer>
  do(action="Back")
  </answer>
- **Finish**
  Terminate the program and optionally print a message.
  **Example**:
  <answer>
  finish(message="Task completed.")
  </answer>


REMEMBER:
- Think before you act: Always analyze the current UI and the best course of action before executing any step, and output in <think> part.
- Only ONE LINE of action in <answer> part per response: Each step must contain exactly one line of executable code.
- Generate execution code strictly according to format requirements.
"""


def get_system_prompt(lang: str = "cn") -> str:
    """
    Get system prompt by language.

    Args:
        lang: Language code, 'cn' for Chinese (default), 'en' for English.

    Returns:
        System prompt string with current date.
    """
    if lang == "cn":
        return get_system_prompt_zh()
    else:
        return get_system_prompt_en()

# =============================================================================
# Default Configuration (matches original Open-AutoGLM)
# =============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    # LLM configuration (matches original Open-AutoGLM ModelConfig)
    "llm": {
        "provider": "OpenAILike",
        "model": "autoglm-phone-9b",
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "temperature": 0.0,
        "top_p": 0.85,
        "frequency_penalty": 0.2,
        "max_tokens": 3000,
    },
    # Agent-specific settings
    # Note: vision is always enabled (AutoGLM requires screenshots)
    # Note: max_steps comes from agent.max_steps in main config
    "lang": "cn",  # cn (18 detailed rules) or en (minimal rules)
    "stream": True,
}


# =============================================================================
# Timing Configuration (matches original Open-AutoGLM)
# =============================================================================

@dataclass
class ActionTimingConfig:
    """Configuration for action handler timing delays."""
    keyboard_switch_delay: float = 1.0
    text_clear_delay: float = 1.0
    text_input_delay: float = 1.0
    keyboard_restore_delay: float = 1.0


@dataclass
class DeviceTimingConfig:
    """Configuration for device operation timing delays."""
    default_tap_delay: float = 1.0
    default_double_tap_delay: float = 1.0
    double_tap_interval: float = 0.1
    default_long_press_delay: float = 1.0
    default_swipe_delay: float = 1.0
    default_back_delay: float = 1.0
    default_home_delay: float = 1.0
    default_launch_delay: float = 1.0


@dataclass
class TimingConfig:
    """Master timing configuration."""
    action: ActionTimingConfig = field(default_factory=ActionTimingConfig)
    device: DeviceTimingConfig = field(default_factory=DeviceTimingConfig)


TIMING_CONFIG = TimingConfig()


# =============================================================================
# Screenshot Data Class (matches original Open-AutoGLM)
# =============================================================================

@dataclass
class Screenshot:
    """Represents a captured screenshot (matches original interface)."""
    base64_data: str
    width: int
    height: int
    is_sensitive: bool = False


# =============================================================================
# Device Factory Wrapper (wraps DroidRun tools to match original interface)
# =============================================================================

class DeviceFactoryWrapper:
    """
    Wraps DroidRun Tools to provide the same interface as original Open-AutoGLM's
    DeviceFactory. All methods are async but match the original signatures and
    return types.
    """

    def __init__(self, tools, loop: asyncio.AbstractEventLoop):
        """
        Initialize wrapper.

        Args:
            tools: DroidRun Tools instance
            loop: Event loop for running async operations
        """
        self.tools = tools
        self.loop = loop
        self._current_app = "System Home"

    async def get_screenshot(self, timeout: int = 10) -> Screenshot:
        """
        Get screenshot matching original interface.

        Returns:
            Screenshot object with base64_data, width, height, is_sensitive
        """
        try:
            _, screenshot_bytes = await self.tools.take_screenshot()
            if screenshot_bytes:
                base64_data = base64.b64encode(screenshot_bytes).decode("utf-8")
                return Screenshot(
                    base64_data=base64_data,
                    width=self.tools.screen_width or 1080,
                    height=self.tools.screen_height or 2400,
                    is_sensitive=False,
                )
        except Exception as e:
            logger.warning(f"Screenshot failed: {e}")

        # Return fallback black image
        return self._create_fallback_screenshot()

    def _create_fallback_screenshot(self, is_sensitive: bool = False) -> Screenshot:
        """Create a black fallback image when screenshot fails."""
        # Create a minimal black PNG (1x1 pixel)
        # In production, you might want a full-size black image
        width = self.tools.screen_width or 1080
        height = self.tools.screen_height or 2400

        try:
            from io import BytesIO

            from PIL import Image

            black_img = Image.new("RGB", (width, height), color="black")
            buffered = BytesIO()
            black_img.save(buffered, format="PNG")
            base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        except ImportError:
            # Minimal 1x1 black PNG if PIL not available
            base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        return Screenshot(
            base64_data=base64_data,
            width=width,
            height=height,
            is_sensitive=is_sensitive,
        )

    async def get_current_app(self) -> str:
        """
        Get current app name.

        Returns:
            App name string (or "System Home" if unknown)
        """
        # Try to get from tools state
        if hasattr(self.tools, "current_package") and self.tools.current_package:
            return self.tools.current_package

        # Try to extract from clickable elements
        if hasattr(self.tools, "clickable_elements_cache") and self.tools.clickable_elements_cache:
            first_elem = self.tools.clickable_elements_cache[0] if self.tools.clickable_elements_cache else {}
            pkg = first_elem.get("package", "")
            if pkg:
                return pkg

        return self._current_app

    async def tap(
        self, x: int, y: int, delay: Optional[float] = None
    ) -> None:
        """
        Tap at coordinates with post-action delay.

        Args:
            x: X coordinate (pixels)
            y: Y coordinate (pixels)
            delay: Delay after tap (default: 1.0s)
        """
        if delay is None:
            delay = TIMING_CONFIG.device.default_tap_delay

        await self.tools.tap_by_coordinates(x, y)
        await asyncio.sleep(delay)

    async def double_tap(
        self, x: int, y: int, delay: Optional[float] = None
    ) -> None:
        """
        Double tap at coordinates.

        Args:
            x: X coordinate (pixels)
            y: Y coordinate (pixels)
            delay: Delay after double tap (default: 1.0s)
        """
        if delay is None:
            delay = TIMING_CONFIG.device.default_double_tap_delay

        await self.tools.tap_by_coordinates(x, y)
        await asyncio.sleep(TIMING_CONFIG.device.double_tap_interval)
        await self.tools.tap_by_coordinates(x, y)
        await asyncio.sleep(delay)

    async def long_press(
        self, x: int, y: int, duration_ms: int = 3000, delay: Optional[float] = None
    ) -> None:
        """
        Long press at coordinates.

        Args:
            x: X coordinate (pixels)
            y: Y coordinate (pixels)
            duration_ms: Press duration in milliseconds
            delay: Delay after long press (default: 1.0s)
        """
        if delay is None:
            delay = TIMING_CONFIG.device.default_long_press_delay

        # Long press = swipe from same point to same point
        await self.tools.swipe(x, y, x, y, duration_ms=duration_ms)
        await asyncio.sleep(delay)

    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: Optional[int] = None,
        delay: Optional[float] = None,
    ) -> None:
        """
        Swipe from start to end coordinates.

        Args:
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            duration_ms: Swipe duration (auto-calculated if None)
            delay: Delay after swipe (default: 1.0s)
        """
        if delay is None:
            delay = TIMING_CONFIG.device.default_swipe_delay

        if duration_ms is None:
            # Calculate duration based on distance (matches original)
            dist_sq = (start_x - end_x) ** 2 + (start_y - end_y) ** 2
            duration_ms = int(dist_sq / 1000)
            duration_ms = max(1000, min(duration_ms, 2000))  # Clamp 1000-2000ms

        await self.tools.swipe(start_x, start_y, end_x, end_y, duration_ms=duration_ms)
        await asyncio.sleep(delay)

    async def back(self, delay: Optional[float] = None) -> None:
        """Press back button."""
        if delay is None:
            delay = TIMING_CONFIG.device.default_back_delay

        await self.tools.press_key(4)  # KEYCODE_BACK
        await asyncio.sleep(delay)

    async def home(self, delay: Optional[float] = None) -> None:
        """Press home button."""
        if delay is None:
            delay = TIMING_CONFIG.device.default_home_delay

        await self.tools.press_key(3)  # KEYCODE_HOME
        await asyncio.sleep(delay)

    async def launch_app(self, app_name: str, delay: Optional[float] = None) -> bool:
        """
        Launch an app by name.

        Args:
            app_name: App name or package name

        Returns:
            True if launched successfully
        """
        if delay is None:
            delay = TIMING_CONFIG.device.default_launch_delay

        try:
            await self.tools.start_app(app_name)
            await asyncio.sleep(delay)
            return True
        except Exception as e:
            logger.warning(f"Failed to launch {app_name}: {e}")
            return False

    async def type_text(self, text: str) -> None:
        """
        Type text with keyboard handling.

        Matches original behavior:
        1. Switch to ADB keyboard (handled by DroidRun portal)
        2. Clear existing text
        3. Type new text
        4. Restore keyboard (handled by DroidRun portal)
        """
        # DroidRun's input_text with clear=True handles all this
        await self.tools.input_text(text, index=-1, clear=True)
        await asyncio.sleep(TIMING_CONFIG.action.text_input_delay)

    async def clear_text(self) -> None:
        """Clear text in focused field."""
        # Type empty string with clear flag
        await self.tools.input_text("", index=-1, clear=True)
        await asyncio.sleep(TIMING_CONFIG.action.text_clear_delay)


# =============================================================================
# Message Builder (matches original Open-AutoGLM)
# =============================================================================

class MessageBuilder:
    """Helper class for building OpenAI-compatible conversation messages."""

    @staticmethod
    def create_system_message(content: str) -> Dict[str, Any]:
        """Create a system message."""
        return {"role": "system", "content": content}

    @staticmethod
    def create_user_message(
        text: str, image_base64: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a user message with optional image in OpenAI format.

        Image comes first, then text (matches original).
        """
        content: List[Dict[str, Any]] = []

        if image_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            })

        content.append({"type": "text", "text": text})

        return {"role": "user", "content": content}

    @staticmethod
    def create_assistant_message(content: str) -> Dict[str, Any]:
        """Create an assistant message."""
        return {"role": "assistant", "content": content}

    @staticmethod
    def remove_images_from_message(message: Dict[str, Any]) -> Dict[str, Any]:
        """Remove image content from a message to save context space."""
        if isinstance(message.get("content"), list):
            message["content"] = [
                item for item in message["content"] if item.get("type") == "text"
            ]
        return message

    @staticmethod
    def build_screen_info(current_app: str, **extra_info) -> str:
        """Build screen info JSON string."""
        info = {"current_app": current_app, **extra_info}
        return json.dumps(info, ensure_ascii=False)


# =============================================================================
# Action Parsing (matches original Open-AutoGLM)
# =============================================================================

@dataclass
class ActionResult:
    """Result of an action execution (matches original Open-AutoGLM)."""
    success: bool
    should_finish: bool
    message: Optional[str] = None
    requires_confirmation: bool = False


def parse_action(response: str) -> Dict[str, Any]:
    """
    Parse action from model response.

    Matches original Open-AutoGLM handler.py parse_action exactly.
    """
    logger.debug(f"Parsing action: {response}")
    try:
        response = response.strip()

        # Special case for Type/Type_Name - extract text directly (matches original)
        if response.startswith('do(action="Type"') or response.startswith(
            'do(action="Type_Name"'
        ):
            text = response.split("text=", 1)[1][1:-2]
            action = {"_metadata": "do", "action": "Type", "text": text}
            return action

        elif response.startswith("do"):
            # Use AST parsing instead of eval for safety
            try:
                # Escape special characters (newlines, tabs, etc.) for valid Python syntax
                response = response.replace('\n', '\\n')
                response = response.replace('\r', '\\r')
                response = response.replace('\t', '\\t')

                tree = ast.parse(response, mode="eval")
                if not isinstance(tree.body, ast.Call):
                    raise ValueError("Expected a function call")

                call = tree.body
                # Extract keyword arguments safely
                action: Dict[str, Any] = {"_metadata": "do"}
                for keyword in call.keywords:
                    key = keyword.arg
                    value = ast.literal_eval(keyword.value)
                    action[key] = value

                return action
            except (SyntaxError, ValueError) as e:
                raise ValueError(f"Failed to parse do() action: {e}") from e

        elif response.startswith("finish"):
            action = {
                "_metadata": "finish",
                "message": response.replace("finish(message=", "")[1:-2],
            }
        else:
            raise ValueError(f"Failed to parse action: {response}")
        return action
    except Exception as e:
        raise ValueError(f"Failed to parse action: {e}") from e


def do(**kwargs) -> Dict[str, Any]:
    """Helper function for creating 'do' actions."""
    kwargs["_metadata"] = "do"
    return kwargs


def finish(**kwargs) -> Dict[str, Any]:
    """Helper function for creating 'finish' actions."""
    kwargs["_metadata"] = "finish"
    return kwargs


def parse_response(content: str) -> Tuple[str, str]:
    """
    Parse the model response into thinking and action parts.

    Matches original Open-AutoGLM parsing rules.
    """
    # Rule 1: Check for finish(message=
    if "finish(message=" in content:
        parts = content.split("finish(message=", 1)
        thinking = parts[0].strip()
        action = "finish(message=" + parts[1]
        return thinking, action

    # Rule 2: Check for do(action=
    if "do(action=" in content:
        parts = content.split("do(action=", 1)
        thinking = parts[0].strip()
        action = "do(action=" + parts[1]
        return thinking, action

    # Rule 3: Fallback to legacy XML tag parsing
    if "<answer>" in content:
        parts = content.split("<answer>", 1)
        thinking = parts[0].replace("<think>", "").replace("</think>", "").strip()
        action = parts[1].replace("</answer>", "").strip()
        return thinking, action

    # Rule 4: No markers found, return content as action
    return "", content


# =============================================================================
# Action Handler (matches original Open-AutoGLM)
# =============================================================================

class ActionHandler:
    """
    Handles execution of actions from AI model output.

    Matches original Open-AutoGLM ActionHandler interface.

    Args:
        device: DeviceFactoryWrapper instance for device operations.
        confirmation_callback: Optional callback for sensitive action confirmation.
            Should return True to proceed, False to cancel.
        takeover_callback: Optional callback for takeover requests (login, captcha).
    """

    def __init__(
        self,
        device: DeviceFactoryWrapper,
        confirmation_callback: Optional[Callable[[str], bool]] = None,
        takeover_callback: Optional[Callable[[str], None]] = None,
    ):
        self.device = device
        self.confirmation_callback = confirmation_callback or self._default_confirmation
        self.takeover_callback = takeover_callback or self._default_takeover

    def _convert_relative_to_absolute(
        self, element: List[int], screen_width: int, screen_height: int
    ) -> Tuple[int, int]:
        """Convert relative coordinates (0-1000) to absolute pixels."""
        x = int(element[0] / 1000 * screen_width)
        y = int(element[1] / 1000 * screen_height)
        return x, y

    async def execute(
        self, action: Dict[str, Any], screen_width: int, screen_height: int
    ) -> ActionResult:
        """
        Execute an action from the AI model.

        Args:
            action: The action dictionary from the model.
            screen_width: Current screen width in pixels.
            screen_height: Current screen height in pixels.

        Returns:
            ActionResult indicating success and whether to finish.
        """
        action_type = action.get("_metadata")

        if action_type == "finish":
            return ActionResult(
                success=True, should_finish=True, message=action.get("message")
            )

        if action_type != "do":
            return ActionResult(
                success=False,
                should_finish=True,
                message=f"Unknown action type: {action_type}",
            )

        action_name = action.get("action")
        handler_method = self._get_handler(action_name)

        if handler_method is None:
            return ActionResult(
                success=False,
                should_finish=False,
                message=f"Unknown action: {action_name}",
            )

        try:
            return await handler_method(action, screen_width, screen_height)
        except Exception as e:
            return ActionResult(
                success=False, should_finish=False, message=f"Action failed: {e}"
            )

    def _get_handler(self, action_name: str) -> Optional[Callable]:
        """Get the handler method for an action."""
        handlers = {
            "Launch": self._handle_launch,
            "Tap": self._handle_tap,
            "Type": self._handle_type,
            "Type_Name": self._handle_type,
            "Swipe": self._handle_swipe,
            "Back": self._handle_back,
            "Home": self._handle_home,
            "Double Tap": self._handle_double_tap,
            "Long Press": self._handle_long_press,
            "Wait": self._handle_wait,
            "Take_over": self._handle_takeover,
            "Note": self._handle_note,
            "Call_API": self._handle_call_api,
            "Interact": self._handle_interact,
        }
        return handlers.get(action_name)

    async def _handle_launch(self, action: Dict, width: int, height: int) -> ActionResult:
        """Handle app launch action."""
        app_name = action.get("app")
        if not app_name:
            return ActionResult(False, False, "No app name specified")

        success = await self.device.launch_app(app_name)
        if success:
            return ActionResult(True, False)
        return ActionResult(False, False, f"App not found: {app_name}")

    async def _handle_tap(self, action: Dict, width: int, height: int) -> ActionResult:
        """Handle tap action."""
        element = action.get("element")
        if not element:
            return ActionResult(False, False, "No element coordinates")

        x, y = self._convert_relative_to_absolute(element, width, height)

        # Check for sensitive operation
        if "message" in action:
            if not self.confirmation_callback(action["message"]):
                return ActionResult(
                    success=False,
                    should_finish=True,
                    message="User cancelled sensitive operation",
                )

        await self.device.tap(x, y)
        return ActionResult(True, False)

    async def _handle_type(self, action: Dict, width: int, height: int) -> ActionResult:
        """Handle text input action."""
        text = action.get("text", "")
        await self.device.type_text(text)
        return ActionResult(True, False)

    async def _handle_swipe(self, action: Dict, width: int, height: int) -> ActionResult:
        """Handle swipe action."""
        start = action.get("start")
        end = action.get("end")

        if not start or not end:
            return ActionResult(False, False, "Missing swipe coordinates")

        start_x, start_y = self._convert_relative_to_absolute(start, width, height)
        end_x, end_y = self._convert_relative_to_absolute(end, width, height)

        await self.device.swipe(start_x, start_y, end_x, end_y)
        return ActionResult(True, False)

    async def _handle_back(self, action: Dict, width: int, height: int) -> ActionResult:
        """Handle back button action."""
        await self.device.back()
        return ActionResult(True, False)

    async def _handle_home(self, action: Dict, width: int, height: int) -> ActionResult:
        """Handle home button action."""
        await self.device.home()
        return ActionResult(True, False)

    async def _handle_double_tap(self, action: Dict, width: int, height: int) -> ActionResult:
        """Handle double tap action."""
        element = action.get("element")
        if not element:
            return ActionResult(False, False, "No element coordinates")

        x, y = self._convert_relative_to_absolute(element, width, height)
        await self.device.double_tap(x, y)
        return ActionResult(True, False)

    async def _handle_long_press(self, action: Dict, width: int, height: int) -> ActionResult:
        """Handle long press action."""
        element = action.get("element")
        if not element:
            return ActionResult(False, False, "No element coordinates")

        x, y = self._convert_relative_to_absolute(element, width, height)
        await self.device.long_press(x, y)
        return ActionResult(True, False)

    async def _handle_wait(self, action: Dict, width: int, height: int) -> ActionResult:
        """Handle wait action."""
        duration_str = action.get("duration", "1 seconds")
        try:
            duration = float(duration_str.replace("seconds", "").strip())
        except ValueError:
            duration = 1.0

        await asyncio.sleep(duration)
        return ActionResult(True, False)

    async def _handle_takeover(self, action: Dict, width: int, height: int) -> ActionResult:
        """Handle takeover request (login, captcha, etc.)."""
        message = action.get("message", "User intervention required")
        self.takeover_callback(message)
        return ActionResult(True, False)

    async def _handle_note(self, action: Dict, width: int, height: int) -> ActionResult:
        """Handle note action (placeholder for content recording)."""
        # This action is typically used for recording page content
        # Implementation depends on specific requirements
        return ActionResult(True, False)

    async def _handle_call_api(self, action: Dict, width: int, height: int) -> ActionResult:
        """Handle API call action (placeholder for summarization)."""
        # This action is typically used for content summarization
        # Implementation depends on specific requirements
        return ActionResult(True, False)

    async def _handle_interact(self, action: Dict, width: int, height: int) -> ActionResult:
        """Handle interaction request (user choice needed)."""
        # This action signals that user input is needed
        return ActionResult(True, False, message="User interaction required")

    @staticmethod
    def _default_confirmation(message: str) -> bool:
        """Default confirmation callback using console input."""
        response = input(f"Sensitive operation: {message}\nConfirm? (Y/N): ")
        return response.upper() == "Y"

    @staticmethod
    def _default_takeover(message: str) -> None:
        """Default takeover callback using console input."""
        input(f"{message}\nPress Enter after completing manual operation...")


# =============================================================================
# Main Entry Point
# =============================================================================

async def run(
    tools,
    instruction: str,
    config: Dict[str, Any],
    max_steps: int = 15,
    confirmation_callback: Optional[Callable[[str], bool]] = None,
    takeover_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Run AutoGLM agent matching original Open-AutoGLM implementation.

    Args:
        tools: DroidRun Tools instance
        instruction: Task to complete
        config: Agent config with structure:
            {
                "llm": {
                    "provider": "OpenAI",
                    "model": "gpt-4o",
                    "temperature": 0.0,
                    ...
                },
                "vision": true,
                "lang": "en",
                "stream": true
            }
        max_steps: Max iterations
        confirmation_callback: Optional callback for sensitive action confirmation
        takeover_callback: Optional callback for takeover requests

    Returns:
        {"success": bool, "reason": str, "steps": int}
    """
    # Merge with defaults (deep merge for nested llm config)
    cfg = {**DEFAULT_CONFIG}
    for key, value in config.items():
        if key == "llm" and isinstance(value, dict):
            # Deep merge llm config
            cfg["llm"] = {**cfg.get("llm", {}), **value}
        else:
            cfg[key] = value

    # Extract LLM config (nested or flat for backward compatibility)
    llm_cfg = cfg.get("llm", {})
    if not llm_cfg:
        # Backward compatibility: flat config (provider/model at top level)
        llm_cfg = {
            "provider": cfg.get("provider"),
            "model": cfg.get("model"),
            "temperature": cfg.get("temperature"),
            "max_tokens": cfg.get("max_tokens"),
            "top_p": cfg.get("top_p"),
            "frequency_penalty": cfg.get("frequency_penalty"),
            "base_url": cfg.get("base_url"),
            "api_key": cfg.get("api_key"),
        }
        # Remove None values
        llm_cfg = {k: v for k, v in llm_cfg.items() if v is not None}

    # Create LLM from config
    llm = load_llm(
        provider=llm_cfg.get("provider", "OpenAI"),
        model=llm_cfg.get("model", "gpt-4o"),
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 3000),
        top_p=llm_cfg.get("top_p", 0.85),
        frequency_penalty=llm_cfg.get("frequency_penalty", 0.2),
        base_url=llm_cfg.get("base_url"),
        api_key=llm_cfg.get("api_key"),
    )

    lang = cfg.get("lang", "cn")
    stream = cfg.get("stream", True)

    # Get system prompt with date (matches original)
    system_prompt = get_system_prompt(lang=lang)

    # Create device wrapper
    loop = asyncio.get_running_loop()
    device = DeviceFactoryWrapper(tools, loop)

    # Stateful conversation context (matches original)
    context: List[Dict[str, Any]] = []

    logger.info(f"🤖 AutoGLM: {instruction}")

    for step in range(max_steps):
        step_start = time.time()
        logger.info(f"📍 Step {step + 1}/{max_steps}")

        # Get current screen state
        await tools.get_state()
        w = tools.screen_width or 1080
        h = tools.screen_height or 2400

        # Get screenshot (matches original interface)
        screenshot = await device.get_screenshot()

        # Get current app (matches original)
        current_app = await device.get_current_app()

        # Build screen info (matches original format)
        screen_info = MessageBuilder.build_screen_info(current_app)

        # Build messages (matches original flow)
        if step == 0:
            # First step: system message + user message with task + screen info
            context.append(MessageBuilder.create_system_message(system_prompt))
            text_content = f"{instruction}\n\n{screen_info}"
            context.append(MessageBuilder.create_user_message(
                text_content, screenshot.base64_data
            ))
        else:
            # Subsequent steps: user message with screen info
            text_content = f"** Screen Info **\n\n{screen_info}"
            context.append(MessageBuilder.create_user_message(
                text_content, screenshot.base64_data
            ))

        # Convert to LlamaIndex format and call LLM
        try:
            response = await acall_with_retries(
                llm,
                to_chat_messages(context),
                stream=stream,
            )
            response_text = str(response)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {"success": False, "reason": f"LLM error: {e}", "steps": step + 1}

        # Parse thinking and action (matches original)
        thinking, action_str = parse_response(response_text)

        # Only log thinking if not streaming (streaming already printed it)
        if thinking and not stream:
            logger.info(f"💭 {thinking[:200]}{'...' if len(thinking) > 200 else ''}")

        # Remove image from current user message BEFORE adding assistant (matches original agent.py:205)
        context[-1] = MessageBuilder.remove_images_from_message(context[-1])

        # Add assistant response to context (matches original format)
        context.append(
            MessageBuilder.create_assistant_message(
                f"<think>{thinking}</think><answer>{action_str}</answer>"
            )
        )

        # Parse action
        try:
            action = parse_action(action_str)
        except ValueError as e:
            logger.warning(f"Failed to parse action: {e}")
            action = {"_metadata": "finish", "message": action_str}

        action_name = action.get("action", action.get("_metadata", "unknown"))
        # Log action with key details
        if action_name in ("Tap", "Double Tap", "Long Press"):
            coords = action.get("element", [])
            logger.info(f"⚡ {action_name} {coords}")
        elif action_name == "Swipe":
            start, end = action.get("start", []), action.get("end", [])
            logger.info(f"⚡ {action_name} {start} → {end}")
        elif action_name == "Type":
            text = action.get("text", "")[:30]
            logger.info(f"⚡ {action_name}: \"{text}{'...' if len(action.get('text', '')) > 30 else ''}\"")
        elif action_name == "Launch":
            logger.info(f"⚡ {action_name}: {action.get('app', '')}")
        elif action_name == "finish":
            logger.info(f"⚡ {action_name}: {action.get('message', '')[:50]}")
        else:
            logger.info(f"⚡ {action_name}")

        # Create action handler and execute (matches original interface)
        handler = ActionHandler(
            device=device,
            confirmation_callback=confirmation_callback,
            takeover_callback=takeover_callback,
        )

        result = await handler.execute(action, screenshot.width, screenshot.height)

        step_time = time.time() - step_start
        logger.debug(f"   ⏱️ {step_time:.1f}s")

        # Check if finished
        if result.should_finish:
            reason = result.message or action.get("message", "Task completed")
            logger.info(f"✅ Done ({step + 1} steps): {reason}")
            return {"success": result.success, "reason": reason, "steps": step + 1}

    logger.warning(f"⚠️ Max steps ({max_steps}) reached")
    return {"success": False, "reason": "Max steps reached", "steps": max_steps}
