
import pytest
import asyncio
from typing import List, Any, Dict, Optional

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, CompletionResponse
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context

from droidrun.agent.codeact.codeact_agent import CodeActAgent
from droidrun.agent.codeact.events import TaskThinkingEvent
from droidrun.agent.context.agent_persona import AgentPersona
from droidrun.tools import Tools
from typing import Tuple

class MockTools(Tools):
    def take_screenshot(self) -> Tuple[str, bytes]:
        return "screenshot.png", b""

    def get_state(self) -> Dict[str, Any]:
        return {}

    def list_packages(self, include_system_apps: bool = False) -> List[str]:
        return []

    def tap_by_index(self, index: int) -> str:
        return "tapped"

    def input_text(self, text: str, element_id: int = -1) -> str:
        return "inputted text"

    def swipe(
        self,
        direction: str,
        start_x: int = -1,
        start_y: int = -1,
        end_x: int = -1,
        end_y: int = -1,
        duration: int = 100,
    ) -> str:
        return "swiped"

    def press_key(self, key: str) -> str:
        return "pressed key"

    def start_app(self, package_name: str) -> str:
        return "started app"

    def back(self) -> str:
        return "pressed back"

    def remember(self, information: str) -> str:
        self.memory = information
        return "remembered"

    def get_memory(self) -> str:
        return self.memory

    def drag(self, start_x, start_y, end_x, end_y, duration) -> str:
        return "dragged"

    def complete(self, success: bool, reason: str) -> str:
        self.finished = True
        self.success = success
        self.reason = reason
        return "completed"

class MockLLM(LLM):
    def __init__(self, token_usage: dict):
        super().__init__()
        self._token_usage = token_usage

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text="```python\nprint('Hello, world!')\n```")

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(role="assistant", content="```python\nprint('Hello, world!')\n```"),
            raw={"usage": self._token_usage}
        )

    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        await asyncio.sleep(0)
        return self.chat(messages, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        await asyncio.sleep(0)
        return self.complete(prompt, **kwargs)

    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any):
        pass

    def stream_complete(self, prompt: str, **kwargs: Any):
        pass

    async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any):
        pass

    async def astream_complete(self, prompt: str, **kwargs: Any):
        pass

    def metadata(self):
        return {}


@pytest.mark.asyncio
async def test_token_tracking_in_trajectory():
    # 1. Setup
    token_usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    mock_llm = MockLLM(token_usage=token_usage)

    # Mock Tools and AgentPersona
    mock_tools = MockTools()
    mock_persona = AgentPersona(
        name="TestPersona",
        description="A test persona.",
        expertise_areas=["testing"],
        system_prompt="You are a test agent.",
        user_prompt="Your goal is {goal}",
        allowed_tools=["print"],
        required_context=[]
    )

    # Instantiate the agent
    agent = CodeActAgent(
        llm=mock_llm,
        persona=mock_persona,
        vision=False,
        tools_instance=mock_tools,
        all_tools_list={"print": print}
    )

    # 2. Run the agent for one step
    handler = agent.run(input="Say hello", remembered_info=None)

    captured_events = []
    async for event in handler.stream_events():
        captured_events.append(event)

    # 3. Verify
    thinking_events = [e for e in captured_events if isinstance(e, TaskThinkingEvent)]
    assert len(thinking_events) > 0

    first_thinking_event = thinking_events[0]
    assert first_thinking_event.tokens is not None
    assert first_thinking_event.tokens["prompt_tokens"] == 10
    assert first_thinking_event.tokens["completion_tokens"] == 20
    assert first_thinking_event.tokens["total_tokens"] == 30
