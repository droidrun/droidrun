from abc import ABC, abstractmethod
from typing import TypedDict, List, Dict, Any, Optional
from . import BaseLLMReasoner

class TaskContext(TypedDict):
    task: str
    history: List[Dict[str, Any]]
    available_tools: List[str]
    current_ui_state: str
    current_phone_state: str
    memories: List[Dict[str, str]]
    screenshot_data: bytes


class TaskResult(TypedDict):
    success: bool
    error: Optional[str]
    steps: List[Dict[str, Any]]
    action_count: int
    agent_used: str
    confidence: float


class BaseAgent(ABC):
    """Base class for all expert agents."""

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        api_key: str = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        vision: bool = False,
        base_url: str = None,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.vision = vision
        self.base_url = base_url

    @abstractmethod
    async def execute_task(self, context: TaskContext) -> TaskResult:
        pass

    def _create_agent(self, prompt: str) -> BaseLLMReasoner:
        return BaseLLMReasoner(
            llm_provider=self.llm_provider,
            model_name=self.llm_model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            vision=self.vision,
            base_url=self.base_url,
        )
