import contextlib
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.llms import LLM, ChatResponse
from pydantic import BaseModel

logger = logging.getLogger("droidrun")
SUPPORTED_PROVIDERS = [
    "Gemini",
    "GoogleGenAI",
    "GenAI",
    "OpenAI",
    "openai_llm",
    "Anthropic",
    "Ollama",
    "DeepSeek",
]


class UsageResult(BaseModel):
    request_tokens: int
    response_tokens: int
    total_tokens: int
    requests: int


def get_usage_from_response(provider: str, chat_rsp: ChatResponse) -> UsageResult:
    rsp = chat_rsp.raw
    if not rsp:
        raise ValueError("No raw response in chat response")

    if provider == "Gemini" or provider == "GoogleGenAI" or provider == "GenAI":
        return UsageResult(
            request_tokens=rsp["usage_metadata"]["prompt_token_count"],
            response_tokens=rsp["usage_metadata"]["candidates_token_count"],
            total_tokens=rsp["usage_metadata"]["total_token_count"],
            requests=1,
        )
    elif provider == "OpenAI" or provider == "OpenAILike" or provider == "openai_llm":
        from openai.types import CompletionUsage as OpenAIUsage

        usage: OpenAIUsage = rsp.usage
        return UsageResult(
            request_tokens=usage.prompt_tokens,
            response_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            requests=1,
        )
    elif provider == "Anthropic_LLM":
        from anthropic.types import Usage as AnthropicUsage

        usage: AnthropicUsage = rsp["usage"]
        return UsageResult(
            request_tokens=usage.input_tokens,
            response_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            requests=1,
        )
    elif provider == "Ollama":
        # Ollama response format uses different field names
        prompt_eval_count = rsp.get("prompt_eval_count", 0)
        eval_count = rsp.get("eval_count", 0)
        return UsageResult(
            request_tokens=prompt_eval_count,
            response_tokens=eval_count,
            total_tokens=prompt_eval_count + eval_count,
            requests=1,
        )
    elif provider == "DeepSeek":
        # DeepSeek follows OpenAI-compatible format
        usage = rsp.usage
        if not usage:
            usage = {}
        return UsageResult(
            request_tokens=usage.prompt_tokens or 0,
            response_tokens=usage.completion_tokens or 0,
            total_tokens=usage.total_tokens or 0,
            requests=1,
        )

    raise ValueError(f"Unsupported provider: {provider}")


class TokenCountingHandler(BaseCallbackHandler):
    """Token counting handler for LLamaIndex LLM calls."""

    def __init__(self, provider: str):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.provider = provider
        self.request_tokens: int = 0
        self.response_tokens: int = 0
        self.total_tokens: int = 0
        self.requests: int = 0

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "TokenCountingHandler"

    @property
    def usage(self) -> UsageResult:
        return UsageResult(
            request_tokens=self.request_tokens,
            response_tokens=self.response_tokens,
            total_tokens=self.total_tokens,
            requests=self.requests,
        )

    def _get_event_usage(self, payload: Dict[str, Any]) -> UsageResult:
        if EventPayload.RESPONSE not in payload:
            raise ValueError("No response in payload")

        chat_rsp: ChatResponse = payload.get(EventPayload.RESPONSE)
        return get_usage_from_response(self.provider, chat_rsp)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""
        return event_id or str(uuid4())

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""
        try:
            usage = self._get_event_usage(payload)

            self.request_tokens += usage.request_tokens
            self.response_tokens += usage.response_tokens
            self.total_tokens += usage.total_tokens
            self.requests += usage.requests
        except Exception as e:
            self.requests += 1
            logger.warning(
                f"Error tracking usage for provider {self.provider}: {e}",
                extra={"provider": self.provider},
            )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
        pass


@contextlib.contextmanager
def llm_callback(llm: LLM, *args: List[BaseCallbackHandler]):
    for arg in args:
        llm.callback_manager.add_handler(arg)
    yield
    for arg in args:
        llm.callback_manager.remove_handler(arg)


def create_tracker(llm: LLM) -> TokenCountingHandler:
    provider = llm.__class__.__name__
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Tracking not yet supported for provider: {provider}")

    return TokenCountingHandler(provider)


def track_usage(llm: LLM) -> TokenCountingHandler:
    """Track token usage for an LLM instance across all requests.

    This function:
    - Creates a new TokenCountingHandler for the LLM provider
    - Registers that handler as an LLM callback to monitor all requests
    - Returns the handler for accessing cumulative usage statistics

    The handler counts tokens for total LLM usage across all requests. For fine-grained
    per-request counting, use either:
    - `create_tracker()` with `llm_callback()` context manager for temporary tracking
    - `get_usage_from_response()` to extract usage from individual responses

    Args:
        llm: The LLamaIndex LLM instance to track usage for

    Returns:
        TokenCountingHandler: The registered handler that accumulates usage statistics

    Raises:
        ValueError: If the LLM provider is not supported for tracking

    Example:
        >>> llm = OpenAI()
        >>> tracker = track_usage(llm)
        >>> # ... make LLM calls ...
        >>> print(f"Total tokens used: {tracker.usage.total_tokens}")
    """
    provider = llm.__class__.__name__
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Tracking not yet supported for provider: {provider}")

    tracker = TokenCountingHandler(provider)
    llm.callback_manager.add_handler(tracker)
    return tracker
