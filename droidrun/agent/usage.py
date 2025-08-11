from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.llms import LLM, ChatResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from uuid import uuid4
import logging

logger = logging.getLogger("droidrun")
SUPPORTED_PROVIDERS = ["Gemini", "GoogleGenAI", "OpenAI", "Anthropic"]


class UsageResult(BaseModel):
    request_tokens: int
    response_tokens: int
    total_tokens: int
    requests: int


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
        if not EventPayload.RESPONSE in payload:
            raise ValueError("No response in payload")

        chat_rsp: ChatResponse = payload.get(EventPayload.RESPONSE)
        rsp = chat_rsp.raw
        if not rsp:
            raise ValueError("No raw response")

        print(f"rsp: {rsp.__class__.__name__}")

        if self.provider == "Gemini" or self.provider == "GoogleGenAI":
            return UsageResult(
                request_tokens=rsp["usage_metadata"]["prompt_token_count"],
                response_tokens=rsp["usage_metadata"]["candidates_token_count"],
                total_tokens=rsp["usage_metadata"]["total_token_count"],
                requests=1,
            )
        elif self.provider == "OpenAI":
            from openai.types import CompletionUsage as OpenAIUsage

            usage: OpenAIUsage = rsp.usage
            return UsageResult(
                request_tokens=usage.prompt_tokens,
                response_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                requests=1,
            )
        elif self.provider == "Anthropic":
            from anthropic.types import Usage as AnthropicUsage

            usage: AnthropicUsage = rsp["usage"]
            return UsageResult(
                request_tokens=usage.input_tokens,
                response_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
                requests=1,
            )

        raise ValueError(f"Unsupported provider: {self.provider}")

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


def with_callback(llm: LLM, *args: List[BaseCallbackHandler]):
    llm.callback_manager = CallbackManager(args)
    return llm


def track_usage(llm: LLM):
    provider = llm.__class__.__name__
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Tracking not yet supported for provider: {provider}")

    tracker = TokenCountingHandler(provider)
    with_callback(llm, tracker)
    return tracker
