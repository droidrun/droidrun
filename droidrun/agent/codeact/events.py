from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Event
from typing import Optional
from ..context.episodic_memory import EpisodicMemory

class TaskInputEvent(Event):
    input: list[ChatMessage]



class TaskThinkingEvent(Event):
    thoughts: Optional[str] = None
    code: Optional[str] = None  
    tokens: Optional[dict] = None

class TaskExecutionEvent(Event):
    code: str
    globals: dict[str, str] = {}
    locals: dict[str, str] = {}

class TaskExecutionResultEvent(Event):
    output: str

class TaskEndEvent(Event):
    success: bool
    reason: str

class EpisodicMemoryEvent(Event):
    episodic_memory: EpisodicMemory