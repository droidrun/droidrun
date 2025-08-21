from dataclasses import dataclass, field
from droidrun.agent.context.agent_persona import AgentPersona
from typing import List, Optional

@dataclass
class EpisodicMemoryStep:
    chat_history: str
    response: str
    timestamp: float
    screenshot: Optional[bytes]
    tokens: Optional[dict] = None

@dataclass 
class EpisodicMemory:
    persona: AgentPersona
    steps: List[EpisodicMemoryStep] = field(default_factory=list)