from llama_index.core.workflow import Event
from droidrun.agent.context import Reflection, Task
from typing import List, Optional, Any

class CodeActExecuteEvent(Event):
    task: Task
    reflection: Optional[Reflection]

class CodeActResultEvent(Event):
    success: bool
    reason: str
    output: Any
    steps: int

class ReasoningLogicEvent(Event):
    reflection: Optional[Reflection] = None
    force_planning: bool = False

class FinalizeEvent(Event):
    success: bool
    # deprecated. use output instead.
    reason: str
    output: Any
    # deprecated. use tasks instead.
    task: List[Task]
    tasks: List[Task]
    steps: int = 1

class TaskRunnerEvent(Event):
    pass

class ReflectionEvent(Event):
    task: Task
    pass