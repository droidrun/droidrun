"""
DroidAgent - A wrapper class that coordinates the planning and execution of tasks
to achieve a user's goal on an Android device.
"""

import logging
from typing import List, Optional, Type

from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import step, StartEvent, StopEvent, Workflow, Context
from llama_index.core.workflow.handler import WorkflowHandler
from pydantic import BaseModel
from droidrun.agent.droid.events import *
from droidrun.agent.codeact import CodeActAgent
from droidrun.agent.codeact.events import EpisodicMemoryEvent
from droidrun.agent.planner import PlannerAgent
from droidrun.agent.context.task_manager import TaskManager
from droidrun.agent.utils.trajectory import Trajectory
from droidrun.agent.utils.structured_output import (
    coerce_to_model,
    schema_instruction,
)
from droidrun.tools import Tools, describe_tools
from droidrun.agent.common.events import ScreenshotEvent, MacroEvent, RecordUIStateEvent
from droidrun.agent.common.default import MockWorkflow
from droidrun.agent.context import ContextInjectionManager
from droidrun.agent.context.agent_persona import AgentPersona
from droidrun.agent.context.personas import DEFAULT
from droidrun.agent.oneflows.reflector import Reflector
from droidrun.telemetry import (
    capture,
    flush,
    DroidAgentInitEvent,
    DroidAgentFinalizeEvent,
)
from droidrun.agent.usage import track_usage

logger = logging.getLogger("droidrun")


class DroidAgent(Workflow):
    """
    A wrapper class that coordinates between PlannerAgent (creates plans) and
        CodeActAgent (executes tasks) to achieve a user's goal.
    """

    @staticmethod
    def _configure_default_logging(debug: bool = False):
        """
        Configure default logging for DroidAgent if no handlers are present.
        This ensures logs are visible when using DroidAgent directly.
        """
        # Only configure if no handlers exist (avoid duplicate configuration)
        if not logger.handlers:
            # Create a console handler
            handler = logging.StreamHandler()

            # Set format
            if debug:
                formatter = logging.Formatter(
                    "%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"
                )
            else:
                formatter = logging.Formatter("%(message)s")

            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if debug else logging.INFO)
            logger.propagate = False

    def __init__(
        self,
        goal: str,
        llm: LLM,
        tools: Tools,
        personas: List[AgentPersona] = [DEFAULT],
        max_steps: int = 15,
        timeout: int = 1000,
        vision: bool = False,
        reasoning: bool = False,
        reflection: bool = False,
        enable_tracing: bool = False,
        debug: bool = False,
        save_trajectories:  str = "none",  
        excluded_tools: List[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the DroidAgent wrapper.

        Args:
            goal: The user's goal or command to execute
            llm: The language model to use for both agents
            max_steps: Maximum number of steps for both agents
            timeout: Timeout for agent execution in seconds
            reasoning: Whether to use the PlannerAgent for complex reasoning (True)
                      or send tasks directly to CodeActAgent (False)
            reflection: Whether to reflect on steps the CodeActAgent did to give the PlannerAgent advice
            enable_tracing: Whether to enable Arize Phoenix tracing
            debug: Whether to enable verbose debug logging
            save_trajectories: Trajectory saving level. Can be:
                - "none" (no saving)
                - "step" (save per step)
                - "action" (save per action)
            **kwargs: Additional keyword arguments to pass to the agents
        """
        self.user_id = kwargs.pop("user_id", None)
        super().__init__(timeout=timeout, *args, **kwargs)
        # Configure default logging if not already configured
        self._configure_default_logging(debug=debug)

        # Setup global tracing first if enabled
        if enable_tracing:
            try:
                from llama_index.core import set_global_handler

                set_global_handler("arize_phoenix")
                logger.info("🔍 Arize Phoenix tracing enabled globally")
            except ImportError:
                logger.warning("⚠️ Arize Phoenix package not found, tracing disabled")
                enable_tracing = False

        self.goal = goal
        self.llm = llm
        self.vision = vision
        self.max_steps = max_steps
        self.max_codeact_steps = max_steps
        self.timeout = timeout
        self.reasoning = reasoning
        self.reflection = reflection
        self.debug = debug

        self.event_counter = 0
        # Handle backward compatibility: bool -> str mapping
        if isinstance(save_trajectories, bool):
            self.save_trajectories = "step" if save_trajectories else "none"
        else:
            # Validate string values
            valid_values = ["none", "step", "action"]
            if save_trajectories not in valid_values:
                logger.warning(f"Invalid save_trajectories value: {save_trajectories}. Using 'none' instead.")
                self.save_trajectories = "none"
            else:
                self.save_trajectories = save_trajectories
        
        self.trajectory = Trajectory(goal=goal)
        self.task_manager = TaskManager()
        self.task_iter = None

        
        self.cim = ContextInjectionManager(personas=personas)
        self.current_episodic_memory = None

        # variables for tracking token usage
        self.token_tracker = track_usage(self.llm)
        self.request_tokens = self.token_tracker.usage.request_tokens  # 0 initially
        self.response_tokens = self.token_tracker.usage.response_tokens  # 0 initially
        self.total_tokens = self.token_tracker.usage.total_tokens  # 0 initially

        logger.info("🤖 Initializing DroidAgent...")
        logger.info(f"💾 Trajectory saving level: {self.save_trajectories}")
        
        self.tool_list = describe_tools(tools, excluded_tools)
        self.tools_instance = tools
        
        self.tools_instance.save_trajectories = self.save_trajectories

        if self.reasoning:
            logger.info("📝 Initializing Planner Agent...")
            self.planner_agent = PlannerAgent(
                goal=goal,
                llm=llm,
                vision=vision,
                personas=personas,
                task_manager=self.task_manager,
                tools_instance=tools,
                timeout=timeout,
                debug=debug,
            )
            self.add_workflows(planner_agent=self.planner_agent)
            self.max_codeact_steps = 5

            if self.reflection:
                self.reflector = Reflector(llm=llm, debug=debug)

        else:
            logger.debug(
                "🚫 Planning disabled - will execute tasks directly with CodeActAgent"
            )
            self.planner_agent = None

        capture(
            DroidAgentInitEvent(
                goal=goal,
                llm=llm.class_name(),
                tools=",".join(self.tool_list),
                personas=",".join([p.name for p in personas]),
                max_steps=max_steps,
                timeout=timeout,
                vision=vision,
                reasoning=reasoning,
                reflection=reflection,
                enable_tracing=enable_tracing,
                debug=debug,
                save_trajectories=save_trajectories,
            ),
            self.user_id,
        )

        logger.info("✅ DroidAgent initialized successfully.")

    def run(self, *args, **kwargs) -> WorkflowHandler:
        """
        Run the DroidAgent workflow.
        """
        return super().run(*args, **kwargs)



    def set_output_schema(self, schema: Type[BaseModel]) -> None:
        """Set a Pydantic schema for structured output.

        When set, the agent will attempt to validate and return a model instance
        from the final result instead of a plain dict.
        """
        self._output_schema = schema

    def _build_structured_output(self, data: dict) -> Optional[BaseModel]:
        """Build a structured output using the configured Pydantic schema (v2)."""
        if not self._output_schema:
            return None
        try:
            return coerce_to_model(self._output_schema, data)
        except Exception as e:
            logger.exception(
                "Structured output validation failed: schema=%s, keys=%s, error=%s",
                getattr(self._output_schema, "__name__", str(self._output_schema)),
                sorted(list(data.keys())),
                e,
            )
            return None

    def get_output_schema_instruction(self) -> Optional[str]:
        """Return a short instruction for LLM prompts based on the configured schema."""
        if not self._output_schema:
            return None
        try:
            return schema_instruction(self._output_schema)
        except Exception as e:
            logger.exception(
                "Failed building schema instruction: schema=%s, error=%s",
                getattr(self._output_schema, "__name__", str(self._output_schema)),
                e,
            )
            return None

    @step
    async def execute_task(
        self, ctx: Context, ev: CodeActExecuteEvent
    ) -> CodeActResultEvent:
        """
        Execute a single task using the CodeActAgent.

        Args:
            task: Task dictionary with description and status

        Returns:
            Tuple of (success, reason)
        """
        task: Task = ev.task
        reflection = ev.reflection if ev.reflection is not None else None
        persona = self.cim.get_persona(task.agent_type)

        logger.info(f"🔧 Executing task: {task.description}")

        try:
            codeact_agent = CodeActAgent(
                llm=self.llm,
                persona=persona,
                vision=self.vision,
                max_steps=self.max_codeact_steps,
                all_tools_list=self.tool_list,
                tools_instance=self.tools_instance,
                debug=self.debug,
                timeout=self.timeout,
            )

            handler = codeact_agent.run(
                input=task.description,
                remembered_info=self.tools_instance.memory,
                reflection=reflection,
            )

            async for nested_ev in handler.stream_events():
                self.handle_stream_event(nested_ev, ctx)

            result = await handler

            # Carry structured output forward if provided by CodeActAgent
            output_payload = result.get("output", result.get("reason"))

            if "success" in result and result["success"]:
                return CodeActResultEvent(
                    success=True,
                    reason=result["reason"],
                    output=output_payload,
                    task=task,
                    steps=len(result.get("codeact_steps", [])),  # Convert to integer
                )
            else:
                return CodeActResultEvent(
                    success=False,
                    reason=result.get("reason", ""),
                    output=output_payload,
                    task=task,
                    steps=len(result.get("codeact_steps", [])),  # Convert to integer
                )

        except Exception as e:
            logger.error(f"Error during task execution: {e}")
            if self.debug:
                import traceback

                logger.error(traceback.format_exc())
            err = f"Error: {str(e)}"
            return CodeActResultEvent(
                success=False,
                reason=err,
                output=err,  # Required field
                task=task,
                steps=0  # Must be an integer
            )

    @step
    async def handle_codeact_execute(self, ctx: Context, ev: CodeActResultEvent) -> FinalizeEvent | ReflectionEvent | ReasoningLogicEvent:
        try:
            task = ev.task
            
            # Store structured output if available
            if isinstance(ev.output, dict):
                self._last_structured_output = ev.output
            
            if not self.reasoning:
                return FinalizeEvent(
                    success=ev.success,
                    reason=ev.reason,
                    output=ev.output if getattr(ev, "output", None) is not None else ev.reason,
                    task=[task],
                    tasks=[task],
                    steps=ev.steps,
                )
            
            # If we have successful output in reasoning mode, finalize
            if ev.success and hasattr(self, '_last_structured_output') and self._last_structured_output.get('success', False):
                tasks = self.task_manager.get_task_history()
                return FinalizeEvent(
                    success=True,
                    reason="Task completed successfully",
                    output=self._last_structured_output,
                    task=tasks,
                    tasks=tasks,
                    steps=ev.steps,
                )
                
            if self.reflection and ev.success:
                return ReflectionEvent(task=task)

            # Reasoning is enabled but reflection is disabled.
            # Success: mark complete and proceed to next step in reasoning loop.
            # Failure: mark failed and trigger planner immediately without advancing to the next queued task.
            if ev.success:
                self.task_manager.complete_task(task, message=ev.reason)
                return ReasoningLogicEvent()
            else:
                self.task_manager.fail_task(task, failure_reason=ev.reason)
                return ReasoningLogicEvent(force_planning=True)

        except Exception as e:
            logger.error(f"❌ Error during DroidAgent execution: {e}")
            if self.debug:
                import traceback

                logger.error(traceback.format_exc())
            tasks = self.task_manager.get_task_history()
            return FinalizeEvent(
                success=False,
                reason=str(e),
                output=str(e),
                task=tasks,
                tasks=tasks,
                steps=self.step_counter,
            )

    @step
    async def reflect(
        self, ctx: Context, ev: ReflectionEvent
    ) -> ReasoningLogicEvent | CodeActExecuteEvent:

        task = ev.task
        if ev.task.agent_type == "AppStarterExpert":
            self.task_manager.complete_task(task)
            return ReasoningLogicEvent()

        reflection = await self.reflector.reflect_on_episodic_memory(
            episodic_memory=self.current_episodic_memory, goal=task.description
        )

        if reflection.goal_achieved:
            self.task_manager.complete_task(task)
            return ReasoningLogicEvent()

        else:
            self.task_manager.fail_task(task)
            return ReasoningLogicEvent(reflection=reflection)

    @step
    async def handle_reasoning_logic(
        self,
        ctx: Context,
        ev: ReasoningLogicEvent,
        planner_agent: Workflow = MockWorkflow(),
    ) -> FinalizeEvent | CodeActExecuteEvent:
        try:
            if self.step_counter >= self.max_steps:
                output = {
                    "success": False,
                    "output": f"Reached maximum number of steps ({self.max_steps})",
                    "reason": f"Reached maximum number of steps ({self.max_steps})"
                }
                tasks = self.task_manager.get_task_history()
                # Store structured output for schema validation
                self._last_structured_output = output
                return FinalizeEvent(
                    success=False,
                    reason=output["reason"],
                    output=output,  # Pass structured output
                    task=tasks,
                    tasks=tasks,
                    steps=self.step_counter,
                )
            
            # If we have a completed task with structured output, finalize
            if hasattr(self, '_last_structured_output') and self._last_structured_output.get('success', False):
                tasks = self.task_manager.get_task_history()
                return FinalizeEvent(
                    success=True,
                    reason="Task completed successfully",
                    output=self._last_structured_output,
                    task=tasks,
                    tasks=tasks,
                    steps=self.step_counter,
                )

            self.step_counter += 1

            if ev.reflection:
                handler = planner_agent.run(
                    remembered_info=self.tools_instance.memory, reflection=ev.reflection
                )
            else:
                if not ev.force_planning and self.task_iter:
                    try:
                        task = next(self.task_iter)
                        return CodeActExecuteEvent(task=task, reflection=None)
                    except StopIteration as e:
                        logger.info("Planning next steps...")

                logger.debug(f"Planning step {self.step_counter}/{self.max_steps}")

                handler = planner_agent.run(
                    remembered_info=self.tools_instance.memory, reflection=None
                )

            async for nested_ev in handler.stream_events():
                self.handle_stream_event(nested_ev, ctx)

            result = await handler

            self.tasks = self.task_manager.get_all_tasks()
            self.task_iter = iter(self.tasks)

            if self.task_manager.goal_completed:
                logger.info(f"✅ Goal completed: {self.task_manager.message}")
                tasks = self.task_manager.get_task_history()
                return FinalizeEvent(
                    success=True,
                    reason=self.task_manager.message,
                    output=self.task_manager.message,
                    task=tasks,
                    tasks=tasks,
                    steps=self.step_counter,
                )
            if not self.tasks:
                logger.warning("No tasks generated by planner")
                output = "Planner did not generate any tasks"
                tasks = self.task_manager.get_task_history()
                return FinalizeEvent(
                    success=False,
                    reason=output,
                    output=output,
                    task=tasks,
                    tasks=tasks,
                    steps=self.step_counter,
                )

            return CodeActExecuteEvent(task=next(self.task_iter), reflection=None)

        except Exception as e:
            logger.error(f"❌ Error during DroidAgent execution: {e}")
            if self.debug:
                import traceback

                logger.error(traceback.format_exc())
            tasks = self.task_manager.get_task_history()
            return FinalizeEvent(
                success=False,
                reason=str(e),
                output=str(e),
                task=tasks,
                tasks=tasks,
                steps=self.step_counter,
            )

    @step
    async def start_handler(
        self, ctx: Context, ev: StartEvent
    ) -> CodeActExecuteEvent | ReasoningLogicEvent:
        """
        Main execution loop that coordinates between planning and execution.

        Returns:
            Dict containing the execution result
        """
        logger.info(f"🚀 Running DroidAgent to achieve goal: {self.goal}")
        ctx.write_event_to_stream(ev)

        self.step_counter = 0
        self.retry_counter = 0

        if not self.reasoning:
            logger.info(f"🔄 Direct execution mode - executing goal: {self.goal}")
            task = Task(
                description=self.goal,
                status=self.task_manager.STATUS_PENDING,
                agent_type="Default",
            )

            return CodeActExecuteEvent(task=task, reflection=None)

        return ReasoningLogicEvent()

    @step
    async def finalize(self, ctx: Context, ev: FinalizeEvent) -> StopEvent:
        ctx.write_event_to_stream(ev)
        capture(
            DroidAgentFinalizeEvent(
                tasks=",".join([f"{t.agent_type}:{t.description}" for t in ev.task]),
                success=ev.success,
                output=ev.output,
                steps=ev.steps,
            ),
            self.user_id,
        )
        flush()

        result = {
            "success": ev.success,
            # deprecated. use output instead.
            "reason": ev.reason,
            "output": ev.output,
            "steps": ev.steps,
        }

        # If a schema is configured, attempt to coerce structured output
        if getattr(self, "_output_schema", None):
            # Case 1: direct dict output from execution
            if isinstance(ev.output, dict):
                model = coerce_to_model(self._output_schema, ev.output)
                if model is not None:
                    if self.trajectory and self.save_trajectories != "none":
                        self.trajectory.save_trajectory()
                    return StopEvent(model)
            # Case 2: reasoning-mode string output but we captured a structured payload earlier
            if self.reasoning and hasattr(self, "_last_structured_output") and isinstance(self._last_structured_output, dict):
                model = coerce_to_model(self._output_schema, self._last_structured_output)
                if model is not None:
                    if self.trajectory and self.save_trajectories != "none":
                        self.trajectory.save_trajectory()
                    return StopEvent(model)

        if self.trajectory and self.save_trajectories != "none":
            self.trajectory.save_trajectory()

        return StopEvent(result)

    def handle_stream_event(self, ev: Event, ctx: Context):

        if isinstance(ev, EpisodicMemoryEvent):
            self.current_episodic_memory = ev.episodic_memory
            return
        
  

        if not isinstance(ev, StopEvent):
            ctx.write_event_to_stream(ev)

            if isinstance(ev, ScreenshotEvent):
                self.trajectory.screenshots.append(ev.screenshot)
            elif isinstance(ev, MacroEvent):
                self.trajectory.macro.append(ev)
            elif isinstance(ev, RecordUIStateEvent):
                self.trajectory.ui_states.append(ev.ui_state)

            else:
                # appending token count for each event
                # current event token count = total token count - token count upto previous event
                tokens_data = {
                    "prompt_tokens": self.token_tracker.usage.request_tokens
                    - self.request_tokens,
                    "completion_tokens": self.token_tracker.usage.response_tokens
                    - self.response_tokens,
                    "total_tokens": self.token_tracker.usage.total_tokens
                    - self.total_tokens,
                }
                ev.tokens = tokens_data

                # Debug: Verify the tokens attribute was set
                logger.debug(f"Set tokens on {ev.__class__.__name__}: {tokens_data}")
                logger.debug(f"Event now has tokens attribute: {hasattr(ev, 'tokens')}")
                if hasattr(ev, "tokens"):
                    logger.debug(f"Event tokens value: {ev.tokens}")

                # updating the self token counts
                self.request_tokens = self.token_tracker.usage.request_tokens
                self.response_tokens = self.token_tracker.usage.response_tokens
                self.total_tokens = self.token_tracker.usage.total_tokens

                self.trajectory.events.append(ev)
