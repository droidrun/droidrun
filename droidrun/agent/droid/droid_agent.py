"""
DroidAgent - A wrapper class that coordinates the planning and execution of tasks
to achieve a user's goal on an Android device.

Architecture:
- When reasoning=False: Uses CodeActAgent directly
- When reasoning=True: Uses Manager (planning) + Executor (action) workflows
"""

import logging
from typing import TYPE_CHECKING, Optional, Type
import llama_index.core
from pydantic import BaseModel
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from llama_index.core.workflow.handler import WorkflowHandler
from workflows.events import Event

from droidrun.agent.codeact import CodeActAgent
from droidrun.agent.codeact.events import EpisodicMemoryEvent
from droidrun.agent.common.events import MacroEvent, RecordUIStateEvent, ScreenshotEvent
from droidrun.agent.context.task_manager import Task, TaskManager
from droidrun.agent.droid.events import (
    CodeActExecuteEvent,
    CodeActResultEvent,
    DroidAgentState,
    ExecutorInputEvent,
    ExecutorResultEvent,
    FinalizeEvent,
    ManagerInputEvent,
    ManagerPlanEvent,
    ScripterExecutorInputEvent,
    ScripterExecutorResultEvent,
)
from droidrun.agent.executor import ExecutorAgent
from droidrun.agent.manager import ManagerAgent
from droidrun.agent.scripter import ScripterAgent
from droidrun.agent.oneflows.structured_output_agent import StructuredOutputAgent
from droidrun.agent.utils.async_utils import wrap_async_tools
from droidrun.agent.utils.llm_loader import load_agent_llms, validate_llm_dict
from droidrun.agent.utils.tools import (
    ATOMIC_ACTION_SIGNATURES,
    build_custom_tools,
    resolve_tools_instance,
)
from droidrun.agent.utils.trajectory import Trajectory
from droidrun.agent.memory import (
    TrajectoryMemory,
    create_memory_client,
    parse_trajectory_for_manual,
)
from droidrun.config_manager.config_manager import (
    AgentConfig,
    CredentialsConfig,
    DeviceConfig,
    DroidRunConfig,
    LoggingConfig,
    TelemetryConfig,
    ToolsConfig,
    TracingConfig,
)
from droidrun.credential_manager import load_credential_manager
from droidrun.telemetry import (
    DroidAgentFinalizeEvent,
    DroidAgentInitEvent,
    capture,
    flush,
)
from droidrun.telemetry.phoenix import arize_phoenix_callback_handler

if TYPE_CHECKING:
    from droidrun.tools import Tools

logger = logging.getLogger("droidrun")


class DroidAgent(Workflow):
    """
    A wrapper class that coordinates between agents to achieve a user's goal.

    Reasoning modes:
    - reasoning=False: Uses CodeActAgent directly for immediate execution
    - reasoning=True: Uses ManagerAgent (planning) + ExecutorAgent (actions)
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
        config: DroidRunConfig | None = None,
        llms: dict[str, LLM] | LLM | None = None,
        agent_config: AgentConfig | None = None,
        device_config: DeviceConfig | None = None,
        tools: "Tools | ToolsConfig | None" = None,
        logging_config: LoggingConfig | None = None,
        tracing_config: TracingConfig | None = None,
        telemetry_config: TelemetryConfig | None = None,
        custom_tools: dict = None,
        memory_client: Optional[TrajectoryMemory] = None,
        reference_trajectory_manual: Optional[str] = None,
        credentials: "CredentialsConfig | dict | None" = None,
        variables: dict | None = None,
        output_model: Type[BaseModel] | None = None,
        timeout: int = 1000,
        *args,
        **kwargs,
    ):
        """
        Initialize the DroidAgent wrapper.

        Args:
            goal: User's goal or command
            config: Full config (required if llms not provided)
            llms: Optional dict of agent-specific LLMs or single LLM for all.
                  If not provided, LLMs will be loaded from config profiles.
            agent_config: Agent config override (optional)
            device_config: Device config override (optional)
            tools: Either a Tools instance (for custom/pre-configured tools),
                   ToolsConfig (for config-based creation), or None (use default).
                   Renamed from tools_config to support both instances and config.
            logging_config: Logging config override (optional)
            tracing_config: Tracing config override (optional)
            telemetry_config: Telemetry config override (optional)
            custom_tools: Custom tool definitions
            memory_client: Optional trajectory memory client override
            reference_trajectory_manual: Optional reference manual to inject into planning
            credentials: Either CredentialsConfig (from config.credentials),
                        dict of credentials {"SECRET_ID": "value"}, or None
            variables: Optional dict of custom variables accessible throughout execution
            output_model: Optional Pydantic model for structured output extraction from final answer
            timeout: Workflow timeout in seconds
        """

        self.user_id = kwargs.pop("user_id", None)
        self.runtype = kwargs.pop("runtype", "developer")
        self.goal = goal
        self.reference_trajectory_manual = reference_trajectory_manual
        self.shared_state = DroidAgentState(instruction=goal, err_to_manager_thresh=2)
        self.output_model = output_model
        base_config = config

        # Store custom variables in shared state
        if variables:
            self.shared_state.custom_variables = variables

        # Load credential manager (supports both config and direct dict)
        # Priority: explicit credentials param > base_config.credentials
        credentials_source = (
            credentials
            if credentials is not None
            else (base_config.credentials if base_config else None)
        )
        credential_manager = load_credential_manager(credentials_source)

        # Resolve tools instance (supports Tools instance, ToolsConfig, or None)
        # Use tools param or fallback to base_config.tools
        tools_fallback = (
            tools if tools is not None else (base_config.tools if base_config else None)
        )
        resolved_device_config = device_config or (
            base_config.device if base_config else DeviceConfig()
        )
        tools_instance, tools_config_resolved = resolve_tools_instance(
            tools=tools_fallback,
            device_config=resolved_device_config,
            tools_config_fallback=base_config.tools if base_config else None,
            credential_manager=credential_manager,
        )

        # Build final config with resolved tools config
        self.config = DroidRunConfig(
            agent=agent_config or (base_config.agent if base_config else AgentConfig()),
            device=resolved_device_config,
            tools=tools_config_resolved,
            logging=logging_config
            or (base_config.logging if base_config else LoggingConfig()),
            tracing=tracing_config
            or (base_config.tracing if base_config else TracingConfig()),
            telemetry=telemetry_config
            or (base_config.telemetry if base_config else TelemetryConfig()),
            llm_profiles=base_config.llm_profiles if base_config else {},
            credentials=base_config.credentials if base_config else CredentialsConfig(),
        )

        llm_override_keys = [
            "custom_provider",
            "custom_model",
            "temperature",
            "base_url",
            "api_base",
        ]
        llm_kwargs = {}
        for key in llm_override_keys:
            if key in kwargs:
                llm_kwargs[key] = kwargs.pop(key)

        super().__init__(*args, timeout=timeout, **kwargs)

        self._configure_default_logging(debug=self.config.logging.debug)

        # Load LLMs if not provided
        if llms is None:
            if config is None:
                raise ValueError(
                    "Either 'llms' or 'config' must be provided. "
                    "If llms is not provided, config is required to load LLMs from profiles."
                )

            logger.info("🔄 Loading LLMs from config (llms not provided)...")

            llms = load_agent_llms(
                config=self.config,
                output_model=output_model,
                **llm_kwargs,
            )
        if isinstance(llms, dict):
            validate_llm_dict(self.config, llms, output_model=output_model)
        elif isinstance(llms, LLM):
            pass
        else:
            raise ValueError(f"Invalid LLM type: {type(llms)}")

        if self.config.tracing.enabled:
            try:
                handler = arize_phoenix_callback_handler()
                llama_index.core.global_handler = handler
                logger.info("🔍 Arize Phoenix tracing enabled globally")
            except ImportError:
                logger.warning(
                    "⚠️  Arize Phoenix is not installed.\n"
                    "    To enable Phoenix integration, install with:\n"
                    "    • If installed via tool: `uv tool install droidrun[phoenix]`"
                    "    • If installed via pip: `uv pip install droidrun[phoenix]`\n"
                )

        self.timeout = timeout
        self.memory_client: TrajectoryMemory = memory_client or create_memory_client(
            config=self.config
        )

        if isinstance(llms, dict):
            self.manager_llm = llms.get("manager")
            self.executor_llm = llms.get("executor")
            self.codeact_llm = llms.get("codeact")
            self.text_manipulator_llm = llms.get("text_manipulator")
            self.app_opener_llm = llms.get("app_opener")
            self.scripter_llm = llms.get("scripter", self.codeact_llm)
            self.structured_output_llm = llms.get("structured_output", self.codeact_llm)

            logger.info("📚 Using agent-specific LLMs from dictionary")
        else:
            logger.info("📚 Using single LLM for all agents")
            self.manager_llm = llms
            self.executor_llm = llms
            self.codeact_llm = llms
            self.text_manipulator_llm = llms
            self.app_opener_llm = llms
            self.scripter_llm = llms
            self.structured_output_llm = llms

        self.trajectory = Trajectory(goal=self.shared_state.instruction)
        self.task_manager = TaskManager()
        self.task_iter = None
        self.current_episodic_memory = None

        self.atomic_tools = ATOMIC_ACTION_SIGNATURES.copy()

        # Build custom tools (credentials + open_app + user custom tools)
        auto_custom_tools = build_custom_tools(credential_manager)
        self.custom_tools = {**auto_custom_tools, **(custom_tools or {})}

        logger.info("🤖 Initializing DroidAgent...")
        logger.info(f"💾 Trajectory saving: {self.config.logging.save_trajectory}")

        self.tools_instance = tools_instance
        self.tools_instance.save_trajectories = self.config.logging.save_trajectory
        # Set LLMs on tools instance for helper tools
        self.tools_instance.app_opener_llm = self.app_opener_llm
        self.tools_instance.text_manipulator_llm = self.text_manipulator_llm

        if (
            self.config.agent.reasoning
            and self.memory_client.enabled
            and not self.reference_trajectory_manual
        ):
            manual = self.memory_client.fetch_reference_manual(goal)
            if manual:
                logger.info("📖 Loaded reference trajectory from memory service.")
                self.reference_trajectory_manual = manual

        if self.config.agent.reasoning:
            logger.info("📝 Initializing Manager and Executor Agents...")
            self.manager_agent = ManagerAgent(
                llm=self.manager_llm,
                tools_instance=tools_instance,
                shared_state=self.shared_state,
                agent_config=self.config.agent,
                custom_tools=self.custom_tools,
                output_model=self.output_model,
                timeout=timeout,
                reference_trajectory_manual=self.reference_trajectory_manual,
            )
            self.executor_agent = ExecutorAgent(
                llm=self.executor_llm,
                tools_instance=tools_instance,
                shared_state=self.shared_state,
                agent_config=self.config.agent,
                custom_tools=self.custom_tools,
                timeout=timeout,
            )
            self.planner_agent = None
        else:
            logger.debug("🚫 Reasoning disabled - executing directly with CodeActAgent")
            self.manager_agent = None
            self.executor_agent = None
            self.planner_agent = None

        atomic_tools = list(ATOMIC_ACTION_SIGNATURES.keys())

        capture(
            DroidAgentInitEvent(
                goal=self.shared_state.instruction,
                llms={
                    "manager": (
                        self.manager_llm.class_name() if self.manager_llm else "None"
                    ),
                    "executor": (
                        self.executor_llm.class_name() if self.executor_llm else "None"
                    ),
                    "codeact": (
                        self.codeact_llm.class_name() if self.codeact_llm else "None"
                    ),
                    "text_manipulator": (
                        self.text_manipulator_llm.class_name()
                        if self.text_manipulator_llm
                        else "None"
                    ),
                    "app_opener": (
                        self.app_opener_llm.class_name()
                        if self.app_opener_llm
                        else "None"
                    ),
                },
                tools=",".join(atomic_tools + ["remember", "complete"]),
                max_steps=self.config.agent.max_steps,
                timeout=timeout,
                vision={
                    "manager": self.config.agent.manager.vision,
                    "executor": self.config.agent.executor.vision,
                    "codeact": self.config.agent.codeact.vision,
                },
                reasoning=self.config.agent.reasoning,
                enable_tracing=self.config.tracing.enabled,
                debug=self.config.logging.debug,
                save_trajectories=self.config.logging.save_trajectory,
                runtype=self.runtype,
            ),
            self.user_id,
        )

        logger.info("✅ DroidAgent initialized successfully.")

    def run(self, *args, **kwargs) -> WorkflowHandler:
        """
        Run the DroidAgent workflow.
        """
        return super().run(*args, **kwargs)

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

        logger.info(f"🔧 Executing task: {task.description}")

        try:
            codeact_agent = CodeActAgent(
                llm=self.codeact_llm,
                agent_config=self.config.agent,
                tools_instance=self.tools_instance,
                custom_tools=self.custom_tools,
                atomic_tools=self.atomic_tools,
                debug=self.config.logging.debug,
                shared_state=self.shared_state,
                safe_execution_config=self.config.safe_execution,
                output_model=self.output_model,
                timeout=self.timeout,
            )

            handler = codeact_agent.run(
                input=task.description,
                remembered_info=self.tools_instance.memory,
            )

            async for nested_ev in handler.stream_events():
                self.handle_stream_event(nested_ev, ctx)

            result = await handler

            if "success" in result and result["success"]:
                return CodeActResultEvent(
                    success=True,
                    reason=result["reason"],
                    task=task,
                )
            else:
                return CodeActResultEvent(
                    success=False,
                    reason=result["reason"],
                    task=task,
                )

        except Exception as e:
            logger.error(f"Error during task execution: {e}")
            if self.config.logging.debug:
                import traceback

                logger.error(traceback.format_exc())
            return CodeActResultEvent(
                success=False, reason=f"Error: {str(e)}", task=task
            )

    @step
    async def handle_codeact_execute(
        self, ctx: Context, ev: CodeActResultEvent
    ) -> FinalizeEvent:
        try:
            return FinalizeEvent(success=ev.success, reason=ev.reason)
        except Exception as e:
            logger.error(f"❌ Error during DroidAgent execution: {e}")
            if self.config.logging.debug:
                import traceback

                logger.error(traceback.format_exc())
            return FinalizeEvent(
                success=False,
                reason=str(e),
            )

    @step
    async def start_handler(
        self, ctx: Context, ev: StartEvent
    ) -> CodeActExecuteEvent | ManagerInputEvent:
        """
        Main execution loop that coordinates between planning and execution.

        Returns:
            Event to trigger next step based on reasoning mode
        """
        logger.info(
            f"🚀 Running DroidAgent to achieve goal: {self.shared_state.instruction}"
        )
        ctx.write_event_to_stream(ev)

        self.tools_instance._set_context(ctx)

        if not hasattr(self, "_tools_wrapped") and not self.config.agent.reasoning:

            self.atomic_tools = wrap_async_tools(self.atomic_tools)
            self.custom_tools = wrap_async_tools(self.custom_tools)

            self._tools_wrapped = True
            logger.debug("✅ Async tools wrapped for synchronous execution contexts")

        if not self.config.agent.reasoning:
            logger.info(
                f"🔄 Direct execution mode - executing goal: {self.shared_state.instruction}"
            )
            task = Task(
                description=self.shared_state.instruction,
                status=self.task_manager.STATUS_PENDING,
                agent_type="Default",
            )
            return CodeActExecuteEvent(task=task)

        logger.info("🧠 Reasoning mode - initializing Manager/Executor workflow")
        return ManagerInputEvent()

    # ========================================================================
    # Manager/Executor Workflow Steps
    # ========================================================================

    @step
    async def run_manager(
        self, ctx: Context, ev: ManagerInputEvent
    ) -> ManagerPlanEvent | FinalizeEvent:
        """
        Run Manager planning phase.

        Pre-flight checks for termination before running manager.
        The Manager analyzes current state and creates a plan with subgoals.
        """
        if self.shared_state.step_number >= self.config.agent.max_steps:
            logger.warning(f"⚠️ Reached maximum steps ({self.config.agent.max_steps})")
            return FinalizeEvent(
                success=False,
                reason=f"Reached maximum steps ({self.config.agent.max_steps})",
            )

        logger.info(
            f"📋 Running Manager for planning... (step {self.shared_state.step_number}/{self.config.agent.max_steps})"
        )

        # Run Manager workflow
        handler = self.manager_agent.run()

        # Stream nested events
        async for nested_ev in handler.stream_events():
            self.handle_stream_event(nested_ev, ctx)

        result = await handler

        # Manager already updated shared_state, just return event with results
        return ManagerPlanEvent(
            plan=result["plan"],
            current_subgoal=result["current_subgoal"],
            thought=result["thought"],
            manager_answer=result.get("manager_answer", ""),
        )

    @step
    async def handle_manager_plan(
        self, ctx: Context, ev: ManagerPlanEvent
    ) -> ExecutorInputEvent | ScripterExecutorInputEvent | FinalizeEvent:
        """
        Process Manager output and decide next step.

        Checks if task is complete, if ScripterAgent should run, or if Executor should take action.
        """
        # Check for answer-type termination
        if ev.manager_answer.strip():
            logger.info(f"💬 Manager provided answer: {ev.manager_answer}")
            self.shared_state.progress_status = f"Answer: {ev.manager_answer}"

            return FinalizeEvent(success=True, reason=ev.manager_answer)

        # Check for <script> tag in current_subgoal, then extract from full plan
        if "<script>" in ev.current_subgoal:
            # Found script tag in subgoal - now search the entire plan
            start_idx = ev.plan.find("<script>")
            end_idx = ev.plan.find("</script>")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                # Extract content between first <script> and first </script> in plan
                task = ev.plan[start_idx + len("<script>") : end_idx].strip()
                logger.info(f"🐍 Routing to ScripterAgent: {task[:80]}...")
                return ScripterExecutorInputEvent(task=task)
            else:
                # <script> found in subgoal but not properly closed in plan - log warning
                logger.warning(
                    "⚠️ Found <script> in subgoal but not properly closed in plan, treating as regular subgoal"
                )

        # Continue to Executor with current subgoal
        logger.info(f"▶️  Proceeding to Executor with subgoal: {ev.current_subgoal}")
        return ExecutorInputEvent(current_subgoal=ev.current_subgoal)

    @step
    async def run_executor(
        self, ctx: Context, ev: ExecutorInputEvent
    ) -> ExecutorResultEvent:
        """
        Run Executor action phase.

        The Executor selects and executes a specific action for the current subgoal.
        """
        logger.info("⚡ Running Executor for action...")

        # Run Executor workflow (Executor will update shared_state directly)
        handler = self.executor_agent.run(subgoal=ev.current_subgoal)

        # Stream nested events
        async for nested_ev in handler.stream_events():
            self.handle_stream_event(nested_ev, ctx)

        result = await handler

        # Update coordination state after execution
        self.shared_state.action_history.append(result["action"])
        self.shared_state.summary_history.append(result["summary"])
        self.shared_state.action_outcomes.append(result["outcome"])
        self.shared_state.error_descriptions.append(result["error"])
        self.shared_state.last_action = result["action"]
        self.shared_state.last_summary = result["summary"]
        self.shared_state.last_action_thought = result.get("thought", "")
        self.shared_state.action_pool.append(result["action_json"])

        return ExecutorResultEvent(
            action=result["action"],
            outcome=result["outcome"],
            error=result["error"],
            summary=result["summary"],
        )

    @step
    async def handle_executor_result(
        self, ctx: Context, ev: ExecutorResultEvent
    ) -> ManagerInputEvent:
        """
        Process Executor result and continue.

        Checks for error escalation and loops back to Manager.
        Note: Max steps check is now done in run_manager pre-flight.
        """
        # Check error escalation and reset flag when errors are resolved
        err_thresh = self.shared_state.err_to_manager_thresh

        if len(self.shared_state.action_outcomes) >= err_thresh:
            latest = self.shared_state.action_outcomes[-err_thresh:]
            error_count = sum(1 for o in latest if not o)
            if error_count == err_thresh:
                logger.warning(f"⚠️ Error escalation: {err_thresh} consecutive errors")
                self.shared_state.error_flag_plan = True
            else:
                if self.shared_state.error_flag_plan:
                    logger.info("✅ Error resolved - resetting error flag")
                self.shared_state.error_flag_plan = False

        self.shared_state.step_number += 1
        logger.info(
            f"🔄 Step {self.shared_state.step_number}/{self.config.agent.max_steps} complete, looping to Manager"
        )

        return ManagerInputEvent()

    # ========================================================================
    # Script Executor Workflow Steps
    # ========================================================================

    @step
    async def run_scripter(
        self, ctx: Context, ev: ScripterExecutorInputEvent
    ) -> ScripterExecutorResultEvent:
        """
        Instantiate and run ScripterAgent for off-device operations.
        """
        logger.info(f"🐍 Starting ScripterAgent for task: {ev.task[:2000]}...")

        # Create fresh ScripterAgent instance for this task
        scripter_agent = ScripterAgent(
            llm=self.scripter_llm,
            agent_config=self.config.agent,
            shared_state=self.shared_state,
            task=ev.task,
            safe_execution_config=self.config.safe_execution,
            timeout=self.timeout,
        )

        # Run ScripterAgent workflow
        handler = scripter_agent.run()

        # Stream nested events
        async for nested_ev in handler.stream_events():
            ctx.write_event_to_stream(nested_ev)

        result = await handler

        # Store in shared state
        script_record = {
            "task": ev.task,
            "message": result["message"],
            "success": result["success"],
            "code_executions": result.get("code_executions", 0),
        }
        self.shared_state.scripter_history.append(script_record)
        self.shared_state.last_scripter_message = result["message"]
        self.shared_state.last_scripter_success = result["success"]

        logger.info(f"🐍 ScripterAgent finished: {result['message'][:2000]}...")

        return ScripterExecutorResultEvent(
            task=ev.task,
            message=result["message"],
            success=result["success"],
            code_executions=result.get("code_executions", 0),
        )

    @step
    async def handle_scripter_result(
        self, ctx: Context, ev: ScripterExecutorResultEvent
    ) -> ManagerInputEvent:
        """
        Process ScripterAgent result and loop back to Manager.
        """
        if ev.success:
            logger.info(
                f"✅ Script completed successfully in {ev.code_executions} steps"
            )
        else:
            logger.warning(f"⚠️ Script failed or reached max steps: {ev.message}")

        # Increment DroidAgent step counter
        self.shared_state.step_number += 1
        logger.info(
            f"🔄 Step {self.shared_state.step_number}/{self.config.agent.max_steps} complete, looping to Manager"
        )

        # Loop back to Manager (script result in shared_state)
        return ManagerInputEvent()

    # ========================================================================
    # End Manager/Executor/Script Workflow Steps
    # ========================================================================

    @step
    async def finalize(self, ctx: Context, ev: FinalizeEvent) -> StopEvent:
        ctx.write_event_to_stream(ev)
        capture(
            DroidAgentFinalizeEvent(
                success=ev.success,
                reason=ev.reason,
                steps=self.shared_state.step_number,
                unique_packages_count=len(self.shared_state.visited_packages),
                unique_activities_count=len(self.shared_state.visited_activities),
            ),
            self.user_id,
        )
        flush()

        # Base result with answer
        result = {
            "success": ev.success,
            "reason": ev.reason,
            "steps": self.shared_state.step_number,
            "structured_output": None,
        }

        if self.memory_client.enabled and self.trajectory:
            try:
                manual = parse_trajectory_for_manual(self.trajectory)
                current_status = manual.get("status") or None
                if current_status in {None, "unknown"}:
                    manual["status"] = "success" if ev.success else "failure"

                should_upload = bool(manual.get("execution_steps")) or bool(
                    manual.get("initial_plan")
                )

                if should_upload:
                    existing_manual = self.memory_client.find_trajectory(self.goal)
                    existing_status = (
                        (existing_manual.get("status") or "").lower()
                        if existing_manual
                        else ""
                    )
                    existing_id = (
                        existing_manual.get("trajectory_id")
                        if existing_manual
                        else None
                    )

                    new_status = (manual.get("status") or "").lower()

                    if existing_status == "success" and new_status == "success":
                        logger.debug(
                            "Skipping trajectory upload: existing successful manual already covers goal '%s'.",
                            self.goal,
                        )
                    else:
                        if existing_id:
                            manual["trajectory_id"] = existing_id
                        uploaded = self.memory_client.add_trajectory(manual)
                        if uploaded:
                            logger.debug(
                                "Trajectory %s to memory service with status '%s'.",
                                "updated" if existing_id else "uploaded",
                                manual.get("status"),
                            )
                else:
                    logger.debug(
                        "Skipping trajectory upload: no execution steps or plan captured for goal '%s'.",
                        self.goal,
                    )
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Skipping trajectory memory upload: %s", exc)

        # Extract structured output if model was provided
        if self.output_model is not None and ev.reason:
            logger.info("🔄 Running structured output extraction...")

            try:
                structured_agent = StructuredOutputAgent(
                    llm=self.structured_output_llm,
                    pydantic_model=self.output_model,
                    answer_text=ev.reason,
                    timeout=self.timeout,
                )

                handler = structured_agent.run()

                # Stream nested events
                async for nested_ev in handler.stream_events():
                    ctx.write_event_to_stream(nested_ev)

                extraction_result = await handler

                if extraction_result["success"]:
                    result["structured_output"] = extraction_result["structured_output"]
                    logger.info("✅ Structured output added to final result")
                else:
                    logger.warning(
                        f"⚠️  Structured extraction failed: {extraction_result['error_message']}"
                    )

            except Exception as e:
                logger.error(f"❌ Error during structured extraction: {e}")
                if self.config.logging.debug:
                    import traceback

                    logger.error(traceback.format_exc())

        if self.trajectory and self.config.logging.save_trajectory != "none":
            self.trajectory.save_trajectory()

        self.tools_instance._set_context(None)

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
                self.trajectory.events.append(ev)
