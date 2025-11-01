"""
DroidAgent - A wrapper class that coordinates the planning and execution of tasks
to achieve a user's goal on an Android device.

Architecture:
- When reasoning=False: Uses CodeActAgent directly
- When reasoning=True: Uses Manager (planning) + Executor (action) workflows
"""

import logging
from typing import TYPE_CHECKING, Type, Awaitable

from pydantic import BaseModel
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from workflows.events import Event
from workflows.handler import WorkflowHandler
from droidrun.agent.codeact import CodeActAgent
from droidrun.agent.common.events import MacroEvent, RecordUIStateEvent, ScreenshotEvent
from droidrun.agent.droid.events import (
    CodeActExecuteEvent,
    CodeActResultEvent,
    ExecutorInputEvent,
    ExecutorResultEvent,
    FinalizeEvent,
    ManagerInputEvent,
    ManagerPlanEvent,
    ResultEvent,
    ScripterExecutorInputEvent,
    ScripterExecutorResultEvent,
)
from droidrun.agent.droid.state import DroidAgentState
from droidrun.agent.executor import ExecutorAgent
from droidrun.agent.manager import ManagerAgent
from droidrun.agent.scripter import ScripterAgent
from droidrun.agent.oneflows.structured_output_agent import StructuredOutputAgent
from droidrun.agent.utils.async_utils import wrap_async_tools
from droidrun.agent.utils.llm_loader import load_agent_llms, validate_llm_dict
from droidrun.agent.utils.prompt_resolver import PromptResolver
from droidrun.agent.utils.tools import (
    ATOMIC_ACTION_SIGNATURES,
    build_custom_tools,
    resolve_tools_instance,
)
from droidrun.agent.utils.tracing_setup import setup_tracing
from droidrun.agent.utils.trajectory import Trajectory
from droidrun.config_manager.config_manager import (
    AgentConfig,
    CredentialsConfig,
    DeviceConfig,
    DroidrunConfig,
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
        config: DroidrunConfig | None = None,
        llms: dict[str, LLM] | LLM | None = None,
        tools: "Tools | None" = None,
        custom_tools: dict = None,
        credentials: dict | None = None,
        variables: dict | None = None,
        output_model: Type[BaseModel] | None = None,
        prompts: dict[str, str] | None = None,
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
            tools: Either a Tools instance (for custom/pre-configured tools) or None (use default from config).
            custom_tools: Custom tool definitions
            credentials: Dict of credentials {"SECRET_ID": "value"} or None (will use config.credentials if available)
            variables: Optional dict of custom variables accessible throughout execution
            output_model: Optional Pydantic model for structured output extraction from final answer
            prompts: Optional dict of custom Jinja2 prompt templates to override defaults.
                    Keys: "codeact_system", "codeact_user", "manager_system", "executor_system", "scripter_system"
                    Values: Jinja2 template strings (NOT file paths)
            timeout: Workflow timeout in seconds
        """

        self.user_id = kwargs.pop("user_id", None)
        self.runtype = kwargs.pop("runtype", "developer")
        self.shared_state = DroidAgentState(
            instruction=goal,
            err_to_manager_thresh=2,
            user_id=self.user_id,
            runtype=self.runtype,
        )
        self.output_model = output_model
        base_config = config

        # Initialize prompt resolver for custom prompts
        self.prompt_resolver = PromptResolver(custom_prompts=prompts)

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
        resolved_device_config = base_config.device if base_config else DeviceConfig()
        tools_instance, tools_config_resolved = resolve_tools_instance(
            tools=tools_fallback,
            device_config=resolved_device_config,
            tools_config_fallback=base_config.tools if base_config else None,
            credential_manager=credential_manager,
        )

        # Build final config with resolved tools config
        self.config = DroidrunConfig(
            agent=base_config.agent if base_config else AgentConfig(),
            device=resolved_device_config,
            tools=tools_config_resolved,
            logging=base_config.logging if base_config else LoggingConfig(),
            tracing=base_config.tracing if base_config else TracingConfig(),
            telemetry=base_config.telemetry if base_config else TelemetryConfig(),
            llm_profiles=base_config.llm_profiles if base_config else {},
            credentials=base_config.credentials if base_config else CredentialsConfig(),
        )

        super().__init__(*args, timeout=timeout, **kwargs)

        self._configure_default_logging(debug=self.config.logging.debug)

        setup_tracing(self.config.tracing)

        # Load LLMs if not provided
        if llms is None:
            if config is None:
                raise ValueError(
                    "Either 'llms' or 'config' must be provided. "
                    "If llms is not provided, config is required to load LLMs from profiles."
                )

            logger.info("🔄 Loading LLMs from config (llms not provided)...")

            llms = load_agent_llms(
                config=self.config, output_model=output_model, **kwargs
            )
        if isinstance(llms, dict):
            validate_llm_dict(self.config, llms, output_model=output_model)
        elif isinstance(llms, LLM):
            pass
        else:
            raise ValueError(f"Invalid LLM type: {type(llms)}")

        self.timeout = timeout

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

        # TODO: Pass shared_state to tools_instance to allow custom tools to access
        # custom_variables and other shared state. Currently tools only have access
        # to context via _set_context(). Consider: self.tools_instance.shared_state = self.shared_state

        if self.config.agent.reasoning:
            logger.info("📝 Initializing Manager and Executor Agents...")
            self.manager_agent = ManagerAgent(
                llm=self.manager_llm,
                tools_instance=tools_instance,
                shared_state=self.shared_state,
                agent_config=self.config.agent,
                custom_tools=self.custom_tools,
                output_model=self.output_model,
                prompt_resolver=self.prompt_resolver,
                timeout=timeout,
            )
            self.executor_agent = ExecutorAgent(
                llm=self.executor_llm,
                tools_instance=tools_instance,
                shared_state=self.shared_state,
                agent_config=self.config.agent,
                custom_tools=self.custom_tools,
                prompt_resolver=self.prompt_resolver,
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
                custom_prompts=prompts,
            ),
            self.user_id,
        )

        logger.info("✅ DroidAgent initialized successfully.")

    def run(self, *args, **kwargs) -> Awaitable[ResultEvent] | WorkflowHandler:
        handler = super().run(*args, **kwargs)  # type: ignore[assignment]
        return handler

    @step
    async def execute_task(
        self, ctx: Context, ev: CodeActExecuteEvent
    ) -> CodeActResultEvent:
        """
        Execute a single task using the CodeActAgent.

        Args:
            instruction: task of what the agent shall do

        Returns:
            Tuple of (success, reason)
        """

        logger.info(f"🔧 Executing task: {ev.instruction}")

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
                prompt_resolver=self.prompt_resolver,
                timeout=self.timeout,
            )

            handler = codeact_agent.run(
                input=ev.instruction,
                remembered_info=self.tools_instance.memory,
            )

            async for nested_ev in handler.stream_events():
                self.handle_stream_event(nested_ev, ctx)

            result = await handler

            if "success" in result and result["success"]:
                event = CodeActResultEvent(
                    success=True,
                    reason=result["reason"],
                    instruction=ev.instruction,
                )
                ctx.write_event_to_stream(event)
                return event
            else:
                event = CodeActResultEvent(
                    success=False,
                    reason=result["reason"],
                    instruction=ev.instruction,
                )
                ctx.write_event_to_stream(event)
                return event

        except Exception as e:
            logger.error(f"Error during task execution: {e}")
            if self.config.logging.debug:
                import traceback

                logger.error(traceback.format_exc())
            event = CodeActResultEvent(
                success=False, reason=f"Error: {str(e)}", instruction=ev.instruction
            )
            ctx.write_event_to_stream(event)
            return event

    @step
    async def handle_codeact_execute(
        self, ctx: Context, ev: CodeActResultEvent
    ) -> FinalizeEvent:
        try:
            event = FinalizeEvent(success=ev.success, reason=ev.reason)
            # ctx.write_event_to_stream(event)
            return event
        except Exception as e:
            logger.error(f"❌ Error during DroidAgent execution: {e}")
            if self.config.logging.debug:
                import traceback

                logger.error(traceback.format_exc())
            event = FinalizeEvent(
                success=False,
                reason=str(e),
            )
            # ctx.write_event_to_stream(event)
            return event

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
            # Get the current running event loop to pass to wrapped tools
            import asyncio

            loop = asyncio.get_running_loop()

            self.atomic_tools = wrap_async_tools(self.atomic_tools, loop=loop)
            self.custom_tools = wrap_async_tools(self.custom_tools, loop=loop)

            self._tools_wrapped = True
            logger.debug("✅ Async tools wrapped for synchronous execution contexts")

        if not self.config.agent.reasoning:
            logger.info(
                f"🔄 Direct execution mode - executing goal: {self.shared_state.instruction}"
            )
            event = CodeActExecuteEvent(instruction=self.shared_state.instruction)
            ctx.write_event_to_stream(event)
            return event

        logger.info("🧠 Reasoning mode - initializing Manager/Executor workflow")
        event = ManagerInputEvent()
        ctx.write_event_to_stream(event)
        return event

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
            event = FinalizeEvent(
                success=False,
                reason=f"Reached maximum steps ({self.config.agent.max_steps})",
            )
            # ctx.write_event_to_stream(event)
            return event

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
        event = ManagerPlanEvent(
            plan=result["plan"],
            current_subgoal=result["current_subgoal"],
            thought=result["thought"],
            manager_answer=result.get("manager_answer", ""),
            success=result.get("success"),
        )
        ctx.write_event_to_stream(event)
        return event

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
            # Use success field from manager, default to True if not set for backward compatibility
            success = ev.success if ev.success is not None else True
            logger.info(
                f"💬 Manager provided answer (success={success}): {ev.manager_answer}"
            )
            self.shared_state.progress_status = f"Answer: {ev.manager_answer}"

            event = FinalizeEvent(success=success, reason=ev.manager_answer)
            # ctx.write_event_to_stream(event)
            return event

        # Check for <script> tag in current_subgoal, then extract from full plan
        if "<script>" in ev.current_subgoal:
            # Found script tag in subgoal - now search the entire plan
            start_idx = ev.plan.find("<script>")
            end_idx = ev.plan.find("</script>")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                # Extract content between first <script> and first </script> in plan
                task = ev.plan[start_idx + len("<script>") : end_idx].strip()
                logger.info(f"🐍 Routing to ScripterAgent: {task[:80]}...")
                event = ScripterExecutorInputEvent(task=task)
                ctx.write_event_to_stream(event)
                return event
            else:
                # <script> found in subgoal but not properly closed in plan - log warning
                logger.warning(
                    "⚠️ Found <script> in subgoal but not properly closed in plan, treating as regular subgoal"
                )

        # Continue to Executor with current subgoal
        logger.info(f"▶️  Proceeding to Executor with subgoal: {ev.current_subgoal}")
        event = ExecutorInputEvent(current_subgoal=ev.current_subgoal)
        ctx.write_event_to_stream(event)
        return event

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

        event = ExecutorResultEvent(
            action=result["action"],
            outcome=result["outcome"],
            error=result["error"],
            summary=result["summary"],
        )
        ctx.write_event_to_stream(event)
        return event

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

        event = ManagerInputEvent()
        ctx.write_event_to_stream(event)
        return event

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
            self.handle_stream_event(nested_ev, ctx)

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

        event = ScripterExecutorResultEvent(
            task=ev.task,
            message=result["message"],
            success=result["success"],
            code_executions=result.get("code_executions", 0),
        )
        ctx.write_event_to_stream(event)
        return event

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
        event = ManagerInputEvent()
        ctx.write_event_to_stream(event)
        return event

    # ========================================================================
    # End Manager/Executor/Script Workflow Steps
    # ========================================================================

    @step
    async def finalize(self, ctx: Context, ev: FinalizeEvent) -> ResultEvent:
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
        result = ResultEvent(
            success=ev.success,
            reason=ev.reason,
            steps=self.shared_state.step_number,
            structured_output=None,
        )

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
                    self.handle_stream_event(nested_ev, ctx)

                extraction_result = await handler

                if extraction_result["success"]:
                    result.structured_output = extraction_result["structured_output"]
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

        return result

    def handle_stream_event(self, ev: Event, ctx: Context):
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
