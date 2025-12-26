"""
DroidAgent - A wrapper class that coordinates the planning and execution of tasks
to achieve a user's goal on an Android device.

Architecture:
- When reasoning=False: Uses CodeActAgent directly
- When reasoning=True: Uses Manager (planning) + Executor (action) workflows
"""

import logging
from typing import TYPE_CHECKING, Type, Awaitable, Union

from pydantic import BaseModel
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from workflows.events import Event
from workflows.handler import WorkflowHandler
from droidrun.agent.codeact import CodeActAgent
from droidrun.agent.codeact.events import CodeActOutputEvent, CodeActResponseEvent
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
    TextManipulatorInputEvent,
    TextManipulatorResultEvent,
)
from droidrun.agent.droid.state import DroidAgentState
from droidrun.agent.executor import ExecutorAgent
from droidrun.agent.manager import ManagerAgent, StatelessManagerAgent
from droidrun.agent.oneflows.text_manipulator import run_text_manipulation_agent
from droidrun.agent.scripter import ScripterAgent
from droidrun.agent.oneflows.structured_output_agent import StructuredOutputAgent
from droidrun.agent.trajectory import TrajectoryWriter
from droidrun.agent.utils.llm_loader import (
    load_agent_llms,
    merge_llms_with_config,
    validate_llm_dict,
)
from droidrun.agent.utils.prompt_resolver import PromptResolver
from droidrun.agent.utils.tools import (
    ATOMIC_ACTION_SIGNATURES,
    build_custom_tools,
    filter_atomic_actions,
    filter_custom_tools,
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
from droidrun.credential_manager import FileCredentialManager, CredentialManager
from droidrun.telemetry import (
    DroidAgentFinalizeEvent,
    DroidAgentInitEvent,
    capture,
    flush,
)
from droidrun.agent.utils.tracing_setup import (
    apply_session_context,
    record_langfuse_screenshot,
)
from opentelemetry import trace

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
        credentials: Union[dict, "CredentialManager", None] = None,
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
            credentials: Dict {"SECRET_ID": "value"}, CredentialManager instance, or None (will use config.credentials if available)
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
            else (config.credentials if config else None)
        )

        # If already a CredentialManager instance, use it. Otherwise wrap in FileCredentialManager
        if isinstance(credentials_source, CredentialManager):
            self.credential_manager = credentials_source
        elif credentials_source is not None:
            cm = FileCredentialManager(credentials_source)
            # Only assign if it actually loaded secrets (handles disabled case)
            self.credential_manager = cm if cm.secrets else None
        else:
            self.credential_manager = None

        self.tools_param = tools
        self.tools_fallback = (
            tools if tools is not None else (config.tools if config else None)
        )
        self.resolved_device_config = config.device if config else DeviceConfig()

        self.config = DroidrunConfig(
            agent=config.agent if config else AgentConfig(),
            device=self.resolved_device_config,
            tools=config.tools if config else ToolsConfig(),
            logging=config.logging if config else LoggingConfig(),
            tracing=config.tracing if config else TracingConfig(),
            telemetry=config.telemetry if config else TelemetryConfig(),
            llm_profiles=config.llm_profiles if config else {},
            credentials=config.credentials if config else CredentialsConfig(),
        )

        self.tools_instance = None

        super().__init__(*args, timeout=timeout, **kwargs)

        self._configure_default_logging(debug=self.config.logging.debug)

        setup_tracing(self.config.tracing, agent=self)

        # Load LLMs if not provided
        if llms is None:
            if config is None:
                raise ValueError(
                    "Either 'llms' or 'config' must be provided. "
                    "If llms is not provided, config is required to load LLMs from profiles."
                )

            logger.debug("🔄 Loading LLMs from config (llms not provided)...")

            llms = load_agent_llms(
                config=self.config, output_model=output_model, **kwargs
            )
        if isinstance(llms, dict):
            # allow users to provide a partial dict of LLMs. Merge any missing ones from configuration defaults.
            llms = merge_llms_with_config(
                self.config, llms, output_model=output_model, **kwargs
            )

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

            logger.debug("📚 Using agent-specific LLMs from dictionary")
        else:
            logger.debug("📚 Using single LLM for all agents")
            self.manager_llm = llms
            self.executor_llm = llms
            self.codeact_llm = llms
            self.text_manipulator_llm = llms
            self.app_opener_llm = llms
            self.scripter_llm = llms
            self.structured_output_llm = llms

        self.trajectory = Trajectory(
            goal=self.shared_state.instruction,
            base_path=self.config.logging.trajectory_path,
        )
        self.trajectory_writer = TrajectoryWriter(queue_size=300)

        self.atomic_tools = ATOMIC_ACTION_SIGNATURES.copy()

        # Store user custom tools, will build auto tools (credentials + open_app)
        self.user_custom_tools = custom_tools or {}
        self.custom_tools = {}

        if self.user_custom_tools:
            logger.debug(f"🔧 User custom tools: {list(self.user_custom_tools.keys())}")

        logger.debug("🤖 Initializing DroidAgent...")
        logger.debug(f"💾 Trajectory saving: {self.config.logging.save_trajectory}")

        if self.config.agent.reasoning:
            # Choose between stateful and stateless manager
            if self.config.agent.manager.stateless:
                logger.debug("📝 Initializing StatelessManager and Executor Agents...")
                ManagerClass = StatelessManagerAgent
            else:
                logger.debug("📝 Initializing Manager and Executor Agents...")
                ManagerClass = ManagerAgent

            self.manager_agent = ManagerClass(
                llm=self.manager_llm,
                tools_instance=None,
                shared_state=self.shared_state,
                agent_config=self.config.agent,
                custom_tools=self.custom_tools,
                output_model=self.output_model,
                prompt_resolver=self.prompt_resolver,
                tracing_config=self.config.tracing,
                timeout=self.timeout,
            )
            self.executor_agent = ExecutorAgent(
                llm=self.executor_llm,
                tools_instance=None,
                shared_state=self.shared_state,
                agent_config=self.config.agent,
                custom_tools=self.custom_tools,
                prompt_resolver=self.prompt_resolver,
                timeout=self.timeout,
            )
        else:
            self.manager_agent = None
            self.executor_agent = None

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

        logger.debug("✅ DroidAgent initialized successfully.")

    def run(self, *args, **kwargs) -> Awaitable[ResultEvent] | WorkflowHandler:
        apply_session_context()
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

        logger.debug(f"🔧 Executing task: {ev.instruction}")

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
                tracing_config=self.config.tracing,
            )

            handler = codeact_agent.run(
                input=ev.instruction,
                remembered_info=self.tools_instance.memory,
            )

            # Track current step's action for history
            current_action = {}
            current_output = ""

            async for nested_ev in handler.stream_events():
                self.handle_stream_event(nested_ev, ctx)

                # Capture thought and code from CodeActResponseEvent
                if isinstance(nested_ev, CodeActResponseEvent):
                    current_action = {
                        "thought": nested_ev.thought or "",
                        "code": nested_ev.code or "",
                    }

                if isinstance(nested_ev, CodeActOutputEvent):
                    current_output = nested_ev.output or ""

                    # Record action and summary for this step
                    self.shared_state.action_history.append(current_action.copy())
                    self.shared_state.summary_history.append(current_output)
                    self.shared_state.action_outcomes.append(True)  # Successful execution

                    # Reset for next iteration
                    current_action = {}

                    if self.config.logging.save_trajectory != "none":
                        self.shared_state.step_number += 1
                        self.trajectory_writer.write(
                            self.trajectory,
                            stage=f"codeact_step_{self.shared_state.step_number}",
                        )

            result = await handler

            if "success" in result and result["success"]:
                return CodeActResultEvent(
                    success=True,
                    reason=result["reason"],
                    instruction=ev.instruction,
                )

            else:
                return CodeActResultEvent(
                    success=False,
                    reason=result["reason"],
                    instruction=ev.instruction,
                )

        except Exception as e:
            logger.error(f"Error during task execution: {e}")
            if self.config.logging.debug:
                import traceback

                logger.error(traceback.format_exc())
            return CodeActResultEvent(
                success=False, reason=f"Error: {str(e)}", instruction=ev.instruction
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
        logger.info(
            f"🚀 Running DroidAgent to achieve goal: {self.shared_state.instruction}"
        )
        ctx.write_event_to_stream(ev)

        await self.trajectory_writer.start()

        # Build and filter tools (single source of truth for tool filtering)
        auto_custom_tools = await build_custom_tools(self.credential_manager)
        disabled_tools = (
            self.config.tools.disabled_tools
            if self.config.tools and self.config.tools.disabled_tools
            else []
        )

        self.atomic_tools = filter_atomic_actions(disabled_tools)
        filtered_custom = filter_custom_tools(
            {**auto_custom_tools, **self.user_custom_tools},
            disabled_tools,
        )
        self.custom_tools.clear()
        self.custom_tools.update(filtered_custom)

        if self.tools_instance is None:
            # Determine if vision is enabled based on the active agent role
            if self.config.agent.reasoning:
                vision_enabled = self.config.agent.manager.vision
            else:
                vision_enabled = self.config.agent.codeact.vision

            tools_instance, tools_config_resolved = await resolve_tools_instance(
                tools=self.tools_fallback,
                device_config=self.resolved_device_config,
                tools_config_fallback=self.config.tools,
                credential_manager=self.credential_manager,
                vision_enabled=vision_enabled,
            )

            self.tools_instance = tools_instance
            self.config.tools = tools_config_resolved

            self.tools_instance.save_trajectories = self.config.logging.save_trajectory
            self.tools_instance.app_opener_llm = self.app_opener_llm
            self.tools_instance.text_manipulator_llm = self.text_manipulator_llm
            self.tools_instance.streaming = self.config.agent.streaming

        # Update sub-agents with tools (outside the if block - works for both auto-created and pre-provided)
        if self.config.agent.reasoning and self.executor_agent:
            self.manager_agent.tools_instance = self.tools_instance
            self.executor_agent.tools_instance = self.tools_instance
            self.executor_agent.atomic_tools = self.atomic_tools

        self.tools_instance._set_context(ctx)

        if self.config.logging.save_trajectory != "none":
            self.trajectory_writer.write(self.trajectory, stage="init")

        if not self.config.agent.reasoning:
            logger.debug(
                f"🔄 Direct execution mode - executing goal: {self.shared_state.instruction}"
            )
            event = CodeActExecuteEvent(instruction=self.shared_state.instruction)
            ctx.write_event_to_stream(event)
            return event

        logger.debug("🧠 Reasoning mode - initializing Manager/Executor workflow")
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
            return FinalizeEvent(
                success=False,
                reason=f"Reached maximum steps ({self.config.agent.max_steps})",
            )

        logger.info(
            f"🔄 Step {self.shared_state.step_number + 1}/{self.config.agent.max_steps}"
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
    ) -> (
        ExecutorInputEvent
        | ScripterExecutorInputEvent
        | FinalizeEvent
        | TextManipulatorInputEvent
    ):
        """
        Process Manager output and decide next step.

        Checks if task is complete, if ScripterAgent should run, or if Executor should take action.
        """
        # Check for answer-type termination
        if ev.manager_answer.strip():
            # Use success field from manager, default to True if not set for backward compatibility
            success = ev.success if ev.success is not None else True
            self.shared_state.progress_summary = f"Answer: {ev.manager_answer}"

            return FinalizeEvent(success=success, reason=ev.manager_answer)

        # Check for <script> tag in current_subgoal, then extract from full plan
        if "<script>" in ev.current_subgoal:
            # Found script tag in subgoal - now search the entire plan
            start_idx = ev.plan.find("<script>")
            end_idx = ev.plan.find("</script>")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                # Extract content between first <script> and first </script> in plan
                task = ev.plan[start_idx + len("<script>") : end_idx].strip()
                logger.debug(f"🐍 Routing to ScripterAgent: {task[:80]}...")
                event = ScripterExecutorInputEvent(task=task)
                ctx.write_event_to_stream(event)
                return event
            else:
                # <script> found in subgoal but not properly closed in plan - log warning
                logger.warning(
                    "⚠️ Found <script> in subgoal but not properly closed in plan, treating as regular subgoal"
                )
        if "TEXT_TASK" in ev.current_subgoal:
            return TextManipulatorInputEvent(
                task=ev.current_subgoal.replace("TEXT_TASK:", "")
                .replace("TEXT_TASK", "")
                .strip()
            )

        # Continue to Executor with current subgoal
        logger.debug(f"▶️  Proceeding to Executor with subgoal: {ev.current_subgoal}")
        return ExecutorInputEvent(current_subgoal=ev.current_subgoal)


    @step
    async def run_text_manipulator(
        self, ctx: Context, ev: TextManipulatorInputEvent
    ) -> TextManipulatorResultEvent:
        logger.debug(f"🔍 Running TextManipulatorAgent for task: {ev.task}")

        if not self.shared_state.focused_text:
            logger.warning("⚠️ No focused text available, using empty string")
            current_text = ""
        else:
            current_text = self.shared_state.focused_text

        try:
            text_to_type, code_ran = await run_text_manipulation_agent(
                instruction=self.shared_state.instruction,
                current_subgoal=ev.task,
                current_text=current_text,
                overall_plan=self.shared_state.plan,
                llm=self.text_manipulator_llm,
                stream=self.config.agent.streaming,
            )

            return TextManipulatorResultEvent(
                task=ev.task, text_to_type=text_to_type, code_ran=code_ran
            )

        except Exception as e:
            logger.error(f"❌ TextManipulator agent failed: {e}")
            if self.config.logging.debug:
                import traceback

                logger.error(traceback.format_exc())

            return TextManipulatorResultEvent(
                task=ev.task, text_to_type="", code_ran=""
            )

    @step
    async def handle_text_manipulator_result(
        self, ctx: Context, ev: TextManipulatorResultEvent
    ) -> ManagerInputEvent:
        if not ev.text_to_type or not ev.text_to_type.strip():
            logger.warning("⚠️ TextManipulator returned empty text, treating as no-op")
            self.shared_state.last_summary = "Text manipulation returned empty result"
            self.shared_state.action_outcomes.append(False)
        else:
            try:
                result = await self.tools_instance.input_text(
                    ev.text_to_type, clear=True
                )

                if (
                    not result
                    or "error" in result.lower()
                    or "failed" in result.lower()
                ):
                    logger.warning(f"⚠️ Text input may have failed: {result}")
                    self.shared_state.last_summary = (
                        f"Text manipulation attempted but may have failed: {result}"
                    )
                    self.shared_state.action_outcomes.append(False)
                else:
                    logger.debug(
                        f"✅ Text manipulator successfully typed {len(ev.text_to_type)} characters"
                    )
                    self.shared_state.last_summary = f"Text manipulation successful: typed {len(ev.text_to_type)} characters"
                    self.shared_state.action_outcomes.append(True)
            except Exception as e:
                logger.error(f"❌ Error during text input: {e}")
                self.shared_state.last_summary = f"Text manipulation error: {str(e)}"
                self.shared_state.action_outcomes.append(False)

        text_manipulation_record = {
            "task": ev.task,
            "code_ran": ev.code_ran,
            "text_length": len(ev.text_to_type) if ev.text_to_type else 0,
            "success": (
                self.shared_state.action_outcomes[-1]
                if self.shared_state.action_outcomes
                else False
            ),
        }

        self.shared_state.text_manipulation_history.append(text_manipulation_record)
        self.shared_state.last_text_manipulation_success = text_manipulation_record[
            "success"
        ]

        self.shared_state.step_number += 1

        if self.config.logging.save_trajectory != "none":
            self.trajectory_writer.write(
                self.trajectory, stage=f"step_{self.shared_state.step_number}"
            )

        return ManagerInputEvent()

    @step
    async def run_executor(
        self, ctx: Context, ev: ExecutorInputEvent
    ) -> ExecutorResultEvent:
        """
        Run Executor action phase.

        The Executor selects and executes a specific action for the current subgoal.
        """
        logger.debug("⚡ Running Executor for action...")

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
                    logger.debug("✅ Error resolved - resetting error flag")
                self.shared_state.error_flag_plan = False

        self.shared_state.step_number += 1

        if self.config.logging.save_trajectory != "none":
            self.trajectory_writer.write(
                self.trajectory, stage=f"step_{self.shared_state.step_number}"
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
        logger.debug(f"🐍 Starting ScripterAgent for task: {ev.task[:2000]}...")

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
            logger.debug(
                f"✅ Script completed successfully in {ev.code_executions} steps"
            )
        else:
            logger.warning(f"⚠️ Script failed or reached max steps: {ev.message}")

        # Increment DroidAgent step counter
        self.shared_state.step_number += 1

        if self.config.logging.save_trajectory != "none":
            self.trajectory_writer.write(
                self.trajectory, stage=f"step_{self.shared_state.step_number}"
            )

        # Loop back to Manager (script result in shared_state)
        return ManagerInputEvent()

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
        await flush()

        # Base result with answer
        result = ResultEvent(
            success=ev.success,
            reason=ev.reason,
            steps=self.shared_state.step_number,
            structured_output=None,
        )

        # Extract structured output if model was provided
        if self.output_model is not None and ev.reason:
            logger.debug("🔄 Running structured output extraction...")

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
                    logger.debug("✅ Structured output added to final result")
                else:
                    logger.warning(
                        f"⚠️  Structured extraction failed: {extraction_result['error_message']}"
                    )

            except Exception as e:
                logger.error(f"❌ Error during structured extraction: {e}")
                if self.config.logging.debug:
                    import traceback

                    logger.error(traceback.format_exc())

        # Capture final screenshot before saving trajectory
        if self.config.logging.save_trajectory != "none":
            try:
                screenshot_result = await self.tools_instance.take_screenshot()
                if isinstance(screenshot_result, tuple):
                    success, screenshot = screenshot_result
                    if success and screenshot:
                        ctx.write_event_to_stream(
                            ScreenshotEvent(screenshot=screenshot)
                        )
                        vision_any = (
                            self.config.agent.manager.vision
                            or self.config.agent.executor.vision
                            or self.config.agent.codeact.vision
                        )
                        parent_span = trace.get_current_span()
                        record_langfuse_screenshot(
                            screenshot,
                            parent_span=parent_span,
                            screenshots_enabled=self.config.tracing.langfuse_screenshots,
                            vision_enabled=vision_any,
                        )
                        logger.debug("📸 Final screenshot captured")
                elif screenshot_result:
                    ctx.write_event_to_stream(
                        ScreenshotEvent(screenshot=screenshot_result)
                    )
                    vision_any = (
                        self.config.agent.manager.vision
                        or self.config.agent.executor.vision
                        or self.config.agent.codeact.vision
                    )
                    parent_span = trace.get_current_span()
                    record_langfuse_screenshot(
                        screenshot_result,
                        parent_span=parent_span,
                        screenshots_enabled=self.config.tracing.langfuse_screenshots,
                        vision_enabled=vision_any,
                    )
                    logger.debug("📸 Final screenshot captured")
            except Exception as e:
                logger.warning(f"Failed to capture final screenshot: {e}")

            self.trajectory_writer.write_final(
                self.trajectory, self.config.logging.trajectory_gifs
            )
            await self.trajectory_writer.stop()
            logger.info(f"📁 Trajectory saved: {self.trajectory.trajectory_folder}")

            # Upload to GCP if enabled
            if self.config.logging.gcp.enabled:
                try:
                    from droidrun.agent.trajectory.gcp_upload import upload_trajectory_to_gcp

                    # Delete local files after upload unless keep_local is True
                    cleanup_local = not self.config.logging.gcp.keep_local

                    # Build execution log from action history and summaries
                    execution_log_lines = []
                    execution_log_lines.append(f"Goal: {self.shared_state.instruction}")
                    execution_log_lines.append(f"Total Steps: {self.shared_state.step_number}")
                    execution_log_lines.append("=" * 80)
                    execution_log_lines.append("")

                    for i, summary in enumerate(self.shared_state.summary_history):
                        step_num = i + 1
                        action = self.shared_state.action_history[i] if i < len(self.shared_state.action_history) else {}
                        outcome = self.shared_state.action_outcomes[i] if i < len(self.shared_state.action_outcomes) else None
                        status = "PASS" if outcome else "FAIL" if outcome is not None else "N/A"

                        execution_log_lines.append(f"🔄 Step {step_num}/{self.config.agent.max_steps}")
                        if action:
                            thought = action.get("thought", "")
                            if thought:
                                execution_log_lines.append(f"CodeAct response:")
                                execution_log_lines.append(thought)
                            code = action.get("code", "")
                            if code:
                                execution_log_lines.append(f"```python")
                                execution_log_lines.append(code)
                                execution_log_lines.append(f"```")
                        execution_log_lines.append(f"💡 Execution result:")
                        execution_log_lines.append(f"{summary}")
                        execution_log_lines.append(f"[{status}]")
                        execution_log_lines.append("")

                    # Add final result
                    execution_log_lines.append("=" * 80)
                    execution_log_lines.append(f"Final Result: {'SUCCESS' if result.success else 'FAILED'}")
                    execution_log_lines.append(f"Reason: {result.reason}")

                    execution_log = "\n".join(execution_log_lines)

                    gcp_result = upload_trajectory_to_gcp(
                        trajectory_folder=str(self.trajectory.trajectory_folder),
                        bucket_name=self.config.logging.gcp.bucket_name,
                        product_id=self.config.logging.gcp.product_id,
                        test_run_id=self.config.logging.gcp.test_run_id,
                        tcue_id=self.config.logging.gcp.tcue_id,
                        cleanup_local=cleanup_local,
                        execution_log=execution_log,
                    )
                    if gcp_result["success"]:
                        if gcp_result["local_deleted"]:
                            logger.info(f"☁️  Uploaded to GCP: {gcp_result['gcp_base_path']} (local deleted)")
                        else:
                            logger.info(f"☁️  Uploaded to GCP: {gcp_result['gcp_base_path']}")
                    else:
                        logger.warning(
                            f"⚠️  GCP upload partial: {len(gcp_result['errors'])} errors"
                        )
                except ImportError as e:
                    logger.warning(f"⚠️  GCP upload skipped (missing dependency): {e}")
                except Exception as e:
                    logger.error(f"❌ GCP upload failed: {e}")
                    if self.config.logging.debug:
                        import traceback
                        logger.error(traceback.format_exc())

        self.tools_instance._set_context(None)

        return result

    def handle_stream_event(self, ev: Event, ctx: Context):
        if not isinstance(ev, StopEvent):
            ctx.write_event_to_stream(ev)

            if isinstance(ev, ScreenshotEvent):
                self.trajectory.screenshot_queue.append(ev.screenshot)
                self.trajectory.screenshot_count += 1
            elif isinstance(ev, MacroEvent):
                self.trajectory.macro.append(ev)
            elif isinstance(ev, RecordUIStateEvent):
                self.trajectory.ui_states.append(ev.ui_state)
            else:
                self.trajectory.events.append(ev)
