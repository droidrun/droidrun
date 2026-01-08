import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Optional, Type

from pydantic import BaseModel
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from opentelemetry import trace

from droidrun.agent.codeact.events import (
    CodeActInputEvent,
    CodeActResponseEvent,
    CodeActCodeEvent,
    CodeActOutputEvent,
    CodeActEndEvent,
)
from droidrun.agent.common.constants import LLM_HISTORY_LIMIT
from droidrun.agent.common.events import RecordUIStateEvent, ScreenshotEvent
from droidrun.agent.usage import get_usage_from_response
from droidrun.agent.utils.chat_utils import (
    to_chat_messages,
    extract_code_and_thought,
    limit_history,
)
from droidrun.agent.utils.executer import ExecuterState, SimpleCodeExecutor
from droidrun.agent.utils.inference import acall_with_retries
from droidrun.agent.utils.prompt_resolver import PromptResolver
from droidrun.agent.utils.tracing_setup import record_langfuse_screenshot
from droidrun.agent.utils.tools import (
    ATOMIC_ACTION_SIGNATURES,
    build_custom_tool_descriptions,
)
from droidrun.config_manager.config_manager import AgentConfig, TracingConfig
from droidrun.config_manager.prompt_loader import PromptLoader
from droidrun.tools import Tools

if TYPE_CHECKING:
    from droidrun.agent.droid import DroidAgentState

logger = logging.getLogger("droidrun")


class CodeActAgent(Workflow):
    """
    Agent that generates and executes Python code using atomic actions.

    Uses ReAct cycle: Thought -> Code -> Observation -> repeat until complete().
    Messages stored as list[dict], converted to ChatMessage only for LLM calls.
    """

    def __init__(
        self,
        llm: LLM,
        agent_config: AgentConfig,
        tools_instance: Tools,
        custom_tools: dict = None,
        atomic_tools: dict = None,
        debug: bool = False,
        shared_state: Optional["DroidAgentState"] = None,
        safe_execution_config=None,
        output_model: Type[BaseModel] | None = None,
        prompt_resolver: Optional[PromptResolver] = None,
        tracing_config: TracingConfig | None = None,
        *args,
        **kwargs,
    ):
        assert llm, "llm must be provided."
        super().__init__(*args, **kwargs)

        self.llm = llm
        self.agent_config = agent_config
        self.config = agent_config.codeact
        self.max_steps = agent_config.max_steps
        self.vision = agent_config.codeact.vision
        self.debug = debug
        self.tools = tools_instance
        self.shared_state = shared_state
        self.output_model = output_model
        self.prompt_resolver = prompt_resolver or PromptResolver()
        self.tracing_config = tracing_config

        self.system_prompt: dict | None = None
        self.code_exec_counter = 0
        self.remembered_info: list[str] | None = None

        # Build tool list from atomic + custom tools
        if atomic_tools is None:
            atomic_tools = ATOMIC_ACTION_SIGNATURES

        merged_signatures = {**atomic_tools, **(custom_tools or {})}

        self.tool_list = {}
        for action_name, signature in merged_signatures.items():
            func = signature["function"]
            if inspect.iscoroutinefunction(func):

                async def async_wrapper(
                    *args, f=func, ti=tools_instance, ss=shared_state, **kwargs
                ):
                    return await f(*args, tools=ti, shared_state=ss, **kwargs)

                self.tool_list[action_name] = async_wrapper
            else:

                def sync_wrapper(
                    *args, f=func, ti=tools_instance, ss=shared_state, **kwargs
                ):
                    return f(*args, tools=ti, shared_state=ss, **kwargs)

                self.tool_list[action_name] = sync_wrapper

        self.tool_list["remember"] = tools_instance.remember
        self.tool_list["complete"] = tools_instance.complete

        # Build tool descriptions
        self.tool_descriptions = build_custom_tool_descriptions(atomic_tools)
        custom_descriptions = build_custom_tool_descriptions(custom_tools or {})
        if custom_descriptions:
            self.tool_descriptions += "\n" + custom_descriptions
        self.tool_descriptions += (
            "\n- remember(information: str): Remember information for later use"
        )
        self.tool_descriptions += (
            "\n- complete(success: bool, reason: str): Mark task as complete"
        )

        self._available_secrets = []
        self._output_schema = None
        if self.output_model is not None:
            self._output_schema = self.output_model.model_json_schema()

        # Initialize code executor
        safe_mode = self.config.safe_execution
        safe_config = safe_execution_config

        self.executor = SimpleCodeExecutor(
            locals={},
            tools=self.tool_list,
            globals={"__builtins__": __builtins__},
            safe_mode=safe_mode,
            allowed_modules=(
                safe_config.get_allowed_modules() if safe_config and safe_mode else None
            ),
            blocked_modules=(
                safe_config.get_blocked_modules() if safe_config and safe_mode else None
            ),
            allowed_builtins=(
                safe_config.get_allowed_builtins()
                if safe_config and safe_mode
                else None
            ),
            blocked_builtins=(
                safe_config.get_blocked_builtins()
                if safe_config and safe_mode
                else None
            ),
            event_loop=None,
        )

        logger.debug("CodeActAgent initialized.")

    async def _build_system_prompt(self) -> dict:
        """Build system prompt message."""
        # Determine click mode for template context
        click_mode = getattr(self.agent_config, 'click_mode', None)
        click_mode_str = click_mode.value if click_mode else "index"
        
        template_context = {
            "tool_descriptions": self.tool_descriptions,
            "available_secrets": self._available_secrets,
            "variables": (
                self.shared_state.custom_variables if self.shared_state else {}
            ),
            "output_schema": self._output_schema,
            "click_mode": click_mode_str,
        }
        
        custom_system_prompt = self.prompt_resolver.get_prompt("codeact_system")
        if custom_system_prompt:
            system_text = PromptLoader.render_template(
                custom_system_prompt,
                template_context,
            )
        else:
            system_text = await PromptLoader.load_prompt(
                self.agent_config.get_codeact_system_prompt_path(),
                template_context,
            )
        return {"role": "system", "content": [{"text": system_text}]}

    async def _build_user_prompt(self, goal: str) -> dict:
        """Build initial user prompt message."""
        custom_user_prompt = self.prompt_resolver.get_prompt("codeact_user")
        if custom_user_prompt:
            user_text = PromptLoader.render_template(
                custom_user_prompt,
                {
                    "goal": goal,
                    "variables": (
                        self.shared_state.custom_variables if self.shared_state else {}
                    ),
                },
            )
        else:
            user_text = await PromptLoader.load_prompt(
                self.agent_config.get_codeact_user_prompt_path(),
                {
                    "goal": goal,
                    "variables": (
                        self.shared_state.custom_variables if self.shared_state else {}
                    ),
                },
            )
        return {"role": "user", "content": [{"text": user_text}]}

    @step
    async def prepare_chat(self, ctx: Context, ev: StartEvent) -> CodeActInputEvent:
        """Initialize message history with goal."""
        self.tools._set_context(ctx)
        logger.debug("Preparing chat for task execution...")

        # Get available secrets
        if hasattr(self.tools, "credential_manager") and self.tools.credential_manager:
            self._available_secrets = await self.tools.credential_manager.get_keys()

        # Build system prompt (lazy load)
        if self.system_prompt is None:
            self.system_prompt = await self._build_system_prompt()

        # Get goal and build user message
        user_input = ev.get("input", default=None)
        assert user_input, "User input cannot be empty."

        user_message = await self._build_user_prompt(user_input)
        self.shared_state.message_history.clear()
        self.shared_state.message_history.append(user_message)

        # Store remembered info if provided
        remembered_info = ev.get("remembered_info", default=None)
        if remembered_info:
            self.remembered_info = remembered_info
            memory_text = "\n### Remembered Information:\n"
            for idx, item in enumerate(remembered_info, 1):
                memory_text += f"{idx}. {item}\n"
            # Append to first user message
            self.shared_state.message_history[0]["content"].append({"text": memory_text})

        return CodeActInputEvent()

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: CodeActInputEvent
    ) -> CodeActResponseEvent | CodeActEndEvent:
        """Get device state, call LLM, return response."""
        ctx.write_event_to_stream(ev)

        # Check max steps
        if self.shared_state.step_number + 1 > self.max_steps:
            event = CodeActEndEvent(
                success=False,
                reason=f"Reached max step count of {self.max_steps} steps",
                code_executions=self.code_exec_counter,
            )
            ctx.write_event_to_stream(event)
            return event

        logger.info(f"🔄 Step {self.shared_state.step_number + 1}/{self.max_steps}")

        # Capture screenshot if needed
        screenshot = None
        if self.vision or (
            hasattr(self.tools, "save_trajectories")
            and self.tools.save_trajectories != "none"
        ):
            try:
                result = await self.tools.take_screenshot()
                if isinstance(result, tuple):
                    success, screenshot = result
                    if not success:
                        logger.warning("Screenshot capture failed")
                        screenshot = None
                else:
                    screenshot = result

                if screenshot:
                    ctx.write_event_to_stream(ScreenshotEvent(screenshot=screenshot))
                    parent_span = trace.get_current_span()
                    record_langfuse_screenshot(
                        screenshot,
                        parent_span=parent_span,
                        screenshots_enabled=bool(
                            self.tracing_config
                            and self.tracing_config.langfuse_screenshots
                        ),
                        vision_enabled=self.vision,
                    )
                    await ctx.store.set("screenshot", screenshot)
                    logger.debug("📸 Screenshot captured for CodeAct")
            except Exception as e:
                logger.warning(f"Failed to capture screenshot: {e}")

        # Get device state
        try:
            formatted_text, focused_text, a11y_tree, phone_state = (
                await self.tools.get_state()
            )

            # Update shared state
            self.shared_state.formatted_device_state = formatted_text
            self.shared_state.focused_text = focused_text
            self.shared_state.a11y_tree = a11y_tree
            self.shared_state.phone_state = phone_state

            # Extract and store package/app name (using unified update method)
            self.shared_state.update_current_app(
                package_name=phone_state.get("packageName", "Unknown"),
                activity_name=phone_state.get("currentApp", "Unknown"),
            )

            # Stream formatted state for trajectory
            ctx.write_event_to_stream(RecordUIStateEvent(ui_state=a11y_tree))

            # Add device state to last user message
            self.shared_state.message_history[-1]["content"].append({"text": f"\n{formatted_text}\n"})

        except Exception as e:
            logger.warning(f"⚠️ Error retrieving state from the connected device: {e}")
            if self.debug:
                logger.error("State retrieval error details:", exc_info=True)

        # Add screenshot to message if vision enabled
        if self.vision and screenshot:
            self.shared_state.message_history[-1]["content"].append({"image": screenshot})

        # Limit history and prepare for LLM
        limited_history = limit_history(
            self.shared_state.message_history, LLM_HISTORY_LIMIT * 2, preserve_first=True
        )

        # Build final messages: system + history
        messages_to_send = [self.system_prompt] + limited_history
        chat_messages = to_chat_messages(messages_to_send)

        # Call LLM
        logger.info("[yellow]CodeAct response:[/yellow]")
        response = await acall_with_retries(
            self.llm, chat_messages, stream=self.agent_config.streaming
        )

        if response is None:
            return CodeActEndEvent(
                success=False,
                reason="LLM response is None. This is a critical error.",
                code_executions=self.code_exec_counter,
            )

        # Extract usage
        usage = None
        try:
            usage = get_usage_from_response(self.llm.class_name(), response)
        except Exception as e:
            logger.warning(f"Could not get usage: {e}")

        # Store assistant response
        response_text = response.message.content
        self.shared_state.message_history.append({"role": "assistant", "content": [{"text": response_text}]})
        self.shared_state.step_number += 1

        # Extract thought and code
        code, thought = extract_code_and_thought(response_text)

        # Update unified state
        self.shared_state.last_thought = thought

        event = CodeActResponseEvent(thought=thought, code=code, usage=usage)
        ctx.write_event_to_stream(event)
        return event


    @step
    async def handle_llm_output(
        self, ctx: Context, ev: CodeActResponseEvent
    ) -> CodeActCodeEvent | CodeActInputEvent:
        """Route to execution or request code if missing."""
        if not ev.thought:
            logger.warning("LLM provided code without thoughts.")
            # Add reminder to get thoughts
            goal = self.shared_state.message_history[0]["content"][0].get("text", "")[:200]
            no_thoughts_text = (
                "Your previous response provided code without explaining your reasoning first. "
                "Remember to always describe your thought process and plan *before* providing the code block.\n\n"
                "The code you provided will be executed below.\n\n"
                f"Now, describe the next step you will take to address the original goal."
            )
            self.shared_state.message_history.append({"role": "user", "content": [{"text": no_thoughts_text}]})
        else:
            logger.debug(f"Reasoning: {ev.thought}")

        if ev.code:
            event = CodeActCodeEvent(code=ev.code)
            ctx.write_event_to_stream(event)
            return event
        else:
            # No code - ask for it
            no_code_text = (
                "No code was provided. If you want to mark task as complete "
                "(whether it failed or succeeded), use complete(success: bool, reason: str) "
                "function within a code block ```python\n```."
            )
            self.shared_state.message_history.append({"role": "user", "content": [{"text": no_code_text}]})
            return CodeActInputEvent()

    @step
    async def execute_code(
        self, ctx: Context, ev: CodeActCodeEvent
    ) -> CodeActOutputEvent | CodeActEndEvent:
        """Execute the code and return result."""
        code = ev.code
        logger.debug(f"Executing:\n```\n{code}\n```")

        try:
            self.code_exec_counter += 1
            result = await self.executor.execute(
                ExecuterState(ui_state=await ctx.store.get("ui_state", None)), code
            )
            logger.info("[dim]💡 Execution result:[/dim]")
            logger.info(f"{result}")
            await asyncio.sleep(self.agent_config.after_sleep_action)

            # Check if complete() was called
            if self.tools.finished:
                logger.debug("✅ Task marked as complete via complete() function")

                # Validate completion state
                success = (
                    self.tools.success if self.tools.success is not None else False
                )
                reason = (
                    self.tools.reason
                    if self.tools.reason
                    else "Task completed without reason"
                )
                self.tools.finished = False

                event = CodeActEndEvent(
                    success=success,
                    reason=reason,
                    code_executions=self.code_exec_counter,
                )
                ctx.write_event_to_stream(event)
                return event

            # Update remembered info
            self.remembered_info = self.tools.memory

            event = CodeActOutputEvent(output=str(result))
            ctx.write_event_to_stream(event)
            return event

        except Exception as e:
            logger.error(f"💥 Action failed: {e}")
            if self.debug:
                logger.error("Exception details:", exc_info=True)

            event = CodeActOutputEvent(output=f"Error during execution: {e}")
            ctx.write_event_to_stream(event)
            return event

    @step
    async def handle_execution_result(
        self, ctx: Context, ev: CodeActOutputEvent
    ) -> CodeActInputEvent:
        """Add execution result to history and loop back."""
        output = ev.output or "Code executed, but produced no output."

        # Add execution output as user message
        observation_text = f"Execution Result:\n```\n{output}\n```"
        self.shared_state.message_history.append({"role": "user", "content": [{"text": observation_text}]})

        return CodeActInputEvent()

    @step
    async def finalize(self, ev: CodeActEndEvent, ctx: Context) -> StopEvent:
        self.tools.finished = False
        ctx.write_event_to_stream(ev)

        return StopEvent(
            result={
                "success": ev.success,
                "reason": ev.reason,
                "code_executions": ev.code_executions,
            }
        )
