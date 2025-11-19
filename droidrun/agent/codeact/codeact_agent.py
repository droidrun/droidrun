import asyncio
import json
import logging
import re
import warnings
from typing import TYPE_CHECKING, List, Optional, Type, Union

from pydantic import BaseModel

# Suppress all warnings for llama-index import (version compatibility)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse
import inspect
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import Memory
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from droidrun.agent.codeact.events import (
    TaskEndEvent,
    TaskExecutionEvent,
    TaskExecutionResultEvent,
    TaskInputEvent,
    TaskThinkingEvent,
)
from droidrun.agent.common.constants import LLM_HISTORY_LIMIT
from droidrun.agent.common.events import RecordUIStateEvent, ScreenshotEvent
from droidrun.agent.usage import get_usage_from_response
from droidrun.agent.utils import chat_utils
from droidrun.agent.utils.executer import ExecuterState, SimpleCodeExecutor
from droidrun.agent.utils.prompt_resolver import PromptResolver
from droidrun.agent.utils.tools import (
    ATOMIC_ACTION_SIGNATURES,
    build_custom_tool_descriptions,
    get_atomic_tool_descriptions,
)
from droidrun.config_manager.config_manager import AgentConfig
from droidrun.config_manager.prompt_loader import PromptLoader
from droidrun.tools import Tools

if TYPE_CHECKING:
    from droidrun.agent.droid import DroidAgentState

logger = logging.getLogger("droidrun")


class CodeActAgent(Workflow):
    """
    An agent that uses a ReAct-like cycle (Thought -> Code -> Observation)
    to solve problems requiring code execution. It extracts code from
    Markdown blocks and uses specific step types for tracking.
    """

    def __init__(
        self,
        llm: LLM,
        agent_config: AgentConfig,
        tools_instance: "Tools",
        custom_tools: dict = None,
        atomic_tools: dict = None,
        debug: bool = False,
        shared_state: Optional["DroidAgentState"] = None,
        safe_execution_config=None,
        output_model: Type[BaseModel] | None = None,
        prompt_resolver: Optional[PromptResolver] = None,
        *args,
        **kwargs,
    ):
        assert llm, "llm must be provided."
        super().__init__(*args, **kwargs)

        self.llm = llm
        self.agent_config = agent_config
        self.config = agent_config.codeact  # Shortcut to codeact config
        self.max_steps = agent_config.max_steps
        self.vision = agent_config.codeact.vision
        self.debug = debug
        self.tools = tools_instance
        self.shared_state = shared_state
        self.output_model = output_model
        self.prompt_resolver = prompt_resolver or PromptResolver()

        self.chat_memory = None
        self.remembered_info = None

        self.goal = None
        self.code_exec_counter = 0

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
        self.tool_descriptions = get_atomic_tool_descriptions()
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
        self.system_prompt = None

        # Get safety settings
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

        logger.info("✅ CodeActAgent initialized successfully.")

    @step
    async def prepare_chat(self, ctx: Context, ev: StartEvent) -> TaskInputEvent:
        """Prepare chat history from user input."""
        # macro tools context
        self.tools._set_context(ctx)

        logger.info("💬 Preparing chat for task execution...")

        if hasattr(self.tools, "credential_manager") and self.tools.credential_manager:
            self._available_secrets = await self.tools.credential_manager.get_keys()

        # Load system prompt on first call (lazy loading)
        if self.system_prompt is None:
            custom_system_prompt = self.prompt_resolver.get_prompt("codeact_system")
            if custom_system_prompt:
                system_prompt_text = PromptLoader.render_template(
                    custom_system_prompt,
                    {
                        "tool_descriptions": self.tool_descriptions,
                        "available_secrets": self._available_secrets,
                        "variables": (
                            self.shared_state.custom_variables
                            if self.shared_state
                            else {}
                        ),
                        "output_schema": self._output_schema,
                    },
                )
            else:
                system_prompt_text = await PromptLoader.load_prompt(
                    self.agent_config.get_codeact_system_prompt_path(),
                    {
                        "tool_descriptions": self.tool_descriptions,
                        "available_secrets": self._available_secrets,
                        "variables": (
                            self.shared_state.custom_variables
                            if self.shared_state
                            else {}
                        ),
                        "output_schema": self._output_schema,
                    },
                )
            self.system_prompt = ChatMessage(role="system", content=system_prompt_text)

        self.chat_memory: Memory = await ctx.store.get(
            "chat_memory", default=Memory.from_defaults()
        )

        user_input = ev.get("input", default=None)
        assert user_input, "User input cannot be empty."

        if ev.remembered_info:
            self.remembered_info = ev.remembered_info

        logger.debug("  - Adding goal to memory.")
        goal = user_input

        # Format user prompt with goal
        custom_user_prompt = self.prompt_resolver.get_prompt("codeact_user")
        if custom_user_prompt:
            user_prompt_text = PromptLoader.render_template(
                custom_user_prompt,
                {
                    "goal": goal,
                    "variables": (
                        self.shared_state.custom_variables if self.shared_state else {}
                    ),
                },
            )
        else:
            user_prompt_text = await PromptLoader.load_prompt(
                self.agent_config.get_codeact_user_prompt_path(),
                {
                    "goal": goal,
                    "variables": (
                        self.shared_state.custom_variables if self.shared_state else {}
                    ),
                },
            )
        self.user_message = ChatMessage(role="user", content=user_prompt_text)

        # No thoughts prompt
        no_thoughts_text = f"""Your previous response provided code without explaining your reasoning first. Remember to always describe your thought process and plan *before* providing the code block.

The code you provided will be executed below.

Now, describe the next step you will take to address the original goal: {goal}"""
        self.no_thoughts_prompt = ChatMessage(role="user", content=no_thoughts_text)

        await self.chat_memory.aput(self.user_message)

        await ctx.store.set("chat_memory", self.chat_memory)
        input_messages = self.chat_memory.get_all()
        return TaskInputEvent(input=input_messages)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: TaskInputEvent
    ) -> TaskThinkingEvent | TaskEndEvent:
        """Handle LLM input."""
        chat_history = ev.input
        assert len(chat_history) > 0, "Chat history cannot be empty."
        ctx.write_event_to_stream(ev)

        if self.shared_state.step_number + 1 > self.max_steps:
            return TaskEndEvent(
                success=False,
                reason=f"Reached max step count of {self.max_steps} steps",
            )

        logger.info(f"🧠 Step {self.shared_state.step_number + 1}: Thinking...")

        model = self.llm.class_name()

        if "remember" in self.tool_list and self.remembered_info:
            await ctx.store.set("remembered_info", self.remembered_info)
            chat_history = await chat_utils.add_memory_block(
                self.remembered_info, chat_history
            )

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
                    await ctx.store.set("screenshot", screenshot)
                    logger.debug("📸 Screenshot captured for CodeAct")
            except Exception as e:
                logger.warning(f"Failed to capture screenshot: {e}")

        if self.vision and screenshot and model != "DeepSeek":
            chat_history = await chat_utils.add_screenshot_image_block(
                screenshot, chat_history
            )

        # Get and format device state using unified formatter
        try:
            # Get raw state from device - returns 4 values directly
            formatted_text, focused_text, a11y_tree, phone_state = (
                await self.tools.get_state()
            )

            # Update shared_state if available
            assert self.shared_state is not None, "Shared state is not set"
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

            # Add device state to chat using new chat_utils function
            # This injects into LAST user message, doesn't create new message
            chat_history = await chat_utils.add_device_state_block(
                formatted_text, chat_history
            )

        except Exception as e:
            logger.warning(f"⚠️ Error retrieving state from the connected device: {e}")
            if self.debug:
                logger.error("State retrieval error details:", exc_info=True)

        response = await self._get_llm_response(ctx, chat_history)
        if response is None:
            return TaskEndEvent(
                success=False, reason="LLM response is None. This is a critical error."
            )

        try:
            usage = get_usage_from_response(self.llm.class_name(), response)
        except Exception as e:
            logger.warning(f"Could not get llm usage from response: {e}")
            usage = None

        await self.chat_memory.aput(response.message)
        self.shared_state.step_number += 1

        code, thoughts = chat_utils.extract_code_and_thought(response.message.content)

        event = TaskThinkingEvent(thoughts=thoughts, code=code, usage=usage)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def handle_llm_output(
        self, ctx: Context, ev: TaskThinkingEvent
    ) -> Union[TaskExecutionEvent, TaskInputEvent]:
        """Handle LLM output."""
        logger.debug("⚙️ Handling LLM output...")
        code = ev.code
        thoughts = ev.thoughts

        if not thoughts:
            logger.warning(
                "🤔 LLM provided code without thoughts. Adding reminder prompt."
            )
            await self.chat_memory.aput(self.no_thoughts_prompt)
        else:
            logger.info(f"🤔 Reasoning: {thoughts}")

        if code:
            return TaskExecutionEvent(code=code)
        else:
            message = ChatMessage(
                role="user",
                content="No code was provided. If you want to mark task as complete (whether it failed or succeeded), use complete(success:bool, reason:str) function within a code block ```pythn\n```.",
            )
            await self.chat_memory.aput(message)
            return TaskInputEvent(input=self.chat_memory.get_all())

    @step
    async def execute_code(
        self, ctx: Context, ev: TaskExecutionEvent
    ) -> Union[TaskExecutionResultEvent, TaskEndEvent]:
        """Execute the code and return the result."""
        code = ev.code
        assert code, "Code cannot be empty."
        logger.info("⚡ Executing action...")
        logger.info(f"Code to execute:\n```python\n{code}\n```")

        try:
            self.code_exec_counter += 1
            result = await self.executor.execute(
                ExecuterState(ui_state=await ctx.store.get("ui_state", None)), code
            )
            logger.info(f"💡 Code execution successful. Result: {result}")
            await asyncio.sleep(self.agent_config.after_sleep_action)

            # Check if complete() was called
            if self.tools.finished:
                logger.info("✅ Task marked as complete via complete() function")

                # Validate completion state
                success = (
                    self.tools.success if self.tools.success is not None else False
                )
                reason = (
                    self.tools.reason
                    if self.tools.reason
                    else "Task completed without reason"
                )

                # Reset finished flag for next execution
                self.tools.finished = False

                logger.info(f"  - Success: {success}")
                logger.info(f"  - Reason: {reason}")

                return TaskEndEvent(success=success, reason=reason)

            self.remembered_info = self.tools.memory

            return TaskExecutionResultEvent(output=str(result))

        except Exception as e:
            logger.error(f"💥 Action failed: {e}")
            if self.debug:
                logger.error("Exception details:", exc_info=True)
            error_message = f"Error during execution: {e}"

            event = TaskExecutionResultEvent(output=error_message)
            ctx.write_event_to_stream(event)
            return event

    @step
    async def handle_execution_result(
        self, ctx: Context, ev: TaskExecutionResultEvent
    ) -> TaskInputEvent:
        """Handle the execution result. Currently it just returns InputEvent."""
        logger.debug("📊 Handling execution result...")
        # Get the output from the event
        output = ev.output
        if output is None:
            output = "Code executed, but produced no output."
            logger.warning("  - Execution produced no output.")
        else:
            logger.debug(
                f"  - Execution output: {output[:100]}..."
                if len(output) > 100
                else f"  - Execution output: {output}"
            )
        # Add the output to memory as an user message (observation)
        observation_message = ChatMessage(
            role="user", content=f"Execution Result:\n```\n{output}\n```"
        )
        await self.chat_memory.aput(observation_message)

        return TaskInputEvent(input=self.chat_memory.get_all())

    @step
    async def finalize(self, ev: TaskEndEvent, ctx: Context) -> StopEvent:
        """Finalize the workflow."""
        self.tools.finished = False
        await ctx.store.set("chat_memory", self.chat_memory)

        result = {}
        result.update(
            {
                "success": ev.success,
                "reason": ev.reason,
                "code_executions": self.code_exec_counter,
            }
        )

        return StopEvent(result)

    async def _get_llm_response(
        self, ctx: Context, chat_history: List[ChatMessage]
    ) -> ChatResponse | None:
        logger.debug("🔍 Getting LLM response...")
        limited_history = self._limit_history(chat_history)
        messages_to_send = [self.system_prompt] + limited_history
        messages_to_send = [chat_utils.message_copy(msg) for msg in messages_to_send]
        try:
            response = await self.llm.achat(messages=messages_to_send)
            logger.debug("🔍 Received LLM response.")

            filtered_chat_history = []
            for msg in limited_history:
                filtered_msg = chat_utils.message_copy(msg)
                if hasattr(filtered_msg, "blocks") and filtered_msg.blocks:
                    filtered_msg.blocks = [
                        block
                        for block in filtered_msg.blocks
                        if not isinstance(block, chat_utils.ImageBlock)
                    ]
                filtered_chat_history.append(filtered_msg)

            assert hasattr(
                response, "message"
            ), f"LLM response does not have a message attribute.\nResponse: {response}"
        except Exception as e:
            if (
                self.llm.class_name() == "Gemini_LLM"
                and "You exceeded your current quota" in str(e)
            ):
                s = str(e._details[2])
                match = re.search(r"seconds:\s*(\d+)", s)
                if match:
                    seconds = int(match.group(1)) + 1
                    logger.error(f"Rate limit error. Retrying in {seconds} seconds...")
                    await asyncio.sleep(seconds)
                else:
                    logger.error("Rate limit error. Retrying in 5 seconds...")
                    await asyncio.sleep(40)
                logger.debug("🔍 Retrying call to LLM...")
                response = await self.llm.achat(messages=messages_to_send)
            elif self.llm.class_name() == "Anthropic_LLM" and "overloaded_error" in str(
                e
            ):
                # Use exponential backoff for Anthropic errors
                if not hasattr(self, "_anthropic_retry_count"):
                    self._anthropic_retry_count = 0
                self._anthropic_retry_count += 1
                seconds = min(2**self._anthropic_retry_count, 60)  # Cap at 60 seconds
                logger.error(
                    f"Anthropic overload error. Retrying in {seconds} seconds... (attempt {self._anthropic_retry_count})"
                )
                await asyncio.sleep(seconds)
                logger.debug("🔍 Retrying call to LLM...")
                response = await self.llm.achat(messages=messages_to_send)
                self._anthropic_retry_count = 0  # Reset on success
            else:
                logger.error(f"Could not get an answer from LLM: {repr(e)}")
                raise e
        logger.debug("  - Received response from LLM.")
        return response

    def _limit_history(self, chat_history: List[ChatMessage]) -> List[ChatMessage]:
        if LLM_HISTORY_LIMIT <= 0:
            return chat_history

        max_messages = LLM_HISTORY_LIMIT * 2
        if len(chat_history) <= max_messages:
            return chat_history

        preserved_head: List[ChatMessage] = []
        if chat_history and chat_history[0].role == "user":
            preserved_head = [chat_history[0]]

        tail = chat_history[-max_messages:]
        if preserved_head and preserved_head[0] in tail:
            preserved_head = []

        return preserved_head + tail
