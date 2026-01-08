"""
ExecutorAgent - Action execution workflow.

This agent is responsible for:
- Taking a specific subgoal from the Manager
- Analyzing the current UI state
- Selecting and executing appropriate actions
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Optional

from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from droidrun.agent.executor.events import (
    ExecutorActionEvent,
    ExecutorContextEvent,
    ExecutorResponseEvent,
    ExecutorActionResultEvent,
)
from droidrun.agent.executor.prompts import parse_executor_response
from droidrun.agent.usage import get_usage_from_response
from droidrun.agent.utils.chat_utils import to_chat_messages
from droidrun.agent.utils.inference import acall_with_retries
from droidrun.agent.utils.prompt_resolver import PromptResolver
from droidrun.agent.utils.tools import (
    ATOMIC_ACTION_SIGNATURES,
    click,
    click_at,
    click_area,
    filter_atomic_actions,
    long_press,
    open_app,
    swipe,
    system_button,
    type,
    wait,
)
from droidrun.config_manager.config_manager import AgentConfig
from droidrun.config_manager.prompt_loader import PromptLoader

if TYPE_CHECKING:
    from droidrun.agent.droid import DroidAgentState

logger = logging.getLogger("droidrun")


class ExecutorAgent(Workflow):
    """
    Action execution agent that performs specific actions.

    Single-turn agent: receives subgoal, selects action, executes it.
    Uses dict messages, converts to ChatMessage at LLM call time.
    """

    def __init__(
        self,
        llm: LLM,
        tools_instance,
        shared_state: "DroidAgentState",
        agent_config: AgentConfig,
        custom_tools: dict = None,
        atomic_tools: dict = None,
        prompt_resolver: Optional[PromptResolver] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.agent_config = agent_config
        self.config = agent_config.executor
        self.vision = agent_config.get_effective_vision("executor")
        self.tools_instance = tools_instance
        self.shared_state = shared_state
        self.prompt_resolver = prompt_resolver or PromptResolver()

        self.custom_tools = custom_tools if custom_tools is not None else {}
        
        # Build atomic tools based on click_mode from interaction_mode
        if atomic_tools is not None:
            self.atomic_tools = atomic_tools
        else:
            # Filter atomic actions based on click_mode
            click_mode = agent_config.interaction_mode.click_mode
            self.atomic_tools = filter_atomic_actions([], click_mode=click_mode)

        logger.debug("ExecutorAgent initialized.")

    @step
    async def prepare_context(
        self, ctx: Context, ev: StartEvent
    ) -> ExecutorContextEvent:
        """Prepare executor context and prompt."""
        self.tools_instance._set_context(ctx)

        subgoal = ev.get("subgoal", "")
        logger.debug(f"🧠 Executor thinking about action for: {subgoal}")

        # Build action history (last 5)
        action_history = []
        if self.shared_state.action_history:
            n = min(5, len(self.shared_state.action_history))
            action_history = [
                {"action": act, "summary": summ, "outcome": outcome, "error": err}
                for act, summ, outcome, err in zip(
                    self.shared_state.action_history[-n:],
                    self.shared_state.summary_history[-n:],
                    self.shared_state.action_outcomes[-n:],
                    self.shared_state.error_descriptions[-n:],
                    strict=True,
                )
            ]

        # Get available secrets
        available_secrets = []
        if (
            hasattr(self.tools_instance, "credential_manager")
            and self.tools_instance.credential_manager
        ):
            available_secrets = await self.tools_instance.credential_manager.get_keys()

        # Build prompt variables
        variables = {
            "instruction": self.shared_state.instruction,
            "app_card": "",
            "device_state": self.shared_state.formatted_device_state,
            "plan": self.shared_state.plan,
            "subgoal": subgoal,
            "progress_status": self.shared_state.progress_summary,
            "atomic_actions": {**self.atomic_tools, **self.custom_tools},
            "action_history": action_history,
            "available_secrets": available_secrets,
            "variables": self.shared_state.custom_variables,
        }

        custom_prompt = self.prompt_resolver.get_prompt("executor_system")
        if custom_prompt:
            prompt_text = PromptLoader.render_template(custom_prompt, variables)
        else:
            prompt_text = await PromptLoader.load_prompt(
                self.agent_config.get_executor_system_prompt_path(),
                variables,
            )

        # Build message as dict
        messages = [{"role": "user", "content": [{"text": prompt_text}]}]

        # Add screenshot if vision enabled
        if self.vision:
            screenshot = self.shared_state.screenshot
            if screenshot is not None:
                messages[0]["content"].append({"image": screenshot})
                logger.debug("📸 Using screenshot for Executor")
            else:
                logger.warning("⚠️ Vision enabled but no screenshot available")
        await ctx.store.set("executor_messages", messages)
        event = ExecutorContextEvent(subgoal=subgoal)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def get_response(
        self, ctx: Context, ev: ExecutorContextEvent
    ) -> ExecutorResponseEvent:
        """Get LLM response."""
        logger.debug("Executor getting LLM response...")

        # Get messages from context
        messages = await ctx.store.get("executor_messages")

        # Convert to ChatMessage and call LLM
        chat_messages = to_chat_messages(messages)

        try:
            logger.info("[green]Executor response:[/green]")
            response = await acall_with_retries(
                self.llm, chat_messages, stream=self.agent_config.streaming
            )
            response_text = str(response)
        except Exception as e:
            raise RuntimeError(f"Error calling LLM in executor: {e}") from e

        # Extract usage
        usage = None
        try:
            usage = get_usage_from_response(self.llm.class_name(), response)
        except Exception as e:
            logger.warning(f"Could not get usage: {e}")

        event = ExecutorResponseEvent(response=response_text, usage=usage)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def process_response(
        self, ctx: Context, ev: ExecutorResponseEvent
    ) -> ExecutorActionEvent:
        """Parse LLM response and extract action."""
        logger.debug("⚙️ Processing executor response...")

        response_text = ev.response

        try:
            parsed = parse_executor_response(response_text)
        except Exception as e:
            logger.error(f"❌ Failed to parse executor response: {e}")
            return ExecutorActionEvent(
                action_json=json.dumps({"action": "invalid"}),
                thought=f"Failed to parse response: {str(e)}",
                description="Invalid response format from LLM",
                full_response=response_text,
            )

        # Update unified state
        self.shared_state.last_thought = parsed["thought"]

        event = ExecutorActionEvent(
            action_json=parsed["action"],
            thought=parsed["thought"],
            description=parsed["description"],
            full_response=response_text,
        )

        ctx.write_event_to_stream(event)
        return event

    @step
    async def execute(
        self, ctx: Context, ev: ExecutorActionEvent
    ) -> ExecutorActionResultEvent:
        """Execute the action."""
        logger.debug(f"⚡ Executing action: {ev.description}")

        try:
            action_dict = json.loads(ev.action_json)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse action JSON: {e}")
            event = ExecutorActionResultEvent(
                action={"action": "invalid"},
                success=False,
                error=f"Invalid action JSON: {str(e)}",
                summary="Failed to parse action",
                thought=ev.thought,
                full_response=ev.full_response,
            )
            ctx.write_event_to_stream(event)
            return event

        success, error, summary = await self._execute_action(action_dict, ev.description)

        await asyncio.sleep(self.agent_config.after_sleep_action)

        logger.debug(f"{'✅' if success else '❌'} Execution complete: {summary}")

        event = ExecutorActionResultEvent(
            action=action_dict,
            success=success,
            error=error,
            summary=summary,
            thought=ev.thought,
            full_response=ev.full_response,
        )
        ctx.write_event_to_stream(event)
        return event


    async def _execute_action(
        self, action_dict: dict, description: str
    ) -> tuple[bool, str, str]:
        """Execute action and return (success, error, summary)."""
        action_type = action_dict.get("action", "unknown")

        # Check custom tools first
        if action_type in self.custom_tools:
            return await self._execute_custom_tool(action_type, action_dict)

        try:
            if action_type == "click":
                index = action_dict.get("index")
                if index is None:
                    return False, "Missing 'index' parameter", "Failed: click requires index"
                await click(index, tools=self.tools_instance)
                return True, "", f"Clicked element at index {index}"

            elif action_type == "long_press":
                index = action_dict.get("index")
                if index is None:
                    return False, "Missing 'index' parameter", "Failed: long_press requires index"
                success = await long_press(index, tools=self.tools_instance)
                if success:
                    return True, "", f"Long pressed element at index {index}"
                return False, "Long press failed", f"Failed to long press at index {index}"

            elif action_type == "type":
                text = action_dict.get("text")
                index = action_dict.get("index", -1)
                clear = action_dict.get("clear", False)
                if text is None:
                    return False, "Missing 'text' parameter", "Failed: type requires text"
                await type(text, index, clear=clear, tools=self.tools_instance)
                return True, "", f"Typed '{text}' into element at index {index}"

            elif action_type == "system_button":
                button = action_dict.get("button")
                if button is None:
                    return False, "Missing 'button' parameter", "Failed: system_button requires button"
                result = await system_button(button, tools=self.tools_instance)
                if "Error" in result:
                    return False, result, f"Failed to press {button} button"
                return True, "", f"Pressed {button} button"

            elif action_type == "swipe":
                coordinate = action_dict.get("coordinate")
                coordinate2 = action_dict.get("coordinate2")
                duration = action_dict.get("duration", 1.0)

                if coordinate is None or coordinate2 is None:
                    return False, "Missing coordinate parameters", "Failed: swipe requires coordinates"

                if not isinstance(coordinate, list) or len(coordinate) != 2:
                    return False, f"Invalid coordinate: {coordinate}", "Failed: coordinate must be [x, y]"
                if not isinstance(coordinate2, list) or len(coordinate2) != 2:
                    return False, f"Invalid coordinate2: {coordinate2}", "Failed: coordinate2 must be [x, y]"

                success = await swipe(coordinate, coordinate2, duration, tools=self.tools_instance)
                if success:
                    return True, "", f"Swiped from {coordinate} to {coordinate2}"
                return False, "Swipe failed", f"Failed to swipe from {coordinate} to {coordinate2}"

            elif action_type == "wait":
                duration = action_dict.get("duration")
                if duration is None:
                    return False, "Missing 'duration' parameter", "Failed: wait requires duration"
                await wait(duration)
                return True, "", f"Waited for {duration} seconds"

            elif action_type == "open_app":
                text = action_dict.get("text")
                if text is None:
                    return False, "Missing 'text' parameter", "Failed: open_app requires text"
                await open_app(text, tools=self.tools_instance)
                return True, "", f"Opened app: {text}"

            elif action_type == "click_at":
                x = action_dict.get("x")
                y = action_dict.get("y")
                if x is None or y is None:
                    return False, "Missing 'x' or 'y' parameter", "Failed: click_at requires x and y"
                result = await click_at(x, y, tools=self.tools_instance)
                return True, "", f"Clicked at coordinates ({x}, {y}): {result}"

            elif action_type == "click_area":
                x1 = action_dict.get("x1")
                y1 = action_dict.get("y1")
                x2 = action_dict.get("x2")
                y2 = action_dict.get("y2")
                if None in (x1, y1, x2, y2):
                    return False, "Missing coordinate parameters", "Failed: click_area requires x1, y1, x2, y2"
                result = await click_area(x1, y1, x2, y2, tools=self.tools_instance)
                return True, "", f"Clicked area center [{x1},{y1},{x2},{y2}]: {result}"

            else:
                return False, f"Unknown action type: {action_type}", f"Failed: unknown action '{action_type}'"

        except Exception as e:
            logger.error(f"Exception during action execution: {e}", exc_info=True)
            return False, f"Exception: {str(e)}", f"Failed to execute {action_type}: {str(e)}"

    async def _execute_custom_tool(
        self, action_type: str, action_dict: dict
    ) -> tuple[bool, str, str]:
        """Execute custom tool."""
        try:
            tool_spec = self.custom_tools[action_type]
            tool_func = tool_spec["function"]

            tool_args = {k: v for k, v in action_dict.items() if k != "action"}

            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(
                    **tool_args,
                    tools=self.tools_instance,
                    shared_state=self.shared_state,
                )
            else:
                result = tool_func(
                    **tool_args,
                    tools=self.tools_instance,
                    shared_state=self.shared_state,
                )

            summary = f"Executed custom tool '{action_type}'"
            if result is not None:
                summary += f": {str(result)}"

            return True, "", summary

        except TypeError as e:
            error_msg = f"Invalid arguments for custom tool '{action_type}': {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg, f"Failed: {action_type}"

        except Exception as e:
            error_msg = f"Error executing custom tool '{action_type}': {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            return False, error_msg, f"Failed: {action_type}"

    @step
    async def finalize(self, ctx: Context, ev: ExecutorActionResultEvent) -> StopEvent:
        """Return executor results to parent workflow."""
        logger.debug("✅ Executor execution complete")

        return StopEvent(
            result={
                "action": ev.action,
                "outcome": ev.success,
                "error": ev.error,
                "summary": ev.summary,
                "thought": ev.thought,
            }
        )
