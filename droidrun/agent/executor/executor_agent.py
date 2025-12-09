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

from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from droidrun.agent.executor.events import (
    ExecutorActionEvent,
    ExecutorActionResultEvent,
    ExecutorContextEvent,
    ExecutorResponseEvent,
)
from droidrun.agent.executor.prompts import parse_executor_response
from droidrun.agent.usage import get_usage_from_response
from droidrun.agent.utils.inference import acall_with_retries
from droidrun.agent.utils.prompt_resolver import PromptResolver
from droidrun.agent.utils.tools import (
    ATOMIC_ACTION_SIGNATURES,
    click,
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

    The Executor:
    1. Receives a subgoal from the Manager
    2. Analyzes current UI state and context
    3. Selects an appropriate action to take
    4. Executes the action on the device
    5. Reports the outcome
    """

    def __init__(
        self,
        llm: LLM,
        tools_instance,
        shared_state: "DroidAgentState",
        agent_config: AgentConfig,
        custom_tools: dict = None,
        prompt_resolver: Optional[PromptResolver] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.agent_config = agent_config
        self.config = agent_config.executor
        self.vision = agent_config.executor.vision
        self.tools_instance = tools_instance
        self.shared_state = shared_state
        self.prompt_resolver = prompt_resolver or PromptResolver()

        self.custom_tools = custom_tools if custom_tools is not None else {}

        logger.info("✅ ExecutorAgent initialized successfully.")

    @step
    async def prepare_context(
        self, ctx: Context, ev: StartEvent
    ) -> ExecutorContextEvent:
        """
        Prepare executor context and prompt.

        This step:
        1. Extracts subgoal from event
        2. Builds action history context
        3. Builds system prompt with all variables
        4. Stores messages for next step
        """
        # macro tools context
        self.tools_instance._set_context(ctx)

        subgoal = ev.get("subgoal", "")
        logger.info(f"🧠 Executor thinking about action for: {subgoal}")

        # Prepare action history as structured data (last 5 actions)
        action_history = []
        if self.shared_state.action_history:
            n = min(5, len(self.shared_state.action_history))
            action_history = [
                {"action": act, "summary": summ, "outcome": outcome, "error": err_des}
                for act, summ, outcome, err_des in zip(
                    self.shared_state.action_history[-n:],
                    self.shared_state.summary_history[-n:],
                    self.shared_state.action_outcomes[-n:],
                    self.shared_state.error_descriptions[-n:],
                    strict=True,
                )
            ]

        # Get available secrets from credential manager
        available_secrets = []
        if (
            hasattr(self.tools_instance, "credential_manager")
            and self.tools_instance.credential_manager
        ):
            available_secrets = await self.tools_instance.credential_manager.get_keys()

        # Let Jinja2 handle all formatting
        variables = {
            "instruction": self.shared_state.instruction,
            "app_card": "",  # TODO: optionally implement app card loader
            "device_state": self.shared_state.formatted_device_state,
            "plan": self.shared_state.plan,
            "subgoal": subgoal,
            "progress_status": self.shared_state.progress_status,
            "atomic_actions": {**ATOMIC_ACTION_SIGNATURES, **self.custom_tools},
            "action_history": action_history,
            "available_secrets": available_secrets,
            "variables": self.shared_state.custom_variables,
        }

        custom_executor_prompt = self.prompt_resolver.get_prompt("executor_system")
        if custom_executor_prompt:
            system_prompt = PromptLoader.render_template(
                custom_executor_prompt, variables
            )
        else:
            system_prompt = await PromptLoader.load_prompt(
                self.agent_config.get_executor_system_prompt_path(),
                variables,
            )

        blocks = [TextBlock(text=system_prompt)]
        if self.vision:
            screenshot = self.shared_state.screenshot
            if screenshot is not None:
                blocks.append(ImageBlock(image=screenshot))
                logger.debug("📸 Using screenshot for Executor")
            else:
                logger.warning("⚠️ Vision enabled but no screenshot available")
        messages = [ChatMessage(role="user", blocks=blocks)]

        logger.debug("✅ Executor context prepared")

        event = ExecutorContextEvent(messages=messages, subgoal=subgoal)
        ctx.write_event_to_stream(event)

        return event

    @step
    async def get_response(
        self, ctx: Context, ev: ExecutorContextEvent
    ) -> ExecutorResponseEvent:
        """
        Executor thinks and gets LLM response.

        This step:
        1. Calls LLM with prepared messages
        2. Returns raw response for parsing
        """
        logger.info("🧠 Executor getting LLM response...")

        # Receive messages from event (not shared_state)
        messages = ev.messages

        try:
            response = await acall_with_retries(self.llm, messages)
            response_text = str(response)
        except Exception as e:
            raise RuntimeError(f"Error calling LLM in executor: {e}") from e

        # Extract usage from response
        try:
            usage = get_usage_from_response(self.llm.class_name(), response)
        except Exception as e:
            logger.warning(f"Could not get llm usage from response: {e}")
            usage = None

        logger.debug("✅ Executor finished thinking, sending response for parsing")

        # Emit event to stream and return for next step
        event = ExecutorResponseEvent(response_text=response_text, usage=usage)
        ctx.write_event_to_stream(event)

        return event

    @step
    async def process_response(
        self, ctx: Context, ev: ExecutorResponseEvent
    ) -> ExecutorActionEvent:
        """
        Parse LLM response and extract action.

        This step:
        1. Parses the raw LLM response
        2. Extracts action, thought, and description
        3. Returns action event for execution
        """
        logger.info("⚙️ Processing executor response...")

        response_text = ev.response_text

        # Parse response
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

        logger.info(f"💡 Thought: {parsed['thought']}")
        logger.info(f"🎯 Action: {parsed['action']}")
        logger.debug(f"  - Description: {parsed['description']}")

        event = ExecutorActionEvent(
            action_json=parsed["action"],
            thought=parsed["thought"],
            description=parsed["description"],
            full_response=response_text,
        )

        # Write event to stream for web interface
        ctx.write_event_to_stream(event)

        return event

    @step
    async def execute(
        self, ctx: Context, ev: ExecutorActionEvent
    ) -> ExecutorActionResultEvent:
        """
        Execute the selected action using the tools instance.

        Maps action JSON to appropriate tool calls and handles execution.
        """
        logger.info(f"⚡ Executing action: {ev.description}")

        # Parse action JSON
        try:
            action_dict = json.loads(ev.action_json)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse action JSON: {e}")
            return ExecutorActionResultEvent(
                action={"action": "invalid"},
                outcome=False,
                error=f"Invalid action JSON: {str(e)}",
                summary="Failed to parse action",
                thought=ev.thought,
                action_json=ev.action_json,
                full_response=ev.full_response,
            )

        outcome, error, summary = await self._execute_action(
            action_dict, ev.description
        )

        await asyncio.sleep(self.agent_config.after_sleep_action)

        logger.info(f"{'✅' if outcome else '❌'} Execution complete: {summary}")

        result_event = ExecutorActionResultEvent(
            action=action_dict,
            outcome=outcome,
            error=error,
            summary=summary,
            thought=ev.thought,
            action_json=ev.action_json,
            full_response=ev.full_response,
        )

        # Write event to stream for web interface
        ctx.write_event_to_stream(result_event)

        return result_event

    async def _execute_action(
        self, action_dict: dict, description: str
    ) -> tuple[bool, str, str]:
        """
        Execute a single action based on the action dictionary.

        Args:
            action_dict: Dictionary containing action type and parameters
            description: Human-readable description of the action

        Returns:
            Tuple of (outcome: bool, error: str, summary: str)
        """

        action_type = action_dict.get("action", "unknown")

        # Check custom_tools first (before atomic actions)
        if action_type in self.custom_tools:
            return await self._execute_custom_tool(action_type, action_dict)

        try:
            if action_type == "click":
                index = action_dict.get("index")
                if index is None:
                    return (
                        False,
                        "Missing 'index' parameter",
                        "Failed: click requires index",
                    )

                result = await click(index, tools=self.tools_instance)
                return True, "None", f"Clicked element at index {index}"

            elif action_type == "long_press":
                index = action_dict.get("index")
                if index is None:
                    return (
                        False,
                        "Missing 'index' parameter",
                        "Failed: long_press requires index",
                    )

                success = await long_press(index, tools=self.tools_instance)
                if success:
                    return True, "None", f"Long pressed element at index {index}"
                else:
                    return (
                        False,
                        "Long press failed",
                        f"Failed to long press at index {index}",
                    )

            elif action_type == "type":
                text = action_dict.get("text")
                index = action_dict.get("index", -1)

                if text is None:
                    return (
                        False,
                        "Missing 'text' parameter",
                        "Failed: type requires text",
                    )

                result = await type(text, index, tools=self.tools_instance)
                return True, "None", f"Typed '{text}' into element at index {index}"

            elif action_type == "system_button":
                button = action_dict.get("button")
                if button is None:
                    return (
                        False,
                        "Missing 'button' parameter",
                        "Failed: system_button requires button",
                    )

                result = await system_button(button, tools=self.tools_instance)
                if "Error" in result:
                    return False, result, f"Failed to press {button} button"
                return True, "None", f"Pressed {button} button"

            elif action_type == "swipe":
                coordinate = action_dict.get("coordinate")
                coordinate2 = action_dict.get("coordinate2")
                duration = action_dict.get("duration", 1.0)  # Default to 1.0 seconds

                if coordinate is None or coordinate2 is None:
                    return (
                        False,
                        "Missing coordinate parameters",
                        "Failed: swipe requires coordinate and coordinate2",
                    )

                # Validate coordinate format before calling swipe
                if not isinstance(coordinate, list) or len(coordinate) != 2:
                    return (
                        False,
                        f"Invalid coordinate format: {coordinate}",
                        "Failed: coordinate must be [x, y]",
                    )
                if not isinstance(coordinate2, list) or len(coordinate2) != 2:
                    return (
                        False,
                        f"Invalid coordinate2 format: {coordinate2}",
                        "Failed: coordinate2 must be [x, y]",
                    )

                success = await swipe(
                    coordinate, coordinate2, duration, tools=self.tools_instance
                )
                if success:
                    return (
                        True,
                        "None",
                        f"Swiped from {coordinate} to {coordinate2} over {duration}s",
                    )
                else:
                    return (
                        False,
                        "Swipe failed",
                        f"Failed to swipe from {coordinate} to {coordinate2}",
                    )

            elif action_type == "wait":
                duration = action_dict.get("duration")
                if duration is None:
                    return (
                        False,
                        "Missing 'duration' parameter",
                        "Failed: wait requires duration",
                    )

                result = await wait(duration)
                return True, "None", f"Waited for {duration} seconds"

            elif action_type == "open_app":
                text = action_dict.get("text")
                if text is None:
                    return (
                        False,
                        "Missing 'text' parameter",
                        "Failed: open_app requires text",
                    )

                result = await open_app(text, tools=self.tools_instance)
                return True, "None", f"Opened app: {text}"

            else:
                return (
                    False,
                    f"Unknown action type: {action_type}",
                    f"Failed: unknown action '{action_type}'",
                )

        except Exception as e:
            logger.error(f"❌ Exception during action execution: {e}", exc_info=True)
            return (
                False,
                f"Exception: {str(e)}",
                f"Failed to execute {action_type}: {str(e)}",
            )

    async def _execute_custom_tool(
        self, action_type: str, action_dict: dict
    ) -> tuple[bool, str, str]:
        """
        Execute a custom tool based on the action dictionary.

        Args:
            action_type: The custom tool name
            action_dict: Dictionary containing action parameters

        Returns:
            Tuple of (outcome: bool, error: str, summary: str)
        """
        try:
            tool_spec = self.custom_tools[action_type]
            tool_func = tool_spec["function"]

            # Extract arguments (exclude 'action' key)
            tool_args = {k: v for k, v in action_dict.items() if k != "action"}

            # Execute the custom tool function
            # Pass tools and shared_state as keyword arguments for flexible signatures
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

            # Success case
            summary = f"Executed custom tool '{action_type}'"
            if result is not None:
                summary += f": {str(result)}"

            return True, "None", summary

        except TypeError as e:
            # Likely missing or wrong arguments
            error_msg = f"Invalid arguments for custom tool '{action_type}': {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg, f"Failed: {action_type}"

        except Exception as e:
            # General execution error
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
                "outcome": ev.outcome,
                "error": ev.error,
                "summary": ev.summary,
                "thought": ev.thought,
                "action_json": ev.action_json,
                "full_response": ev.full_response,
            }
        )
