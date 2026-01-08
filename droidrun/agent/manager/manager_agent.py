"""
ManagerAgent - Planning and reasoning workflow.

This agent is responsible for:
- Analyzing the current state
- Creating plans and subgoals
- Tracking progress
- Deciding when tasks are complete
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional, Type

from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel
from opentelemetry import trace

from droidrun.agent.common.events import RecordUIStateEvent, ScreenshotEvent
from droidrun.agent.manager.events import (
    ManagerContextEvent,
    ManagerPlanDetailsEvent,
    ManagerResponseEvent,
)
from droidrun.agent.manager.prompts import parse_manager_response
from droidrun.agent.usage import get_usage_from_response
from droidrun.agent.utils.chat_utils import (
    to_chat_messages,
    filter_empty_messages,
)
import copy
from droidrun.agent.utils.inference import acall_with_retries
from droidrun.agent.utils.tracing_setup import record_langfuse_screenshot
from droidrun.agent.utils.prompt_resolver import PromptResolver
from droidrun.agent.utils.tools import build_custom_tool_descriptions
from droidrun.app_cards.app_card_provider import AppCardProvider
from droidrun.app_cards.providers import (
    CompositeAppCardProvider,
    LocalAppCardProvider,
    ServerAppCardProvider,
)
from droidrun.config_manager.prompt_loader import PromptLoader

if TYPE_CHECKING:
    from droidrun.agent.droid import DroidAgentState
    from droidrun.config_manager.config_manager import AgentConfig, TracingConfig
    from droidrun.tools import Tools


logger = logging.getLogger("droidrun")


class ManagerAgent(Workflow):
    """
    Planning and reasoning agent that decides what to do next.

    The Manager:
    1. Analyzes current device state and action history
    2. Creates plans with specific subgoals
    3. Tracks progress and completed steps
    4. Decides when tasks are complete or need to provide answers
    """

    def __init__(
        self,
        llm: LLM,
        tools_instance: "Tools | None",
        shared_state: "DroidAgentState",
        agent_config: "AgentConfig",
        custom_tools: dict = None,
        output_model: Type[BaseModel] | None = None,
        prompt_resolver: Optional[PromptResolver] = None,
        tracing_config: "TracingConfig | None" = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.config = agent_config.manager
        self.vision = agent_config.get_effective_vision("manager")
        self.tools_instance = tools_instance
        self.shared_state = shared_state
        self.custom_tools = custom_tools if custom_tools is not None else {}
        self.output_model = output_model
        self.agent_config = agent_config
        self.app_card_config = self.agent_config.app_cards
        self.prompt_resolver = prompt_resolver or PromptResolver()
        self.tracing_config = tracing_config

        # Initialize app card provider
        self.app_card_provider: AppCardProvider = self._initialize_app_card_provider()

        logger.debug("ManagerAgent initialized.")

    def _initialize_app_card_provider(self) -> AppCardProvider:
        """Initialize app card provider based on configuration mode."""
        if not self.app_card_config.enabled:

            class DisabledProvider(AppCardProvider):
                async def load_app_card(
                    self, package_name: str, instruction: str = ""
                ) -> str:
                    return ""

            return DisabledProvider()

        mode = self.app_card_config.mode.lower()

        if mode == "local":
            return LocalAppCardProvider(
                app_cards_dir=self.app_card_config.app_cards_dir
            )
        elif mode == "server":
            if not self.app_card_config.server_url:
                logger.warning("Server mode but no server_url, falling back to local")
                return LocalAppCardProvider(
                    app_cards_dir=self.app_card_config.app_cards_dir
                )
            return ServerAppCardProvider(
                server_url=self.app_card_config.server_url,
                timeout=self.app_card_config.server_timeout,
                max_retries=self.app_card_config.server_max_retries,
            )
        elif mode == "composite":
            if not self.app_card_config.server_url:
                logger.warning("Composite mode but no server_url, falling back to local")
                return LocalAppCardProvider(
                    app_cards_dir=self.app_card_config.app_cards_dir
                )
            return CompositeAppCardProvider(
                server_url=self.app_card_config.server_url,
                app_cards_dir=self.app_card_config.app_cards_dir,
                server_timeout=self.app_card_config.server_timeout,
                server_max_retries=self.app_card_config.server_max_retries,
            )
        else:
            logger.warning(f"Unknown app_card mode '{mode}', falling back to local")
            return LocalAppCardProvider(
                app_cards_dir=self.app_card_config.app_cards_dir
            )

    async def _build_system_prompt(self, has_text_to_modify: bool) -> str:
        """Build system prompt with all context."""
        # Build error history if needed
        error_history = None
        if self.shared_state.error_flag_plan:
            k = self.shared_state.err_to_manager_thresh
            error_history = [
                {"action": act, "summary": summ, "error": err_des}
                for act, summ, err_des in zip(
                    self.shared_state.action_history[-k:],
                    self.shared_state.summary_history[-k:],
                    self.shared_state.error_descriptions[-k:],
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

        # Output schema if provided
        output_schema = None
        if self.output_model is not None:
            output_schema = self.output_model.model_json_schema()

        variables = {
            "instruction": self.shared_state.instruction,
            "device_date": await self.tools_instance.get_date(),
            "app_card": self.shared_state.app_card,
            "important_notes": "",  # TODO: implement
            "error_history": error_history,
            "text_manipulation_enabled": has_text_to_modify,
            "custom_tools_descriptions": build_custom_tool_descriptions(
                self.custom_tools
            ),
            "scripter_execution_enabled": self.agent_config.scripter.enabled,
            "scripter_max_steps": self.agent_config.scripter.max_steps,
            "available_secrets": available_secrets,
            "variables": self.shared_state.custom_variables,
            "output_schema": output_schema,
        }

        custom_prompt = self.prompt_resolver.get_prompt("manager_system")
        if custom_prompt:
            return PromptLoader.render_template(custom_prompt, variables)
        else:
            return await PromptLoader.load_prompt(
                self.agent_config.get_manager_system_prompt_path(),
                variables,
            )

    def _build_user_message_content(self) -> str:
        """Build user message content with last action context."""
        parts = []

        # Add last thought
        if self.shared_state.last_thought:
            parts.append(f"<thought>\n{self.shared_state.last_thought}\n</thought>\n")

        # Add last action
        if self.shared_state.last_action:
            action_str = json.dumps(self.shared_state.last_action)
            parts.append(f"<last_action>\n{action_str}\n</last_action>\n")

        # Add last action summary
        if self.shared_state.last_summary:
            parts.append(
                f"<last_action_description>\n{self.shared_state.last_summary}\n</last_action_description>\n"
            )

        return "".join(parts)

    def _build_messages_with_context(
        self, system_prompt: str, screenshot: bytes | None = None
    ) -> list[dict]:
        """
        Build messages from history and inject current context.

        Args:
            system_prompt: System prompt text
            screenshot: Current screenshot if vision enabled

        Returns:
            List of message dicts ready for conversion
        """

        # Start with system message
        messages = [{"role": "system", "content": [{"text": system_prompt}]}]

        # Add accumulated message history (deep copy to avoid mutation)
        messages.extend(copy.deepcopy(self.shared_state.message_history))

        # Find last user message
        user_indices = [i for i, msg in enumerate(messages) if msg["role"] == "user"]

        if user_indices:
            last_user_idx = user_indices[-1]

            # Add memory to last user message
            current_memory = (self.shared_state.memory or "").strip()
            if current_memory:
                messages[last_user_idx]["content"].append(
                    {"text": f"\n<memory>\n{current_memory}\n</memory>\n"}
                )

            # Add current device state
            current_state = self.shared_state.formatted_device_state.strip()
            if current_state:
                messages[last_user_idx]["content"].append(
                    {"text": f"\n<device_state>\n{current_state}\n</device_state>\n"}
                )

            # Add screenshot if vision enabled
            if screenshot and self.vision:
                messages[last_user_idx]["content"].append({"image": screenshot})

            # Add script result if available
            if self.shared_state.last_scripter_message:
                status = "SUCCESS" if self.shared_state.last_scripter_success else "FAILED"
                script_context = (
                    f'\n<script_result status="{status}">\n'
                    f"{self.shared_state.last_scripter_message}\n"
                    f"</script_result>\n"
                )
                messages[last_user_idx]["content"].append({"text": script_context})
                self.shared_state.last_scripter_message = ""

            # Add previous device state to second-to-last user message
            if len(user_indices) >= 2:
                second_last_idx = user_indices[-2]
                prev_state = self.shared_state.previous_formatted_device_state.strip()
                if prev_state:
                    messages[second_last_idx]["content"].append(
                        {"text": f"\n<device_state>\n{prev_state}\n</device_state>\n"}
                    )

        messages = filter_empty_messages(messages)
        return messages

    async def _validate_and_retry(
        self, messages: list[dict], initial_response: str
    ) -> str:
        """Validate LLM response and retry if needed."""
        output = initial_response
        parsed = parse_manager_response(output)

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            error_message = None

            if parsed["answer"] and not parsed["plan"]:
                if parsed["success"] is None:
                    error_message = (
                        'You must include success="true" or success="false" attribute '
                        "in the <request_accomplished> tag.\n"
                        'Example: <request_accomplished success="true">Task completed</request_accomplished>\n'
                        "Retry again."
                    )
                else:
                    break  # Valid
            elif parsed["plan"] and parsed["answer"]:
                error_message = (
                    "You cannot use both request_accomplished tag while the plan is not finished. "
                    "If you want to use request_accomplished tag, please make sure the plan is finished.\n"
                    "Retry again."
                )
            elif not parsed["plan"]:
                error_message = (
                    "You must provide a plan to complete the task. "
                    "Please provide a plan with the correct format."
                )
            else:
                break  # Valid: plan without answer

            if error_message:
                retry_count += 1
                logger.warning(
                    f"Manager response invalid (retry {retry_count}/{max_retries}): {error_message}"
                )

                # Build retry messages
                retry_messages = messages + [
                    {"role": "assistant", "content": [{"text": output}]},
                    {"role": "user", "content": [{"text": error_message}]},
                ]

                chat_messages = to_chat_messages(retry_messages)

                try:
                    response = await acall_with_retries(
                        self.llm, chat_messages, stream=self.agent_config.streaming
                    )
                    output = response.message.content
                    parsed = parse_manager_response(output)
                except Exception as e:
                    logger.error(f"LLM retry failed: {e}")
                    break

        return output

    # ========================================================================
    # Workflow Steps
    # ========================================================================

    @step
    async def prepare_context(
        self, ctx: Context, ev: StartEvent
    ) -> ManagerContextEvent:
        """Gather context and prepare manager prompt."""
        logger.debug("💬 Preparing manager context...")

        # Get and format device state
        formatted_text, focused_text, a11y_tree, phone_state = (
            await self.tools_instance.get_state()
        )

        # Update shared state (previous ← current, current ← new)
        self.shared_state.previous_formatted_device_state = (
            self.shared_state.formatted_device_state
        )
        self.shared_state.formatted_device_state = formatted_text
        self.shared_state.focused_text = focused_text
        self.shared_state.a11y_tree = a11y_tree
        self.shared_state.phone_state = phone_state

        # Update package/activity tracking
        self.shared_state.update_current_app(
            package_name=phone_state.get("packageName", "Unknown"),
            activity_name=phone_state.get("currentApp", "Unknown"),
        )

        # Stream UI state for trajectory
        ctx.write_event_to_stream(RecordUIStateEvent(ui_state=a11y_tree))

        # Load app card
        if self.app_card_config.enabled:
            try:
                self.shared_state.app_card = await self.app_card_provider.load_app_card(
                    package_name=self.shared_state.current_package_name,
                    instruction=self.shared_state.instruction,
                )
            except Exception as e:
                logger.warning(f"Error loading app card: {e}")
                self.shared_state.app_card = ""
        else:
            self.shared_state.app_card = ""

        # Capture screenshot if needed
        screenshot = None
        if self.vision or (
            hasattr(self.tools_instance, "save_trajectories")
            and self.tools_instance.save_trajectories != "none"
        ):
            try:
                result = await self.tools_instance.take_screenshot()
                if isinstance(result, tuple):
                    success, screenshot = result
                    if not success:
                        logger.warning("📸 Screenshot capture failed")
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
                    logger.debug("📸 Screenshot captured for Manager")
            except Exception as e:
                logger.warning(f"Failed to capture screenshot: {e}")

        # Detect text manipulation mode
        focused_text_clean = focused_text.replace("'", "").strip()
        has_text_to_modify = focused_text_clean != ""

        # Store for next step
        self.shared_state.has_text_to_modify = has_text_to_modify
        self.shared_state.screenshot = screenshot

        # Build user message and add to history
        user_content = self._build_user_message_content()
        self.shared_state.message_history.append(
            {"role": "user", "content": [{"text": user_content}]}
        )

        event = ManagerContextEvent()
        ctx.write_event_to_stream(event)
        return event

    @step
    async def get_response(
        self, ctx: Context, ev: ManagerContextEvent
    ) -> ManagerResponseEvent:
        """Get LLM response."""
        logger.debug("🧠 Manager thinking about the plan...")

        has_text_to_modify = self.shared_state.has_text_to_modify
        screenshot = self.shared_state.screenshot

        # Build system prompt
        system_prompt = await self._build_system_prompt(has_text_to_modify)

        # Build messages with context
        messages = self._build_messages_with_context(
            system_prompt=system_prompt, screenshot=screenshot
        )

        # Convert and call LLM
        chat_messages = to_chat_messages(messages)

        try:
            logger.info("[cyan]📋 Manager response:[/cyan]")
            response = await acall_with_retries(
                self.llm, chat_messages, stream=self.agent_config.streaming
            )
            output = response.message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"Error calling LLM in manager: {e}") from e

        # Extract usage
        usage = None
        try:
            usage = get_usage_from_response(self.llm.class_name(), response)
        except Exception as e:
            logger.warning(f"Could not get usage: {e}")

        output = await self._validate_and_retry(messages, output)

        event = ManagerResponseEvent(response=output, usage=usage)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def process_response(
        self, ctx: Context, ev: ManagerResponseEvent
    ) -> ManagerPlanDetailsEvent:
        """Parse LLM response and update state."""
        logger.debug("⚙️ Processing manager response...")

        output = ev.response
        parsed = parse_manager_response(output)

        # Update memory (append)
        memory_update = parsed.get("memory", "").strip()
        if memory_update:
            if self.shared_state.memory:
                self.shared_state.memory += "\n" + memory_update
            else:
                self.shared_state.memory = memory_update

        # Append assistant response to message history
        self.shared_state.message_history.append(
            {"role": "assistant", "content": [{"text": output}]}
        )

        # Update unified state fields
        self.shared_state.previous_plan = self.shared_state.plan
        self.shared_state.plan = parsed["plan"]
        self.shared_state.current_subgoal = parsed["current_subgoal"]
        self.shared_state.last_thought = parsed["thought"]
        self.shared_state.manager_answer = parsed["answer"]

        if parsed.get("progress_summary"):
            self.shared_state.progress_summary = parsed["progress_summary"]

        event = ManagerPlanDetailsEvent(
            plan=parsed["plan"],
            subgoal=parsed["current_subgoal"],
            thought=parsed["thought"],
            answer=parsed["answer"],
            memory_update=memory_update,
            progress_summary=parsed.get("progress_summary", ""),
            success=parsed["success"],
            full_response=output,
        )
        ctx.write_event_to_stream(event)
        return event

    @step
    async def finalize(self, ctx: Context, ev: ManagerPlanDetailsEvent) -> StopEvent:
        logger.debug("✅ Manager planning complete")

        return StopEvent(
            result={
                "plan": ev.plan,
                "current_subgoal": ev.subgoal,
                "thought": ev.thought,
                "manager_answer": ev.answer,
                "memory_update": ev.memory_update,
                "success": ev.success,
            }
        )
