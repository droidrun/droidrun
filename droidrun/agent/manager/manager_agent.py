"""
ManagerAgent - Planning and reasoning workflow.

This agent is responsible for:
- Analyzing the current state
- Creating plans and subgoals
- Tracking progress
- Deciding when tasks are complete
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Type

from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel

from droidrun.agent.common.events import RecordUIStateEvent, ScreenshotEvent
from droidrun.agent.manager.events import (
    ManagerContextEvent,
    ManagerPlanDetailsEvent,
    ManagerResponseEvent,
)
from droidrun.agent.manager.prompts import parse_manager_response
from droidrun.agent.usage import get_usage_from_response
from droidrun.agent.utils.chat_utils import (
    remove_empty_messages,
    convert_messages_to_chatmessages,
)
from droidrun.agent.utils.device_state_formatter import format_device_state
from droidrun.agent.utils.inference import acall_with_retries
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
    from droidrun.config_manager.config_manager import AgentConfig
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.config = agent_config.manager
        self.vision = self.config.vision
        self.tools_instance = tools_instance
        self.shared_state = shared_state
        self.custom_tools = custom_tools if custom_tools is not None else {}
        self.output_model = output_model
        self.agent_config = agent_config
        self.app_card_config = self.agent_config.app_cards
        self.prompt_resolver = prompt_resolver or PromptResolver()

        # Initialize app card provider based on mode
        self.app_card_provider: AppCardProvider = self._initialize_app_card_provider()

        logger.info("✅ ManagerAgent initialized successfully.")

    def _initialize_app_card_provider(self) -> AppCardProvider:
        """Initialize app card provider based on configuration mode."""
        if not self.app_card_config.enabled:
            # Return a dummy provider that always returns empty string
            class DisabledProvider(AppCardProvider):
                async def load_app_card(
                    self, package_name: str, instruction: str = ""
                ) -> str:
                    return ""

            return DisabledProvider()

        mode = self.app_card_config.mode.lower()

        if mode == "local":
            logger.info(
                f"Initializing local app card provider (dir: {self.app_card_config.app_cards_dir})"
            )
            return LocalAppCardProvider(
                app_cards_dir=self.app_card_config.app_cards_dir
            )

        elif mode == "server":
            if not self.app_card_config.server_url:
                logger.warning(
                    "Server mode enabled but no server_url configured, falling back to local"
                )
                return LocalAppCardProvider(
                    app_cards_dir=self.app_card_config.app_cards_dir
                )

            logger.info(
                f"Initializing server app card provider (url: {self.app_card_config.server_url})"
            )
            return ServerAppCardProvider(
                server_url=self.app_card_config.server_url,
                timeout=self.app_card_config.server_timeout,
                max_retries=self.app_card_config.server_max_retries,
            )

        elif mode == "composite":
            if not self.app_card_config.server_url:
                logger.warning(
                    "Composite mode enabled but no server_url configured, falling back to local"
                )
                return LocalAppCardProvider(
                    app_cards_dir=self.app_card_config.app_cards_dir
                )

            logger.info(
                f"Initializing composite app card provider "
                f"(server: {self.app_card_config.server_url}, local: {self.app_card_config.app_cards_dir})"
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

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _build_system_prompt(self, has_text_to_modify: bool) -> str:
        """Build system prompt with all context."""

        # Prepare error history as structured data (if needed)
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

        # Get available secrets from credential manager
        available_secrets = []
        if (
            hasattr(self.tools_instance, "credential_manager")
            and self.tools_instance.credential_manager
        ):
            available_secrets = await self.tools_instance.credential_manager.get_keys()

        # Prepare output structure schema if provided
        output_schema = None
        if self.output_model is not None:
            output_schema = self.output_model.model_json_schema()

        # Let Jinja2 handle all formatting and conditionals
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

        custom_manager_prompt = self.prompt_resolver.get_prompt("manager_system")
        if custom_manager_prompt:
            return PromptLoader.render_template(custom_manager_prompt, variables)
        else:
            return await PromptLoader.load_prompt(
                self.agent_config.get_manager_system_prompt_path(),
                variables,
            )

    def _build_messages_with_context(
        self, system_prompt: str, screenshot: str = None
    ) -> list[dict]:
        """
        Build messages from history and inject current context.

        Args:
            system_prompt: System prompt to use
            screenshot: Path to current screenshot (if vision enabled)

        Returns:
            List of message dicts ready for conversion
        """
        import copy

        # Start with system message
        messages = [{"role": "system", "content": [{"text": system_prompt}]}]

        # Add accumulated message history (deep copy to avoid mutation)
        messages.extend(copy.deepcopy(self.shared_state.message_history))

        # ====================================================================
        # Inject memory, device state, screenshot to LAST user message
        # ====================================================================
        # Find last user message index
        user_indices = [i for i, msg in enumerate(messages) if msg["role"] == "user"]

        if user_indices:
            last_user_idx = user_indices[-1]

            # Add memory to last user message
            current_memory = (self.shared_state.memory or "").strip()
            if current_memory:
                if (
                    messages[last_user_idx]["content"]
                    and "text" in messages[last_user_idx]["content"][0]
                ):
                    messages[last_user_idx]["content"][0][
                        "text"
                    ] += f"\n<memory>\n{current_memory}\n</memory>\n"
                else:
                    messages[last_user_idx]["content"].insert(
                        0, {"text": f"<memory>\n{current_memory}\n</memory>\n"}
                    )

            # Add CURRENT device state to last user message (use unified state)
            current_state = self.shared_state.formatted_device_state.strip()
            if current_state:
                if (
                    messages[last_user_idx]["content"]
                    and "text" in messages[last_user_idx]["content"][0]
                ):
                    messages[last_user_idx]["content"][0][
                        "text"
                    ] += f"\n<device_state>\n{current_state}\n</device_state>\n"
                else:
                    messages[last_user_idx]["content"].insert(
                        0,
                        {"text": f"<device_state>\n{current_state}\n</device_state>\n"},
                    )

            # Add screenshot to last user message
            if screenshot and self.vision:
                messages[last_user_idx]["content"].append({"image": screenshot})

            # Add script result if available
            if self.shared_state.last_scripter_message:
                status = (
                    "SUCCESS" if self.shared_state.last_scripter_success else "FAILED"
                )
                script_context = (
                    f'\n<script_result status="{status}">\n'
                    f"{self.shared_state.last_scripter_message}\n"
                    f"</script_result>\n"
                )

                if (
                    messages[last_user_idx]["content"]
                    and "text" in messages[last_user_idx]["content"][0]
                ):
                    messages[last_user_idx]["content"][0]["text"] += script_context
                else:
                    messages[last_user_idx]["content"].insert(
                        0, {"text": script_context}
                    )

                # Clear after injection (avoid duplicate injection)
                self.shared_state.last_scripter_message = ""

            # Add PREVIOUS device state to SECOND-TO-LAST user message (if exists)
            if len(user_indices) >= 2:
                second_last_user_idx = user_indices[-2]
                prev_state = self.shared_state.previous_formatted_device_state.strip()

                if prev_state:
                    if (
                        messages[second_last_user_idx]["content"]
                        and "text" in messages[second_last_user_idx]["content"][0]
                    ):
                        messages[second_last_user_idx]["content"][0][
                            "text"
                        ] += f"\n<device_state>\n{prev_state}\n</device_state>\n"
                    else:
                        messages[second_last_user_idx]["content"].insert(
                            0,
                            {
                                "text": f"<device_state>\n{prev_state}\n</device_state>\n"
                            },
                        )
        messages = remove_empty_messages(messages)
        return messages

    async def _validate_and_retry_llm_call(
        self, ctx: Context, initial_messages: list[dict], initial_response: str
    ) -> str:
        """
        Validate LLM response and retry if needed.

        Args:
            ctx: Workflow context
            initial_messages: Messages sent to LLM
            initial_response: Initial LLM response

        Returns:
            Final validated response (may be same as initial or from retry)
        """

        output_planning = initial_response
        parsed = parse_manager_response(output_planning)

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            # Validation rules
            error_message = None

            if parsed["answer"] and not parsed["plan"]:
                # Check if success attribute is present
                if parsed["success"] is None:
                    error_message = 'You must include success="true" or success="false" attribute in the <request_accomplished> tag.\nExample: <request_accomplished success="true">Task completed successfully</request_accomplished>\nRetry again.'
                else:
                    # Valid: answer without plan (task complete)
                    break
            elif parsed["plan"] and parsed["answer"]:
                error_message = "You cannot use both request_accomplished tag while the plan is not finished. If you want to use request_accomplished tag, please make sure the plan is finished.\nRetry again."
            elif not parsed["plan"]:
                error_message = "You must provide a plan to complete the task. Please provide a plan with the correct format."
            else:
                # Valid: plan without answer
                break

            if error_message:
                retry_count += 1
                logger.warning(
                    f"Manager response invalid (retry {retry_count}/{max_retries}): {error_message}"
                )

                # Retry with error message
                retry_messages = initial_messages + [
                    {"role": "assistant", "content": [{"text": output_planning}]},
                    {"role": "user", "content": [{"text": error_message}]},
                ]

                chat_messages = convert_messages_to_chatmessages(retry_messages)

                try:
                    response = await acall_with_retries(self.llm, chat_messages)
                    output_planning = response.message.content
                    parsed = parse_manager_response(output_planning)
                except Exception as e:
                    logger.error(f"LLM retry failed: {e}")
                    break  # Give up retrying

        return output_planning

    # ========================================================================
    # Workflow Steps
    # ========================================================================

    @step
    async def prepare_context(
        self, ctx: Context, ev: StartEvent
    ) -> ManagerContextEvent:
        """
        Gather context and prepare manager prompt.

        This step:
        1. Gets current device state (UI elements, screenshot)
        2. Detects text manipulation mode
        3. Builds message history entry with last action
        4. Stores context for get_response() step
        """
        logger.info("💬 Preparing manager context...")

        # ====================================================================
        # Step 1: Get and format device state using unified formatter
        # ====================================================================
        raw_state = await self.tools_instance.get_state()
        formatted_text, focused_text, a11y_tree, phone_state = format_device_state(
            raw_state
        )

        # Update shared state (previous ← current, current ← new)
        self.shared_state.previous_formatted_device_state = (
            self.shared_state.formatted_device_state
        )
        self.shared_state.formatted_device_state = formatted_text
        self.shared_state.focused_text = focused_text
        self.shared_state.a11y_tree = a11y_tree
        self.shared_state.phone_state = phone_state

        # Extract and store package/app name
        self.shared_state.update_current_app(
            package_name=phone_state.get("packageName", "Unknown"),
            activity_name=phone_state.get("currentApp", "Unknown"),
        )

        # Stream UI state for trajectory
        ctx.write_event_to_stream(RecordUIStateEvent(ui_state=a11y_tree))

        # ====================================================================
        # Step 1.5: Load app card (blocking)
        # ====================================================================
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
        # ====================================================================
        # Step 2: Capture screenshot if needed for vision or trajectories
        # ====================================================================
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
                        logger.warning("Screenshot capture failed")
                        screenshot = None
                else:
                    screenshot = result

                if screenshot:
                    ctx.write_event_to_stream(ScreenshotEvent(screenshot=screenshot))
                    logger.debug("📸 Screenshot captured for Manager")
            except Exception as e:
                logger.warning(f"Failed to capture screenshot: {e}")
                screenshot = None

        # ====================================================================
        # Step 3: Detect text manipulation mode
        # ====================================================================
        # Show text manipulation section when there's focused text with content
        # (TEXT_TASK is now only for complex text editing, not simple clearing)
        focused_text_clean = focused_text.replace("'", "").strip()
        has_text_to_modify = focused_text_clean != ""

        # ====================================================================
        # Step 5: Build user message entry
        # ====================================================================
        parts = []

        # Add context from last action
        if self.shared_state.finish_thought:
            parts.append(f"<thought>\n{self.shared_state.finish_thought}\n</thought>\n")

        if self.shared_state.last_action:
            import json

            action_str = json.dumps(self.shared_state.last_action)
            parts.append(f"<last_action>\n{action_str}\n</last_action>\n")

        if self.shared_state.last_summary:
            parts.append(
                f"<last_action_description>\n{self.shared_state.last_summary}\n</last_action_description>\n"
            )

        self.shared_state.message_history.append(
            {"role": "user", "content": [{"text": "".join(parts)}]}
        )

        # Store has_text_to_modify and screenshot for next step
        self.shared_state.has_text_to_modify = has_text_to_modify
        self.shared_state.screenshot = screenshot

        logger.debug(
            f"  - Device state prepared (text_modify={has_text_to_modify}, screenshot={screenshot is not None})"
        )
        event = ManagerContextEvent()
        ctx.write_event_to_stream(event)
        return event

    @step
    async def get_response(
        self, ctx: Context, ev: ManagerContextEvent
    ) -> ManagerResponseEvent:
        """
        Manager thinks and gets LLM response.

        This step:
        1. Builds system prompt with all context
        2. Builds messages from history with injected context
        3. Calls LLM
        4. Validates and retries if needed
        5. Returns raw validated response
        """
        logger.info("🧠 Manager thinking about the plan...")

        has_text_to_modify = self.shared_state.has_text_to_modify
        screenshot = self.shared_state.screenshot

        # ====================================================================
        # Step 1: Build system prompt (app card already loaded in prepare_context)
        # ====================================================================
        system_prompt = await self._build_system_prompt(has_text_to_modify)

        # ====================================================================
        # Step 2: Build messages with context
        # ====================================================================
        messages = self._build_messages_with_context(
            system_prompt=system_prompt, screenshot=screenshot
        )

        # ====================================================================
        # Step 3: Convert messages and call LLM
        # ====================================================================
        chat_messages = convert_messages_to_chatmessages(messages)

        try:
            response = await acall_with_retries(self.llm, chat_messages)
            output_planning = response.message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"Error calling LLM in manager: {e}") from e

        # Extract usage from response
        try:
            usage = get_usage_from_response(self.llm.class_name(), response)
        except Exception as e:
            logger.warning(f"Could not get llm usage from response: {e}")
            usage = None

        # ====================================================================
        # Step 4: Validate and retry if needed
        # ====================================================================
        output_planning = await self._validate_and_retry_llm_call(
            ctx=ctx, initial_messages=messages, initial_response=output_planning
        )

        logger.debug("✅ Manager finished thinking, sending response for processing")

        # Emit event to stream and return for next step
        event = ManagerResponseEvent(output_planning=output_planning, usage=usage)
        ctx.write_event_to_stream(event)

        return event

    @step
    async def process_response(
        self, ctx: Context, ev: ManagerResponseEvent
    ) -> ManagerPlanDetailsEvent:
        """
        Parse LLM response and update agent state.

        This step:
        1. Parses the raw LLM response
        2. Updates shared state (memory, message history, planning fields)
        3. Logs the results
        4. Returns processed plan event
        """
        logger.info("⚙️ Processing manager response...")

        output_planning = ev.output_planning

        # ====================================================================
        # Step 1: Parse response
        # ====================================================================
        parsed = parse_manager_response(output_planning)

        # ====================================================================
        # Step 2: Update state
        # ====================================================================
        memory_update = parsed.get("memory", "").strip()

        # Update memory (append, not replace)
        if memory_update:
            if self.shared_state.memory:
                self.shared_state.memory += "\n" + memory_update
            else:
                self.shared_state.memory = memory_update

        # Append assistant response to message history
        self.shared_state.message_history.append(
            {"role": "assistant", "content": [{"text": output_planning}]}
        )

        # Update planning fields
        self.shared_state.plan = parsed["plan"]
        self.shared_state.current_subgoal = parsed["current_subgoal"]
        self.shared_state.finish_thought = parsed["thought"]
        self.shared_state.manager_answer = parsed["answer"]

        logger.info(f"📝 Plan: {parsed['plan'][:100]}...")
        logger.debug(f"  - Current subgoal: {parsed['current_subgoal']}")
        logger.debug(
            f"  - Manager answer: {parsed['answer'][:50] if parsed['answer'] else 'None'}"
        )

        event = ManagerPlanDetailsEvent(
            plan=parsed["plan"],
            current_subgoal=parsed["current_subgoal"],
            thought=parsed["thought"],
            manager_answer=parsed["answer"],
            memory_update=memory_update,
            success=parsed["success"],
            full_response=output_planning,
        )

        # Write event to stream for web interface
        ctx.write_event_to_stream(event)

        return event

    @step
    async def finalize(self, ctx: Context, ev: ManagerPlanDetailsEvent) -> StopEvent:
        """Return manager results to parent workflow."""
        logger.debug("✅ Manager planning complete")

        return StopEvent(
            result={
                "plan": ev.plan,
                "current_subgoal": ev.current_subgoal,
                "thought": ev.thought,
                "manager_answer": ev.manager_answer,
                "memory_update": ev.memory_update,
                "success": ev.success,
            }
        )
