"""
ManagerAgent - Planning and reasoning workflow.

This agent is responsible for:
- Analyzing the current state
- Creating plans and subgoals
- Tracking progress
- Deciding when tasks are complete
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from llama_index.core.base.llms.types import TextBlock

from droidrun.agent.manager.events import ManagerInternalPlanEvent, ManagerThinkingEvent
from droidrun.agent.manager.prompts import parse_manager_response
from droidrun.agent.utils import convert_messages_to_chatmessages
from droidrun.agent.utils.chat_utils import remove_empty_messages, message_copy
from droidrun.agent.utils.device_state_formatter import format_device_state
from droidrun.agent.utils.inference import acall_with_retries
from droidrun.agent.utils.tools import build_custom_tool_descriptions
from droidrun.app_cards.app_card_provider import AppCardProvider
from droidrun.app_cards.providers import (
    CompositeAppCardProvider,
    LocalAppCardProvider,
    ServerAppCardProvider,
)
from droidrun.config_manager.prompt_loader import PromptLoader

if TYPE_CHECKING:
    from droidrun.agent.droid.events import DroidAgentState
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
        tools_instance: "Tools",
        shared_state: "DroidAgentState",
        agent_config: "AgentConfig",
        custom_tools: dict = None,
        reference_trajectory_manual: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.config = agent_config.manager
        self.vision = self.config.vision
        self.tools_instance = tools_instance
        self.shared_state = shared_state
        self.custom_tools = custom_tools or {}
        self.agent_config = agent_config
        self.app_card_config = self.agent_config.app_cards
        self.reference_trajectory_manual = reference_trajectory_manual

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

    def _build_system_prompt(self, has_text_to_modify: bool) -> str:
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

        # Let Jinja2 handle all formatting and conditionals
        return PromptLoader.load_prompt(
            self.agent_config.get_manager_system_prompt_path(),
            {
                "instruction": self.shared_state.instruction,
                "device_date": self.tools_instance.get_date(),
                "app_card": self.shared_state.app_card,
                "important_notes": "",  # TODO: implement
                "error_history": error_history,
                "text_manipulation_enabled": has_text_to_modify,
                "custom_tools_descriptions": build_custom_tool_descriptions(
                    self.custom_tools
                ),
                "scripter_execution_enabled": self.agent_config.scripter.enabled,
                "scripter_max_steps": self.agent_config.scripter.max_steps,
            },
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
    async def prepare_input(self, ctx: Context, ev: StartEvent) -> ManagerThinkingEvent:
        """
        Gather context and prepare manager prompt.

        This step:
        1. Gets current device state (UI elements, screenshot)
        2. Detects text manipulation mode
        3. Builds message history entry with last action
        4. Stores context for think() step
        """
        logger.info("💬 Preparing manager input...")

        # ====================================================================
        # Step 1: Get and format device state using unified formatter
        # ====================================================================
        raw_state = self.tools_instance.get_state()
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

        # ====================================================================
        # Step 1.5: Start loading app card in background (only if package/instruction changed)
        # ====================================================================
        if self.app_card_config.enabled:
            current_package = self.shared_state.current_package_name
            current_instruction = self.shared_state.instruction

            # Check if we need to start a new loading task
            package_changed = current_package != self.shared_state._app_card_package
            instruction_changed = (
                current_instruction != self.shared_state._app_card_instruction
            )

            if package_changed or instruction_changed:
                # Cancel old task if it exists and is still running (non-blocking)
                if (
                    self.shared_state.app_card_loading_task
                    and not self.shared_state.app_card_loading_task.done()
                ):
                    self.shared_state.app_card_loading_task.cancel()

                # Start new loading task
                loading_task = asyncio.create_task(
                    self.app_card_provider.load_app_card(
                        package_name=current_package, instruction=current_instruction
                    )
                )
                self.shared_state.app_card_loading_task = loading_task
                self.shared_state._app_card_package = current_package
                self.shared_state._app_card_instruction = current_instruction
        # ====================================================================
        # Step 2: Capture screenshot if vision enabled
        # ====================================================================
        screenshot = None
        if self.vision:
            try:
                result = self.tools_instance.take_screenshot()
                if isinstance(result, tuple):
                    success, screenshot = result
                    if not success:
                        screenshot = None

                else:
                    screenshot = result
                logger.debug("📸 Screenshot captured for Manager")
            except Exception as e:
                logger.warning(f"Failed to capture screenshot: {e}")
                screenshot = None

        # ====================================================================
        # Step 3: Detect text manipulation mode
        # ====================================================================
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
        return ManagerThinkingEvent()

    @step
    async def think(
        self, ctx: Context, ev: ManagerThinkingEvent
    ) -> ManagerInternalPlanEvent:
        """
        Manager reasons and creates plan.

        This step:
        1. Builds system prompt with all context
        2. Builds messages from history with injected context
        3. Calls LLM
        4. Validates and retries if needed
        5. Parses response
        6. Updates state (memory, message history)
        """
        logger.info("🧠 Manager thinking about the plan...")

        has_text_to_modify = self.shared_state.has_text_to_modify
        screenshot = self.shared_state.screenshot

        # ====================================================================
        # Try to get app card from previous iteration's loading task
        # ====================================================================
        if self.app_card_config.enabled and self.shared_state.app_card_loading_task:
            try:
                # Wait briefly for the background task to complete (0.1s timeout)
                self.shared_state.app_card = await asyncio.wait_for(
                    self.shared_state.app_card_loading_task, timeout=0.1
                )
            except asyncio.TimeoutError:
                # Task not ready yet, use empty string
                self.shared_state.app_card = ""
            except asyncio.CancelledError:
                # Task was cancelled (app/instruction changed), use empty string
                logger.debug("App card task was cancelled")
                self.shared_state.app_card = ""
            except Exception as e:
                logger.warning(f"Error getting app card: {e}")
                self.shared_state.app_card = ""
        else:
            self.shared_state.app_card = ""

        # ====================================================================
        # Step 1: Build system prompt
        # ====================================================================
        system_prompt = self._build_system_prompt(has_text_to_modify)

        # ====================================================================
        # Step 2: Build messages with context
        # ====================================================================
        messages = self._build_messages_with_context(
            system_prompt=system_prompt, screenshot=screenshot
        )

        if self.reference_trajectory_manual:
            user_indices = [i for i, msg in enumerate(messages) if msg["role"] == "user"]
            if user_indices:
                last_user_idx = user_indices[-1]
                content = messages[last_user_idx].setdefault("content", [])
                content.insert(0, {"text": f"\n{self.reference_trajectory_manual}\n"})

        # ====================================================================
        # Step 3: Convert messages and call LLM
        # ====================================================================
        chat_messages = convert_messages_to_chatmessages(messages)

        if self.reference_trajectory_manual and chat_messages:
            chat_messages = chat_messages.copy()
            chat_messages[-1] = message_copy(chat_messages[-1])
            reference_block = TextBlock(text=f"\n{self.reference_trajectory_manual}\n")
            chat_messages[-1].blocks.insert(0, reference_block)

            if logger.isEnabledFor(logging.DEBUG):
                debug_dir = Path.cwd() / "logs"
                debug_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                log_path = debug_dir / f"planner_messages_{ts}.json"
                try:
                    with log_path.open("w", encoding="utf-8") as fh:
                        json.dump(
                            {
                                "reference_injected": True,
                                "messages": [
                                    {
                                        "role": msg.role,
                                        "content": [
                                            block.model_dump()
                                            for block in getattr(msg, "blocks", [])
                                        ],
                                    }
                                    for msg in chat_messages
                                ],
                            },
                            fh,
                            indent=2,
                        )
                    logger.debug(
                        "Saved planner message snapshot with reference trajectory to %s",
                        log_path,
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    logger.debug(
                        "Failed to write planner message snapshot: %s",
                        exc,
                    )

        try:
            response = await acall_with_retries(self.llm, chat_messages)
            output_planning = response.message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"Error calling LLM in manager: {e}") from e

        # ====================================================================
        # Step 4: Validate and retry if needed
        # ====================================================================
        output_planning = await self._validate_and_retry_llm_call(
            ctx=ctx, initial_messages=messages, initial_response=output_planning
        )

        # ====================================================================
        # Step 5: Parse response
        # ====================================================================
        parsed = parse_manager_response(output_planning)

        # ====================================================================
        # Step 6: Update state
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

        event = ManagerInternalPlanEvent(
            plan=parsed["plan"],
            current_subgoal=parsed["current_subgoal"],
            thought=parsed["thought"],
            manager_answer=parsed["answer"],
            memory_update=memory_update,
        )

        # Write event to stream for web interface
        ctx.write_event_to_stream(event)

        return event

    @step
    async def finalize(self, ctx: Context, ev: ManagerInternalPlanEvent) -> StopEvent:
        """Return manager results to parent workflow."""
        logger.debug("✅ Manager planning complete")

        return StopEvent(
            result={
                "plan": ev.plan,
                "current_subgoal": ev.current_subgoal,
                "thought": ev.thought,
                "manager_answer": ev.manager_answer,
                "memory_update": ev.memory_update,
            }
        )
