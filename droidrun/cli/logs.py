"""
Rich text logging and event handling for the DroidRun CLI.

This module provides a custom logging handler that renders a rich terminal UI
with live updates for agent execution progress, event streaming, and status display.
"""

import logging
import time
from typing import List

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner

from droidrun.agent.codeact.events import (
    TaskEndEvent,
    TaskExecutionEvent,
    TaskExecutionResultEvent,
    TaskInputEvent,
    TaskThinkingEvent,
)
from droidrun.agent.common.events import RecordUIStateEvent, ScreenshotEvent
from droidrun.agent.droid.events import (
    CodeActExecuteEvent,
    CodeActResultEvent,
    FinalizeEvent,
    TaskRunnerEvent,
)
from droidrun.agent.executor.events import (
    ExecutorActionEvent,
    ExecutorActionResultEvent,
)
from droidrun.agent.manager.events import (
    ManagerContextEvent,
    ManagerPlanDetailsEvent,
    ManagerResponseEvent,
)


class LogHandler(logging.Handler):
    """
    Custom logging handler with rich terminal UI rendering.

    Displays agent execution progress with live-updating panels showing:
    - Activity logs (scrolling log output)
    - Current goal
    - Current step/status with spinner

    Args:
        goal: User's goal/command being executed
        current_step: Initial step message (default: "Initializing...")
        rich_text: Enable rich terminal UI (default: True). If False, uses simple console output.
    """

    def __init__(
        self, goal: str, current_step: str = "Initializing...", rich_text: bool = True
    ):
        super().__init__()

        self.goal = goal
        self.current_step = current_step
        self.is_completed = False
        self.is_success = False
        self.rich_text = rich_text

        if self.rich_text:
            self.spinner = Spinner("dots")
            self.console = Console()
            self.layout = self._create_layout()
            self.logs: List[str] = []
        else:
            self.console = Console()
            self.logs: List[str] = []

        # Event deduplication: map signature -> last seen timestamp
        # This suppresses immediate duplicate events (same type + key fields) which
        # previously caused repeated rendering of success/failure messages.
        self._event_last_seen: dict[str, float] = {}
        # Window (seconds) during which an identical event is considered duplicate
        self._duplicate_window = 1.0

    def emit(self, record):
        msg = self.format(record)
        lines = msg.splitlines()

        if self.rich_text:
            for line in lines:
                self.logs.append(line)
                # Optionally, limit the log list size
                if len(self.logs) > 100:
                    self.logs.pop(0)
            self.rerender()
        else:
            # Simple console output for non-rich mode
            for line in lines:
                self.console.print(line)

    def render(self):
        if self.rich_text:
            return Live(self.layout, refresh_per_second=4, console=self.console)
        else:
            # Return a no-op context manager for non-rich mode
            from contextlib import nullcontext

            return nullcontext()

    def rerender(self):
        if self.rich_text:
            self._update_layout(
                self.layout,
                self.logs,
                self.current_step,
                self.goal,
                self.is_completed,
                self.is_success,
            )

    def update_step(self, step: str):
        self.current_step = step
        if self.rich_text:
            self.rerender()
        else:
            # Simple console output for status updates
            status_symbol = "⚡"
            if self.is_completed:
                status_symbol = "✓" if self.is_success else "✗"
            self.console.print(f"{status_symbol} {step}")

    def _create_layout(self):
        """Create a layout with logs at top and status at bottom"""
        layout = Layout()
        layout.split(
            Layout(name="logs"),
            Layout(name="goal", size=3),
            Layout(name="status", size=3),
        )
        return layout

    def _update_layout(
        self,
        layout: Layout,
        log_list: List[str],
        step_message: str,
        goal: str = None,
        completed: bool = False,
        success: bool = False,
    ):
        """Update the layout with current logs and step information"""
        import shutil

        from rich.text import Text

        # Cache terminal size to avoid frequent recalculation
        try:
            terminal_height = shutil.get_terminal_size().lines
        except:  # noqa: E722
            terminal_height = 24  # fallback

        # Reserve space for panels and borders (more conservative estimate)
        other_components_height = 10  # goal panel + status panel + borders + padding
        available_log_lines = max(8, terminal_height - other_components_height)

        # Only show recent logs, but ensure we don't flicker
        visible_logs = (
            log_list[-available_log_lines:]
            if len(log_list) > available_log_lines
            else log_list
        )

        # Ensure we always have some content to prevent panel collapse
        if not visible_logs:
            visible_logs = ["Initializing..."]

        log_content = "\n".join(visible_logs)

        layout["logs"].update(
            Panel(
                log_content,
                title=f"Activity Log ({len(log_list)} entries)",
                border_style="blue",
                title_align="left",
                padding=(0, 1),
                height=available_log_lines + 2,
            )
        )

        if goal:
            goal_text = Text(goal, style="bold")
            layout["goal"].update(
                Panel(
                    goal_text,
                    title="Goal",
                    border_style="magenta",
                    title_align="left",
                    padding=(0, 1),
                    height=3,
                )
            )

        step_display = Text()

        if completed:
            if success:
                step_display.append("✓ ", style="bold green")
                panel_title = "Completed"
                panel_style = "green"
            else:
                step_display.append("✗ ", style="bold red")
                panel_title = "Failed"
                panel_style = "red"
        else:
            step_display.append("⚡ ", style="bold yellow")
            panel_title = "Status"
            panel_style = "yellow"

        step_display.append(step_message)

        layout["status"].update(
            Panel(
                step_display,
                title=panel_title,
                border_style=panel_style,
                title_align="left",
                padding=(0, 1),
                height=3,
            )
        )

    def handle_event(self, event):
        """Handle streaming events from the agent workflow."""
        logger = logging.getLogger("droidrun")
        # here is the forced removal in case any more duplicate events are emitted in future.
        # built a signature for event
        try:
            sig_parts = [event.__class__.__name__]
            for attr in ("success", "reason", "task", "message", "post_number"):
                sig_parts.append(str(getattr(event, attr, "")))
            sig = "|".join(sig_parts)
        except Exception:
            sig = event.__class__.__name__

        now = time.time()
        last = self._event_last_seen.get(sig)
        if last and (now - last) < self._duplicate_window:  # 1 sec gap
            # if duplicate detected within that window , ignore the second one.
            # showed it in debug log
            logger.debug(f"Suppressed duplicate event: {sig}")
            return
        self._event_last_seen[sig] = now

        # Log different event types with proper names
        if isinstance(event, ScreenshotEvent):
            logger.debug("📸 Taking screenshot...")

        elif isinstance(event, RecordUIStateEvent):
            logger.debug("✏️ Recording UI state")

        # Manager events (reasoning mode - planning)
        elif isinstance(event, ManagerContextEvent):
            self.current_step = "Manager preparing context..."
            logger.info("🧠 Manager preparing context...")

        elif isinstance(event, ManagerResponseEvent):
            self.current_step = "Manager received response..."
            logger.debug("📥 Manager received LLM response")

        elif isinstance(event, ManagerPlanDetailsEvent):
            self.current_step = "Plan created"
            # Show thought (concise reasoning)
            if hasattr(event, "thought") and event.thought:
                thought_preview = (
                    event.thought[:120] + "..."
                    if len(event.thought) > 120
                    else event.thought
                )
                logger.info(f"💭 Thought: {thought_preview}")

            # Show current subgoal (what we're working on next)
            if hasattr(event, "current_subgoal") and event.current_subgoal:
                subgoal_preview = (
                    event.current_subgoal[:150] + "..."
                    if len(event.current_subgoal) > 150
                    else event.current_subgoal
                )
                logger.info(f"📋 Next step: {subgoal_preview}")

            # Show answer if provided (task complete)
            if hasattr(event, "manager_answer") and event.manager_answer:
                answer_preview = (
                    event.manager_answer[:200] + "..."
                    if len(event.manager_answer) > 200
                    else event.manager_answer
                )
                logger.info(f"💬 Answer: {answer_preview}")

            # Debug: show memory updates
            if hasattr(event, "memory_update") and event.memory_update:
                logger.debug(f"🧠 Memory: {event.memory_update[:100]}...")

        # Executor events (reasoning mode - action execution)
        elif isinstance(event, ExecutorActionEvent):
            self.current_step = "Selecting action..."
            # Show what action was chosen
            if hasattr(event, "description") and event.description:
                logger.info(f"🎯 Action: {event.description}")

            # Debug: show executor's reasoning
            if hasattr(event, "thought") and event.thought:
                thought_preview = (
                    event.thought[:120] + "..."
                    if len(event.thought) > 120
                    else event.thought
                )
                logger.debug(f"💭 Reasoning: {thought_preview}")

        elif isinstance(event, ExecutorActionResultEvent):
            # Show result with appropriate emoji
            if hasattr(event, "outcome") and hasattr(event, "summary"):
                if event.outcome:
                    self.current_step = "Action completed"
                    logger.info(f"✅ {event.summary}")
                else:
                    self.current_step = "Action failed"
                    error_msg = (
                        event.error if hasattr(event, "error") else "Unknown error"
                    )
                    logger.info(f"❌ {event.summary} ({error_msg})")

        # CodeAct events (direct mode)
        elif isinstance(event, TaskInputEvent):
            self.current_step = "Processing task input..."
            logger.info("💬 Task input received...")

        elif isinstance(event, TaskThinkingEvent):
            if hasattr(event, "thoughts") and event.thoughts:
                thoughts_preview = (
                    event.thoughts[:150] + "..."
                    if len(event.thoughts) > 150
                    else event.thoughts
                )
                logger.info(f"🧠 Thinking: {thoughts_preview}")
            if hasattr(event, "code") and event.code:
                logger.info("💻 Executing action code")
                logger.debug(f"{event.code}")

        elif isinstance(event, TaskExecutionEvent):
            self.current_step = "Executing action..."
            logger.info("⚡ Executing action...")

        elif isinstance(event, TaskExecutionResultEvent):
            if hasattr(event, "output") and event.output:
                output = str(event.output)
                if "Error" in output or "Exception" in output:
                    output_preview = (
                        output[:100] + "..." if len(output) > 100 else output
                    )
                    logger.info(f"❌ Action error: {output_preview}")
                else:
                    output_preview = (
                        output[:100] + "..." if len(output) > 100 else output
                    )
                    logger.info(f"⚡ Action result: {output_preview}")

        elif isinstance(event, TaskEndEvent):
            if hasattr(event, "success") and hasattr(event, "reason"):
                if event.success:
                    self.current_step = event.reason
                    logger.info(f"✅ Task completed: {event.reason}")
                else:
                    self.current_step = "Task failed"
                    logger.info(f"❌ Task failed: {event.reason}")

        # Droid coordination events
        elif isinstance(event, CodeActExecuteEvent):
            self.current_step = "Executing task..."
            logger.info("🔧 Starting task execution...")

        elif isinstance(event, CodeActResultEvent):
            if hasattr(event, "success") and hasattr(event, "reason"):
                if event.success:
                    self.current_step = event.reason
                    logger.info(f"✅ Task completed: {event.reason}")
                else:
                    self.current_step = "Task failed"
                    logger.info(f"❌ Task failed: {event.reason}")

        elif isinstance(event, TaskRunnerEvent):
            self.current_step = "Processing tasks..."
            logger.info("🏃 Processing task queue...")

        elif isinstance(event, FinalizeEvent):
            if hasattr(event, "success") and hasattr(event, "reason"):
                self.is_completed = True
                self.is_success = event.success
                if event.success:
                    self.current_step = f"Success: {event.reason}"
                    logger.info(f"🎉 Goal achieved: {event.reason}")
                else:
                    self.current_step = f"Failed: {event.reason}"
                    logger.info(f"❌ Goal failed: {event.reason}")

        else:
            logger.debug(f"🔄 {event.__class__.__name__}")
