"""
Trajectory utilities for DroidRun agents.

This module provides helper functions for working with agent trajectories,
including saving, loading, and analyzing them.
"""

import json
import logging
import os
import time
import uuid
from typing import Dict, List, Any
from PIL import Image
import io
from llama_index.core.workflow import Event

logger = logging.getLogger("droidrun")


def make_serializable(obj):
    """Recursively make objects JSON serializable."""
    if hasattr(obj, "__class__") and obj.__class__.__name__ == "ChatMessage":
        # Extract the text content from the ChatMessage
        if hasattr(obj, "content") and obj.content is not None:
            return {"role": obj.role.value, "content": obj.content}
        # If content is not available, try extracting from blocks
        elif hasattr(obj, "blocks") and obj.blocks:
            text_content = ""
            for block in obj.blocks:
                if hasattr(block, "text"):
                    text_content += block.text
            return {"role": obj.role.value, "content": text_content}
        else:
            return str(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        # Handle other custom objects by converting to dict
        return {
            k: make_serializable(v)
            for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }
    else:
        return obj


class Trajectory:

    def __init__(self, goal: str = None):
        """Initializes an empty trajectory class.

        Args:
            goal: The goal/prompt that this trajectory is trying to achieve
        """
        self.events: List[Event] = [] 
        self.screenshots: List[bytes] = []
        self.ui_states: List[Dict[str, Any]] = []
        self.macro: List[Event] = []
        self.goal = goal or "DroidRun automation sequence"

    def set_goal(self, goal: str) -> None:
        """Update the goal/description for this trajectory.

        Args:
            goal: The new goal/prompt description
        """
        self.goal = goal

    def create_screenshot_gif(self, output_path: str, duration: int = 1000) -> str:
        """
        Create a GIF from a list of screenshots.

        Args:
            output_path: Base path for the GIF (without extension)
            duration: Duration for each frame in milliseconds

        Returns:
            Path to the created GIF file, or None if no screenshots available
        """
        if len(self.screenshots) == 0:
            logger.info("📷 No screenshots available for GIF creation")
            return None

        images = []
        for screenshot in self.screenshots:
            img_data = screenshot
            img = Image.open(io.BytesIO(img_data))
            images.append(img)

        # Save as GIF
        gif_path = os.path.join(output_path, "trajectory.gif")
        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0
        )

        return gif_path

    def get_trajectory(self) -> List[Dict[str, Any]]:
        # Save main trajectory events
        serializable_events = []
        for event in self.events:
            event_dict = {
                "type": event.__class__.__name__,
                **{
                    k: make_serializable(v)
                    for k, v in event.__dict__.items()
                    if not k.startswith("_")
                },
            }
            serializable_events.append(event_dict)

        return serializable_events

    def save_trajectory(
        self,
        directory: str = "trajectories",
    ) -> str:
        """
        Save trajectory steps to a JSON file and create a GIF of screenshots if available.
        Also saves the macro sequence as a separate file for replay.
        Creates a dedicated folder for each trajectory containing all related files.

        Args:
            directory: Base directory to save the trajectory files

        Returns:
            Path to the trajectory folder
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        trajectory_folder = os.path.join(directory, f"{timestamp}_{unique_id}")
        os.makedirs(trajectory_folder, exist_ok=True)

        serializable_events = self.get_trajectory()

        trajectory_json_path = os.path.join(trajectory_folder, "trajectory.json")
        with open(trajectory_json_path, "w") as f:
            json.dump(serializable_events, f, indent=2)

        # Save macro sequence as a separate file for replay
        if self.macro:
            macro_data = []
            for macro_event in self.macro:
                macro_dict = {
                    "type": macro_event.__class__.__name__,
                    **{
                        k: make_serializable(v)
                        for k, v in macro_event.__dict__.items()
                        if not k.startswith("_")
                    },
                }
                macro_data.append(macro_dict)

            macro_json_path = os.path.join(trajectory_folder, "macro.json")
            with open(macro_json_path, "w") as f:
                json.dump(
                    {
                        "version": "1.0",
                        "description": self.goal,
                        "timestamp": timestamp,
                        "total_actions": len(macro_data),
                        "actions": macro_data,
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"💾 Saved macro sequence with {len(macro_data)} actions to {macro_json_path}"
            )

        gif_path = self.create_screenshot_gif(
            os.path.join(trajectory_folder, "screenshots")
        )
        if gif_path:
            logger.info(f"🎬 Saved screenshot GIF to {gif_path}")

        logger.info(f"📁 Trajectory saved to folder: {trajectory_folder}")

        if len(self.ui_states) != len(self.screenshots):
            logger.warning("UI states and screenshots are not the same length!")

        os.makedirs(os.path.join(trajectory_folder, "ui_states"), exist_ok=True)
        for idx, ui_state in enumerate(self.ui_states):
            ui_states_path = os.path.join(trajectory_folder, "ui_states", f"{idx}.json")
            with open(ui_states_path, "w", encoding="utf-8") as f:
                json.dump(ui_state, f, ensure_ascii=False, indent=2)
        return trajectory_folder

    @staticmethod
    def load_trajectory_folder(trajectory_folder: str) -> Dict[str, Any]:
        """
        Load trajectory data from a trajectory folder.

        Args:
            trajectory_folder: Path to the trajectory folder

        Returns:
            Dictionary containing trajectory data, macro data, and file paths
        """
        result = {
            "trajectory_data": None,
            "macro_data": None,
            "gif_path": None,
            "folder_path": trajectory_folder,
        }

        try:
            # Load main trajectory
            trajectory_json_path = os.path.join(trajectory_folder, "trajectory.json")
            if os.path.exists(trajectory_json_path):
                with open(trajectory_json_path, "r") as f:
                    result["trajectory_data"] = json.load(f)
                logger.info(f"📖 Loaded trajectory data from {trajectory_json_path}")

            # Load macro sequence
            macro_json_path = os.path.join(trajectory_folder, "macro.json")
            if os.path.exists(macro_json_path):
                with open(macro_json_path, "r") as f:
                    result["macro_data"] = json.load(f)
                logger.info(f"📖 Loaded macro data from {macro_json_path}")

            # Check for GIF
            gif_path = os.path.join(trajectory_folder, "screenshots.gif")
            if os.path.exists(gif_path):
                result["gif_path"] = gif_path
                logger.info(f"🎬 Found screenshot GIF at {gif_path}")

            return result

        except Exception as e:
            logger.error(f"❌ Error loading trajectory folder {trajectory_folder}: {e}")
            return result

    @staticmethod
    def load_macro_sequence(macro_file_path: str) -> Dict[str, Any]:
        """
        Load a macro sequence from a saved macro file.

        Args:
            macro_file_path: Path to the macro JSON file (can be full path or trajectory folder)

        Returns:
            Dictionary containing the macro sequence data
        """
        # Check if it's a folder path - if so, look for macro.json inside
        if os.path.isdir(macro_file_path):
            macro_file_path = os.path.join(macro_file_path, "macro.json")

        try:
            with open(macro_file_path, "r") as f:
                macro_data = json.load(f)

            logger.info(
                f"📖 Loaded macro sequence with {macro_data.get('total_actions', 0)} actions from {macro_file_path}"
            )
            return macro_data
        except FileNotFoundError:
            logger.error(f"❌ Macro file not found: {macro_file_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing macro file {macro_file_path}: {e}")
            return {}

    @staticmethod
    def get_macro_summary(macro_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of a macro sequence.

        Args:
            macro_data: The macro data dictionary

        Returns:
            Dictionary with statistics about the macro
        """
        if not macro_data or "actions" not in macro_data:
            return {"error": "Invalid macro data"}

        actions = macro_data["actions"]

        # Count action types
        action_types = {}
        for action in actions:
            action_type = action.get("action_type", "unknown")
            action_types[action_type] = action_types.get(action_type, 0) + 1

        # Calculate duration if timestamps are available
        timestamps = [
            action.get("timestamp") for action in actions if action.get("timestamp")
        ]
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0

        return {
            "version": macro_data.get("version", "unknown"),
            "description": macro_data.get("description", "No description"),
            "total_actions": len(actions),
            "action_types": action_types,
            "duration_seconds": round(duration, 2) if duration > 0 else None,
            "timestamp": macro_data.get("timestamp", "unknown"),
        }

    @staticmethod
    def print_macro_summary(macro_file_path: str) -> None:
        """
        Print a summary of a macro sequence.

        Args:
            macro_file_path: Path to the macro JSON file or trajectory folder
        """
        macro_data = Trajectory.load_macro_sequence(macro_file_path)
        if not macro_data:
            print("❌ Could not load macro data")
            return

        summary = Trajectory.get_macro_summary(macro_data)

        print("=== Macro Summary ===")
        print(f"File: {macro_file_path}")
        print(f"Version: {summary.get('version', 'unknown')}")
        print(f"Description: {summary.get('description', 'No description')}")
        print(f"Timestamp: {summary.get('timestamp', 'unknown')}")
        print(f"Total actions: {summary.get('total_actions', 0)}")
        if summary.get("duration_seconds"):
            print(f"Duration: {summary['duration_seconds']} seconds")
        print("Action breakdown:")
        for action_type, count in summary.get("action_types", {}).items():
            print(f"  - {action_type}: {count}")
        print("=====================")

    @staticmethod
    def print_trajectory_folder_summary(trajectory_folder: str) -> None:
        """
        Print a comprehensive summary of a trajectory folder.

        Args:
            trajectory_folder: Path to the trajectory folder
        """
        folder_data = Trajectory.load_trajectory_folder(trajectory_folder)

        print("=== Trajectory Folder Summary ===")
        print(f"Folder: {trajectory_folder}")
        print(
            f"Trajectory data: {'✅ Available' if folder_data['trajectory_data'] else '❌ Missing'}"
        )
        print(
            f"Macro data: {'✅ Available' if folder_data['macro_data'] else '❌ Missing'}"
        )
        print(
            f"Screenshot GIF: {'✅ Available' if folder_data['gif_path'] else '❌ Missing'}"
        )

        if folder_data["macro_data"]:
            print("\n--- Macro Summary ---")
            summary = Trajectory.get_macro_summary(folder_data["macro_data"])
            print(f"Description: {summary.get('description', 'No description')}")
            print(f"Total actions: {summary.get('total_actions', 0)}")
            if summary.get("duration_seconds"):
                print(f"Duration: {summary['duration_seconds']} seconds")
            print("Action breakdown:")
            for action_type, count in summary.get("action_types", {}).items():
                print(f"  - {action_type}: {count}")

        if folder_data["trajectory_data"]:
            print(f"\n--- Trajectory Summary ---")
            print(f"Total events: {len(folder_data['trajectory_data'])}")

        print("=================================")

    def print_trajectory_summary(self, trajectory_data: Dict[str, Any]) -> None:
        """
        Print a summary of a trajectory.

        Args:
            trajectory_data: The trajectory data dictionary
        """
        stats = self.get_trajectory_statistics(trajectory_data)

        print("=== Trajectory Summary ===")
        print(f"Goal: {trajectory_data.get('goal', 'Unknown')}")
        print(f"Success: {trajectory_data.get('success', False)}")
        print(f"Reason: {trajectory_data.get('reason', 'Unknown')}")
        print(f"Total steps: {stats['total_steps']}")
        print("Step breakdown:")
        for step_type, count in stats["step_types"].items():
            print(f"  - {step_type}: {count}")
        print(f"Planning steps: {stats['planning_steps']}")
        print(f"Execution steps: {stats['execution_steps']}")
        print(f"Successful executions: {stats['successful_executions']}")
        print(f"Failed executions: {stats['failed_executions']}")
        print("==========================")


def get_trajectory_statistics(
    trajectory_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Get statistics about a trajectory.

    Args:
        trajectory_steps: The trajectory list of steps

    Returns:
        Dictionary with statistics about the trajectory
    """

    # Count different types of steps
    step_types = {}
    for step in trajectory_steps:
        step_type = step.get("type", "unknown")
        step_types[step_type] = step_types.get(step_type, 0) + 1

    # Count planning vs execution steps
    planning_steps = sum(
        count
        for step_type, count in step_types.items()
        if step_type.startswith("planner_")
    )
    execution_steps = sum(
        count
        for step_type, count in step_types.items()
        if step_type.startswith("codeact_")
    )

    # Count successful vs failed executions
    successful_executions = sum(
        1
        for step in trajectory_steps
        if step.get("type") == "codeact_execution" and step.get("success", False)
    )
    failed_executions = sum(
        1
        for step in trajectory_steps
        if step.get("type") == "codeact_execution" and not step.get("success", True)
    )

    # Return statistics
    return {
        "total_steps": len(trajectory_steps),
        "step_types": step_types,
        "planning_steps": planning_steps,
        "execution_steps": execution_steps,
        "successful_executions": successful_executions,
        "failed_executions": failed_executions,
    }


# Example usage:
"""
# Save a trajectory with a specific goal (automatically creates folder structure)
trajectory = Trajectory(goal="Open settings and check battery level")
# ... add events and screenshots to trajectory ...
folder_path = trajectory.save_trajectory()

# Or update the goal later
trajectory.set_goal("Navigate to Settings and find device info")

# Load entire trajectory folder
folder_data = Trajectory.load_trajectory_folder(folder_path)
trajectory_events = folder_data['trajectory_data']
macro_actions = folder_data['macro_data']
gif_path = folder_data['gif_path']

# Load just the macro from folder
macro_data = Trajectory.load_macro_sequence(folder_path)

# Print summaries
Trajectory.print_trajectory_folder_summary(folder_path)
Trajectory.print_macro_summary(folder_path)

# Example folder structure created:
# trajectories/
# └── trajectory_20250108_143052/
#     ├── trajectory.json      # Full trajectory events
#     ├── macro.json          # Macro sequence with goal as description
#     └── screenshots.gif     # Screenshot animation
"""
