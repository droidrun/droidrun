"""
Trajectory utilities for DroidRun agents.

This module provides helper functions for working with agent trajectories,
including saving, loading, and analyzing them.
"""

import json
import logging
import os
import time
from typing import Dict, List, Any
from PIL import Image
import io
from llama_index.core.workflow import Event

logger = logging.getLogger("droidrun")


class Trajectory:

    def __init__(self):
        """Initializes an empty trajectory class."""
        self.events: List[Event] = []
        self.screenshots: List[bytes] = []

    def create_screenshot_gif(self, output_path: str, duration: int = 1000) -> str:
        """
        Create a GIF from a list of screenshots.

        Args:
            output_path: Base path for the GIF (without extension)
            duration: Duration for each frame in milliseconds

        Returns:
            Path to the created GIF file
        """
        if len(self.screenshots) == 0:
            return None

        images = []
        for screenshot in self.screenshots:
            img_data = screenshot
            img = Image.open(io.BytesIO(img_data))
            images.append(img)

        # Save as GIF
        gif_path = f"{output_path}.gif"
        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0
        )

        return gif_path

    def save_trajectory(
        self,
        directory: str = "trajectories",
    ) -> str:
        """
        Save trajectory steps to a JSON file and create a GIF of screenshots if available.

        Args:
            directory: Directory to save the trajectory files

        Returns:
            Path to the saved trajectory file
        """
        os.makedirs(directory, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(directory, f"trajectory_{timestamp}")

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
                result = {}
                for k, v in obj.__dict__.items():
                    if not k.startswith("_"):
                        try:
                            result[k] = make_serializable(v)
                        except (TypeError, ValueError) as e:
                            # If serialization fails, convert to string representation
                            logger.warning(f"Failed to serialize attribute {k}: {e}")
                            result[k] = str(v)
                return result
            else:
                try:
                    # Test if the object is JSON serializable
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    # If not serializable, convert to string
                    return str(obj)

        serializable_events = []
        for event in self.events:
            # Debug: Check if tokens attribute exists
            if hasattr(event, "tokens"):
                logger.debug(
                    f"Event {event.__class__.__name__} has tokens: {event.tokens}"
                )
            else:
                logger.debug(
                    f"Event {event.__class__.__name__} does NOT have tokens attribute"
                )

            # Start with the basic event structure
            event_dict = {"type": event.__class__.__name__}

            # Add all attributes from __dict__
            for k, v in event.__dict__.items():
                if not k.startswith("_"):
                    try:
                        event_dict[k] = make_serializable(v)
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Failed to serialize attribute {k}: {e}")
                        event_dict[k] = str(v)

            # Explicitly check for and add tokens attribute if it exists
            if hasattr(event, "tokens") and "tokens" not in event_dict:
                logger.debug(
                    f"Manually adding tokens attribute for {event.__class__.__name__}"
                )
                event_dict["tokens"] = make_serializable(event.tokens)

            # Debug: Check if tokens is in the serialized event
            if "tokens" in event_dict:
                logger.debug(
                    f"Serialized event contains tokens: {event_dict['tokens']}"
                )
            else:
                logger.debug(f"Serialized event does NOT contain tokens")

            serializable_events.append(event_dict)

        json_path = f"{base_path}.json"
        with open(json_path, "w") as f:
            json.dump(serializable_events, f, indent=2)

        self.create_screenshot_gif(base_path)

        return json_path

    def get_trajectory_statistics(trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about a trajectory.

        Args:
            trajectory_data: The trajectory data dictionary

        Returns:
            Dictionary with statistics about the trajectory
        """
        trajectory_steps = trajectory_data.get("trajectory_steps", [])

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
            "goal_achieved": trajectory_data.get("success", False),
        }

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
