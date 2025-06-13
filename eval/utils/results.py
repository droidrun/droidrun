"""
Result handling utilities for AndroidWorld benchmarks.
"""

import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
from droidrun.agent.droid import DroidAgent

logger = logging.getLogger("android_world_bench")

class ResultManager:
    """Manages benchmark results."""
    
    def __init__(self, results_dir: str = "eval_results"):
        """Initialize the result manager.
        
        Args:
            results_dir: Directory to save results to
        """
        self.results_dir = results_dir
        self.results = []
        
        # Create results directory if it doesn't exist
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize summary
        self.summary = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "tasks": [],
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_time": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Try to load existing summary
        summary_file = os.path.join(results_dir, "summary.json")
        if os.path.exists(summary_file):
            try:
                with open(summary_file, "r") as f:
                    self.summary = json.load(f)
                logger.info(f"Loaded existing summary from {summary_file}")
            except Exception as e:
                logger.error(f"Error loading summary file: {e}")
    
    def save_task_result(self, result: Dict[str, Any]):
        """Save a task result.
        
        Args:
            result: Task result data
        """
        # Add to results list
        self.results.append(result)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = result["task_name"].replace(" ", "_")
        filename = f"{timestamp}_{task_name}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            with open(filepath, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved result for task '{task_name}' to {filepath}")
        except Exception as e:
            logger.error(f"Error saving result file: {e}")
        
        # Update summary
        self._update_summary(result)
    
    def _update_summary(self, result: Dict[str, Any]):
        """Update the summary with a new result.
        
        Args:
            result: Task result data
        """
        # Update total tasks
        self.summary["total_tasks"] += 1
        
        # Update successful tasks
        if result.get("success", False):
            self.summary["successful_tasks"] += 1
        
        # Update success rate
        if self.summary["total_tasks"] > 0:
            self.summary["success_rate"] = self.summary["successful_tasks"] / self.summary["total_tasks"]
        
        # Add simplified task result
        summary_task = {
            "task_name": result["task_name"],
            "success": result.get("success", False),
            "steps_taken": result.get("steps_taken", 0),
            "execution_time": result.get("execution_time", 0),
            "timestamp": result.get("timestamp", datetime.now().isoformat())
        }
        
        # Add trajectory statistics if available
        if "trajectory_stats" in result:
            summary_task["trajectory_stats"] = result["trajectory_stats"]
            
            # Initialize trajectory summary stats if not present
            if "trajectory_summary" not in self.summary:
                self.summary["trajectory_summary"] = {
                    "avg_total_steps": 0,
                    "avg_planning_steps": 0,
                    "avg_execution_steps": 0,
                    "tasks_with_trajectory": 0
                }
            
            # Update trajectory summary
            self.summary["trajectory_summary"]["tasks_with_trajectory"] += 1
            tasks_with_traj = self.summary["trajectory_summary"]["tasks_with_trajectory"]
            
            # Calculate running averages
            current_avg_total = self.summary["trajectory_summary"]["avg_total_steps"]
            current_avg_planning = self.summary["trajectory_summary"]["avg_planning_steps"]
            current_avg_execution = self.summary["trajectory_summary"]["avg_execution_steps"]
            
            new_total = result["trajectory_stats"]["total_steps"]
            new_planning = result["trajectory_stats"]["planning_steps"]
            new_execution = result["trajectory_stats"]["execution_steps"]
            
            # Update running averages
            self.summary["trajectory_summary"]["avg_total_steps"] = (current_avg_total * (tasks_with_traj - 1) + new_total) / tasks_with_traj
            self.summary["trajectory_summary"]["avg_planning_steps"] = (current_avg_planning * (tasks_with_traj - 1) + new_planning) / tasks_with_traj
            self.summary["trajectory_summary"]["avg_execution_steps"] = (current_avg_execution * (tasks_with_traj - 1) + new_execution) / tasks_with_traj
            
        self.summary["tasks"].append(summary_task)
        
        # Update averages
        total_steps = sum(t.get("steps_taken", 0) for t in self.summary["tasks"])
        total_time = sum(t.get("execution_time", 0) for t in self.summary["tasks"])
        
        if self.summary["total_tasks"] > 0:
            self.summary["avg_steps"] = total_steps / self.summary["total_tasks"]
            self.summary["avg_time"] = total_time / self.summary["total_tasks"]
        
        # Save updated summary
        self._save_summary()
    
    def _save_summary(self):
        """Save the summary to file."""
        summary_file = os.path.join(self.results_dir, "summary.json")
        try:
            with open(summary_file, "w") as f:
                json.dump(self.summary, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving summary file: {e}")
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        if not self.results:
            logger.info("No results to summarize")
            return
        
        print("\nBenchmark Summary:")
        print("-----------------")
        print(f"Total tasks run: {self.summary['total_tasks']}")
        print(f"Successful tasks: {self.summary['successful_tasks']}")
        print(f"Success rate: {self.summary['success_rate']:.2%}")
        print(f"Average steps per task: {self.summary['avg_steps']:.2f}")
        print(f"Average execution time: {self.summary['avg_time']:.2f} seconds")
        
        # Add trajectory statistics if available
        has_trajectory = any('trajectory_stats' in r for r in self.results)
        if has_trajectory:
            avg_total_steps = sum(r.get('trajectory_stats', {}).get('total_steps', 0) for r in self.results) / len(self.results)
            avg_planning_steps = sum(r.get('trajectory_stats', {}).get('planning_steps', 0) for r in self.results) / len(self.results)
            avg_execution_steps = sum(r.get('trajectory_stats', {}).get('execution_steps', 0) for r in self.results) / len(self.results)
            
            print("\nTrajectory Statistics:")
            print("---------------------")
            print(f"Average trajectory steps: {avg_total_steps:.2f}")
            print(f"Average planning steps: {avg_planning_steps:.2f}")
            print(f"Average execution steps: {avg_execution_steps:.2f}")
        
        print("\nTask Results:")
        print("------------")
        
        # Print table header
        header = f"{'Task Name':<50} {'Success':<10} {'Steps':<8} {'Time (s)':<10}"
        if has_trajectory:
            header += f"{'Traj Steps':<12}"
        print(header)
        
        print("-" * (80 + (12 if has_trajectory else 0)))
        
        # Print each task result
        for r in self.results:
            task_name = r["task_name"]
            success = "✓" if r.get("success", False) else "✗"
            steps = r.get("steps_taken", 0)
            time_s = f"{r.get('execution_time', 0):.2f}"
            
            row = f"{task_name:<50} {success:<10} {steps:<8} {time_s:<10}"
            if has_trajectory:
                traj_steps = r.get('trajectory_stats', {}).get('total_steps', 0)
                row += f"{traj_steps:<12}"
            
            print(row)
        
        print(f"\nResults saved to: {self.results_dir}")


def create_task_result(task_name: str, task_description: str) -> Dict[str, Any]:
    """Create a new task result object.
    
    Args:
        task_name: Name of the task
        task_description: Description of the task
        
    Returns:
        Task result object
    """
    return {
        "task_name": task_name,
        "task_description": task_description,
        "success": False,
        "agent_success": False,
        "steps_taken": 0,
        "max_steps": 0,
        "execution_time": 0,
        "logs": [],
        "timestamp": datetime.now().isoformat(),
        "error": None,
        "trajectory": [],  # Will store the agent's execution trajectory
        "trajectory_stats": {  # Will store statistics about the trajectory
            "total_steps": 0,
            "planning_steps": 0,
            "execution_steps": 0
        }
    }


def update_result_from_agent(result: Dict[str, Any], agent_result: Any, agent: DroidAgent) -> Dict[str, Any]:
    """Update a task result with information from an agent run.
    
    Args:
        result: Task result to update
        agent_result: Result from agent run
        agent: Agent instance
        
    Returns:
        Updated task result
    """
    # Extract information from agent result if available
    if agent_result is not None:
        if isinstance(agent_result, dict):
            # Update with values from result dict
            if "steps" in agent_result:
                result["steps_taken"] = agent_result["steps"]
            if "success" in agent_result:
                result["agent_success"] = agent_result["success"]
            if "logs" in agent_result:
                result["logs"] = agent_result["logs"]
            # Capture final thought if available
            if "reason" in agent_result:
                result["final_thought"] = agent_result["reason"]
            
            # Save trajectory information if available
            if "trajectory" in agent_result:
                result["trajectory"] = agent_result["trajectory"]
                
                # Calculate trajectory statistics
                plan_steps = sum(1 for step in agent_result["trajectory"] if step["type"].startswith("planner_"))
                code_steps = sum(1 for step in agent_result["trajectory"] if step["type"].startswith("codeact_"))
                
                result["trajectory_stats"] = {
                    "total_steps": len(agent_result["trajectory"]),
                    "planning_steps": plan_steps,
                    "execution_steps": code_steps
                }
                
                logger.info(f"Captured agent trajectory with {len(agent_result['trajectory'])} steps")
                logger.info(f"  - Planning steps: {plan_steps}")
                logger.info(f"  - Execution steps: {code_steps}")
        
        # Check if agent marked task as complete via tools_instance
        if hasattr(agent.tools_instance, 'finished') and agent.tools_instance.finished:
            result["complete_success"] = agent.tools_instance.success
            result["complete_reason"] = getattr(agent.tools_instance, 'reason', None)
            logger.info(f"Agent marked task as complete: success={agent.tools_instance.success}")
    
    return result


class ProgressTracker:
    """Tracks progress of benchmark runs."""
    
    def __init__(self, progress_file: str):
        """Initialize the progress tracker.
        
        Args:
            progress_file: Path to progress file
        """
        self.progress_file = progress_file
        self.completed_tasks = 0
        
        # Ensure directory exists if progress_file includes a directory path
        dirname = os.path.dirname(progress_file)
        if dirname:  # Only create directories if there's actually a directory path
            os.makedirs(dirname, exist_ok=True)
        
        # Try to load existing progress
        self._load_progress()
    
    def _load_progress(self):
        """Load progress from file."""
        if not os.path.exists(self.progress_file):
            logger.info(f"Progress file not found: {self.progress_file}")
            return
        
        try:
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
                self.completed_tasks = progress_data.get("completed_tasks", 0)
                last_updated = progress_data.get("last_updated", "unknown")
                logger.info(f"Loaded progress: {self.completed_tasks} tasks completed (last updated: {last_updated})")
        except Exception as e:
            logger.error(f"Error loading progress file: {e}")
    
    def update_progress(self, completed_tasks: int):
        """Update progress with the number of completed tasks.
        
        Args:
            completed_tasks: Number of completed tasks
        """
        self.completed_tasks = completed_tasks
        
        try:
            progress_data = {
                "completed_tasks": self.completed_tasks,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=4)
            logger.info(f"Updated progress: {self.completed_tasks} tasks completed")
        except Exception as e:
            logger.error(f"Error updating progress file: {e}")
    
    def get_completed_tasks(self) -> int:
        """Get the number of completed tasks.
        
        Returns:
            Number of completed tasks
        """
        return self.completed_tasks 