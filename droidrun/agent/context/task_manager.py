import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import copy

@dataclass
class Task:
    """
    Represents a single task with its properties.
    """
    description: str
    status: str
    agent_type: str
    # Optional fields to carry success/failure context back to the planner
    message: Optional[str] = None
    failure_reason: Optional[str] = None
    

class TaskManager:
    """
    Manages a list of tasks for an agent, each with a status and assigned specialized agent.
    """
    STATUS_PENDING = "pending"      
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    VALID_STATUSES = {
        STATUS_PENDING,
        STATUS_COMPLETED,
        STATUS_FAILED
    }
    def __init__(self):
        """Initializes an empty task list."""
        self.tasks: List[Task] = []
        self.goal_completed = False 
        self.message = None
        self.task_history = [] 
        self.file_path = os.path.join(os.path.dirname(__file__), "todo.txt")

    def get_all_tasks(self) -> List[Task]:
        return self.tasks
        
    def get_task_history(self):
        return self.task_history

    def get_current_task(self) -> Optional[Task]:
        """Return the first task with status "pending" from the task list."""
        for task in self.tasks:
            if task.status == self.STATUS_PENDING:
                return task
        return None

    def complete_task(self, task: Task, message: Optional[str] = None):
        task = copy.deepcopy(task)
        task.status = self.STATUS_COMPLETED
        task.message = message
        self.task_history.append(task)

    def fail_task(self, task: Task, failure_reason: Optional[str] = None):
        task = copy.deepcopy(task)
        task.status = self.STATUS_FAILED
        task.failure_reason = failure_reason
        self.task_history.append(task)

    def get_completed_tasks(self) -> list[dict]:
        return [task for task in self.task_history if task.status == self.STATUS_COMPLETED]

    def get_failed_tasks(self) -> list[dict]:
        return [task for task in self.task_history if task.status == self.STATUS_FAILED]
    
    def get_task_history(self) -> list[dict]:
        return self.task_history


    def save_to_file(self):
        """Saves the current task list to a Markdown file."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                for i, task in enumerate(self.tasks, 1):
                    f.write(f"Task {i}: {task.description}\n")
                    f.write(f"Status: {task.status}\n")
                    f.write(f"Agent: {task.agent_type}\n")
                    f.write("-" * 40 + "\n")
        except Exception as e:
            print(f"Error saving tasks to file: {e}")



    def set_tasks_with_agents(self, task_assignments: List[Dict[str, str]]):
        """
        Clears the current task list and sets new tasks with their assigned agents.
        
        Args:
            task_assignments: A list of dictionaries, each containing:
                            - 'task': The task description string
                            - 'agent': The agent type
                            
        Example:
            task_manager.set_tasks_with_agents([
                {'task': 'Open Gmail app', 'agent': 'AppStarterExpert'},
                {'task': 'Navigate to compose email', 'agent': 'UIExpert'}
            ])
        """
        try:
            self.tasks = []
            for i, assignment in enumerate(task_assignments):
                if not isinstance(assignment, dict) or 'task' not in assignment:
                    raise ValueError(f"Each task assignment must be a dictionary with 'task' key at index {i}.")
                
                task_description = assignment['task']
                if not isinstance(task_description, str) or not task_description.strip():
                    raise ValueError(f"Task description must be a non-empty string at index {i}.")
                
                agent_type = assignment.get('agent', 'Default')
                
                task_obj = Task(
                    description=task_description.strip(),
                    status=self.STATUS_PENDING,
                    agent_type=agent_type
                )
                
                self.tasks.append(task_obj)
            
            print(f"Tasks set with agents: {len(self.tasks)} tasks added.")
            self.save_to_file()
        except Exception as e:
            print(f"Error setting tasks with agents: {e}")

    def complete_goal(self, message: str):
        """
        Marks the goal as completed, use this whether the task completion was successful or on failure.
        This method should be called when the task is finished, regardless of the outcome.

        Args:
            message: The message to be logged.
        """
        self.goal_completed = True
        self.message = message
        print(f"Goal completed: {message}")
