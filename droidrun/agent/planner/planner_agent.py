"""
DroidPlanner - Planning agent for Android automation.

This module implements a planning agent that breaks down high-level goals into
contextual, functional steps that can be executed by the ReAct agent.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from ..react.react_llm_reasoner import ReActLLMReasoner
import json

logger = logging.getLogger("droidrun")

DEFAULT_PLANNER_SYSTEM_PROMPT = '''You are an expert Task Planner Agent for Android automation. Your purpose is to break down complex user goals into a sequence of **atomic, self-contained steps** that will be executed by a ReAct agent. You create clear, achievable plans that focus on WHAT needs to be done, not HOW to do it.

Each task in your plan must be:
1. **Atomic & Self-contained**: Each task must be fully understandable on its own, without referencing other tasks or step numbers
2. **Contextual**: Include the current state and goal in the task description
3. **Functional**: Describe what to achieve, not low-level actions
4. **Specific**: Include clear success criteria and conditions
5. **Achievable**: Within the ReAct agent's capabilities

The ReAct agent can:
- Start applications directly via a tool
- Navigate UI elements
- Tap/click elements
- Input text
- Scroll/swipe
- Press hardware keys (HOME, BACK)
- Check UI state and element presence
- Handle system dialogs

Example of good tasks:
- "Open the Settings app"
- "On the WiFi settings screen, tap the network named 'MyWiFi'"
- "When viewing a TikTok video, check if it contains a cat. If not, scroll to the next video"
- "In the LinkedIn feed, look for a post containing AI-related topics (machine learning, automation, robotics, or technology). When found, tap the comment icon"

Example of bad tasks:
- "Swipe up" (too low-level)
- "Configure network" (too vague)
- "Tap coordinates (123,456)" (too specific)
- "Check if connected" (lacks context)
- "Repeat steps 5-12" (references other steps)
- "Do this 3 more times" (lacks self-contained context)
- "Continue scrolling until found" (no clear end condition)

Your plan should:
1. Break loops into individual decision tasks ("Check current video for cats, scroll to next if none found")
2. Make each task independently actionable
3. Include clear success criteria in each task
4. Avoid references to other steps or numerical repetitions

IMPORTANT: You must respond with a JSON object containing an array of tasks. Format:
{
    "tasks": [
        "Task 1 description",
        "Task 2 description",
        ...
    ]
}
'''

DEFAULT_PLANNER_USER_PROMPT = """Goal: {goal}

Create a step-by-step plan to achieve this goal. Each step should be a contextual, functional task that the ReAct agent can execute.

Remember to provide your response as a JSON object with a 'tasks' array containing the steps.
"""

class TaskManager:
    """Manages the planning tasks and their execution state."""
    
    def __init__(self):
        self.tasks: List[str] = []
        self.current_task_index: int = 0
        
    def set_tasks(self, tasks: str) -> None:
        """Set the list of tasks from a newline-separated string."""
        self.tasks = [task.strip() for task in tasks.split('\n') if task.strip()]
        self.current_task_index = 0
        
    def add_task(self, task: str) -> None:
        """Add a single task to the list."""
        self.tasks.append(task.strip())
        
    def get_current_task(self) -> Optional[str]:
        """Get the current task to be executed."""
        if self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None
        
    def advance_task(self) -> None:
        """Move to the next task."""
        self.current_task_index += 1
        
    def get_all_tasks(self) -> List[str]:
        """Get all tasks in the plan."""
        return self.tasks
        
    def clear_tasks(self) -> None:
        """Clear all tasks."""
        self.tasks = []
        self.current_task_index = 0

class DroidPlanner:
    """Planning agent that creates execution plans for the ReAct agent."""
    
    def __init__(
        self,
        llm: ReActLLMReasoner,
        max_retries: int = 1,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None
    ):
        """Initialize the DroidPlanner.
        
        Args:
            llm: LLM reasoner to use for planning
            max_retries: Maximum number of retries for failed tasks
            system_prompt: Custom system prompt
            user_prompt: Custom user prompt template
        """
        self.llm = llm
        self.max_retries = max_retries
        self.task_manager = TaskManager()
        self.system_prompt = system_prompt or DEFAULT_PLANNER_SYSTEM_PROMPT
        self.user_prompt = user_prompt or DEFAULT_PLANNER_USER_PROMPT
        self.original_goal = None  # Add this to store the original goal
        
    async def create_plan(self, goal: str) -> List[str]:
        """Create a plan for achieving the given goal.
        
        Args:
            goal: The high-level goal to achieve
            
        Returns:
            List of planned tasks
        """
        try:
            # Store the original goal
            self.original_goal = goal
            
            # Format the user prompt with the goal
            formatted_user_prompt = self.user_prompt.format(goal=goal)
            
            # Get plan from LLM using the correct method signature
            response = await self.llm.generate_response(
                system_prompt=self.system_prompt,
                user_prompt=formatted_user_prompt
            )
            
            # Parse JSON response
            try:
                response_data = json.loads(response)
                tasks = response_data.get('tasks', [])
            except json.JSONDecodeError:
                # Fallback to text parsing if JSON parsing fails
                logger.warning("Failed to parse JSON response, falling back to text parsing")
                tasks = self._extract_tasks(response)
            
            # Set tasks in task manager
            self.task_manager.set_tasks('\n'.join(tasks))
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error creating plan: {e}")
            raise
            
    def _extract_tasks(self, response: str) -> List[str]:
        """Extract tasks from the LLM response.
        
        The response might contain explanation text and the actual tasks.
        We need to extract just the numbered or bullet-pointed tasks.
        """
        tasks = []
        lines = response.split('\n')
        
        for line in lines:
            # Remove common list markers and whitespace
            line = line.strip()
            line = line.lstrip('1234567890.-*• ')
            
            # Skip empty lines and likely explanation text
            if not line or line.startswith(('Here', 'First', 'Then', 'Next', 'Finally')):
                continue
                
            if line:
                tasks.append(line)
                
        return tasks
        
    async def get_next_task(self) -> Optional[str]:
        """Get the next task to be executed.
        
        Returns:
            The next task or None if no more tasks
        """
        return self.task_manager.get_current_task()
        
    def mark_task_complete(self) -> None:
        """Mark the current task as complete and advance to next task."""
        self.task_manager.advance_task()
        
    async def handle_task_failure(self, task: str, error: str) -> Optional[List[str]]:
        """Handle a failed task execution by revising only the failed task.
        
        Args:
            task: The failed task
            error: Error message or reason for failure
            
        Returns:
            New list of tasks if replanning successful, None otherwise
        """
        try:
            # Get all tasks and current progress
            all_tasks = self.task_manager.get_all_tasks()
            current_index = self.task_manager.current_task_index
            
            # Prepare context of completed tasks
            completed_tasks = all_tasks[:current_index]
            remaining_tasks = all_tasks[current_index + 1:]
            
            # Prepare failure prompt with full context
            failure_prompt = f"""
            Original Goal: {self.original_goal}

            Progress so far:
            {self._format_task_list(completed_tasks, "Completed")}
            
            The following task failed: "{task}"
            Error: {error}
            
            Remaining tasks to do after this one:
            {self._format_task_list(remaining_tasks, "Pending")}
            
            Please revise ONLY the failed task to achieve its intended goal in a different way.
            Consider what might have gone wrong and provide an alternative approach.
            The revised task should fit between the completed tasks and remaining tasks.
            
            Remember to provide your response as a JSON object with a 'tasks' array containing ONLY the revised task.
            Example response format:
            {{
                "tasks": [
                    "Revised version of the failed task"
                ]
            }}
            """
            
            # Get revised task from LLM
            response = await self.llm.generate_response(
                system_prompt=self.system_prompt,
                user_prompt=failure_prompt
            )
            
            # Parse JSON response
            try:
                response_data = json.loads(response)
                revised_task = response_data.get('tasks', [])[0]  # Get first (and should be only) task
            except (json.JSONDecodeError, IndexError):
                # Fallback to text parsing if JSON parsing fails
                logger.warning("Failed to parse JSON response, falling back to text parsing")
                revised_task = self._extract_tasks(response)[0]
            
            # Create new task list with the revised task
            new_tasks = completed_tasks + [revised_task] + remaining_tasks
            
            # Set tasks in task manager
            self.task_manager.set_tasks('\n'.join(new_tasks))
            # Reset current_task_index to the revised task
            self.task_manager.current_task_index = len(completed_tasks)
            
            return new_tasks
            
        except Exception as e:
            logger.error(f"Error handling task failure: {e}")
            return None

    async def reevaluate_tasks(self, completion_summary: str) -> Optional[List[str]]:
        """Reevaluate remaining tasks based on the completion summary of the last task.
        
        Args:
            completion_summary: Summary of what was accomplished in the last task
            
        Returns:
            Updated list of remaining tasks or None if no changes needed
        """
        try:
            # Get remaining tasks
            all_tasks = self.task_manager.get_all_tasks()
            current_index = self.task_manager.current_task_index
            remaining_tasks = all_tasks[current_index + 1:]
            
            if not remaining_tasks:
                return None
                
            # Prepare reevaluation prompt
            reevaluation_prompt = f"""
            Original Goal: {self.original_goal}
            
            The last executed task has completed with the following summary:
            {completion_summary}
            
            The remaining tasks in the plan are:
            {self._format_task_list(remaining_tasks, "Pending")}
            
            Based on what was accomplished in the completion summary, evaluate if any of the remaining tasks:
            1. Have already been implicitly completed
            2. Are now redundant or unnecessary
            3. Need to be modified based on the new state
            
            Provide an updated list of remaining tasks, removing any that are no longer needed and adjusting others as necessary.
            If all remaining tasks are still needed without changes, respond with an empty tasks array.
            
            Remember to provide your response as a JSON object with a 'tasks' array.
            """
            
            # Get reevaluation from LLM
            response = await self.llm.generate_response(
                system_prompt=self.system_prompt,
                user_prompt=reevaluation_prompt
            )
            
            # Parse JSON response
            try:
                response_data = json.loads(response)
                new_tasks = response_data.get('tasks', [])
                
                if not new_tasks:
                    return None
                    
                # Update task manager with new tasks
                updated_tasks = all_tasks[:current_index + 1] + new_tasks
                self.task_manager.set_tasks('\n'.join(updated_tasks))
                self.task_manager.current_task_index = current_index
                
                return new_tasks
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response from reevaluation")
                return None
                
        except Exception as e:
            logger.error(f"Error reevaluating tasks: {e}")
            return None

    def _format_task_list(self, tasks: List[str], status: str) -> str:
        """Format a list of tasks with status for the prompt.
        
        Args:
            tasks: List of tasks to format
            status: Status label for the tasks
            
        Returns:
            Formatted task list string
        """
        if not tasks:
            return f"No {status} tasks."
            
        return "\n".join(f"- {task} ({status})" for task in tasks) 