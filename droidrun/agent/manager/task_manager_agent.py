"""
Task Manager Agent - Delegates tasks to specialized agents.

This module implements a manager agent that routes tasks to the most appropriate
specialized agent based on the task's nature using LLM reasoning.
"""

import logging
from typing import Optional, Dict, Any, List, TypedDict
from ..base.base_llm_reasoner import BaseLLMReasoner
from ..base import BaseAgent, TaskContext, TaskResult

logger = logging.getLogger("manager")

DEFAULT_MANAGER_SYSTEM_PROMPT = """You are an expert Task Manager that specializes in delegating tasks to the most appropriate specialized agent. Your job is to analyze a task and determine which expert agent would be best suited to handle it.

Available Expert Agents:

1. App Starter Agent
   - Specializes in launching Android applications
   - Excellent at determining correct package names
   - Best for tasks involving starting or launching apps
   - Example tasks: "open settings app", "launch camera", "start google maps"

2. ReAct Agent
   - Specializes in UI navigation and interaction
   - Can handle complex UI workflows
   - Best for tasks involving:
     * Navigating menus and screens
     * Filling forms
     * Interacting with UI elements
     * Reading and verifying UI state
   - Example tasks: "navigate to wifi settings", "enter password in the field", "toggle airplane mode"

Analyze the task and respond with a JSON object containing:
{
    "selected_agent": "app_starter" or "react",
    "confidence": float between 0 and 1,
    "reasoning": "Brief explanation of why this agent was selected"
}
"""


class SelectAgentResult(TypedDict):
    selected_agent: str
    confidence: float
    reasoning: str


class TaskManagerAgentConfig(TypedDict):
    """Configuration for a task manager agent."""

    name: str
    agent: BaseAgent
    description: str


class TaskManagerAgent:
    """Manager agent that delegates tasks to specialized agents."""

    def _generate_system_prompt(self, agents: List[TaskManagerAgentConfig]) -> str:
        agent_list = []
        for i, agent in enumerate(agents):
            agent_list.append(f"{i+1}. {agent['name']}\n {agent['description']}")

        available_agents = "\n".join(agent_list)
        agent_names = " or ".join(f'"{agent["name"]}"' for agent in agents)

        return f"""You are an expert Task Manager that specializes in delegating tasks to the most appropriate specialized agent. Your job is to analyze a task and determine which expert agent would be best suited to handle it.

Available Expert Agents:

{available_agents}

Analyze the task and respond with a JSON object containing:
{
    "selected_agent": {agent_names},
    "confidence": float between 0 and 1,
    "reasoning": "Brief explanation of why this agent was selected"
}
"""

    def __init__(
        self,
        llm: BaseLLMReasoner,
        agents: List[TaskManagerAgentConfig],
        fallback_agent: str,
        device_serial: Optional[str] = None,
    ):
        """Initialize the task manager agent.

        Args:
            llm: LLM reasoner for agent selection
            react_agent: ReAct agent for UI navigation tasks
            app_starter_agent: AppStarter agent for app launching tasks
            device_serial: Optional device serial number
            system_prompt: Optional custom system prompt
        """
        self.llm = llm
        self.device_serial = device_serial
        self.system_prompt = self._generate_system_prompt(agents)
        self.fallback_agent = fallback_agent

    async def execute_task(self, ctx: TaskContext) -> TaskResult:
        """Execute a task by delegating to the appropriate agent.

        Args:
            task: The task description from the planner

        Returns:
            Dictionary containing:
            - success: Whether the task was successful
            - steps: List of steps taken (if using ReAct agent)
            - action_count: Number of actions taken (if using ReAct agent)
            - error: Error message if task failed
            - agent_used: Which agent was selected for the task
            - confidence: Confidence in the agent selection
        """
        try:
            # Determine the best agent for this task using LLM
            agent_selection = await self._select_agent(ctx)
            selected_agent = agent_selection.get("selected_agent")
            confidence = agent_selection.get("confidence", 0)
            reasoning = agent_selection.get("reasoning", "")

            logger.info(f"Selected agent: {selected_agent} (confidence: {confidence})")
            logger.info(f"Reasoning: {reasoning}")

            if selected_agent != self.fallback_agent and confidence >= 0.7:
                agent = await self._resolve_agent(selected_agent)
            else:
                agent = await self._resolve_agent(self.fallback_agent)

            task_result = await agent.execute_task(ctx)

            return task_result

            """if selected_agent == "app_starter" and confidence >= 0.7:
                logger.info("Delegating to AppStarter agent")
                result = await self.app_starter_agent.start_app(task)

                return {
                    "success": result.get("success", False),
                    "error": (
                        None if result.get("success", False) else "Failed to start app"
                    ),
                    "agent_used": "app_starter",
                    "confidence": confidence,
                }

            # Default to ReAct agent for all other cases
            logger.info("Delegating to ReAct agent")
            steps, action_count = await self.react_agent.run(task)

            # Check if task was successful
            task_success = False
            for step in reversed(steps):
                if step.step_type.value == "observation":
                    if "goal achieved" in step.content.lower():
                        task_success = True
                        break
  

            return {
                "success": task_success,
                "steps": steps,
                "action_count": action_count,
                "error": None if task_success else "Task execution failed",
                "agent_used": "react",
                "confidence": confidence,
            }
            """

        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_used": None,
                "confidence": 0,
            }

    async def _resolve_agent(self, agent_name: str) -> BaseAgent:
        """Resolve the agent from the list of available agents.

        Args:
            agent_name: Name of the agent to resolve

        Returns:
            The resolved agent
        """
        for agent in self.agents:
            if agent["name"] == agent_name:
                return agent["agent"]
        raise ValueError(f"Agent {agent_name} not found")

    async def _select_agent(self, ctx: TaskContext) -> SelectAgentResult:
        """Use LLM to select the best agent for the task.

        Args:
            task: Task description

        Returns:
            Dictionary containing selected agent info
        """
        try:
            # Create the user prompt
            user_prompt = f"""
            Task: "{ctx['task']}"
            
            Analyze this task and determine which specialized agent would be best suited to handle it.
            Consider the capabilities of each available agent and the nature of the task.
            Respond with a JSON object containing your selection and reasoning.
            """

            # Get LLM response
            response = await self.llm.generate_response(
                system_prompt=self.system_prompt, user_prompt=user_prompt
            )

            # Parse JSON response
            import json

            try:
                result = json.loads(response)
                return SelectAgentResult(
                    selected_agent=result.get(
                        "selected_agent", self.fallback_agent
                    ),  # Default to react if not specified
                    confidence=float(result.get("confidence", 0.5)),
                    reasoning=result.get("reasoning", "No reasoning provided"),
                )
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "Failed to parse LLM response, defaulting to ReAct agent"
                )
                return SelectAgentResult(
                    selected_agent=self.fallback_agent,
                    confidence=0.5,
                    reasoning="Failed to parse LLM response",
                )

        except Exception as e:
            logger.error(f"Error in agent selection: {e}")
            return SelectAgentResult(
                selected_agent=self.fallback_agent,
                confidence=0.5,
                reasoning=f"Error in selection: {str(e)}",
            )
