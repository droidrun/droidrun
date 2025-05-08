"""
DroidAgent - Main orchestrator for Android automation.

This module implements the main orchestration logic for Android automation,
managing the Planner, ReAct agent and AppStarter agent.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .react.react_llm_reasoner import ReActLLMReasoner
from .react.react_agent import ReActAgent
from .app_starter.app_starter_agent import AppStarterAgent, AppStarterLLMReasoner
from .planner.planner_agent import DroidPlanner
from .manager.task_manager_agent import TaskManagerAgent, TaskManagerAgentConfig

# Set up logger
logger = logging.getLogger("droidrun")

class DroidAgent:
    """Main orchestrator for Android automation."""
    
    def __init__(
        self,
        llm_provider: str = "gemini",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        device_serial: Optional[str] = None,
        max_steps: int = 100,
        temperature: float = 0.2,
        vision: bool = False,
        base_url: Optional[str] = None,
        max_retries: int = 1
    ):
        """Initialize the DroidAgent.
        
        Args:
            llm_provider: LLM provider to use
            model_name: Model name to use
            api_key: API key for the LLM provider
            device_serial: Serial number of the Android device
            max_steps: Maximum steps for ReAct agent
            temperature: Temperature for LLM generation
            vision: Whether to enable vision capabilities
            base_url: Optional base URL for the API
            max_retries: Maximum number of retries for failed tasks
        """
        # Initialize ReAct LLM reasoner
        self.react_llm = ReActLLMReasoner(
            llm_provider=llm_provider,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            vision=vision,
            base_url=base_url
        )
        
        # Initialize AppStarter LLM reasoner
        self.app_starter_llm = AppStarterLLMReasoner(
            llm_provider=llm_provider,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            vision=vision,
            base_url=base_url
        )
        
        # Initialize AppStarter agent
        self.app_starter = AppStarterAgent(
            llm=self.app_starter_llm,
            device_serial=device_serial
        )
        
        # Initialize ReAct agent
        self.react = ReActAgent(
            llm=self.react_llm,
            device_serial=device_serial,
            max_steps=max_steps,
            task=None
        )
        
        # Initialize Planner
        self.planner = DroidPlanner(
            llm=self.react_llm,
            max_retries=max_retries
        )
        
        # Initialize Task Manager Agent
        self.task_manager = TaskManagerAgent(
            llm=self.react_llm,
            agents=[
                TaskManagerAgentConfig(
                    name="AppStarterAgent",
                    agent=self.app_starter,
                    description="Agent that starts apps"
                ),
                # FIXME: ReActAgent is not a BaseAgent. needs to be replaced with CodeActAgent
                TaskManagerAgentConfig(
                    name="ReActAgent",
                    agent=self.react,
                    description="Agent that performs actions on the screen"
                )
            ],
            fallback_agent="ReActAgent",
            device_serial=device_serial
        )
        
        self.device_serial = device_serial
        
    async def run(self, goal: str) -> bool:
        """Run the automation loop.
        
        Args:
            goal: The high-level goal to achieve
            
        Returns:
            True if goal was achieved, False otherwise
        """
        try:
            logger.info(f"Starting automation for goal: {goal}")
            
            # Create plan using the planner
            logger.info("Creating execution plan...")
            tasks = await self.planner.create_plan(goal)
            logger.info(f"Created plan with {len(tasks)} tasks:")
            for i, task in enumerate(tasks, 1):
                logger.info(f"  {i}. {task}")
            
            # Execute each task in the plan using the task manager
            while True:
                current_task = await self.planner.get_next_task()
                if not current_task:
                    logger.info("All tasks completed")
                    return True
                
                logger.info(f"Executing task: {current_task}")
                
                # Use task manager to execute the task
                result = await self.task_manager.execute_task(current_task)
                
                if result["success"]:
                    logger.info(f"Task completed successfully")
                    completion_summary = result.get("summary", "")
                    
                    # Reevaluate remaining tasks based on completion summary
                    if completion_summary:
                        logger.info("Reevaluating remaining tasks based on completion summary...")
                        remaining_tasks = await self.planner.reevaluate_tasks(completion_summary)
                        if remaining_tasks is not None:
                            logger.info("Updated remaining tasks based on completion:")
                            for i, task in enumerate(remaining_tasks, 1):
                                logger.info(f"  {i}. {task}")
                    
                    self.planner.mark_task_complete()
                else:
                    logger.warning(f"Task failed: {current_task}")
                    error_msg = result.get("error", "Unknown error")
                    new_tasks = await self.planner.handle_task_failure(current_task, error_msg)
                    
                    if not new_tasks:
                        logger.error("Could not create recovery plan")
                        return False
                    
                    logger.info("Created recovery plan with new tasks:")
                    for i, task in enumerate(new_tasks, 1):
                        logger.info(f"  {i}. {task}")
            
        except Exception as e:
            logger.error(f"Error in DroidAgent execution: {e}")
            return False
