"""
DroidAgent - Main orchestrator for Android automation.

This module implements the main orchestration logic for Android automation,
managing the ReAct agent and AppStarter agent.
"""

import logging
from typing import Any, Dict, List, Optional

from .react.react_llm_reasoner import ReActLLMReasoner
from .react.react_agent import ReActAgent
from .app_starter.app_starter_agent import AppStarterAgent, AppStarterLLMReasoner

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
        base_url: Optional[str] = None
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
            app_starter=self.app_starter,
            task=None
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
            
            # Run ReAct agent with the goal
            steps, action_count = await self.react.run(goal)
            
            # Check if goal was achieved
            for step in reversed(steps):
                if step.step_type.value == "observation":
                    if "goal achieved" in step.content.lower():
                        logger.info(f"Goal achieved in {action_count} actions")
                        return True
                    break
            
            logger.info(f"Goal not achieved after {action_count} actions")
            return False
            
        except Exception as e:
            logger.error(f"Error in DroidAgent execution: {e}")
            return False
