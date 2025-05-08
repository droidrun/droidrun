"""
App Starter Agent - A specialized agent for starting Android apps.

This module provides a focused agent that determines the correct package name
for an app description and launches it.
"""

import logging
from typing import Any, Dict, List, Optional

from ..base.base_llm_reasoner import BaseLLMReasoner
from droidrun.tools import list_packages, start_app
from ..base.base_agent import BaseAgent, TaskContext, TaskResult

# Set up logger
logger = logging.getLogger("droidrun")

class AppStarterLLMReasoner(BaseLLMReasoner):
    """Specialized LLM reasoner for determining app package names."""
    
    async def determine_package(
        self,
        app_name: str,
        available_packages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Determine the correct package name for an app description.
        
        Args:
            app_name: Name or description of the app to start
            available_packages: List of available package info
            
        Returns:
            Dictionary containing:
            - package_name: The determined package name
            - confidence: Confidence score (0-1)
        """
        system_prompt = f"""
        You are an Android package name specialist. Your task is to determine the correct
        package name for an app based on its name or description.

        The user wants to start: {app_name}

        Available packages:
        {self._format_packages(available_packages)}

        Analyze the packages and return a JSON response with:
        - package_name: The full package name that best matches the request
        - confidence: Number between 0 and 1 indicating your confidence
        """

        user_prompt = f"What is the correct package name for {app_name}?"

        try:
            response = await self.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Error in package analysis: {e}")
            return {
                "package_name": "",
                "confidence": 0,
            }

    def _format_packages(self, packages: List[Dict[str, Any]]) -> str:
        """Format package information for the prompt."""
        if not packages:
            return "No package information available"

        return "\n".join(
            f"- {pkg.get('label', '')} ({pkg.get('package', '')})"
            for pkg in packages
        )

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured format."""
        try:
            import json
            data = json.loads(response)
            
            return {
                "package_name": data.get("package_name", ""),
                "confidence": float(data.get("confidence", 0)),
            }
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {
                "package_name": "",
                "confidence": 0,
            }

class AppStarterAgent(BaseAgent):
    """Agent specialized in starting Android apps."""
    
    def __init__(
        self,
        llm: Optional[AppStarterLLMReasoner] = None,
        device_serial: Optional[str] = None
    ):
        """Initialize the app starter agent."""
        if llm is None:
            raise ValueError("AppStarterLLMReasoner instance is required")
            
        self.reasoner = llm
        self.device_serial = device_serial

    async def execute_task(self, ctx: TaskContext) -> TaskResult:
        """Execute the task."""
        # FIXME: return type does not match TaskResult
        return await self._start_app(ctx["task"])

    async def _start_app(self, app_name: str) -> Dict[str, Any]:
        """Start the specified app.
        
        Args:
            app_name: Name or description of the app to start
            
        Returns:
            Dictionary with the result of the start attempt
        """
        try:
            # Get list of available packages
            packages_result = await list_packages(self.device_serial)
            available_packages = packages_result.get("packages", [])
            
            # Let the LLM determine the package name
            result = await self.reasoner.determine_package(
                app_name=app_name,
                available_packages=available_packages
            )
            
            # If confident enough, try to start the app
            if result["confidence"] >= 0.8 and result["package_name"]:
                start_result = await start_app(
                    package=result["package_name"],
                    serial=self.device_serial
                )
                return {
                    "success": True,
                    "package_name": result["package_name"],
                    "details": start_result
                }
            
            return {
                "success": False,
            }
                
        except Exception as e:
            logger.error(f"Error starting app: {e}")
            return {
                "success": False,
            }
