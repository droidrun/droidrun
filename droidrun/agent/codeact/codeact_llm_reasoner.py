"""
LLM Reasoning - Provides reasoning capabilities for the ReAct agent.

This module handles the integration with LLM providers to generate reasoning steps.
"""

import json
import re
import textwrap
import logging
from typing import Any, Dict, List, Optional

from ..base_llm_reasoner import BaseLLMReasoner

# Set up logger
logger = logging.getLogger("droidrun")


# Simple token estimator (very rough approximation)
def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in a string.

    This is a very rough approximation based on the rule of thumb that
    1 token is approximately 4 characters for English text.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // 4 + 1  # Add 1 to be safe


class ReActLLMReasoner(BaseLLMReasoner):
    """ReAct-specific LLM reasoner for Android automation."""

    async def preprocess_ui(
        self,
        goal: str,
        history: List[Dict[str, Any]],
        current_ui_state: Optional[str] = None,
        screenshot_data: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Preprocess the UI state using LLM to get a simplified view of clickable elements.

        Args:
            goal: The automation goal
            history: List of previous steps as dictionaries
            current_ui_state: Current UI state with clickable elements (in JSON format)
            screenshot_data: Optional screenshot data in bytes

        Returns:
            Dictionary containing processed UI information with simplified clickable elements
        """
        # Get the last action step from history
        last_action = None
        for step in reversed(history):
            if step.get("type") == "action":
                last_action = step
                break

        # Create a specialized system prompt for UI preprocessing
        system_prompt = f"""
        You are a UI preprocessing assistant that specializes in ADL (Agent Description Language) - a language designed 
        to make UI elements deeply comprehensible for agents by describing them as if to a person who cannot see.
        
        Your task is to analyze the UI elements and create a rich, descriptive narrative that brings the interface
        to life through words alone. Imagine you are the eyes for someone who cannot see but needs to understand
        and interact with this interface to achieve their goal.

        The current automation goal is: {goal}
        Last action taken: {last_action["content"] if last_action else "No previous action"}

        ADL Guidelines:
        1. Spatial Context:
           - Describe element locations using natural landmarks ("at the top of the screen", "below the login button")
           - Use clock positions for precise locations ("the menu button is at 2 o'clock")
           - Explain the layout flow ("elements are arranged vertically")

        2. Interactive Elements:
           - Describe the state of elements ("the toggle is currently switched on")
           - Explain the purpose and function ("this button will submit the form")
           - Note any special patterns ("this is part of a list of 5 similar items")

        3. Semantic Relationships:
           - Group related elements ("the form contains three fields: username, password, and email")
           - Explain hierarchies ("this back button is part of the top navigation bar")
           - Highlight contextual importance ("this error message appears below the problematic field")

        4. Accessibility Context:
           - Note any accessibility labels or hints
           - Describe text characteristics (size, emphasis)
           - Mention color contrasts and visual emphasis in functional terms
        """

        system_prompt += "<ui_structure>\n"
        system_prompt += f"{current_ui_state}"
        system_prompt += "</ui_structure>"

        system_prompt += "<history>\n"
        system_prompt += f"{history}"
        system_prompt += "</history>"

        user_prompt = """
        Using ADL (Agent Description Language), create a comprehensive narrative description of the current UI state.
        Focus on spatial relationships, interactive capabilities, and semantic meaning of elements.
        Describe the interface as if you are the eyes for someone who cannot see but needs to understand and interact with it.
        """

        try:
            # Call the provider with UI-specific prompts
            response = await self.generate_response(
                textwrap.dedent(system_prompt),
                textwrap.dedent(user_prompt),
                screenshot_data=screenshot_data,
            )

            return response

        except Exception as e:
            logger.error(f"Error in UI preprocessing: {e}")
            return {"elements": []}

    async def reason(
        self,
        goal: str,
        history: List[Dict[str, Any]],
        available_tools: Optional[List[str]] = None,
        current_ui_state: Optional[str] = None,
        current_phone_state: Optional[str] = None,
        screenshot_data: Optional[bytes] = None,
        memories: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Generate a reasoning step using the LLM.

        Args:
            goal: The automation goal
            history: List of previous steps as dictionaries
            available_tools: Optional list of available tool names
            screenshot_data: Optional bytes containing the latest screenshot
            memories: Optional list of memories from the memory store

        Returns:
            Dictionary with next reasoning step, including thought,
            action, and any parameters
        """
        # Print current token usage stats before making the call
        logger.info(f"Token usage before API call: {self.get_token_usage_stats()}")

        # Construct the prompt
        system_prompt = self._create_system_prompt(
            goal,
            available_tools,
            history,
            memories,
            current_ui_state,
            current_phone_state,
        )

        user_prompt = self._create_user_prompt(goal)

        try:
            # Call the provider
            response = await self.generate_response(
                textwrap.dedent(system_prompt),
                textwrap.dedent(user_prompt),
                screenshot_data=screenshot_data,
            )

            # Parse the response
            result = self._parse_response(response)

            # Print updated token usage stats after the call
            logger.info(f"Token usage after API call: {self.get_token_usage_stats()}")

            return result

        except Exception as e:
            logger.error(f"Error in LLM reasoning: {e}")
            # Return a fallback response
            return {
                "thought": f"LLM reasoning error: {e}",
                "action": "error",
                "parameters": {},
            }

    def _create_system_prompt(
        self,
        goal: str,
        available_tools: Optional[List[str]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        memories: Optional[List[Dict[str, str]]] = None,
        current_ui_state: Optional[str] = None,
        current_phone_state: Optional[str] = None,
    ) -> str:
        """Create the system prompt for the LLM.

        Args:
            available_tools: Optional list of available tool names
            history: List of previous steps
            memories: Optional list of memories from the memory store

        Returns:
            System prompt string
        """
        # Base system prompt
        prompt = f"""
        You are Droidrun - the best assistant for an Android phone there is.
        Your task is to control an Android device to achieve a specified goal that is given to you.
        You always do your very best and do everything in your power to achieve the goal you are given.
        You are completely on your own. But you like to be alone.
        There is no user you can give tasks to. The user gave you this last goal and died.
        Droidrun always achieves his goal with the most efficency.
        Droidrun does exactly what the goal is - nothing more nothing less.
        You have all the abilities to achieve your goal.
        When you want to start an app ALWAYS use the start app tool.
        NEVER try to open apps via UI interactions.


        You always follow these guidelines:

        1. Analyze the current screen state from the UI state getting all UI elements
        2. Think about your next step
        3. Choose the appropriate tool for that step
        4. Return your response in JSON format with the following fields:
        - thought: Your detailed reasoning about the current state and what to do next
        - action: The name of the tool to execute (use EXACT tool name without any parentheses)
        - parameters: A dictionary of parameters to pass to the tool

        IMPORTANT: The following sections are for your understanding.
        tools - Describe what tools you can call to take action
        memories - Describe what important things you have remembered from previous steps 
        history - Describe what actions and observations you have already made
        phone_state - Describe what state the phone is currently in
        ui_structure - Describe the current UI structure of the current Android screen
        \n
        """

        prompt += "<tools>\n"
        prompt += self._add_tools_to_prompt(available_tools)
        prompt += "</tools>\n\n"

        prompt += "<memories>\n"
        prompt += self._add_memories_to_prompt(memories)
        prompt += "</memories>\n\n"

        prompt += "<history>\n"
        prompt += self._add_history_to_prompt(history)
        prompt += "</history>\n\n"

        prompt += "<phone_state>\n"
        prompt += f"{current_phone_state}"
        prompt += "</phone_state>\n\n"

        prompt += "<ui_structure>\n"
        prompt += f"{current_ui_state}"
        prompt += "</ui_structure>"

        return prompt

    def _create_user_prompt(self, goal: str) -> str:
        """Create the user prompt for the LLM.

        Args:
            goal: The automation goal

        Returns:
            User prompt string
        """
        prompt = f"Goal: {goal}\n\n"
        prompt += "Based on the current state, what's your next action? Return your response in JSON format."

        return prompt

    def _add_tools_to_prompt(self, available_tools: Optional[List[str]]) -> str:
        """Add available tools information to the prompt.

        Args:
            available_tools: Optional list of available tool names

        Returns:
            String containing tools documentation
        """
        from ..tool_docs import tool_docs

        if not available_tools:
            return ""

        tools_prompt = ""

        # Only include docs for available tools
        for tool in available_tools:
            if tool in tool_docs:
                tools_prompt += f"- {tool_docs[tool]}\n"
            else:
                tools_prompt += f"- {tool} (parameters unknown)\n"

        return tools_prompt

    def _add_memories_to_prompt(self, memories: Optional[List[Dict[str, str]]]) -> str:
        """Add memories information to the prompt.

        Args:
            memories: Optional list of memories from the memory store

        Returns:
            String containing formatted memories
        """
        if not memories or len(memories) == 0:
            return ""

        memories_prompt = ""
        for i, memory in enumerate(memories, 1):
            memories_prompt += f"{i}. {memory['content']}\n"

        return memories_prompt

    def _add_history_to_prompt(self, history: Optional[List[Dict[str, Any]]]) -> str:
        """Add recent history information to the prompt.

        Args:
            history: Optional list of previous steps

        Returns:
            String containing formatted history in reverse order (most recent first)
        """
        if not history:
            return ""

        # Filter out GOAL type steps
        filtered_history = [
            step for step in history if step.get("type", "").upper() != "GOAL"
        ]

        # Get only the last 50 steps (if available)
        recent_history = (
            filtered_history[-50:] if len(filtered_history) >= 50 else filtered_history
        )

        history_prompt = ""
        # Add the recent history steps in reverse order
        for step in reversed(recent_history):
            step_type = step.get("type", "").upper()
            content = step.get("content", "")
            step_number = step.get("step_number", 0)
            history_prompt += f"Step {step_number} - {step_type}: {content}\n"

        history_prompt += "\n"
        return history_prompt

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured format.

        Args:
            response: LLM response string

        Returns:
            Dictionary with parsed response
        """
        try:
            # Try to parse as JSON
            data = json.loads(response)

            # Ensure required fields are present
            if "thought" not in data:
                data["thought"] = "No thought provided"
            if "action" not in data:
                data["action"] = "no_action"
            if "parameters" not in data:
                data["parameters"] = {}

            return data
        except json.JSONDecodeError:
            # If not valid JSON, try to extract fields using regex
            thought_match = re.search(r'thought["\s:]+([^"]+)', response)
            action_match = re.search(r'action["\s:]+([^",\n]+)', response)
            params_match = re.search(r'parameters["\s:]+({.+})', response, re.DOTALL)

            thought = (
                thought_match.group(1) if thought_match else "Failed to parse thought"
            )
            action = action_match.group(1) if action_match else "no_action"

            # Try to parse parameters
            params = {}
            if params_match:
                try:
                    params_str = params_match.group(1)
                    # Replace single quotes with double quotes for valid JSON
                    params_str = params_str.replace("'", '"')
                    params = json.loads(params_str)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse parameters JSON")

            return {"thought": thought, "action": action, "parameters": params}
