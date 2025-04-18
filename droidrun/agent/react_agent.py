"""
ReAct Agent - Reasoning + Acting agent for controlling Android devices.

This module implements a ReAct agent that can control Android devices through
reasoning about the current state and taking appropriate actions.
"""

import time
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

# Import tools
from droidrun.tools import (
    DeviceManager,
    tap,
    swipe,
    input_text,
    press_key,
    start_app,
    install_app,
    uninstall_app,
    take_screenshot,
    list_packages,
    get_clickables,
    complete,
    extract,
)

# Import LLM reasoning
from .llm_reasoning import LLMReasoner

# Set up logger
logger = logging.getLogger("droidrun")

class ReActStepType(Enum):
    """Types of steps in a ReAct agent's reasoning and acting process."""
    THOUGHT = "thought"  # Internal reasoning step
    ACTION = "action"    # Taking an action
    OBSERVATION = "observation"  # Observing the result
    PLAN = "plan"        # Planning future steps
    GOAL = "goal"        # Setting or refining the goal

class ReActStep:
    """A single step in the ReAct agent's process."""
    
    def __init__(
        self, 
        step_type: ReActStepType, 
        content: str,
    ):
        """Initialize a ReAct step.
        
        Args:
            step_type: The type of step (thought, action, observation)
            content: The content of the step
        """
        self.step_type = step_type
        self.content = content
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the step to a dictionary.
        
        Returns:
            Dict representation of the step
        """
        return {
            "type": self.step_type.value,
            "content": self.content,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        """String representation of the step.
        
        Returns:
            Formatted string representation
        """
        type_str = self.step_type.value.upper()
        
        # Format based on step type
        if self.step_type == ReActStepType.THOUGHT:
            return f"🤔 THOUGHT: {self.content}"
        elif self.step_type == ReActStepType.ACTION:
            return f"🔄 ACTION: {self.content}"
        elif self.step_type == ReActStepType.OBSERVATION:
            return f"👁️ OBSERVATION: {self.content}"
        elif self.step_type == ReActStepType.PLAN:
            return f"📝 PLAN: {self.content}"
        elif self.step_type == ReActStepType.GOAL:
            return f"🎯 GOAL: {self.content}"
        
        return f"{type_str}: {self.content}"

class ReActAgent:
    """ReAct agent for Android device automation."""
    
    def __init__(
        self, 
        task: Optional[str] = None,
        llm: Any = None,
        device_serial: Optional[str] = None,
        max_steps: int = 100,
        vision: bool = False
    ):
        """Initialize the ReAct agent.
        
        Args:
            task: The automation task to perform (same as goal)
            llm: LLM instance to use for reasoning
            device_serial: Serial number of the Android device to control
            max_steps: Maximum number of steps to take
            vision: Whether to enable vision capabilities (screenshot tool)
        """
        self.device_serial = device_serial
        self.goal = task  # Store task as goal for backward compatibility
        self.max_steps = max_steps
        self.llm_instance = llm
        self.vision = vision
        
        # Initialize steps list
        self.steps: List[ReActStep] = []
        
        # Initialize screenshot storage
        self._last_screenshot: Optional[bytes] = None
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Define available tools and their functions
        self.tools: Dict[str, Callable] = {
            # UI interaction
            "tap": tap,
            "swipe": swipe,
            "input_text": input_text,
            "press_key": press_key,
            
            # App management
            "start_app": start_app,
            "install_app": install_app,
            "uninstall_app": uninstall_app,
            "list_packages": list_packages,
            
            # UI analysis
            "get_clickables": get_clickables,
            
            # Data extraction
            "extract": extract,
            
            # Goal management
            "complete": complete,
        }
        
        # Add screenshot tool only if vision is enabled
        if vision:
            self.tools["take_screenshot"] = take_screenshot
            logger.info("Vision capabilities enabled: screenshot tool available")
        else:
            logger.info("Vision capabilities disabled: screenshot tool not available")
        
        # Initialize device manager
        self.device_manager = DeviceManager()
        
        # Initialize LLM reasoner based on provided LLM or defaults
        try:
            # If an llm instance is provided, use it or adapt it
            if llm is not None:
                # Check if the llm is already an LLMReasoner instance
                if hasattr(llm, "reason") and callable(getattr(llm, "reason")):
                    # If it's already an LLMReasoner or compatible, use it directly
                    self.reasoner = llm
                    logger.info(f"Using provided LLM reasoner: provider={getattr(llm, 'llm_provider', 'unknown')}, model={getattr(llm, 'model_name', 'unknown')}")
                else:
                    # Detect the type of LLM and use appropriate initialization
                    llm_provider, model_name, api_key = self._detect_llm_type(llm)
                    
                    self.reasoner = LLMReasoner(
                        llm_provider=llm_provider,
                        model_name=model_name,
                        api_key=api_key,
                        temperature=0.2,
                        max_tokens=2000,
                        vision=self.vision
                    )
            else:
                # Use default OpenAI if no LLM provided
                self.reasoner = LLMReasoner(
                    llm_provider="openai",
                    temperature=0.2,
                    max_tokens=2000,
                    vision=self.vision
                )
            
            self.use_llm = True
        except (ImportError, ValueError) as e:
            logger.warning(f"LLM reasoning not available: {e}")
            self.reasoner = None
            self.use_llm = False
    
    def _detect_llm_type(self, llm_instance: Any) -> Tuple[str, Optional[str], Optional[str]]:
        """Detect the type of LLM instance and return appropriate configuration.
        
        Args:
            llm_instance: The LLM instance provided
            
        Returns:
            Tuple of (provider_name, model_name, api_key)
        """
        # First check if it's our own LLMReasoner
        if hasattr(llm_instance, "llm_provider") and hasattr(llm_instance, "model_name"):
            # It's likely our own LLMReasoner, use its attributes directly
            return (
                llm_instance.llm_provider,
                llm_instance.model_name,
                llm_instance.api_key if hasattr(llm_instance, "api_key") else None
            )
            
        # Default values if not a recognized type
        provider = "openai"
        model = None
        api_key = None
        
        # Check for common attributes to determine provider
        instance_type = str(type(llm_instance))
        
        if "openai" in instance_type.lower():
            provider = "openai"
            if hasattr(llm_instance, "model_name"):
                model = llm_instance.model_name
            elif hasattr(llm_instance, "model"):
                model = llm_instance.model
                
            # Try to extract API key if available
            if hasattr(llm_instance, "api_key"):
                api_key = llm_instance.api_key
        
        elif "anthropic" in instance_type.lower() or "claude" in instance_type.lower():
            provider = "anthropic"
            if hasattr(llm_instance, "model_name"):
                model = llm_instance.model_name
            elif hasattr(llm_instance, "model"):
                model = llm_instance.model
                
            # Try to extract API key if available
            if hasattr(llm_instance, "api_key"):
                api_key = llm_instance.api_key
        
        elif "gemini" in instance_type.lower():
            provider = "gemini"
            if hasattr(llm_instance, "model_name"):
                model = llm_instance.model_name
            elif hasattr(llm_instance, "model"):
                model = llm_instance.model
                
            # Try to extract API key if available
            if hasattr(llm_instance, "api_key"):
                api_key = llm_instance.api_key
        
        logger.info(f"Detected LLM type: {provider}, model: {model}")
        return provider, model, api_key
    
    async def connect(self) -> bool:
        """Connect to the specified device.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            devices = await self.device_manager.list_devices()
            
            if not self.device_serial:
                # If no device specified, use the first one available
                if not devices:
                    logger.error("No devices found")
                    return False
                
                self.device_serial = devices[0].serial
                logger.info(f"Using first available device: {self.device_serial}")
            
            # Check if specified device exists
            device_exists = False
            for device in devices:
                if device.serial == self.device_serial:
                    device_exists = True
                    break
            
            if not device_exists:
                logger.error(f"Device {self.device_serial} not found")
                return False
            
            logger.info(f"Connected to device: {self.device_serial}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to device: {e}")
            return False
    
    async def add_step(
        self, 
        step_type: ReActStepType, 
        content: str,
    ) -> ReActStep:
        """Add a step to the agent's reasoning process.
        
        Args:
            step_type: Type of step
            content: Content of the step
        
        Returns:
            The created ReActStep
        """
        # Create the step
        step = ReActStep(step_type, content)
        
        # Add to steps list
        self.steps.append(step)
        
        # Log the step
        logger.info(str(step))
        
        return step
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name with the given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
        
        Returns:
            The result of tool execution
        
        Raises:
            ValueError: If tool not found or parameter validation fails
        """
        import inspect
        
        if tool_name not in self.tools:
            # Clean up tool name by removing extra parentheses
            cleaned_tool_name = tool_name.replace("()", "")
            if cleaned_tool_name in self.tools:
                tool_name = cleaned_tool_name
            else:
                raise ValueError(f"Tool {tool_name} not found")
        
        tool_func = self.tools[tool_name]
        
        # Add serial number if needed and not provided
        sig = inspect.signature(tool_func)
        if 'serial' in sig.parameters and 'serial' not in kwargs:
            kwargs['serial'] = self.device_serial
            
        try:
            # Execute the tool and capture the result
            result = await tool_func(**kwargs)
            
            # Special handling for formatted results
            if tool_name == "list_packages" and isinstance(result, dict):
                # Format package list for better readability
                message = result.get("message", "")
                packages = result.get("packages", [])
                package_list = "\n".join([f"- {pkg.get('package', '')}" for pkg in packages])
                
                return f"{message}\n{package_list}"
            elif tool_name == "get_clickables" and isinstance(result, dict):
                # Format clickable elements for better readability
                message = result.get("message", "")
                clickable = result.get("clickable_elements", [])
                return clickable
                

            elif tool_name == "take_screenshot" and isinstance(result, tuple) and len(result) >= 2:
                # For screenshots, store the image data for the LLM and return the path
                path, image_data = result
                # Store the screenshot data for the next LLM call
                self._last_screenshot = image_data
                return f"Screenshot captured and available for analysis"
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error: {str(e)}"
    
    async def run(self) -> List[ReActStep]:
        """Run the ReAct agent to achieve the goal.
        
        Returns:
            List of steps taken during execution
        """
        if not self.goal:
            raise ValueError("No goal specified")
        
        # Connect to device
        if not await self.connect():
            await self.add_step(
                ReActStepType.OBSERVATION, 
                "Failed to connect to device"
            )
            return self.steps
        
        # Add initial goal step
        await self.add_step(ReActStepType.GOAL, self.goal)
        
        # Continue with ReAct loop
        step_count = 0
        goal_achieved = False
        
        while step_count < self.max_steps and not goal_achieved:
            # Generate next step using LLM reasoning
            if self.use_llm and self.reasoner:
                try:
                    # Convert steps to dictionaries for the LLM
                    history = [step.to_dict() for step in self.steps]
                    
                    # Get available tool names
                    available_tools = list(self.tools.keys())
                    
                    # Get LLM reasoning, passing the last screenshot if available
                    reasoning_result = await self.reasoner.reason(
                        goal=self.goal,
                        history=history,
                        available_tools=available_tools,
                        screenshot_data=self._last_screenshot
                    )
                    
                    # Clear the screenshot after using it
                    self._last_screenshot = None
                    
                    # Extract thought, action, and parameters
                    thought = reasoning_result.get("thought", "")
                    action = reasoning_result.get("action", "")
                    parameters = reasoning_result.get("parameters", {})
                    
                    # Add thought step
                    thought_step = await self.add_step(
                        ReActStepType.THOUGHT, 
                        thought,
                    )
                    
                    # Add action step
                    action_description = f"{action}({', '.join(f'{k}={v}' for k, v in parameters.items())})"
                    action_step = await self.add_step(ReActStepType.ACTION, action_description)
                    
                    # Execute the action if it's a valid tool
                    result = "No action taken"
                    if action in self.tools:
                        try:
                            # Execute the tool
                            result = await self.execute_tool(action, **parameters)
                            
                            # Check if the complete tool was called
                            if action == "complete":
                                goal_achieved = True
                                # Get token usage stats
                                stats = self.reasoner.get_token_usage_stats()
                                total_tokens = stats['total_tokens']
                                cost = (total_tokens / 1_000_000) * 0.10  # $0.10 per 1M tokens
                                
                                print("\n===== Final Token Usage and Cost =====")
                                print(f"Total Tokens Used: {total_tokens:,}")
                                print(f"Total API Calls: {stats['api_calls']}")
                                print(f"Estimated Cost: ${cost:.4f}")
                                print("===================================\n")

                                print(f"Summary: {result}")
                            
                            if isinstance(result, bytes):
                                result = f"Binary data ({len(result)} bytes)"
                            elif isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], bytes):
                                # For screenshot which returns (path, bytes)
                                result = f"Screenshot saved to {result[0]} ({len(result[1])} bytes)"
                        except Exception as e:
                            result = f"Error: {str(e)}"
                    else:
                        result = f"Invalid action: {action}"
                    
                    # Add the observation step with the result
                    await self.add_step(
                        ReActStepType.OBSERVATION,
                        str(result)
                    )
                    
                    # Check if goal is achieved (let the LLM determine this)
                    if "goal achieved" in thought.lower() or "goal complete" in thought.lower():
                        goal_achieved = True
                    
                except Exception as e:
                    logger.error(f"Error in LLM reasoning: {e}")
                    await self.add_step(
                        ReActStepType.OBSERVATION, 
                        f"Error in LLM reasoning: {e}"
                    )
            # Increment step count
            step_count += 1
        
        # Add final step if goal achieved
        if goal_achieved:
            await self.add_step(
                ReActStepType.OBSERVATION, 
                f"Goal achieved in {step_count} steps."
            )
        elif step_count >= self.max_steps:
            await self.add_step(
                ReActStepType.OBSERVATION, 
                f"Maximum steps ({self.max_steps}) reached without achieving goal."
            )
        
        return self.steps

async def run_agent(
    task: str, 
    llm: Any = None,
    device_serial: Optional[str] = None,
    llm_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    vision: bool = False,
    **kwargs
) -> List[ReActStep]:
    """Run the ReAct agent with the given goal.
    
    Args:
        task: The automation task to perform
        llm: LLM instance to use for reasoning
        device_serial: Serial number of the Android device
        llm_provider: LLM provider name (openai, azure, anthropic, or gemini)
        model_name: Name of the LLM model to use
        api_key: API key for accessing the LLM service
        vision: Whether to enable vision capabilities (screenshot tool)
        **kwargs: Additional provider-specific parameters
    
    Returns:
        List of ReAct steps taken
    """
    # If llm_provider, model_name, and api_key are provided, create an LLMReasoner
    if llm is None and llm_provider and model_name and api_key:
        from .llm_reasoning import LLMReasoner
        logger.info(f"Creating LLMReasoner with provider={llm_provider}, model={model_name}")
        llm = LLMReasoner(
            llm_provider=llm_provider,
            model_name=model_name, 
            api_key=api_key,
            temperature=0.2,
            max_tokens=2000,
            vision=vision,
            **kwargs  # Pass additional provider-specific parameters
        )
    
    agent = ReActAgent(
        task=task,
        llm=llm,
        device_serial=device_serial,
        vision=vision
    )
    
    steps = await agent.run()
    return steps