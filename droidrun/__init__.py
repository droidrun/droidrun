"""
DroidRun - A framework for controlling Android devices through LLM agents.
"""

__version__ = "0.1.0"

# Import main classes for easier access
from droidrun.agent.react_agent import ReActAgent as Agent
from droidrun.agent.react_agent import ReActStep, ReActStepType
from droidrun.llm import OpenAILLM, AnthropicLLM, AzureOpenAILLM, GeminiLLM

# Make main components available at package level
__all__ = [
    "Agent",
    "ReActStep",
    "ReActStepType",
    "OpenAILLM",
    "AnthropicLLM",
    "AzureOpenAILLM",
    "GeminiLLM",
]