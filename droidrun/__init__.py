"""
Droidrun - A framework for controlling Android devices through LLM agents.
"""

__version__ = "0.4.5"

# Import main classes for easier access
from droidrun.agent.droid import DroidAgent
from droidrun.agent.utils.llm_picker import load_llm

# Import macro functionality
from droidrun.macro import MacroPlayer, replay_macro_file, replay_macro_folder
from droidrun.tools import AdbTools, IOSTools, Tools
from droidrun.agent import ResultEvent

# Import configuration classes
from droidrun.config_manager import (
    DroidrunConfig,
    # Agent configs
    AgentConfig,
    CodeActConfig,
    ManagerConfig,
    ExecutorConfig,
    ScripterConfig,
    AppCardConfig,
    # Feature configs
    DeviceConfig,
    LoggingConfig,
    TracingConfig,
    TelemetryConfig,
    ToolsConfig,
    CredentialsConfig,
    SafeExecutionConfig,
    LLMProfile,
)

# Make main components available at package level
__all__ = [
    # Agent
    "DroidAgent",
    "load_llm",
    "ResultEvent",
    # Tools
    "Tools",
    "AdbTools",
    "IOSTools",
    # Macro
    "MacroPlayer",
    "replay_macro_file",
    "replay_macro_folder",
    # Configuration
    "DroidrunConfig",
    "AgentConfig",
    "CodeActConfig",
    "ManagerConfig",
    "ExecutorConfig",
    "ScripterConfig",
    "AppCardConfig",
    "DeviceConfig",
    "LoggingConfig",
    "TracingConfig",
    "TelemetryConfig",
    "ToolsConfig",
    "CredentialsConfig",
    "SafeExecutionConfig",
    "LLMProfile",
]
