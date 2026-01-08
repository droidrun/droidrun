"""
Droidrun - A framework for controlling Android devices through LLM agents.
"""

from importlib.metadata import version

__version__ = version("droidrun")

# Import main classes for easier access
from droidrun.agent import ResultEvent
from droidrun.agent.droid import DroidAgent
from droidrun.agent.utils.llm_picker import load_llm

# Import configuration classes
from droidrun.config_manager import (
    # Agent configs
    AgentConfig,
    AppCardConfig,
    ClickMode,
    CodeActConfig,
    CredentialsConfig,
    # Feature configs
    DeviceConfig,
    DroidrunConfig,
    ExecutorConfig,
    InteractionMode,
    LLMProfile,
    LoggingConfig,
    ManagerConfig,
    SafeExecutionConfig,
    ScripterConfig,
    TelemetryConfig,
    ToolsConfig,
    TracingConfig,
)

# Import macro functionality
from droidrun.macro import MacroPlayer, replay_macro_file, replay_macro_folder
from droidrun.tools import AdbTools, IOSTools, StealthAdbTools, Tools

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
    "StealthAdbTools",
    # Macro
    "MacroPlayer",
    "replay_macro_file",
    "replay_macro_folder",
    # Configuration
    "DroidrunConfig",
    "AgentConfig",
    "ClickMode",
    "InteractionMode",
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

try:
    from droidrun.tools import MobileRunTools

    __all__.append("MobileRunTools")
except ImportError:
    pass
