"""
DroidRun - A framework for controlling Android devices through LLM agents.
"""

__version__ = "0.3.0"

# Import main classes for easier access
from droidrun.agent.utils.llm_picker import load_llm
from droidrun.tools import Tools, AdbTools, IOSTools
from droidrun.agent.droid import DroidAgent

# Import macro functionality
from droidrun.macro import MacroPlayer, replay_macro_file, replay_macro_folder


# Make main components available at package level
__all__ = [
    "DroidAgent",
    "load_llm",
    "Tools",
    "AdbTools",
    "IOSTools",
    "MacroPlayer",
    "replay_macro_file",
    "replay_macro_folder",
]
