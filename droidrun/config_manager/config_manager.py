from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import yaml

from droidrun.config_manager.path_resolver import PathResolver
from droidrun.config_manager.safe_execution import SafeExecutionConfig


# ---------- Helpers / defaults ----------
def _default_config_text() -> str:
    """Generate default config.yaml content with all settings."""
    return """# DroidRun Configuration File
# This file is auto-generated. Edit values as needed.

# === Agent Settings ===
agent:
  # Maximum number of steps per task
  max_steps: 15
  # Enable planning with reasoning mode
  reasoning: false
  # Sleep duration after each action, waits for ui state to be updated (seconds)
  after_sleep_action: 1.0
  # Wait duration for UI to stabilize (seconds)
  wait_for_stable_ui: 0.3
  # Base directory for prompt templates
  prompts_dir: config/prompts

  # CodeAct Agent Configuration
  codeact:
    # Enable vision capabilities (screenshots)
    vision: false
    # System prompt filename (located in prompts_dir/codeact/)
    system_prompt: system.jinja2
    # User prompt filename (located in prompts_dir/codeact/)
    user_prompt: user.jinja2
    # Enable safe code execution (restricts imports and builtins)
    safe_execution: false

  # Manager Agent Configuration
  manager:
    # Enable vision capabilities (screenshots)
    vision: false
    # System prompt filename (located in prompts_dir/manager/)
    system_prompt: system.jinja2

  # Executor Agent Configuration
  executor:
    # Enable vision capabilities (screenshots)
    vision: false
    # System prompt filename (located in prompts_dir/executor/)
    system_prompt: system.jinja2

  # Scripter Agent Configuration
  scripter:
    # Enable scripter execution for off-device operations
    enabled: true
    # Maximum steps per script task
    max_steps: 10
    # Execution timeout per code block (seconds)
    execution_timeout: 30.0
    # System prompt filename (located in prompts_dir/scripter/)
    system_prompt_path: system.jinja2
    # Enable safe code execution (restricts imports and builtins)
    safe_execution: false

  # App Cards Configuration
  app_cards:
    # Enable app-specific instruction cards
    enabled: true
    # Mode: local (file-based), server (HTTP API), or composite (server with local fallback)
    mode: local
    # Directory containing app card files (for local/composite modes)
    app_cards_dir: config/app_cards
    # Server URL for remote app cards (for server/composite modes)
    server_url: null
    # Server request timeout in seconds
    server_timeout: 2.0
    # Number of server retry attempts
    server_max_retries: 2

# === LLM Profiles ===
# Define LLM configurations for each agent type
llm_profiles:
  # Manager: Plans and reasons about task progress
  manager:
    provider: GoogleGenAI
    model: models/gemini-2.5-pro
    temperature: 0.2
    kwargs:
      max_tokens: 8192

  # Executor: Selects and executes atomic actions
  executor:
    provider: GoogleGenAI
    model: models/gemini-2.5-pro
    temperature: 0.1
    kwargs:
      max_tokens: 4096

  # CodeAct: Generates and executes code actions
  codeact:
    provider: GoogleGenAI
    model: models/gemini-2.5-pro
    temperature: 0.2
    kwargs:
      max_tokens: 8192

  # Text Manipulator: Edits text in input fields
  text_manipulator:
    provider: GoogleGenAI
    model: models/gemini-2.5-pro
    temperature: 0.3
    kwargs:
      max_tokens: 4096

  # App Opener: Opens apps by name/description
  app_opener:
    provider: OpenAI
    model: gpt-4o-mini
    temperature: 0.0
    base_url: null
    api_base: null
    kwargs:
      max_tokens: 512
      api_key: YOUR_API_KEY

  # Scripter: Executes Python scripts for off-device operations
  scripter:
    provider: GoogleGenAI
    model: models/gemini-2.5-flash
    temperature: 0.1
    kwargs:
      max_tokens: 4096

  # Structured Output: Extracts structured data from final answers
  structured_output:
    provider: GoogleGenAI
    model: models/gemini-2.5-flash
    temperature: 0.0
    kwargs:
      max_tokens: 2048

# === Device Settings ===
device:
  # Default device serial (null = auto-detect for Android)
  serial: null
  # Platform: android or ios
  platform: android
  # Use TCP communication instead of content provider
  use_tcp: false

# === Telemetry Settings ===
telemetry:
  # Enable anonymous telemetry
  enabled: true

# === Tracing Settings ===
tracing:
  # Enable Arize Phoenix tracing
  enabled: false

# === Logging Settings ===
logging:
  # Enable debug logging
  debug: false
  # Trajectory saving level (none, step, action)
  save_trajectory: none
  rich_text: false

# === Memory Settings ===
memory:
  # Provider: disabled, local_http, remote_http
  provider: disabled
  local_http:
    enabled: false
    base_url: http://localhost:8000
    similarity_threshold: 0.5
    timeout: 5
  remote_http:
    enabled: false
    base_url: null
    auth_token: null
    similarity_threshold: 0.6
    timeout: 10

# === Safe Execution Settings ===
# Applied when agent.codeact.safe_execution or agent.scripter.safe_execution is true
safe_execution:
  # Allow all imports (ignores allowed_modules, respects blocked_modules)
  allow_all_imports: false

  # Allowed modules (empty + allow_all_imports=false = no imports allowed)
  # Example: ['json', 'requests', 're', 'datetime', 'math', 'collections']
  allowed_modules: []

  # Blocked modules (takes precedence over allowed_modules and allow_all_imports)
  # Prevents dangerous file operations, subprocess execution, and code manipulation
  blocked_modules:
    - os
    - sys
    - subprocess
    - shutil
    - pathlib
    - pty
    - fcntl
    - resource
    - pickle
    - shelve
    - marshal
    - imp
    - importlib
    - ctypes
    - code
    - codeop
    - tempfile
    - glob
    - socket
    - socketserver
    - asyncio

  # Allow all builtins (ignores allowed_builtins, respects blocked_builtins)
  allow_all_builtins: false

  # Allowed builtins (empty + allow_all_builtins=false = use safe defaults)
  # Safe defaults include: int, str, list, dict, print, len, range, etc.
  allowed_builtins: []

  # Blocked builtins (takes precedence over allowed_builtins and allow_all_builtins)
  blocked_builtins:
    - open
    - compile
    - exec
    - eval
    - __import__
    - breakpoint
    - exit
    - quit
    - input

# === Tool Settings ===
tools:
  # Enable drag tool
  allow_drag: false

# === Credential Settings ===
credentials:
  # Enable credential manager
  enabled: false
  # Path to credentials file (resolved via PathResolver)
  file_path: credentials.yaml
"""


# Removed: _default_project_config_path() - now using PathResolver


# ---------- Config Schema ----------
@dataclass
class LLMProfile:
    """LLM profile configuration."""

    provider: str = "GoogleGenAI"
    model: str = "models/gemini-2.0-flash-exp"
    temperature: float = 0.2
    base_url: Optional[str] = None
    api_base: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_load_llm_kwargs(self) -> Dict[str, Any]:
        """Convert profile to kwargs for load_llm function."""
        result = {
            "model": self.model,
            "temperature": self.temperature,
        }
        # Add optional URL parameters
        if self.base_url:
            result["base_url"] = self.base_url
        if self.api_base:
            result["api_base"] = self.api_base
        # Merge additional kwargs
        result.update(self.kwargs)
        return result


@dataclass
class CodeActConfig:
    """CodeAct agent configuration."""

    vision: bool = False
    system_prompt: str = "system.jinja2"
    user_prompt: str = "user.jinja2"
    safe_execution: bool = False


@dataclass
class ManagerConfig:
    """Manager agent configuration."""

    vision: bool = False
    system_prompt: str = "system.jinja2"


@dataclass
class ExecutorConfig:
    """Executor agent configuration."""

    vision: bool = False
    system_prompt: str = "system.jinja2"


@dataclass
class ScripterConfig:
    """Scripter agent configuration."""

    enabled: bool = True
    max_steps: int = 10
    execution_timeout: float = 30.0
    system_prompt_path: str = "system.jinja2"
    safe_execution: bool = False


@dataclass
class AppCardConfig:
    """App card configuration."""

    enabled: bool = True
    mode: str = "local"  # local | server | composite
    app_cards_dir: str = "config/app_cards"
    server_url: Optional[str] = None
    server_timeout: float = 2.0
    server_max_retries: int = 2


@dataclass
class AgentConfig:
    """Agent-related configuration."""

    max_steps: int = 15
    reasoning: bool = False
    after_sleep_action: float = 1.0
    wait_for_stable_ui: float = 0.3
    prompts_dir: str = "config/prompts"

    codeact: CodeActConfig = field(default_factory=CodeActConfig)
    manager: ManagerConfig = field(default_factory=ManagerConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    scripter: ScripterConfig = field(default_factory=ScripterConfig)
    app_cards: AppCardConfig = field(default_factory=AppCardConfig)

    def get_codeact_system_prompt_path(self) -> str:
        """Get resolved absolute path to CodeAct system prompt."""
        path = f"{self.prompts_dir}/codeact/{self.codeact.system_prompt}"
        return str(PathResolver.resolve(path, must_exist=True))

    def get_codeact_user_prompt_path(self) -> str:
        """Get resolved absolute path to CodeAct user prompt."""
        path = f"{self.prompts_dir}/codeact/{self.codeact.user_prompt}"
        return str(PathResolver.resolve(path, must_exist=True))

    def get_manager_system_prompt_path(self) -> str:
        """Get resolved absolute path to Manager system prompt."""
        path = f"{self.prompts_dir}/manager/{self.manager.system_prompt}"
        return str(PathResolver.resolve(path, must_exist=True))

    def get_executor_system_prompt_path(self) -> str:
        """Get resolved absolute path to Executor system prompt."""
        path = f"{self.prompts_dir}/executor/{self.executor.system_prompt}"
        return str(PathResolver.resolve(path, must_exist=True))

    def get_scripter_system_prompt_path(self) -> str:
        """Get resolved absolute path to Scripter system prompt."""
        path = f"{self.prompts_dir}/scripter/{self.scripter.system_prompt_path}"
        return str(PathResolver.resolve(path, must_exist=True))


@dataclass
class DeviceConfig:
    """Device-related configuration."""

    serial: Optional[str] = None
    use_tcp: bool = False
    platform: str = "android"  # "android" or "ios"


@dataclass
class TelemetryConfig:
    """Telemetry configuration."""

    enabled: bool = True


@dataclass
class TracingConfig:
    """Tracing configuration."""

    enabled: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""

    debug: bool = False
    save_trajectory: str = "none"
    rich_text: bool = False


@dataclass
class LocalMemoryConfig:
    """Configuration for the local HTTP trajectory memory service."""

    enabled: bool = False
    base_url: str = "http://localhost:8000"
    similarity_threshold: float = 0.5
    timeout: float = 5.0


@dataclass
class RemoteMemoryConfig:
    """Configuration for remote HTTP trajectory memory service."""

    enabled: bool = False
    base_url: Optional[str] = None
    auth_token: Optional[str] = None
    similarity_threshold: float = 0.6
    timeout: float = 10.0


@dataclass
class MemoryConfig:
    """Top-level trajectory memory configuration."""

    provider: str = "disabled"
    local_http: LocalMemoryConfig = field(default_factory=LocalMemoryConfig)
    remote_http: RemoteMemoryConfig = field(default_factory=RemoteMemoryConfig)


@dataclass
class ToolsConfig:
    """Tools configuration."""

    allow_drag: bool = False


@dataclass
class CredentialsConfig:
    """Credentials configuration."""

    enabled: bool = False
    file_path: str = "credentials.yaml"


@dataclass
class DroidRunConfig:
    """Complete DroidRun configuration schema."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    llm_profiles: Dict[str, LLMProfile] = field(default_factory=dict)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)
    safe_execution: SafeExecutionConfig = field(default_factory=SafeExecutionConfig)

    def __post_init__(self):
        """Ensure default profiles exist."""
        if not self.llm_profiles:
            self.llm_profiles = self._default_profiles()

    @staticmethod
    def _default_profiles() -> Dict[str, LLMProfile]:
        """Get default agent specific LLM profiles."""
        return {
            "manager": LLMProfile(
                provider="GoogleGenAI",
                model="models/gemini-2.5-pro",
                temperature=0.2,
                kwargs={},
            ),
            "executor": LLMProfile(
                provider="GoogleGenAI",
                model="models/gemini-2.5-pro",
                temperature=0.1,
                kwargs={},
            ),
            "codeact": LLMProfile(
                provider="GoogleGenAI",
                model="models/gemini-2.5-pro",
                temperature=0.2,
                kwargs={"max_tokens": 8192},
            ),
            "text_manipulator": LLMProfile(
                provider="GoogleGenAI",
                model="models/gemini-2.5-pro",
                temperature=0.3,
                kwargs={},
            ),
            "app_opener": LLMProfile(
                provider="GoogleGenAI",
                model="models/gemini-2.5-pro",
                temperature=0.0,
                kwargs={},
            ),
            "scripter": LLMProfile(
                provider="GoogleGenAI",
                model="models/gemini-2.5-flash",
                temperature=0.1,
                kwargs={"max_tokens": 4096},
            ),
            "structured_output": LLMProfile(
                provider="GoogleGenAI",
                model="models/gemini-2.5-flash",
                temperature=0.0,
                kwargs={"max_tokens": 2048},
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = asdict(self)
        # Convert LLMProfile objects to dicts
        result["llm_profiles"] = {
            name: asdict(profile) for name, profile in self.llm_profiles.items()
        }
        # safe_execution is already converted by asdict
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DroidRunConfig":
        """Create config from dictionary."""
        # Parse LLM profiles
        llm_profiles = {}
        for name, profile_data in data.get("llm_profiles", {}).items():
            llm_profiles[name] = LLMProfile(**profile_data)

        # Parse agent config with sub-configs
        agent_data = data.get("agent", {})

        codeact_data = agent_data.get("codeact", {})
        codeact_config = (
            CodeActConfig(**codeact_data) if codeact_data else CodeActConfig()
        )

        manager_data = agent_data.get("manager", {})
        manager_config = (
            ManagerConfig(**manager_data) if manager_data else ManagerConfig()
        )

        executor_data = agent_data.get("executor", {})
        executor_config = (
            ExecutorConfig(**executor_data) if executor_data else ExecutorConfig()
        )

        script_data = agent_data.get("scripter", {})
        scripter_config = (
            ScripterConfig(**script_data) if script_data else ScripterConfig()
        )

        app_cards_data = agent_data.get("app_cards", {})
        app_cards_config = (
            AppCardConfig(**app_cards_data) if app_cards_data else AppCardConfig()
        )

        agent_config = AgentConfig(
            max_steps=agent_data.get("max_steps", 15),
            reasoning=agent_data.get("reasoning", False),
            after_sleep_action=agent_data.get("after_sleep_action", 1.0),
            wait_for_stable_ui=agent_data.get("wait_for_stable_ui", 0.3),
            prompts_dir=agent_data.get("prompts_dir", "config/prompts"),
            codeact=codeact_config,
            manager=manager_config,
            executor=executor_config,
            scripter=scripter_config,
            app_cards=app_cards_config,
        )

        # Parse memory config
        memory_data = data.get("memory", {})
        memory_config = MemoryConfig(
            provider=memory_data.get("provider", "disabled"),
            local_http=LocalMemoryConfig(**memory_data.get("local_http", {})),
            remote_http=RemoteMemoryConfig(**memory_data.get("remote_http", {})),
        )

        # Parse safe_execution config
        safe_exec_data = data.get("safe_execution", {})
        safe_execution_config = (
            SafeExecutionConfig(**safe_exec_data)
            if safe_exec_data
            else SafeExecutionConfig()
        )

        return cls(
            agent=agent_config,
            llm_profiles=llm_profiles,
            device=DeviceConfig(**data.get("device", {})),
            telemetry=TelemetryConfig(**data.get("telemetry", {})),
            tracing=TracingConfig(**data.get("tracing", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            memory=memory_config,
            tools=ToolsConfig(**data.get("tools", {})),
            credentials=CredentialsConfig(**data.get("credentials", {})),
            safe_execution=safe_execution_config,
        )


# ---------- ConfigManager ----------
class ConfigManager:
    """
    Thread-safe singleton ConfigManager with typed configuration schema.

    Usage:
        from droidrun.config_manager.config_manager import ConfigManager
        from droidrun.agent.utils.llm_picker import load_llms_from_profiles

        # Create config instance (singleton pattern)
        config = ConfigManager()

        # Access typed config objects
        print(config.agent.max_steps)

        # Load all LLMs from profiles
        llms = load_llms_from_profiles(config.llm_profiles)
        manager_llm = llms['manager']
        executor_llm = llms['executor']
        codeact_llm = llms['codeact']

        # Load specific profiles with overrides
        llms = load_llms_from_profiles(
            config.llm_profiles,
            profile_names=['manager', 'executor'],
            manager={'temperature': 0.1}
        )

        # Modify and save
        config.save()
    """

    _instance: Optional["ConfigManager"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, path: Optional[str] = None):
        # ensure singleton
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, path: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return

        self._lock = threading.RLock()

        # Resolution order:
        # 1) Explicit path arg
        # 2) DROIDRUN_CONFIG env var
        # 3) Default "config.yaml" (checks working dir, then package dir)
        if path:
            self.path = PathResolver.resolve(path)
        else:
            env = os.environ.get("DROIDRUN_CONFIG")
            if env:
                self.path = PathResolver.resolve(env)
            else:
                # Default: checks CWD first, then package dir
                self.path = PathResolver.resolve("config.yaml")

        # Initialize with default config
        self._config = DroidRunConfig()
        self.validate_fn: Optional[Callable[[DroidRunConfig], None]] = None

        self._ensure_file_exists()
        self.load_config()

        self._initialized = True

    # ---------------- Typed property access ----------------
    @property
    def config(self) -> DroidRunConfig:
        """Access the internal DroidRunConfig object."""
        with self._lock:
            return self._config

    @property
    def agent(self) -> AgentConfig:
        """Access agent configuration."""
        with self._lock:
            return self._config.agent

    @property
    def device(self) -> DeviceConfig:
        """Access device configuration."""
        with self._lock:
            return self._config.device

    @property
    def telemetry(self) -> TelemetryConfig:
        """Access telemetry configuration."""
        with self._lock:
            return self._config.telemetry

    @property
    def tracing(self) -> TracingConfig:
        """Access tracing configuration."""
        with self._lock:
            return self._config.tracing

    @property
    def logging(self) -> LoggingConfig:
        """Access logging configuration."""
        with self._lock:
            return self._config.logging

    @property
    def memory(self) -> MemoryConfig:
        """Access trajectory memory configuration."""
        with self._lock:
            return self._config.memory

    @property
    def tools(self) -> ToolsConfig:
        """Access tools configuration."""
        with self._lock:
            return self._config.tools

    @property
    def credentials(self) -> CredentialsConfig:
        """Access credentials configuration."""
        with self._lock:
            return self._config.credentials

    @property
    def llm_profiles(self) -> Dict[str, LLMProfile]:
        """Access LLM profiles."""
        with self._lock:
            return self._config.llm_profiles

    # ---------------- LLM Profile Helpers ----------------
    def get_llm_profile(self, profile_name: str) -> LLMProfile:
        """
        Get an LLM profile by name.

        Args:
            profile_name: Name of the profile (fast, mid, smart, custom, etc.)

        Returns:
            LLMProfile object

        Raises:
            KeyError: If profile_name doesn't exist
        """
        with self._lock:
            if profile_name not in self._config.llm_profiles:
                raise KeyError(
                    f"LLM profile '{profile_name}' not found. "
                    f"Available profiles: {list(self._config.llm_profiles.keys())}"
                )

            return self._config.llm_profiles[profile_name]

    # ---------------- I/O ----------------
    def _ensure_file_exists(self) -> None:
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(_default_config_text())

    def load_config(self) -> None:
        """Load YAML from file into memory. Runs validator if registered."""
        with self._lock:
            if not self.path.exists():
                # create starter file and set default config
                self._ensure_file_exists()
                self._config = DroidRunConfig()
                return

            with open(self.path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:
                    try:
                        self._config = DroidRunConfig.from_dict(data)
                    except Exception as e:
                        # If parsing fails, use defaults and log warning
                        import logging

                        logger = logging.getLogger("droidrun")
                        logger.warning(f"Failed to parse config, using defaults: {e}")
                        self._config = DroidRunConfig()
                else:
                    self._config = DroidRunConfig()
            self._run_validation()

    def save(self) -> None:
        """Persist current in-memory config to YAML file."""
        with self._lock:
            with open(self.path, "w", encoding="utf-8") as f:
                yaml.dump(
                    self._config.to_dict(), f, sort_keys=False, default_flow_style=False
                )

    def reload(self) -> None:
        """Reload config from disk (useful when edited externally or via UI)."""
        self.load_config()

    # ---------------- Validation ----------------
    def register_validator(self, fn: Callable[[DroidRunConfig], None]) -> None:
        """
        Register a validation function that takes the config object and raises
        an exception if invalid. The validator is run immediately on registration.
        """
        with self._lock:
            self.validate_fn = fn
            self._run_validation()

    def _run_validation(self) -> None:
        if self.validate_fn is None:
            return
        try:
            self.validate_fn(self._config)
        except Exception as exc:
            raise Exception(f"Validation failed: {exc}") from exc

    def as_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the config dict to avoid accidental mutation."""
        with self._lock:
            import copy

            return copy.deepcopy(self._config.to_dict())

    # Implemented for for config webiu so we can have dropdown prompt selection. but canceled webui plan.
    def list_available_prompts(self, agent_type: str) -> List[str]:
        """
        List all available prompt files for a given agent type.

        Args:
            agent_type: One of "codeact", "manager", "executor"

        Returns:
            List of prompt filenames available in the agent's prompts directory

        Example:
            >>> config.list_available_prompts("manager")
            ['system.jinja2', 'experimental.jinja2', 'minimal.jinja2']
        """
        agent_type = agent_type.lower()
        if agent_type not in ["codeact", "manager", "executor"]:
            raise ValueError(
                f"Invalid agent_type: {agent_type}. Must be one of: codeact, manager, executor"
            )

        # Resolve prompts directory
        prompts_path = f"{self.agent.prompts_dir}/{agent_type}"
        prompts_dir = PathResolver.resolve(prompts_path)

        if not prompts_dir.exists():
            return []

        # List all .md files in the directory
        return sorted([f.name for f in prompts_dir.glob("*.jinja2")])

    # useful for tests to reset singleton state
    @classmethod
    def _reset_instance_for_testing(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __repr__(self) -> str:
        return f"<ConfigManager path={self.path!s}>"
