---
title: 'Configuration'
description: 'Complete DroidAgent configuration guide - all parameters, minimal examples'
---

## Quick Start

```python
from droidrun import DroidAgent, DroidrunConfig

# Minimal (uses defaults)
agent = DroidAgent(goal="Open settings")
result = await agent.run()

# Load from config.yaml
config = DroidrunConfig.from_yaml("config.yaml")
agent = DroidAgent(goal="Open settings", config=config)
result = await agent.run()
```

---

## DroidAgent Parameters

### Required

```python
DroidAgent(
    goal="Your task",  # REQUIRED: Task description
)
```

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `DroidrunConfig \| None` | `None` | Full config object (loads LLMs from profiles if `llms` not provided) |
| `llms` | `dict[str, LLM] \| LLM \| None` | `None` | LLM(s) - dict for per-agent, single LLM for all, or None to load from config |
| `tools` | `Tools \| None` | `None` | Tools instance (AdbTools/IOSTools) |
| `custom_tools` | `dict \| None` | `None` | Custom tool definitions |
| `credentials` | `dict \| None` | `None` | Dict of credential secrets |
| `variables` | `dict \| None` | `None` | Custom variables accessible during execution |
| `output_model` | `Type[BaseModel] \| None` | `None` | Pydantic model for structured output extraction |
| `prompts` | `dict[str, str] \| None` | `None` | Custom Jinja2 prompt templates (NOT file paths) |
| `timeout` | `int` | `1000` | Workflow timeout in seconds |

---

## Configuration Classes

### AgentConfig

```python
from droidrun import AgentConfig, CodeActConfig, ManagerConfig, ExecutorConfig, ScripterConfig, AppCardConfig

AgentConfig(
    # Core settings
    max_steps=15,                    # Max execution steps
    reasoning=True,                 # Enable Manager/Executor workflow
    after_sleep_action=1.0,          # Wait after actions (seconds)
    wait_for_stable_ui=0.3,          # Wait for UI to stabilize (seconds)
    prompts_dir="config/prompts",    # Prompt templates directory

    # Sub-configs
    codeact=CodeActConfig(...),
    manager=ManagerConfig(...),
    executor=ExecutorConfig(...),
    scripter=ScripterConfig(...),
    app_cards=AppCardConfig(...),
)
```

**CodeActConfig**
```python
CodeActConfig(
    vision=True,                    # Enable screenshots
    system_prompt="system.jinja2",   # Filename in prompts_dir/codeact/
    user_prompt="user.jinja2",       # Filename in prompts_dir/codeact/
    safe_execution=True,            # Restrict imports/builtins
)
```

**ManagerConfig**
```python
ManagerConfig(
    vision=True,                    # Enable screenshots
    system_prompt="system.jinja2",   # Filename in prompts_dir/manager/
)
```

**ExecutorConfig**
```python
ExecutorConfig(
    vision=True,                    # Enable screenshots
    system_prompt="system.jinja2",   # Filename in prompts_dir/executor/
)
```

**ScripterConfig**
```python
ScripterConfig(
    enabled=True,                    # Enable off-device Python execution
    max_steps=10,                    # Max scripter steps
    execution_timeout=30.0,          # Code block timeout (seconds)
    system_prompt_path="system.jinja2",  # Filename in prompts_dir/scripter/
    safe_execution=False,            # Restrict imports/builtins
)
```

**AppCardConfig**
```python
AppCardConfig(
    enabled=True,                    # Enable app-specific instructions
    mode="local",                    # "local" | "server" | "composite"
    app_cards_dir="config/app_cards",  # Directory for app card files
    server_url=None,                 # Server URL (for server/composite modes)
    server_timeout=2.0,              # Server request timeout (seconds)
    server_max_retries=2,            # Server retry attempts
)
```

---

### DeviceConfig

```python
from droidrun import DeviceConfig

DeviceConfig(
    serial=None,          # Device serial/IP (None = auto-detect)
    platform="android",   # "android" or "ios"
    use_tcp=False,        # TCP vs content provider communication
)
```

---

### LoggingConfig

```python
from droidrun import LoggingConfig

LoggingConfig(
    debug=True,                   # Enable debug logs
    save_trajectory="none",        # "none" | "step" | "action"
    trajectory_path="trajectories", # Directory for trajectory files
    trajectory_gifs=False,         # Save trajectory as animated GIFs
    rich_text=False,               # Rich text formatting in logs
)
```

---

### TracingConfig

```python
from droidrun import TracingConfig

TracingConfig(
    enabled=True,                      # Enable tracing
    provider="phoenix",                 # "phoenix" or "langfuse"

    # Langfuse settings (only used if provider="langfuse")
    langfuse_secret_key="",             # LANGFUSE_SECRET_KEY env var
    langfuse_public_key="",             # LANGFUSE_PUBLIC_KEY env var
    langfuse_host="",                   # LANGFUSE_HOST env var (e.g., "https://cloud.langfuse.com")
    langfuse_user_id="anonymous",       # User ID for Langfuse tracing
    langfuse_session_id="",             # Empty = auto-generate UUID; custom value to persist across runs
)
```

---

### TelemetryConfig

```python
from droidrun import TelemetryConfig

TelemetryConfig(
    enabled=True,  # Enable anonymous telemetry
)
```

---

### ToolsConfig

```python
from droidrun import ToolsConfig

ToolsConfig(
    allow_drag=True,  # Enable drag tool
)
```

---

### CredentialsConfig

```python
from droidrun import CredentialsConfig

CredentialsConfig(
    enabled=True,                   # Enable credential manager
    file_path="credentials.yaml",    # Path to credentials file
)
```

---

### SafeExecutionConfig

```python
from droidrun import SafeExecutionConfig

SafeExecutionConfig(
    # Imports
    allow_all_imports=False,         # Allow all imports (ignores allowed_modules)
    allowed_modules=[],              # Allowed module names (e.g., ["json", "requests"])
    blocked_modules=[                # Blocked modules (takes precedence)
        "os", "sys", "subprocess", "shutil", "pathlib", "pty", "fcntl",
        "resource", "pickle", "shelve", "marshal", "imp", "importlib",
        "ctypes", "code", "codeop", "tempfile", "glob", "socket",
        "socketserver", "asyncio"
    ],

    # Builtins
    allow_all_builtins=False,        # Allow all builtins (ignores allowed_builtins)
    allowed_builtins=[],             # Allowed builtin names (empty = safe defaults)
    blocked_builtins=[               # Blocked builtins (takes precedence)
        "open", "compile", "exec", "eval", "__import__",
        "breakpoint", "exit", "quit", "input"
    ],
)
```

---

## LLM Configuration

### Single LLM (All Agents)

```python
from llama_index.llms.gemini import Gemini

llm = Gemini(model="models/gemini-2.5-pro", temperature=0.2)
agent = DroidAgent(goal="...", llms=llm)
```

### Per-Agent LLMs

```python
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini

agent = DroidAgent(
    goal="...",
    llms={
        "manager": OpenAI(model="gpt-4o"),                    # Planning
        "executor": Gemini(model="models/gemini-2.5-flash"),  # Action selection
        "codeact": Gemini(model="models/gemini-2.5-pro"),     # Code generation
        "text_manipulator": Gemini(model="models/gemini-2.5-flash"),  # Text input
        "app_opener": OpenAI(model="gpt-4o-mini"),            # App launching
        "scripter": Gemini(model="models/gemini-2.5-flash"),  # Off-device scripts
        "structured_output": Gemini(model="models/gemini-2.5-flash"),  # Output extraction
    }
)
```

**LLM Keys:**
- `manager` - Planning (reasoning mode only)
- `executor` - Action selection (reasoning mode only)
- `codeact` - Code generation (direct mode)
- `scripter` - Off-device Python execution
- `text_manipulator` - Text input helper
- `app_opener` - App launching helper
- `structured_output` - Final output extraction

---

## Custom Tools

```python
def my_tool(param: str, **kwargs) -> str:
    """Tool description."""
    return f"Result: {param}"

agent = DroidAgent(
    goal="...",
    custom_tools={
        "my_tool": {
            "arguments": ["param"],
            "description": "Tool description with usage example",
            "function": my_tool
        }
    }
)
```

---

## Credentials

### Dict Format (Recommended)
```python
agent = DroidAgent(
    goal="...",
    credentials={
        "USERNAME": "alice@example.com",
        "PASSWORD": "secret123"
    }
)
# Agent can call get_username() and get_password()
```

### Config Format
```python
from droidrun import DroidrunConfig, CredentialsConfig

config = DroidrunConfig(
    credentials=CredentialsConfig(
        enabled=True,
        file_path="config/credentials.yaml"
    )
)

agent = DroidAgent(
    goal="...",
    config=config
)
```

---

## Custom Variables

```python
agent = DroidAgent(
    goal="...",
    variables={
        "api_url": "https://api.example.com",
        "user_id": "12345",
        "custom_data": {"key": "value"}
    }
)
# Access in shared_state.custom_variables
```

---

## Structured Output

```python
from pydantic import BaseModel

class FlightInfo(BaseModel):
    airline: str
    flight_number: str
    confirmation_code: str

agent = DroidAgent(
    goal="Book a flight and extract details",
    output_model=FlightInfo
)
result = await agent.run()
print(result.structured_output.airline)  # Typed output
```

---

## Custom Prompts

```python
custom_prompt = """
You are an expert mobile agent.
Goal: {{ instruction }}
Be precise and efficient.
"""

agent = DroidAgent(
    goal="...",
    prompts={
        "codeact_system": custom_prompt,
        "codeact_user": "...",
        "manager_system": "...",
        "executor_system": "...",
        "scripter_system": "..."
    }
)
```

**Template Variables:**
- `{{ instruction }}` - User's goal
- `{{ device_date }}` - Device date/time
- `{{ app_card }}` - App-specific instructions
- `{{ state }}` - Device state
- `{{ history }}` - Action history

---

## Complete Example

```python
from droidrun import (
    DroidAgent, DroidrunConfig,
    AgentConfig, CodeActConfig, DeviceConfig, LoggingConfig, TracingConfig
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from pydantic import BaseModel

# Structured output
class Output(BaseModel):
    name: str
    value: int

# Custom tool
def send_email(to: str, subject: str, **kwargs) -> str:
    """Send email."""
    return f"Sent to {to}"

# Build configuration
config = DroidrunConfig(
    agent=AgentConfig(
        max_steps=30,
        reasoning=True,
        after_sleep_action=1.5,
        codeact=CodeActConfig(vision=True, safe_execution=True)
    ),
    device=DeviceConfig(
        serial="emulator-5554",
        platform="android",
        use_tcp=False
    ),
    logging=LoggingConfig(
        debug=True,
        save_trajectory="action",
        trajectory_gifs=True
    ),
    tracing=TracingConfig(enabled=True),
)

agent = DroidAgent(
    goal="Complex task",
    config=config,

    # LLMs
    llms={
        "manager": OpenAI(model="gpt-4o"),                    # Planning
        "executor": Gemini(model="models/gemini-2.5-flash"),  # Action selection
        "codeact": Gemini(model="models/gemini-2.5-pro"),     # Code generation
        "text_manipulator": Gemini(model="models/gemini-2.5-flash"),  # Text input
        "app_opener": OpenAI(model="gpt-4o-mini"),            # App launching
        "scripter": Gemini(model="models/gemini-2.5-flash"),  # Off-device scripts
        "structured_output": Gemini(model="models/gemini-2.5-flash"),  # Output extraction
    },

    # Custom tools
    custom_tools={
        "send_email": {
            "arguments": ["to", "subject"],
            "description": "Send email to recipient with subject",
            "function": send_email
        }
    },

    # Credentials
    credentials={"USERNAME": "alice", "PASSWORD": "secret"},

    # Variables
    variables={"api_url": "https://api.example.com"},

    # Structured output
    output_model=Output,

    # Timeout
    timeout=600
)

result = await agent.run()
```

---

## YAML Config (CLI)

For CLI usage, create `config.yaml`:

```yaml
agent:
  max_steps: 15
  reasoning: false
  after_sleep_action: 1.0
  wait_for_stable_ui: 0.3
  prompts_dir: config/prompts

  codeact:
    vision: false
    system_prompt: system.jinja2
    user_prompt: user.jinja2
    safe_execution: false

  manager:
    vision: false
    system_prompt: system.jinja2

  executor:
    vision: false
    system_prompt: system.jinja2

  scripter:
    enabled: true
    max_steps: 10
    execution_timeout: 30.0
    system_prompt_path: system.jinja2
    safe_execution: false

  app_cards:
    enabled: true
    mode: local
    app_cards_dir: config/app_cards
    server_url: null
    server_timeout: 2.0
    server_max_retries: 2

llm_profiles:
  manager:
    provider: GoogleGenAI
    model: models/gemini-2.5-pro
    temperature: 0.2
    kwargs:
      max_tokens: 8192

  executor:
    provider: GoogleGenAI
    model: models/gemini-2.5-flash
    temperature: 0.1
    kwargs:
      max_tokens: 4096

  codeact:
    provider: GoogleGenAI
    model: models/gemini-2.5-pro
    temperature: 0.2
    kwargs:
      max_tokens: 8192

  text_manipulator:
    provider: GoogleGenAI
    model: models/gemini-2.5-flash
    temperature: 0.3

  app_opener:
    provider: OpenAI
    model: gpt-4o-mini
    temperature: 0.0

  scripter:
    provider: GoogleGenAI
    model: models/gemini-2.5-flash
    temperature: 0.1

  structured_output:
    provider: GoogleGenAI
    model: models/gemini-2.5-flash
    temperature: 0.0

device:
  serial: null
  platform: android
  use_tcp: false

telemetry:
  enabled: true

tracing:
  enabled: false
  provider: phoenix  # "phoenix" or "langfuse"
  langfuse_secret_key: ""
  langfuse_public_key: ""
  langfuse_host: ""
  langfuse_user_id: anonymous
  langfuse_session_id: ""

logging:
  debug: false
  save_trajectory: none
  trajectory_path: trajectories
  trajectory_gifs: false
  rich_text: false

safe_execution:
  allow_all_imports: false
  allowed_modules: []
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
  allow_all_builtins: false
  allowed_builtins: []
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

tools:
  allow_drag: false

credentials:
  enabled: false
  file_path: credentials.yaml
```

---

## CLI Overrides

```bash
# Override agent settings
droidrun run "Task" --steps 30 --reasoning --vision

# Override device
droidrun run "Task" --device emulator-5554 --tcp

# Override LLM (applies to ALL agents)
droidrun run "Task" --provider GoogleGenAI --model models/gemini-2.5-flash

# Override logging
droidrun run "Task" --debug --save-trajectory action --tracing

# Custom config file
droidrun run "Task" --config /path/to/config.yaml
```

**All CLI Flags:**
- `--config PATH` - Custom config file
- `--device SERIAL` - Device serial/IP
- `--provider PROVIDER` - LLM provider (OpenAI, Ollama, Anthropic, GoogleGenAI, DeepSeek)
- `--model MODEL` - LLM model name
- `--temperature FLOAT` - LLM temperature
- `--steps INT` - Max steps
- `--base_url URL` - API base URL (for Ollama/OpenRouter)
- `--api_base URL` - API base URL (for OpenAI-like)
- `--vision/--no-vision` - Enable/disable vision for all agents
- `--reasoning/--no-reasoning` - Enable/disable reasoning mode
- `--tracing/--no-tracing` - Enable/disable tracing
- `--debug/--no-debug` - Enable/disable debug logs
- `--tcp/--no-tcp` - Enable/disable TCP communication
- `--save-trajectory none|step|action` - Trajectory saving level
- `--ios` - Run on iOS device

---

## Environment Variables

Set API keys via environment variables:

```bash
export GOOGLE_API_KEY=your-key
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key
export DEEPSEEK_API_KEY=your-key
export DROIDRUN_CONFIG=/path/to/config.yaml  # Custom config path
```
