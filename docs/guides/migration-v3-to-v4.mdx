This guide will help you migrate your Droidrun projects from v3 to v4. Version 4 introduces significant changes to installation, configuration, and API design that improve developer experience and simplify usage.

---

## Overview of Major Changes

Droidrun v4 introduces several breaking changes:

1. **Installation method** changed from `pip` to `uv`
2. **Python version requirement** increased from 3.10+ to 3.11+
3. **Configuration system** completely redesigned with `DroidrunConfig`
4. **Agent API** simplified with config-based initialization
5. **LLM initialization** now handled through config profiles
6. **Result format** changed from dictionary to `ResultEvent` object
7. **Credential Manager** feature included for secure handling of sensitive data
8. **App Instruction Cards** for improved app navigation
9. **Custom Tools** support for extending agent capabilities
10. **Output Extraction** using Pydantic models for structured data

---

## Installation Changes

### v3 Installation
```bash
pip install 'droidrun[google,anthropic,openai,deepseek,ollama,dev]'
```

### v4 Installation

**For CLI usage only:**
```bash
uv tool install 'droidrun[google,anthropic,openai,deepseek,ollama,openrouter]'
```

**For CLI + Python integration:**
```bash
uv pip install 'droidrun[google,anthropic,openai,deepseek,ollama,openrouter]'
```

Note: `droidrun` now provides a `--version` flag — verify your installation with `droidrun --version` after installing.

### Migration Steps

1. **Install uv** (if not already installed):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Uninstall v3** (optional but recommended):
   ```bash
   pip uninstall droidrun
   ```

3. **Install v4** using your preferred method above

4. **Update Python version** to 3.11+ if needed

---

## Configuration System Changes

### v3: No Config Files
v3 had no configuration files. All settings were passed directly to the agent.

### v4: Configuration-Driven Approach

v4 uses a `config.yaml` file that is automatically created on first run:

```yaml
# config.yaml (auto-generated)
llm_profiles:
  - name: "gemini-flash"
    provider: "GoogleGenAI"
    model: "models/gemini-2.5-flash"
    
  - name: "gpt-4o"
    provider: "OpenAI"
    model: "gpt-4o"

default_profile: "gemini-flash"
vision: false
reasoning: false
max_steps: 15
```

### Migration Steps

1. **Create config.yaml** (automatically created on first run, or copy from `config_example.yaml`)

2. **Define your LLM profiles** in the config file instead of in code

3. **Set environment variables** for API keys:
   ```bash
   export GOOGLE_API_KEY=your-api-key
   export OPENAI_API_KEY=your-api-key
   export ANTHROPIC_API_KEY=your-api-key
   ```

---

## API Changes

### Agent Initialization

**v3 Code:**
```python
from droidrun import AdbTools, DroidAgent
from llama_index.llms.google_genai import GoogleGenAI

# Manual LLM setup
llm = GoogleGenAI(
    api_key="YOUR_GEMINI_API_KEY",
    model="gemini-2.5-flash",
)

# Manual tools loading
tools = AdbTools()

# Agent creation
agent = DroidAgent(
    goal="Open Settings and check battery level",
    llm=llm,
    tools=tools,
    vision=True,
    reasoning=False,
)
```

**v4 Code:**
```python
from droidrun import DroidAgent
from droidrun.config_manager.config_manager import DroidrunConfig

# Config-based initialization (LLMs loaded automatically)
config = DroidrunConfig()

# Agent creation (tools loaded automatically)
agent = DroidAgent(
    goal="Open Settings and check battery level",
    config=config,
)
```

### Key Differences

| Feature | v3 | v4 |
|---------|----|----|
| **LLM Setup** | Manual instantiation | Loaded from config profiles |
| **Tools Loading** | Manual `AdbTools()` | Automatic from config |
| **Configuration** | Constructor parameters | `DroidrunConfig` object |
| **API Keys** | In code | Environment variables |

---

## Result Object Changes

### v3: Dictionary Result
```python
result = await agent.run()
print(f"Success: {result['success']}")
if result.get('output'):
    print(f"Output: {result['output']}")
```

### v4: ResultEvent Object
```python
result = await agent.run()
print(f"Success: {result.success}")
print(f"Reason: {result.reason}")
print(f"Steps: {result.steps}")
```

### Migration Steps

Update all result access patterns:
- `result['success']` → `result.success`
- `result.get('output')` → `result.reason` (or appropriate attribute)
- Add access to new fields: `result.steps`

---

## CLI Changes

### New CLI Flags

v4 introduces additional flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--config`, `-c` | Custom config path | `config.yaml` |
| `--device`, `-d` | Device serial or IP | Auto-detect |
| `--tcp` | Use TCP instead of content provider | `false` |
| `--save-trajectory` | Save execution data | `none` |

---

## Breaking Changes Summary

### 1. Python Version
- **v3:** Python 3.10+
- **v4:** Python 3.11+
- **Action:** Update your Python installation

### 2. Installation Tool
- **v3:** `pip install`
- **v4:** `uv tool install` or `uv pip install`
- **Action:** Install `uv` and use new commands

### 3. Import Changes
```python
# v3
from droidrun import AdbTools, DroidAgent
from llama_index.llms.google_genai import GoogleGenAI

# v4
from droidrun import DroidAgent
from droidrun.config_manager.config_manager import DroidrunConfig
# No need to import LLM classes directly
```

### 4. Agent Constructor
```python
# v3: Multiple required parameters
agent = DroidAgent(
    goal="...",
    llm=llm,           # Required
    tools=tools,       # Required
    vision=True,
    reasoning=False,
)

# v4: Config-based, simplified
agent = DroidAgent(
    goal="...",
    config=config,     # Single config object
)
```

### 5. LLM Management
- **v3:** Manually instantiate LLM objects
- **v4:** Define profiles in `config.yaml`, loaded automatically
- **Action:** Move LLM configs to `config.yaml`

### 6. Tools Loading
- **v3:** `tools = AdbTools()` required
- **v4:** Tools loaded automatically from config
- **Action:** Remove manual `AdbTools()` instantiation

### 7. Result Access
- **v3:** Dictionary with string keys
- **v4:** Object with attributes
- **Action:** Update result access patterns

---

## Step-by-Step Migration Guide

### Step 1: Update Environment

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux

# Upgrade Python if needed
python --version  # Should be 3.11+

# Uninstall v3
pip uninstall droidrun
```

### Step 2: Install v4

```bash
# For CLI + Python integration
uv pip install 'droidrun[google,anthropic,openai,deepseek,ollama,openrouter]'

# Or for CLI only
uv tool install 'droidrun[google,anthropic,openai,deepseek,ollama,openrouter]'
```

### Step 3: Create Configuration

```bash
# Run any command to auto-generate config.yaml
droidrun ping
```

Edit `config.yaml` to add your LLM profiles:

```yaml
llm_profiles:
  - name: "my-gemini"
    provider: "GoogleGenAI"
    model: "models/gemini-2.5-flash"
    
  - name: "my-gpt4"
    provider: "OpenAI"
    model: "gpt-4o"

default_profile: "my-gemini"
vision: false
reasoning: false
max_steps: 15
```

### Step 4: Set Environment Variables

```bash
export GOOGLE_API_KEY=your-api-key
export OPENAI_API_KEY=your-api-key
export ANTHROPIC_API_KEY=your-api-key
```

### Step 5: Update Your Code

**Before (v3):**
```python
import asyncio
from droidrun import AdbTools, DroidAgent
from llama_index.llms.google_genai import GoogleGenAI

async def main():
    tools = AdbTools()
    llm = GoogleGenAI(
        api_key="YOUR_GEMINI_API_KEY",
        model="gemini-2.5-flash",
    )
    
    agent = DroidAgent(
        goal="Open Settings and check battery level",
        llm=llm,
        tools=tools,
        vision=True,
        reasoning=False,
    )
    
    result = await agent.run()
    print(f"Success: {result['success']}")
    if result.get('output'):
        print(f"Output: {result['output']}")

if __name__ == "__main__":
    asyncio.run(main())
```

**After (v4):**
```python
import asyncio
from droidrun import DroidAgent
from droidrun.config_manager.config_manager import DroidrunConfig

async def main():
    config = DroidrunConfig()
    
    agent = DroidAgent(
        goal="Open Settings and check battery level",
        config=config,
    )
    
    result = await agent.run()
    print(f"Success: {result.success}")
    print(f"Reason: {result.reason}")
    print(f"Steps: {result.steps}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 6: Test Your Migration

```bash
# Test device connection
droidrun ping

# Test simple command
droidrun "Open Settings"

# Test your Python script
python your_script.py
```

---

## New Features in v4

Droidrun v4 introduces several powerful new features that weren't available in v3. Here's a quick overview:

### 1. Structured Output Extraction

Extract typed data from device interactions using Pydantic models:

```python
from pydantic import BaseModel, Field
from droidrun import DroidAgent
from droidrun.config_manager import DroidrunConfig

class ContactInfo(BaseModel):
    name: str = Field(description="Full name")
    phone: str = Field(description="Phone number")
    email: str = Field(description="Email address")

config = DroidrunConfig()

agent = DroidAgent(
    goal="Find John Smith's contact information",
    config=config,
    output_model=ContactInfo,  # NEW in v4
)

result = await agent.run()
contact = result.structured_output  # Typed Pydantic object
print(f"Name: {contact.name}, Phone: {contact.phone}")
```

The agent automatically extracts structured data into your Pydantic model after completing the task. **→ See [Structured Output Documentation](/features/structured-output)** for detailed examples and best practices.

---

### 2. Credential Management

Securely manage passwords, API keys, and tokens without hardcoding them:

```python
from droidrun import DroidAgent
from droidrun.config_manager import DroidrunConfig

# In-memory credentials (recommended for SDK)
credentials = {
    "EMAIL_USER": "user@example.com",
    "EMAIL_PASS": "secret_password",
    "API_KEY": "sk-1234567890"
}

config = DroidrunConfig()

agent = DroidAgent(
    goal="Login to Gmail with my credentials",
    config=config,
    credentials=credentials,  # NEW in v4
)

result = await agent.run()
```

Credentials are never logged or exposed - agents use `type_secret()` to securely input sensitive data. You can also store credentials in a `credentials.yaml` file. **→ See [Credential Management Documentation](/features/credentials)** for YAML configuration and advanced usage.

---

### 3. App Instruction Cards

Teach agents how to use specific apps with custom instruction guides:

```bash
# App cards are enabled by default
droidrun run "Send an email to john@example.com" --reasoning
```

When your agent opens an app (like Gmail, Chrome, WhatsApp), it automatically loads app-specific instructions that help it:
- Navigate the UI efficiently
- Find buttons and features quickly
- Use app-specific shortcuts and search syntax
- Complete tasks with higher success rates

**Create your own app cards:**

```bash
# 1. Create app cards directory
mkdir -p config/app_cards

# 2. Create a markdown file for your app
touch config/app_cards/chrome.md

# 3. Register in app_cards.json
echo '{"com.android.chrome": "chrome.md"}' > config/app_cards/app_cards.json
```

**→ See [App Instruction Cards Documentation](/features/app-cards)** for examples, best practices, and troubleshooting.

---

### 4. Custom Tools

Extend agent capabilities with custom Python functions for external APIs, data processing, or domain-specific logic:

```python
from droidrun import DroidAgent
from droidrun.config_manager import DroidrunConfig

def calculate_tax(amount: float, rate: float, **kwargs) -> str:
    """Calculate tax for a given amount."""
    tax = amount * rate
    total = amount + tax
    return f"Tax: ${tax:.2f}, Total: ${total:.2f}"

custom_tools = {
    "calculate_tax": {
        "arguments": ["amount", "rate"],
        "description": "Calculate tax for a given amount and rate",
        "function": calculate_tax
    }
}

config = DroidrunConfig()

agent = DroidAgent(
    goal="Calculate tax for $100 at 8% rate",
    config=config,
    custom_tools=custom_tools,  # NEW in v4
)

result = await agent.run()
```

Custom tools can access device state, shared memory, and credentials. **→ See [Custom Tools Documentation](/features/custom-tools)** for accessing device state, API integration patterns, and advanced examples.

---

### 5. Advanced Configuration System

v4 introduces granular configuration for all aspects of agent behavior:

```python
from droidrun.config_manager import (
    DroidrunConfig,
    AgentConfig,
    CodeActConfig,
    ManagerConfig,
    ExecutorConfig,
    ScripterConfig,
    AppCardConfig,
    DeviceConfig,
    LoggingConfig,
    TracingConfig
)

config = DroidrunConfig(
    agent=AgentConfig(
        max_steps=30,
        reasoning=True,
        after_sleep_action=1.5,
        codeact=CodeActConfig(vision=True, safe_execution=True),
        manager=ManagerConfig(vision=True),
        executor=ExecutorConfig(vision=False),
        scripter=ScripterConfig(enabled=True, max_steps=10),
        app_cards=AppCardConfig(enabled=True, mode="local")
    ),
    device=DeviceConfig(serial="emulator-5554", use_tcp=False),
    logging=LoggingConfig(debug=True, save_trajectory="action"),
    tracing=TracingConfig(enabled=True)
)

agent = DroidAgent(goal="Complex task", config=config)
```

**→ See [Configuration Reference Documentation](/sdk/droid-agent)** for all available configuration classes and parameters.

---

### 6. Per-Agent LLM Configuration

Use different LLMs for different agent roles (planning, execution, code generation):

```python
from droidrun import DroidAgent
from droidrun.config_manager import DroidrunConfig
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini

config = DroidrunConfig()

agent = DroidAgent(
    goal="Complex multi-step task",
    config=config,
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

This allows you to optimize cost and performance by using appropriate models for each task. **→ See [SDK Configuration Documentation](/sdk/configuration)** for all LLM keys and configuration options.

---

### 7. Custom Variables

Pass non-sensitive contextual data to agents:

```python
from droidrun import DroidAgent
from droidrun.config_manager import DroidrunConfig

config = DroidrunConfig()

agent = DroidAgent(
    goal="Send report to the project lead",
    config=config,
    variables={
        "project_id": "12345",
        "lead_email": "alice@example.com",
        "api_endpoint": "https://api.example.com"
    }
)
```

Variables are accessible throughout execution in `shared_state.custom_variables` and can be used in custom tools or referenced by agents.
 **→ See [Custom Variables Documentation](/features/custom-variables)** for usage patterns.

---

### 8. Tracing and Trajectory Recording

Monitor agent execution with real-time Phoenix tracing and local trajectory recording:
```python
from droidrun import DroidAgent
from droidrun.config_manager import DroidrunConfig, TracingConfig, LoggingConfig

config = DroidrunConfig(
    tracing=TracingConfig(enabled=True),  # Phoenix tracing
    logging=LoggingConfig(save_trajectory="step")  # Local recording
)

agent = DroidAgent(goal="Open Settings", config=config)
result = await agent.run()
```

Or via CLI:
```bash
# Enable Phoenix tracing
droidrun run "Open Settings" --tracing

# Enable trajectory recording (screenshots + UI state)
droidrun run "Open Settings" --save-trajectory step
```

**Phoenix Tracing** provides real-time monitoring of:
- LLM calls with prompts and token usage
- Agent workflow execution
- Tool invocations and results

**Trajectory Recording** saves locally:
- Screenshots per step/action
- UI state and element hierarchies
- Agent reasoning and decisions

Trajectories are saved to `trajectories/` directory for offline debugging. 
**→ See [Tracing Documentation](/features/tracing)** for Phoenix setup, configuration options, and trajectory analysis.

---


*Last updated: November 2025*