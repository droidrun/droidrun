<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/droidrun-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/droidrun.png">
  <img src="./static/droidrun.png"  width="full">
</picture>

[![GitHub stars](https://img.shields.io/github/stars/droidrun/droidrun?style=social)](https://github.com/droidrun/droidrun/stargazers)
[![Discord](https://img.shields.io/discord/1360219330318696488?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://discord.gg/ZZbKEZZkwK)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.droidrun.ai)
[![Twitter Follow](https://img.shields.io/twitter/follow/droid_run?style=social)](https://x.com/droid_run)


DroidRun is a powerful framework for controlling Android devices through LLM agents. It allows you to automate Android device interactions using natural language commands.

## ✨ Features

- Control Android devices with natural language commands
- Supports multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama, DeepSeek)
- Planning capabilities for complex multi-step tasks
- LlamaIndex integration for flexible LLM interactions
- Easy to use CLI with enhanced debugging features
- Extendable Python API for custom automations
- Screenshot analysis for visual understanding of the device
- Execution tracing with Arize Phoenix

## 📦 Installation

### 🚀 Option 1: Install from PyPI (Recommended)

```bash
pip install droidrun
```

### 🔧 Option 2: Install from Source

```bash
git clone https://github.com/droidrun/droidrun.git
cd droidrun
pip install -e .
```

## 📋 Prerequisites

1. An Android device connected via USB or ADB over TCP/IP
2. ADB (Android Debug Bridge) installed and configured
3. DroidRun Portal app installed on your Android device
4. API key for at least one of the supported LLM providers:
   - OpenAI
   - Anthropic
   - Google Gemini

### 🔧 Setting up ADB

ADB (Android Debug Bridge) is required for DroidRun to communicate with your Android device:

1. **Install ADB**:
   - **Windows**: Download [Android SDK Platform Tools](https://developer.android.com/studio/releases/platform-tools) and extract the ZIP file
   - **macOS**: `brew install android-platform-tools`
   - **Linux**: `sudo apt install adb` (Ubuntu/Debian) or `sudo pacman -S android-tools` (Arch)

2. **Add ADB to your PATH**:
   - **Windows**: Add the path to the extracted platform-tools folder to your system's PATH environment variable
   - **macOS/Linux**: Add the following to your ~/.bashrc or ~/.zshrc:
     ```bash
     export PATH=$PATH:/path/to/platform-tools
     ```

3. **Verify ADB installation**:
   ```bash
   adb version
   ```

4. **Enable USB debugging on your Android device**:
   - Go to **Settings → About phone**
   - Tap **Build number** 7 times to enable Developer options
   - Go to **Settings → System → Developer options** (location may vary by device)
   - Enable **USB debugging**

## 🛠️ Setup

### 📱 1. Install DroidRun Portal App

DroidRun requires the DroidRun Portal app to be installed on your Android device:

1. Download the DroidRun Portal APK from the [DroidRun Portal repository](https://github.com/droidrun/droidrun-portal)
2. Use DroidRun to install the portal app:
   ```bash
   droidrun setup --path=/path/to/droidrun-portal.apk
   ```

Alternatively, you can use ADB to install it manually:
```bash
adb install -r /path/to/droidrun-portal.apk
```

### 🔑 2. Set up API keys

Create a `.env` file in your working directory or set environment variables:

```bash
# Choose at least one of these based on your preferred provider
export OPENAI_API_KEY="your_openai_api_key_here"
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
export GEMINI_API_KEY="your_gemini_api_key_here"
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
# For Ollama, no API key is needed
```

To load the environment variables from the `.env` file:

```bash
source .env
```

### 📱 3. Connect to an Android device

Connect your device via USB or set up wireless ADB:

```bash
# List connected devices
droidrun devices

# Connect to a device over Wi-Fi
droidrun connect 192.168.1.100
```

## 💻 Using the CLI

DroidRun's CLI is designed to be simple and intuitive. You can use it in two ways:

### 🚀 Basic Usage

```bash
# Format: droidrun "task description" [options]
droidrun "Open the settings app"
```

### 🔌 With Provider Options

```bash
# Using OpenAI
droidrun "Open the calculator app" --provider OpenAI --model gpt-4o-mini

# Using Anthropic
droidrun "Check the battery level" --provider Anthropic --model claude-3-sonnet-20240229

# Using Gemini
droidrun "Install and open Instagram" --provider Gemini --model models/gemini-2.5-pro-preview-05-06

# Using Ollama (local)
droidrun "Check battery level" --provider Ollama --model llama2
```

### ⚙️ Additional Options

```bash
# Specify a particular device
droidrun "Open Chrome and search for weather" --device abc123

# Enable vision capabilities
droidrun "Analyze what's on the screen" --vision

# Enable planning for complex tasks
droidrun "Find and download a specific app" --reasoning

# Enable execution tracing (requires Phoenix server running)
droidrun "Debug this complex workflow" --tracing

# Set maximum number of steps
droidrun "Open settings and enable dark mode" --steps 20
```

## 📝 Creating a Minimal Test Script

If you want to use DroidRun in your Python code rather than via the CLI, you can create a minimal test script:

```python
#!/usr/bin/env python3
import asyncio
from droidrun.agent.droid import DroidAgent
from droidrun.agent.utils.llm_picker import load_llm
from droidrun.tools import load_tools

async def main():
    # Load tools
    tool_list, tools_instance = await load_tools()
    
    # Load LLM
    llm = load_llm(
        provider_name="Gemini",  # Case sensitive: OpenAI, Ollama, Anthropic, Gemini, DeepSeek
        model="models/gemini-2.5-pro-preview-05-06",
        temperature=0.2
    )
    
    # Create and run the agent
    agent = DroidAgent(
        goal="Open the Settings app and check the Android version",
        llm=llm,
        tools_instance=tools_instance,
        tool_list=tool_list,
        vision=True,      # Enable vision for screen analysis
        reasoning=True    # Enable planning for complex tasks
    )
    
    # Run the agent
    result = await agent.run()
    print(f"Success: {result['success']}")
    if result.get('reason'):
        print(f"Reason: {result['reason']}")

if __name__ == "__main__":
    asyncio.run(main())
```

You can also use LlamaIndex directly:

```python
import asyncio
from llama_index.llms.gemini import Gemini
from droidrun.agent.droid import DroidAgent
from droidrun.tools import load_tools

async def main():
    # Load tools
    tool_list, tools_instance = await load_tools()
    
    # Create LlamaIndex LLM directly
    llm = Gemini(
        model="models/gemini-2.5-pro-preview-05-06",
        temperature=0.2
    )
    
    # Create and run the agent
    agent = DroidAgent(
        goal="Open the Settings app and check the Android version",
        llm=llm,
        tools_instance=tools_instance,
        tool_list=tool_list
    )
    
    # Run the agent
    result = await agent.run()
    print(f"Success: {result['success']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## ❓ Troubleshooting

### 🔑 API Key Issues

If you encounter errors about missing API keys, ensure:
1. You've set the correct environment variable for your chosen provider
2. The API key is valid and has appropriate permissions
3. You've correctly sourced your `.env` file or exported the variables manually

### 📱 Device Connection Issues

If you have trouble connecting to your device:
1. Ensure USB debugging is enabled on your Android device
2. Check that your device is recognized by ADB: `adb devices`
3. For wireless connections, make sure your device and computer are on the same network

### 🤖 LLM Provider Selection

If DroidRun is using the wrong LLM provider:
1. Explicitly specify the provider with `--provider` (in CLI) or `llm_provider=` (in code)
2. When using Gemini, ensure you have set `GEMINI_API_KEY` and specified `--provider gemini`

### 📊 Tracing Issues

If you're using the tracing feature:
1. Make sure to install Arize Phoenix: `pip install "arize-phoenix[llama-index]"`
2. Start the Phoenix server before running your command: `phoenix serve`
3. Access the tracing UI at http://localhost:6006 after execution

### 🎬 Demo Videos

1. **Shopping Assistant**: Watch how DroidRun searches Amazon for headphones and sends the top 3 products to a colleague on WhatsApp.
   
   Prompt: "Go to Amazon, search for headphones and write the top 3 products to my colleague on WhatsApp."
   
   [![Shopping Assistant Demo](https://img.youtube.com/vi/VQK3JcifgwU/0.jpg)](https://www.youtube.com/watch?v=VQK3JcifgwU)

2. **Social Media Automation**: See DroidRun open X (Twitter) and post "Hello World".
   
   Prompt: "Open up X and post Hello World."
   
   [![Social Media Automation Demo](https://img.youtube.com/vi/i4-sDQhzt_M/0.jpg)](https://www.youtube.com/watch?v=i4-sDQhzt_M)

## 💡 Example Use Cases

- Automated UI testing of Android applications
- Creating guided workflows for non-technical users
- Automating repetitive tasks on Android devices
- Remote assistance for less technical users
- Exploring Android UI with natural language commands

## 🗺️ Roadmap

### 🤖 Agent:
- **Improve memory**: Enhance context retention for complex multi-step tasks
- **Expand planning capabilities**: Add support for more complex reasoning strategies
- **Add Integrations**: Support more LLM providers and agent frameworks (LangChain, Agno etc.)

### ⚙️ Automations:
- **Create Automation Scripts**: Generate reusable scripts from agent actions that can be scheduled or shared

### ☁️ Cloud:
- **Hosted version**: Remote device control via web interface without local setup
- **Add-Ons**: Marketplace for extensions serving specific use cases
- **Proxy Hours**: Cloud compute time with tiered pricing for running automations
- **Droidrun AppStore**: Simple installation of Apps on your hosted devices

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 

## Security Checks

To ensure the security of the codebase, we have integrated security checks using `bandit` and `safety`. These tools help identify potential security issues in the code and dependencies.

### Running Security Checks

Before submitting any code, please run the following security checks:

1. **Bandit**: A tool to find common security issues in Python code.
   ```bash
   bandit -r droidrun
   ```

2. **Safety**: A tool to check your installed dependencies for known security vulnerabilities.
   ```bash
   safety check
   ```

You can also run both checks using the provided script:
```bash
python -m droidrun.tools.security_check
```
