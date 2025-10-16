# DroidRun Cloud Device API Reference

Complete reference for using DroidRun's cloud Android device provisioning in your automation framework.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [DroidRunClient](#droidrunclient)
- [CloudAdbTools](#cloudadbtools)
- [Complete Examples](#complete-examples)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

---

## Overview

The DroidRun cloud integration provides two main classes:

1. **`DroidRunClient`** - Low-level API client for device lifecycle management
2. **`CloudAdbTools`** - High-level wrapper that extends `AdbTools` for cloud devices

### Architecture

```
┌─────────────────────────────────────┐
│   Your Automation Code              │
│   (uses Tools interface)            │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼─────┐      ┌────────▼────────┐
│AdbTools │      │ CloudAdbTools   │
│(local)  │      │(cloud)          │
└─────────┘      └────────┬────────┘
                          │
                 ┌────────┴────────┐
                 │                 │
         ┌───────▼──────┐  ┌──────▼────────┐
         │DroidRunClient│  │ AdbTools      │
         │(API client)  │  │ (parent)      │
         └──────────────┘  └───────┬───────┘
                                   │
                           ┌───────▼────────┐
                           │ PortalClient   │
                           │ (content provider)│
                           └────────────────┘
```

### How It Works

1. **Provision**: Create a Limbar Android device via DroidRun API
2. **Wait**: Poll until device is ready and ADB tunnel is established
3. **Connect**: Get cloud serial (e.g., `cloud.droidrun.ai:12345`)
4. **Automate**: Use all standard AdbTools methods (tap, swipe, input_text, etc.)
5. **Terminate**: Clean up device and stop billing

---

## Quick Start

```python
from droidrun.tools.cloud_adb import CloudAdbTools
from droidrun.tools.droidrun_client import DroidRunClient
import os

# Initialize API client
client = DroidRunClient(
    base_url=os.getenv("DROIDRUN_API_URL"),
    service_key=os.getenv("DROIDRUN_SERVICE_KEY")
)

# Use cloud device with context manager (automatic cleanup)
with CloudAdbTools(api_client=client, apps=["com.example.app"]) as tools:
    # All AdbTools methods work!
    state = tools.get_state()
    tools.start_app("com.example.app")
    tools.tap_by_index(5)
    tools.input_text("Hello from cloud!")
    screenshot = tools.take_screenshot()
    tools.complete(success=True, reason="Test passed")
# Device automatically terminated here
```

---

## DroidRunClient

Low-level client for the DroidRun Device API.

### Constructor

```python
DroidRunClient(
    base_url: str,
    service_key: str,
    timeout: int = 30,
    max_retries: int = 3
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | *required* | DroidRun Device API URL (e.g., `"https://device-api.droidrun.ai"`) |
| `service_key` | `str` | *required* | Service key for authentication (sent in `Authorization` header) |
| `timeout` | `int` | `30` | Request timeout in seconds |
| `max_retries` | `int` | `3` | Max retry attempts for failed requests |

#### Example

```python
client = DroidRunClient(
    base_url="https://device-api.droidrun.ai",
    service_key="your-service-key"
)
```

---

### Methods

#### `provision()`

Provision a new cloud Android device.

```python
provision(
    apps: Optional[List[str]] = None,
    files: Optional[List[str]] = None,
    country: str = "US"
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `apps` | `List[str]` | `None` | App package names or APK URLs to pre-install |
| `files` | `List[str]` | `None` | File URLs to pre-push to device (`/sdcard/`) |
| `country` | `str` | `"US"` | ISO country code for device location |

**Returns:**

```python
{
    "id": "device-uuid-here",
    "streamUrl": "wss://stream.example.com/...",  # WebSocket screen streaming
    "serial": "",  # Empty until prerequisites met
    "token": "auth-token-here"
}
```

**Example:**

```python
response = client.provision(
    apps=[
        "com.android.chrome",  # Package name
        "https://storage.example.com/app.apk"  # APK URL
    ],
    files=["https://storage.example.com/data.json"],
    country="US"
)
device_id = response["id"]
```

---

#### `wait_for_prerequisites()`

Wait for device to be ready and get ADB serial.

This is a **long-polling** endpoint - the server waits up to 10 minutes before responding.

```python
wait_for_prerequisites(
    device_id: str,
    timeout: int = 600,
    poll_interval: int = 5
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device_id` | `str` | *required* | Device ID from `provision()` response |
| `timeout` | `int` | `600` | Max seconds to wait (10 minutes) |
| `poll_interval` | `int` | `5` | Seconds between retries if request fails |

**Returns:**

```python
{
    "id": "device-uuid-here",
    "streamUrl": "wss://stream.example.com/...",
    "serial": "cloud.droidrun.ai:12345",  # ADB connection string
    "token": "auth-token-here"
}
```

**Example:**

```python
device = client.wait_for_prerequisites(device_id="abc-123", timeout=600)
serial = device["serial"]  # "cloud.droidrun.ai:12345"

# Now you can connect via adbutils
from adbutils import adb
device = adb.device(serial=serial)
```

**Raises:**

- `TimeoutError` - Device not ready within timeout
- `requests.HTTPError` - API request failed
- `RuntimeError` - Device readiness check failed

---

#### `terminate()`

Terminate a cloud device and stop billing.

**⚠️ IMPORTANT**: Always call this when done to avoid unnecessary charges!

```python
terminate(device_id: str) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `device_id` | `str` | Device ID to terminate |

**Example:**

```python
client.terminate(device_id="abc-123")
```

**Raises:**

- `requests.HTTPError` - API request failed
- `RuntimeError` - Termination failed after retries

---

## CloudAdbTools

High-level wrapper that extends `AdbTools` for cloud devices.

### Constructor

```python
CloudAdbTools(
    api_client: DroidRunClient,
    apps: Optional[List[str]] = None,
    files: Optional[List[str]] = None,
    country: str = "US",
    provision_timeout: int = 600,
    app_opener_llm = None,
    text_manipulator_llm = None
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_client` | `DroidRunClient` | *required* | DroidRunClient instance |
| `apps` | `List[str]` | `None` | Apps to pre-install (package names or APK URLs) |
| `files` | `List[str]` | `None` | Files to pre-push (URLs) |
| `country` | `str` | `"US"` | ISO country code |
| `provision_timeout` | `int` | `600` | Max seconds to wait for device ready |
| `app_opener_llm` | LLM | `None` | LLM for app opening workflow |
| `text_manipulator_llm` | LLM | `None` | LLM for text manipulation |

#### Example

```python
from droidrun.tools.cloud_adb import CloudAdbTools
from droidrun.tools.droidrun_client import DroidRunClient

client = DroidRunClient(base_url="...", service_key="...")

tools = CloudAdbTools(
    api_client=client,
    apps=["com.example.app"],
    country="US"
)
```

---

### Methods

CloudAdbTools inherits **all methods from AdbTools**. Here are the key ones:

#### Device State

```python
get_state() -> Dict[str, Any]
```

Get accessibility tree and phone state.

**Returns:**

```python
{
    "a11y_tree": [...],  # UI element hierarchy
    "phone_state": {
        "currentApp": "com.android.launcher",
        "focusedElement": {...},
        "screenResolution": {"width": 1080, "height": 1920}
    }
}
```

---

#### UI Interaction

```python
tap_by_index(index: int) -> str
```

Tap UI element by index from accessibility tree.

```python
swipe(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration_ms: int = 300
) -> bool
```

Swipe gesture from start to end coordinates.

```python
drag(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration: float = 3
) -> bool
```

Drag and drop gesture.

```python
input_text(
    text: str,
    index: int = -1,
    clear: bool = False
) -> str
```

Input text via Portal keyboard (supports Unicode).

---

#### App Management

```python
start_app(package: str, activity: Optional[str] = None) -> str
```

Launch an app by package name.

```python
list_packages(include_system_apps: bool = False) -> List[str]
```

List installed packages.

```python
get_apps(include_system: bool = True) -> List[Dict[str, str]]
```

Get apps with human-readable labels.

---

#### Key Events

```python
back() -> str
```

Press Android back button.

```python
press_key(keycode: int) -> str
```

Press Android key by keycode (e.g., 66=ENTER, 67=DELETE).

---

#### Screenshot

```python
take_screenshot(hide_overlay: bool = True) -> Tuple[str, bytes]
```

Capture screenshot.

**Returns:** `("PNG", image_bytes)`

---

#### Memory & Completion

```python
remember(information: str) -> str
```

Store information in agent memory.

```python
get_memory() -> List[str]
```

Retrieve stored memory.

```python
complete(success: bool, reason: str = "") -> None
```

Mark task as complete.

---

#### Cloud-Specific Method

```python
terminate() -> None
```

Terminate the cloud device and stop billing.

**⚠️ IMPORTANT**: Always call this or use context manager!

---

### Context Manager

CloudAdbTools supports context managers for automatic cleanup:

```python
with CloudAdbTools(api_client=client, apps=["com.example.app"]) as tools:
    tools.tap_by_index(1)
    tools.swipe(100, 500, 100, 200)
# Device automatically terminated here (even if exception occurs)
```

---

## Complete Examples

### Example 1: Simple Automation

```python
from droidrun.tools.cloud_adb import CloudAdbTools
from droidrun.tools.droidrun_client import DroidRunClient
import os

# Setup
client = DroidRunClient(
    base_url=os.getenv("DROIDRUN_API_URL"),
    service_key=os.getenv("DROIDRUN_SERVICE_KEY")
)

# Run automation
with CloudAdbTools(api_client=client, country="US") as tools:
    # Get current state
    state = tools.get_state()
    current_app = state["phone_state"]["currentApp"]
    print(f"Current app: {current_app}")

    # Open settings
    tools.start_app("com.android.settings")

    # Interact with UI
    tools.tap_by_index(3)
    tools.input_text("WiFi")

    # Take screenshot
    format, screenshot = tools.take_screenshot()
    with open("screenshot.png", "wb") as f:
        f.write(screenshot)

    # Mark complete
    tools.complete(success=True, reason="Settings opened successfully")
```

---

### Example 2: Parallel Device Automation

```python
import concurrent.futures
from droidrun.tools.cloud_adb import CloudAdbTools
from droidrun.tools.droidrun_client import DroidRunClient

client = DroidRunClient(base_url="...", service_key="...")

def run_test_on_device(test_name: str) -> dict:
    """Run test on a dedicated cloud device."""
    with CloudAdbTools(api_client=client) as tools:
        tools.start_app("com.example.app")
        tools.tap_by_index(1)
        state = tools.get_state()
        tools.complete(success=True, reason=f"{test_name} passed")
        return {"test": test_name, "state": state}

# Run 5 tests in parallel on separate devices
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    tests = ["login", "search", "checkout", "settings", "profile"]
    futures = [executor.submit(run_test_on_device, test) for test in tests]
    results = [f.result() for f in futures]

print(f"Completed {len(results)} tests in parallel")
```

---

### Example 3: Manual Lifecycle Control

```python
from droidrun.tools.cloud_adb import CloudAdbTools
from droidrun.tools.droidrun_client import DroidRunClient

client = DroidRunClient(base_url="...", service_key="...")

# Manual provision (no context manager)
tools = CloudAdbTools(
    api_client=client,
    apps=["com.example.app"],
    country="US"
)

try:
    # Your automation
    tools.start_app("com.example.app")
    tools.tap_by_index(5)
    tools.swipe(100, 500, 100, 200)

    # Some logic that might raise exception
    state = tools.get_state()
    if not state["phone_state"]["currentApp"] == "com.example.app":
        raise RuntimeError("App not running!")

    tools.complete(success=True)

except Exception as e:
    print(f"Test failed: {e}")
    tools.complete(success=False, reason=str(e))

finally:
    # CRITICAL: Always terminate in finally block
    tools.terminate()
```

---

### Example 4: Pre-installing APKs

```python
from droidrun.tools.cloud_adb import CloudAdbTools
from droidrun.tools.droidrun_client import DroidRunClient

client = DroidRunClient(base_url="...", service_key="...")

# Provision device with pre-installed APKs
with CloudAdbTools(
    api_client=client,
    apps=[
        "https://storage.example.com/my-app.apk",  # Custom APK
        "https://storage.example.com/test-helper.apk"
    ],
    files=[
        "https://storage.example.com/test-data.json"  # Pushed to /sdcard/
    ]
) as tools:
    # Apps are already installed when device is ready
    tools.start_app("com.mycompany.myapp")
    tools.tap_by_index(1)

    # File is already on device at /sdcard/test-data.json
    # Can access via shell or app
```

---

## Error Handling

### Common Exceptions

| Exception | Cause | Solution |
|-----------|-------|----------|
| `RuntimeError` | Provisioning failed | Check API credentials, Limbar quota |
| `TimeoutError` | Device not ready in time | Increase `provision_timeout` (default 600s) |
| `requests.HTTPError` | API request failed | Check network, API status, service key |
| `ValueError` | Invalid parameters | Check app URLs, country codes |

### Retry Strategy

DroidRunClient has built-in retries:
- **Max retries**: 3 attempts
- **Backoff**: Exponential (2^attempt seconds)
- **Applies to**: provision(), terminate()

### Handling Provisioning Failures

```python
from droidrun.tools.cloud_adb import CloudAdbTools
from droidrun.tools.droidrun_client import DroidRunClient

client = DroidRunClient(base_url="...", service_key="...")

try:
    tools = CloudAdbTools(
        api_client=client,
        apps=["com.example.app"],
        provision_timeout=900  # Increase timeout to 15 min
    )
except RuntimeError as e:
    print(f"Provisioning failed: {e}")
    # Fallback to local device or retry logic
except TimeoutError as e:
    print(f"Device took too long to provision: {e}")
    # Retry or alert
```

### Ensuring Cleanup

**⚠️ Critical:** Always ensure device termination to avoid billing!

```python
# Option 1: Context manager (RECOMMENDED)
with CloudAdbTools(api_client=client) as tools:
    # Automatic cleanup even on exception
    pass

# Option 2: Try/finally
tools = CloudAdbTools(api_client=client)
try:
    # Your code
    pass
finally:
    tools.terminate()  # Always runs

# Option 3: Explicit termination check
tools = CloudAdbTools(api_client=client)
try:
    # Your code
    pass
except Exception as e:
    print(f"Error: {e}")
    raise
finally:
    if not tools._terminated:
        tools.terminate()
```

---

## Best Practices

### 1. Always Use Context Managers

```python
# ✅ GOOD: Automatic cleanup
with CloudAdbTools(api_client=client) as tools:
    tools.tap_by_index(1)

# ❌ BAD: Manual cleanup can be forgotten
tools = CloudAdbTools(api_client=client)
tools.tap_by_index(1)
# Oops, forgot to terminate! Still billing!
```

### 2. Pre-install Apps During Provision

```python
# ✅ GOOD: Apps ready when device starts
with CloudAdbTools(api_client=client, apps=["com.example.app"]) as tools:
    tools.start_app("com.example.app")

# ❌ SLOW: Install after provisioning wastes time
with CloudAdbTools(api_client=client) as tools:
    tools.install_app("/path/to/app.apk")  # Extra time
    tools.start_app("com.example.app")
```

### 3. Use Parallel Provisioning for Test Suites

```python
# ✅ GOOD: Parallel execution
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(run_test, test) for test in tests]

# ❌ SLOW: Sequential execution
for test in tests:
    run_test(test)  # Each device provisions one at a time
```

### 4. Set Appropriate Timeouts

```python
# ✅ GOOD: Realistic timeout
with CloudAdbTools(api_client=client, provision_timeout=600) as tools:
    pass  # 10 min is typical

# ❌ BAD: Too short
with CloudAdbTools(api_client=client, provision_timeout=60) as tools:
    pass  # Will likely timeout
```

### 5. Store Credentials Securely

```python
# ✅ GOOD: Environment variables
import os
client = DroidRunClient(
    base_url=os.getenv("DROIDRUN_API_URL"),
    service_key=os.getenv("DROIDRUN_SERVICE_KEY")
)

# ❌ BAD: Hardcoded
client = DroidRunClient(
    base_url="https://...",
    service_key="my-secret-key-123"  # Don't commit this!
)
```

### 6. Handle Errors Gracefully

```python
# ✅ GOOD: Proper error handling
try:
    with CloudAdbTools(api_client=client) as tools:
        tools.tap_by_index(1)
        tools.complete(success=True)
except RuntimeError as e:
    print(f"Provisioning failed: {e}")
    # Fallback or retry logic
except Exception as e:
    print(f"Test failed: {e}")
    # Report failure

# ❌ BAD: No error handling
with CloudAdbTools(api_client=client) as tools:
    tools.tap_by_index(1)  # What if this fails?
```

### 7. Log Device IDs for Debugging

```python
# ✅ GOOD: Log device info
tools = CloudAdbTools(api_client=client)
print(f"Provisioned device: {tools.device_id}")
print(f"Serial: {tools.cloud_serial}")

try:
    # Your automation
    pass
finally:
    tools.terminate()
```

---

## Additional Resources

- **DroidRun Device API Documentation**: See `README.md` in the API server repository
- **AdbTools Documentation**: See existing framework docs for all inherited methods
- **Limbar Platform**: https://limbar.io
- **Support**: Contact your DroidRun account manager

---

## Changelog

### v1.0.0 (2025-10-15)
- Initial release
- `DroidRunClient` with provision, wait, terminate
- `CloudAdbTools` with full AdbTools compatibility
- Content provider mode for Portal app
- Context manager support
- Automatic cleanup and error handling
