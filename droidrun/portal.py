"""
Portal APK management and device communication utilities.

This module handles downloading, installing, and managing the DroidRun Portal app
on Android devices. It also provides utilities for checking accessibility service
status and managing device communication modes (TCP and content provider).
"""

import contextlib
import os
import tempfile

import requests
from adbutils import AdbDevice, adb
from rich.console import Console

from droidrun.tools import AdbTools

REPO = "droidrun/droidrun-portal"
ASSET_NAME = "droidrun-portal"
GITHUB_API_HOSTS = ["https://api.github.com", "https://ungh.cc"]

PORTAL_PACKAGE_NAME = "com.droidrun.portal"
A11Y_SERVICE_NAME = (
    f"{PORTAL_PACKAGE_NAME}/com.droidrun.portal.DroidrunAccessibilityService"
)


def get_latest_release_assets(debug: bool = False):
    """
    Fetch the latest Portal APK release assets from GitHub.

    Args:
        debug: Enable debug logging

    Returns:
        List of asset dictionaries from the latest GitHub release

    Raises:
        requests.HTTPError: If the GitHub API request fails
    """
    for host in GITHUB_API_HOSTS:
        url = f"{host}/repos/{REPO}/releases/latest"
        response = requests.get(url)
        if response.status_code == 200:
            if debug:
                print(f"Using GitHub release on {host}")
            break

    response.raise_for_status()
    latest_release = response.json()

    if "release" in latest_release:
        assets = latest_release["release"]["assets"]
    else:
        assets = latest_release.get("assets", [])

    return assets


@contextlib.contextmanager
def download_portal_apk(debug: bool = False):
    """
    Download the latest Portal APK from GitHub releases.

    This context manager downloads the APK to a temporary file and yields
    the file path. The file is automatically deleted when the context exits.

    Args:
        debug: Enable debug logging

    Yields:
        str: Path to the downloaded APK file

    Raises:
        Exception: If the Portal APK asset is not found in the release
        requests.HTTPError: If the download fails
    """
    console = Console()
    assets = get_latest_release_assets(debug)

    asset_version = None
    asset_url = None
    for asset in assets:
        if (
            "browser_download_url" in asset
            and "name" in asset
            and asset["name"].startswith(ASSET_NAME)
        ):
            asset_url = asset["browser_download_url"]
            asset_version = asset["name"].split("-")[-1]
            asset_version = asset_version.removesuffix(".apk")
            break
        elif "downloadUrl" in asset and os.path.basename(
            asset["downloadUrl"]
        ).startswith(ASSET_NAME):
            asset_url = asset["downloadUrl"]
            asset_version: str = asset.get("name", os.path.basename(asset_url)).split("-")[-1]
            asset_version = asset_version.removesuffix(".apk")
            break
        else:
            if debug:
                print(asset)

    if not asset_url:
        raise Exception(f"Asset named '{ASSET_NAME}' not found in the latest release.")

    console.print(f"Found Portal APK [bold]{asset_version}[/bold]")
    if debug:
        console.print(f"Asset URL: {asset_url}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".apk")
    try:
        r = requests.get(asset_url, stream=True)
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp.close()
        yield tmp.name
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def enable_portal_accessibility(
    device: AdbDevice, service_name: str = A11Y_SERVICE_NAME
):
    """
    Enable the Portal accessibility service on the device.

    Args:
        device: ADB device connection
        service_name: Full accessibility service name (default: Portal service)

    Note:
        This may fail on some devices due to security restrictions.
        Manual enablement may be required.
    """
    device.shell(f"settings put secure enabled_accessibility_services {service_name}")
    device.shell("settings put secure accessibility_enabled 1")


def check_portal_accessibility(
    device: AdbDevice, service_name: str = A11Y_SERVICE_NAME, debug: bool = False
) -> bool:
    """
    Check if the Portal accessibility service is enabled.

    Args:
        device: ADB device connection
        service_name: Full accessibility service name to check
        debug: Enable debug logging

    Returns:
        True if the accessibility service is enabled, False otherwise
    """
    a11y_services = device.shell("settings get secure enabled_accessibility_services")
    if service_name not in a11y_services:
        if debug:
            print(a11y_services)
        return False

    a11y_enabled = device.shell("settings get secure accessibility_enabled")
    if a11y_enabled != "1":
        if debug:
            print(a11y_enabled)
        return False

    return True


def ping_portal(device: AdbDevice, debug: bool = False):
    """
    Ping the Droidrun Portal to check if it is installed and accessible.
    """
    try:
        packages = device.list_packages()
    except Exception as e:
        raise Exception("Failed to list packages") from e

    if PORTAL_PACKAGE_NAME not in packages:
        if debug:
            print(packages)
        raise Exception("Portal is not installed on the device")

    if not check_portal_accessibility(device, debug=debug):
        device.shell("am start -a android.settings.ACCESSIBILITY_SETTINGS")
        raise Exception(
            "Droidrun Portal is not enabled as an accessibility service on the device"
        )


def ping_portal_content(device: AdbDevice, debug: bool = False):
    """
    Test Portal accessibility via content provider.

    Args:
        device: ADB device connection
        debug: Enable debug logging

    Raises:
        Exception: If Portal is not reachable via content provider
    """
    try:
        state = device.shell("content query --uri content://com.droidrun.portal/state")
        if "Row: 0 result=" not in state:
            raise Exception("Failed to get state from Droidrun Portal")
    except Exception as e:
        raise Exception("Droidrun Portal is not reachable") from e


def ping_portal_tcp(device: AdbDevice, debug: bool = False):
    """
    Test Portal accessibility via TCP mode.

    Args:
        device: ADB device connection
        debug: Enable debug logging

    Raises:
        Exception: If Portal is not reachable via TCP or port forwarding fails
    """
    try:
        AdbTools(serial=device.serial, use_tcp=True)
    except Exception as e:
        raise Exception("Failed to setup TCP forwarding") from e


def set_overlay_offset(device: AdbDevice, offset: int):
    """
    Set the overlay offset using the /overlay_offset portal content provider endpoint.
    """
    try:
        cmd = f'content insert --uri "content://com.droidrun.portal/overlay_offset" --bind offset:i:{offset}'
        device.shell(cmd)
    except Exception as e:
        raise Exception("Error setting overlay offset") from e


def toggle_overlay(device: AdbDevice, visible: bool):
    """toggle the overlay visibility.

    Args:
        device: Device to toggle the overlay on
        visible: Whether to show the overlay

    throws:
        Exception: If the overlay toggle fails
    """
    try:
        device.shell(
            f"am broadcast -a com.droidrun.portal.TOGGLE_OVERLAY --ez overlay_visible {'true' if visible else 'false'}"
        )
    except Exception as e:
        raise Exception("Failed to toggle overlay") from e


def setup_keyboard(device: AdbDevice):
    """
    Set up the DroidRun keyboard as the default input method.
    Simple setup that just switches to DroidRun keyboard without saving/restoring.

    throws:
        Exception: If the keyboard setup fails
    """
    try:
        device.shell("ime enable com.droidrun.portal/.DroidrunKeyboardIME")
        device.shell("ime set com.droidrun.portal/.DroidrunKeyboardIME")
    except Exception as e:
        raise Exception("Error setting up keyboard") from e


def disable_keyboard(
    device: AdbDevice, target_ime: str = "com.droidrun.portal/.DroidrunKeyboardIME"
):
    """
    Disable a specific IME (keyboard) and optionally switch to another.
    By default, disables the DroidRun keyboard.

    Args:
        target_ime: The IME package/activity to disable (default: DroidRun keyboard)

    Returns:
        bool: True if disabled successfully, False otherwise
    """
    try:
        device.shell(f"ime disable {target_ime}")
        return True
    except Exception as e:
        raise Exception("Error disabling keyboard") from e


def test():
    device = adb.device()
    ping_portal(device, debug=False)


if __name__ == "__main__":
    test()
