import requests
import tempfile
import os
import contextlib
from adbutils import adb, AdbDevice

REPO = "droidrun/droidrun-portal"
ASSET_NAME = "droidrun-portal"
GITHUB_API_HOSTS = ["https://api.github.com", "https://ungh.cc"]

PORTAL_PACKAGE_NAME = "com.droidrun.portal"
A11Y_SERVICE_NAME = (
    f"{PORTAL_PACKAGE_NAME}/com.droidrun.portal.DroidrunAccessibilityService"
)


def get_latest_release_assets(debug: bool = False):
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
    assets = get_latest_release_assets(debug)

    asset_url = None
    for asset in assets:
        if (
            "browser_download_url" in asset
            and "name" in asset
            and asset["name"].startswith(ASSET_NAME)
        ):
            asset_url = asset["browser_download_url"]
            break
        elif "downloadUrl" in asset and os.path.basename(
            asset["downloadUrl"]
        ).startswith(ASSET_NAME):
            asset_url = asset["downloadUrl"]
            break
        else:
            if debug:
                print(asset)

    if not asset_url:
        raise Exception(f"Asset named '{ASSET_NAME}' not found in the latest release.")

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


def enable_portal_accessibility(device: AdbDevice):
    device.shell(
        f"settings put secure enabled_accessibility_services {A11Y_SERVICE_NAME}"
    )
    device.shell("settings put secure accessibility_enabled 1")


def check_portal_accessibility(device: AdbDevice, debug: bool = False) -> bool:
    a11y_services = device.shell(
        "settings get secure enabled_accessibility_services"
    )
    if not A11Y_SERVICE_NAME in a11y_services:
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
        raise Exception(f"Failed to list packages: {e}")

    if not PORTAL_PACKAGE_NAME in packages:
        if debug:
            print(packages)
        raise Exception("Portal is not installed on the device")

    if not check_portal_accessibility(device, debug):
        device.shell("am start -a android.settings.ACCESSIBILITY_SETTINGS")
        raise Exception(
            "Droidrun Portal is not enabled as an accessibility service on the device"
        )

    try:
        state = device.shell(
            "content query --uri content://com.droidrun.portal/state"
        )
        if not "Row: 0 result=" in state:
            raise Exception("Failed to get state from Droidrun Portal")

    except Exception as e:
        raise Exception(f"Droidrun Portal is not reachable: {e}")


def test():
    device = adb.device()
    ping_portal(device, debug=False)


if __name__ == "__main__":
    test()
