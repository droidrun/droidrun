"""
DroidRun Device API Client - Python client for provisioning cloud Android devices.

This module provides a simple interface to the DroidRun Device API for managing
on-demand Limbar Android device instances.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("droidrun-tools")


class DroidRunClient:
    """
    Client for DroidRun Device API.

    Handles provisioning, waiting for device readiness, and termination of
    cloud Android devices from Limbar infrastructure.

    Example:
        client = DroidRunClient(
            base_url="https://device-api.droidrun.ai",
            service_key="your-service-key"
        )

        # Provision a device
        response = client.provision(apps=["com.example.app"], country="US")
        device_id = response["id"]

        # Wait for device to be ready
        device = client.wait_for_prerequisites(device_id)
        serial = device["serial"]  # e.g., "cloud.droidrun.ai:12345"

        # Use device with ADB...

        # Clean up
        client.terminate(device_id)
    """

    def __init__(
        self,
        base_url: str,
        service_key: str,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize DroidRun API client.

        Args:
            base_url: Base URL of DroidRun Device API (e.g., "https://device-api.droidrun.ai")
            service_key: Service key for authentication
            timeout: Request timeout in seconds (default: 30)
            max_retries: Max retry attempts for failed requests (default: 3)
        """
        self.base_url = base_url.rstrip("/")
        self.service_key = service_key
        self.timeout = timeout
        self.max_retries = max_retries

        # Set up session with auth header
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": service_key,
                "Content-Type": "application/json",
            }
        )

    def provision(
        self,
        apps: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        country: str = "US",
    ) -> Dict[str, Any]:
        """
        Provision a new cloud Android device.

        This creates a Limbar device instance but does NOT wait for it to be ready.
        Use wait_for_prerequisites() to wait for device readiness.

        Args:
            apps: List of app package names or APK URLs to pre-install
            files: List of file URLs to pre-push to device
            country: ISO country code for device location (default: "US")

        Returns:
            Dictionary with device info:
            {
                "id": "device-uuid",
                "streamUrl": "wss://...",  # WebSocket URL for screen streaming
                "serial": "",  # Empty until prerequisites are met
                "token": "auth-token"
            }

        Raises:
            requests.HTTPError: If API request fails
            RuntimeError: If provisioning fails after retries
        """
        url = f"{self.base_url}/provision"
        payload = {
            "apps": apps or [],
            "files": files or [],
            "country": country,
        }

        logger.info(f"Provisioning device with country={country}, apps={apps}")

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()

                data = response.json()
                logger.info(f"Device provisioned: {data.get('id')}")
                return data

            except requests.RequestException as e:
                logger.warning(f"Provision attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Provisioning failed after {self.max_retries} attempts") from e
                time.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError("Provisioning failed")

    def wait_for_prerequisites(
        self,
        device_id: str,
        timeout: int = 600,
        poll_interval: int = 5,
    ) -> Dict[str, Any]:
        """
        Wait for device to be ready and get ADB serial.

        This is a long-polling endpoint that waits up to 10 minutes on the server side.
        The device is ready when the serial field is populated.

        Args:
            device_id: Device ID from provision() response
            timeout: Max seconds to wait for device (default: 600 = 10 min)
            poll_interval: Seconds between retry attempts if request fails (default: 5)

        Returns:
            Dictionary with device info:
            {
                "id": "device-uuid",
                "streamUrl": "wss://...",
                "serial": "cloud.droidrun.ai:12345",  # ADB connection string
                "token": "auth-token"
            }

        Raises:
            TimeoutError: If device not ready within timeout
            requests.HTTPError: If API request fails
            RuntimeError: If device readiness check fails
        """
        url = f"{self.base_url}/wait-for-prerequisites/{device_id}"
        start_time = time.time()

        logger.info(f"Waiting for device prerequisites: {device_id}")

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"Device {device_id} not ready after {timeout} seconds"
                )

            try:
                # Server-side timeout is 10 minutes, so we set client timeout higher
                response = self.session.get(url, timeout=self.timeout + 600)
                response.raise_for_status()

                data = response.json()

                # Check if device is ready (serial is populated)
                if data.get("serial"):
                    logger.info(
                        f"Device ready: {device_id} -> serial={data['serial']}"
                    )
                    return data

                # Device provisioned but serial still empty - should not happen
                # with long-polling, but handle gracefully
                logger.warning(
                    f"Device {device_id} returned but serial empty, retrying..."
                )
                time.sleep(poll_interval)

            except requests.Timeout:
                # Long-polling timeout is expected, retry immediately
                logger.debug(f"Long-polling timeout, retrying for {device_id}")
                continue

            except requests.RequestException as e:
                logger.warning(f"Wait request failed: {e}, retrying in {poll_interval}s")
                time.sleep(poll_interval)

        raise RuntimeError(f"Failed to get device prerequisites for {device_id}")

    def terminate(self, device_id: str) -> None:
        """
        Terminate a cloud device and stop billing.

        IMPORTANT: Always call this when done with a device to avoid unnecessary charges!

        Args:
            device_id: Device ID to terminate

        Raises:
            requests.HTTPError: If API request fails
            RuntimeError: If termination fails after retries
        """
        url = f"{self.base_url}/terminate/{device_id}"

        logger.info(f"Terminating device: {device_id}")

        for attempt in range(self.max_retries):
            try:
                response = self.session.delete(url, timeout=self.timeout)
                response.raise_for_status()

                logger.info(f"Device terminated: {device_id}")
                return

            except requests.RequestException as e:
                logger.warning(f"Terminate attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"Termination failed after {self.max_retries} attempts. "
                        f"Device {device_id} may still be running!"
                    ) from e
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Failed to terminate device {device_id}")

    def __repr__(self):
        return f"DroidRunClient(base_url='{self.base_url}')"
