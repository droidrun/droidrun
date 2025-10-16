"""
Cloud-enabled AdbTools for on-demand Limbar devices.

This module provides a wrapper around AdbTools that automatically provisions
cloud Android devices from Limbar infrastructure via the DroidRun Device API.
"""

import atexit
import logging
import os
from typing import Optional

from droidrun.config_manager.config_manager import DeviceConfig
from droidrun.tools.adb import AdbTools
from droidrun.tools.droidrun_client import DroidRunClient

logger = logging.getLogger("droidrun-tools")


class CloudAdbTools(AdbTools):
    """
    AdbTools wrapper that provisions on-demand cloud Android devices.

    Extends AdbTools to work with Limbar cloud devices via DroidRun Device API.
    Handles full lifecycle: provision -> connect -> use -> terminate.
    Portal app accessed via content provider through ADB tunnel.
    """

    def __init__(
        self,
        device_config: DeviceConfig,
        app_opener_llm=None,
        text_manipulator_llm=None,
    ) -> None:
        """Initialize CloudAdbTools by provisioning a cloud device."""
        # Read API key from config or environment
        api_key = device_config.cloud_service_key or os.getenv("DROIDRUN_API_KEY")
        if not api_key:
            raise ValueError(
                "Cloud API key not found. Set 'cloud_service_key' in config.yaml "
                "or DROIDRUN_API_KEY environment variable"
            )

        # Create internal API client
        self.api_client = DroidRunClient(
            base_url=device_config.cloud_base_url,
            service_key=api_key,
        )
        self.device_id = None
        self.cloud_serial = None
        self._terminated = False

        # Provision the cloud device
        logger.info("Provisioning cloud device...")
        serial = self._provision_and_connect(
            apps=device_config.cloud_apps,
            files=device_config.cloud_files,
            country=device_config.cloud_country,
            timeout=device_config.cloud_provision_timeout,
        )

        logger.info(f"Cloud device ready: {serial}")

        # Initialize parent AdbTools with cloud serial
        super().__init__(
            serial=serial,
            use_tcp=False,
            app_opener_llm=app_opener_llm,
            text_manipulator_llm=text_manipulator_llm,
        )

        # Register cleanup on exit
        atexit.register(self._cleanup_on_exit)

    def _cleanup_on_exit(self):
        """Automatic cleanup registered with atexit."""
        if not self._terminated and self.device_id:
            try:
                logger.info("Auto-terminating cloud device on exit...")
                self.terminate()
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

    def _provision_and_connect(
        self,
        apps: Optional[list[str]] = None,
        files: Optional[list[str]] = None,
        country: str = "US",
        timeout: int = 600,
    ) -> str:
        """
        Provision cloud device and return ADB serial.

        This method handles the two-step provisioning process:
        1. POST /provision - Creates device instance
        2. GET /wait-for-prerequisites - Waits for device ready and ADB tunnel

        Args:
            apps: Apps to pre-install
            files: Files to pre-push
            country: Device country code
            timeout: Max wait time in seconds

        Returns:
            Cloud ADB serial (e.g., "cloud.droidrun.ai:12345")

        Raises:
            RuntimeError: If provisioning fails or times out
        """
        try:
            # Step 1: Provision device
            provision_response = self.api_client.provision(
                apps=apps, files=files, country=country
            )
            self.device_id = provision_response["id"]
            logger.info(f"Device provisioned: {self.device_id}")

            # Step 2: Wait for device ready
            logger.info("Waiting for device prerequisites...")
            device_data = self.api_client.wait_for_prerequisites(
                device_id=self.device_id, timeout=timeout
            )

            # Step 3: Extract serial
            self.cloud_serial = device_data["serial"]
            if not self.cloud_serial:
                raise RuntimeError("Device ready but serial is empty")

            logger.info(f"Device ready with serial: {self.cloud_serial}")
            return self.cloud_serial

        except Exception as e:
            logger.error(f"Provisioning failed: {e}")
            # Cleanup attempt if we got a device ID
            if self.device_id and not self._terminated:
                try:
                    logger.warning(f"Attempting cleanup for {self.device_id}")
                    self.api_client.terminate(self.device_id)
                    self._terminated = True
                except Exception as cleanup_err:
                    logger.error(f"Cleanup failed: {cleanup_err}")
            raise RuntimeError(f"Failed to provision cloud device: {e}") from e

    def terminate(self):
        """
        Terminate the cloud device and stop billing.

        IMPORTANT: Always call this when done to avoid unnecessary charges!

        This method is idempotent - safe to call multiple times.
        """
        if self._terminated:
            logger.debug("Device already terminated")
            return

        if not self.device_id:
            logger.warning("No device_id to terminate")
            return

        try:
            logger.info(f"Terminating cloud device: {self.device_id}")
            self.api_client.terminate(self.device_id)
            logger.info(f"Device terminated: {self.device_id}")
            self._terminated = True
        except Exception as e:
            logger.error(f"Termination failed: {e}")
            raise
        finally:
            # Clear references even if termination failed
            # (prevent further usage attempts)
            self.device_id = None
            self.cloud_serial = None

    def __enter__(self):
        """Context manager entry - returns self for use in with statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - ensures device cleanup.

        Automatically terminates device even if an exception occurred.
        Does not suppress exceptions - they will be re-raised after cleanup.
        """
        try:
            if not self._terminated:
                self.terminate()
        except Exception as e:
            logger.error(f"Error during context manager cleanup: {e}")
            # Don't suppress original exception
            if exc_type is None:
                raise
        return False  # Don't suppress exceptions

    def __repr__(self):
        status = "terminated" if self._terminated else "active"
        return f"CloudAdbTools(device_id={self.device_id}, serial={self.cloud_serial}, status={status})"
