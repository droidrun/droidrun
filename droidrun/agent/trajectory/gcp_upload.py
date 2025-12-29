"""
GCP upload utilities for DroidRun trajectories.

This module provides functionality to upload trajectory data to Google Cloud Storage,
following Nova's logging pattern with path structure: bucket/product_id/test_run_id/tcue_id/
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger("droidrun")


class GCPStorageWrapper:
    """Wrapper for Google Cloud Storage operations."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy-load the storage client."""
        if self._client is None:
            try:
                from google.cloud import storage
                self._client = storage.Client()
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required for GCP uploads. "
                    "Install it with: pip install google-cloud-storage"
                )
        return self._client

    def get_bucket(self, bucket_name: str):
        """Get a bucket object by name."""
        return self.client.bucket(bucket_name)

    def upload_file(
        self,
        local_path: str,
        bucket_name: str,
        remote_path: str,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload a file to GCS.

        Args:
            local_path: Path to local file
            bucket_name: Name of the GCS bucket
            remote_path: Path in the bucket
            content_type: MIME type (auto-detected if None)

        Returns:
            GCS URI of the uploaded file
        """
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")

        # Auto-detect content type based on extension
        if content_type is None:
            ext = os.path.splitext(local_path)[1].lower()
            content_type = {
                ".json": "application/json",
                ".png": "image/png",
                ".gif": "image/gif",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".txt": "text/plain",
                ".mp4": "video/mp4",
            }.get(ext, "application/octet-stream")

        bucket = self.get_bucket(bucket_name)
        blob = bucket.blob(remote_path)

        # Use binary mode for images and videos, text mode for others
        if content_type.startswith("image/") or content_type.startswith("video/"):
            with open(local_path, "rb") as f:
                blob.upload_from_file(f, content_type=content_type)
        else:
            with open(local_path, "r", encoding="utf-8") as f:
                blob.upload_from_string(f.read(), content_type=content_type)

        gcs_uri = f"gs://{bucket_name}/{remote_path}"
        logger.debug(f"Uploaded {local_path} to {gcs_uri}")
        return gcs_uri

    def upload_directory(
        self,
        local_dir: str,
        bucket_name: str,
        remote_prefix: str,
        file_extensions: Optional[list] = None,
    ) -> list:
        """
        Upload all files from a directory to GCS.

        Args:
            local_dir: Path to local directory
            bucket_name: Name of the GCS bucket
            remote_prefix: Prefix path in the bucket
            file_extensions: List of extensions to include (e.g., ['.json', '.png'])
                           If None, uploads all files

        Returns:
            List of GCS URIs of uploaded files
        """
        local_path = Path(local_dir)
        if not local_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {local_dir}")

        uploaded = []

        for item in local_path.rglob("*"):
            if not item.is_file():
                continue

            # Filter by extension if specified
            if file_extensions and item.suffix.lower() not in file_extensions:
                continue

            # Calculate relative path for remote
            rel_path = item.relative_to(local_path)
            remote_path = f"{remote_prefix}/{rel_path}"

            try:
                uri = self.upload_file(str(item), bucket_name, remote_path)
                uploaded.append(uri)
            except Exception as e:
                logger.warning(f"Failed to upload {item}: {e}")

        return uploaded


def upload_trajectory_to_gcp(
    trajectory_folder: str,
    bucket_name: str,
    product_id: str,
    test_run_id: str,
    tcue_id: str,
    cleanup_local: bool = True,
    execution_log: Optional[str] = None,
) -> dict:
    """
    Upload a trajectory folder to GCP with Nova-style path structure.

    Path structure: bucket/product_id/test_run_id/tcue_id/
        - trajectory.json
        - macro.json
        - detailed_log.log (execution output)
        - screenshots/
            - 0000.png, 0001.png, ...
            - trajectory.gif
        - ui_states/
            - 0000.json, 0001.json, ...

    Args:
        trajectory_folder: Path to local trajectory folder
        bucket_name: Name of the GCS bucket
        product_id: Product identifier
        test_run_id: Test run identifier
        tcue_id: Test case under execution identifier
        cleanup_local: If True, delete local trajectory folder after successful upload
        execution_log: Optional string containing the execution log output to upload as detailed_log.log

    Returns:
        Dictionary with upload results:
        {
            "success": bool,
            "gcp_base_path": str,
            "uploaded_files": list,
            "errors": list,
            "local_deleted": bool
        }
    """
    result = {
        "success": False,
        "gcp_base_path": "",
        "uploaded_files": [],
        "errors": [],
        "local_deleted": False,
    }

    try:
        storage = GCPStorageWrapper()
        folder_path = Path(trajectory_folder)

        if not folder_path.is_dir():
            result["errors"].append(f"Trajectory folder not found: {trajectory_folder}")
            return result

        # Build GCP path: product_id/test_run_id/tcue_id
        gcp_prefix = f"{product_id}/{test_run_id}/{tcue_id}"
        result["gcp_base_path"] = f"gs://{bucket_name}/{gcp_prefix}"

        # Upload trajectory.json
        trajectory_json = folder_path / "trajectory.json"
        if trajectory_json.exists():
            try:
                uri = storage.upload_file(
                    str(trajectory_json),
                    bucket_name,
                    f"{gcp_prefix}/trajectory.json",
                )
                result["uploaded_files"].append(uri)
            except Exception as e:
                result["errors"].append(f"Failed to upload trajectory.json: {e}")

        # Upload macro.json
        macro_json = folder_path / "macro.json"
        if macro_json.exists():
            try:
                uri = storage.upload_file(
                    str(macro_json),
                    bucket_name,
                    f"{gcp_prefix}/macro.json",
                )
                result["uploaded_files"].append(uri)
            except Exception as e:
                result["errors"].append(f"Failed to upload macro.json: {e}")

        # Upload execution log as detailed_log.log
        if execution_log:
            try:
                bucket = storage.get_bucket(bucket_name)
                blob = bucket.blob(f"{gcp_prefix}/detailed_log.log")
                blob.upload_from_string(execution_log, content_type="text/plain")
                uri = f"gs://{bucket_name}/{gcp_prefix}/detailed_log.log"
                result["uploaded_files"].append(uri)
                logger.debug(f"Uploaded execution log to {uri}")
            except Exception as e:
                result["errors"].append(f"Failed to upload detailed_log.log: {e}")

        # Upload screenshots folder
        screenshots_folder = folder_path / "screenshots"
        if screenshots_folder.is_dir():
            try:
                uris = storage.upload_directory(
                    str(screenshots_folder),
                    bucket_name,
                    f"{gcp_prefix}/screenshots",
                    file_extensions=[".png", ".gif", ".mp4"],
                )
                result["uploaded_files"].extend(uris)
            except Exception as e:
                result["errors"].append(f"Failed to upload screenshots: {e}")

        # Upload ui_states folder
        ui_states_folder = folder_path / "ui_states"
        if ui_states_folder.is_dir():
            try:
                uris = storage.upload_directory(
                    str(ui_states_folder),
                    bucket_name,
                    f"{gcp_prefix}/ui_states",
                    file_extensions=[".json"],
                )
                result["uploaded_files"].extend(uris)
            except Exception as e:
                result["errors"].append(f"Failed to upload ui_states: {e}")

        result["success"] = len(result["errors"]) == 0

        if result["success"]:
            logger.info(
                f"✅ Uploaded trajectory to {result['gcp_base_path']} "
                f"({len(result['uploaded_files'])} files)"
            )

            # Delete local trajectory folder after successful upload
            if cleanup_local:
                try:
                    shutil.rmtree(folder_path)
                    result["local_deleted"] = True
                    logger.debug(f"🗑️ Deleted local trajectory folder: {trajectory_folder}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to delete local trajectory folder: {e}")
        else:
            logger.warning(
                f"⚠️ Partial upload to {result['gcp_base_path']}: "
                f"{len(result['uploaded_files'])} files, {len(result['errors'])} errors"
            )

        return result

    except ImportError as e:
        result["errors"].append(str(e))
        logger.error(f"❌ GCP upload failed: {e}")
        return result
    except Exception as e:
        result["errors"].append(str(e))
        logger.error(f"❌ GCP upload failed: {e}")
        return result
