#!/bin/sh

REPO="droidrun/droidrun-portal"
ASSET_NAME="droidrun-portal"

# Get latest release data
latest_release=$(curl -s "https://api.github.com/repos/$REPO/releases/latest")

# Extract download URL for the .apk file
apk_url=$(echo "$latest_release" | grep "browser_download_url" | grep "$ASSET_NAME.*\.apk" | cut -d '"' -f 4)

# Download the APK
if [ -n "$apk_url" ]; then
  echo "Downloading: $apk_url"
  curl -L -o "${ASSET_NAME}.apk" "$apk_url"
else
  echo "APK asset not found in the latest release."
  exit 1
fi