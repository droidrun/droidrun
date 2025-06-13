#!/bin/bash

# Start Emulator
#============================================
./start_emu_headless.sh && \
adb root && \
droidrun setup --path /opt/shared/droidrun-portal.apk && \
cd /opt/shared/droidrun && \
python -m eval.android_world_bench --perform-emulator-setup "$@"
