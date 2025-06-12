#!/bin/bash

# Start Emulator
#============================================
./start_emu_headless.sh && \
adb root && \
cd /opt/shared/droidrun && \
python -m eval.android_world_bench --task-ids "$@" --perform-emulator-setup
