#!/bin/sh
#
# This script launches the swayidle daemon that our focus_helper.py script depends on.
# It is called by the Python script itself.
#

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <marker_path> <timeout_seconds>"
    exit 1
fi

IDLE_MARKER="$1"
IDLE_TIMEOUT="$2"

# We use pgrep to check if a swayidle process with our specific marker file
# is already running. This prevents launching duplicate daemons.
if pgrep -f "swayidle.*touch ${IDLE_MARKER}" > /dev/null; then
    echo "Focus daemon is already running."
    exit 0
fi

echo "[DEAMON] Starting focus daemon..."

# -w: Don't exit after the first idle event.
# timeout <seconds> <command>: Run command when idle for <seconds>.
# resume <command>: Run command when activity resumes.
swayidle -w \
    timeout "${IDLE_TIMEOUT}" "touch '${IDLE_MARKER}'" \
    resume "rm -f '${IDLE_MARKER}'" &

echo "[DEAMON] Focus daemon started in the background."
