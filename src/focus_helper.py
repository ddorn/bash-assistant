import subprocess
import time
from pathlib import Path
from datetime import datetime, time as dt_time

# --- Configuration ---
USAGE_MINUTES_THRESHOLD = 15  # Lock screen after X minutes of activity
CHECK_INTERVAL_SECONDS = 60  # How often the script checks for activity
# Notify at 5, 3, and 1 minute(s) before the screen locks.
NOTIFICATION_MINUTES_BEFORE_LOCK = {1, 3, 5}
# IMPORTANT: This value is passed to the swayidle daemon.
IDLE_TIMEOUT_SECONDS = 60  # Consider user idle no activity for X seconds
IDLE_MARKER_PATH = Path("/tmp/focus_idle_marker")
# The script will look for your image here.
# You can change this to an absolute path if you prefer.
LOCK_IMAGE_PATH = Path(__file__).parent.parent / "data/deepbreaths.jpg"
DAEMON_SCRIPT_PATH = Path(__file__).parent.parent / "scripts/run_focus_daemon.sh"

# This is dangerous! If set to a high value
# the computer will not be usable for the given amount of time.
# Keeps the screen locked for X seconds, re-locking it if unlocked.
# If set to 0, locks only once.
LOCK_DURATION_SECONDS = 30  # ! Read the comment above.

# Time-based override configuration
# Set to None to disable time-based override
OVERRIDE_START_TIME = dt_time(9, 0)  # 9:00 AM
OVERRIDE_END_TIME = dt_time(20, 0)  # 8:00 PM (20:00)
OVERRIDE_END_DATE = datetime(2025, 9, 23)  # Override ends on this date (inclusive)
# ---


def is_override_active() -> bool:
    """Check if the time-based override is currently active."""
    if OVERRIDE_START_TIME is None or OVERRIDE_END_TIME is None:
        return False

    now = datetime.now()

    # Check if we're past the override end date
    if now.date() > OVERRIDE_END_DATE.date():
        return False

    # Check if current time is within the override window
    current_time = now.time()
    return OVERRIDE_START_TIME <= current_time <= OVERRIDE_END_TIME


def send_notification(message: str):
    """Sends a desktop notification using notify-send."""
    summary = "Focus Helper"
    try:
        # We use a summary to ensure notifications are grouped correctly.
        subprocess.run(["notify-send", summary, message], check=True, capture_output=True)
        print(f"Sent notification: {message}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Could not send notification. Is 'notify-send' installed?")


def start_daemon():
    """Starts the swayidle daemon using the shell script."""
    print("Attempting to start the focus daemon...")
    # This will raise an exception and stop the script if the daemon fails to start,
    # which is the desired "fail fast" behavior.
    # We do NOT capture output, as that would cause this call to hang
    # waiting for the backgrounded swayidle process to exit.
    subprocess.run(
        [str(DAEMON_SCRIPT_PATH), str(IDLE_MARKER_PATH), str(IDLE_TIMEOUT_SECONDS)], check=True
    )
    print("Daemon start script executed successfully.")


def main():
    """Main loop to track usage and lock the screen."""
    # Check if override is active
    if is_override_active():
        print("Focus helper is currently disabled due to time-based override.")
        print(
            f"Override active: {OVERRIDE_START_TIME} - {OVERRIDE_END_TIME} until {OVERRIDE_END_DATE.date()}"
        )
        return

    start_daemon()
    print("Starting focus helper script...")

    active_minutes = 0
    while True:
        # The swayidle daemon creates the marker file when the user is idle.
        # If the file doesn't exist, it means the user was active.
        if not IDLE_MARKER_PATH.exists():
            active_minutes += CHECK_INTERVAL_SECONDS / 60
            print(f"User is active. Total active time: {active_minutes:.2f} minutes.")

            minutes_remaining = USAGE_MINUTES_THRESHOLD - active_minutes
            if minutes_remaining in NOTIFICATION_MINUTES_BEFORE_LOCK:
                message = f"Short break in {minutes_remaining} minute(s)."
                send_notification(message)
        else:
            print(f"User is idle. Total active time remains {active_minutes:.2f} minutes.")

        if active_minutes >= USAGE_MINUTES_THRESHOLD:
            print(f"Usage threshold of {USAGE_MINUTES_THRESHOLD} minutes reached. Locking screen.")
            keep_screen_locked_for(LOCK_DURATION_SECONDS)
            # Reset the counter after a break
            active_minutes = 0
            # Also, ensure the idle marker is gone so we don't count the break
            # as idle time that carries over.
            IDLE_MARKER_PATH.unlink(missing_ok=True)

        time.sleep(CHECK_INTERVAL_SECONDS)


def lock_screen():
    command = ["swaylock"]
    if LOCK_IMAGE_PATH.exists():
        command.extend(["-i", str(LOCK_IMAGE_PATH)])
    else:
        print(f"Warning: Lock image not found at {LOCK_IMAGE_PATH}. Locking without image.")

    try:
        subprocess.run(command, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Could not lock screen. Is 'swaylock' installed?")


def keep_screen_locked_for(seconds: int):
    """Keep the screen locked for a given number of seconds. Re-locking if unlocked."""
    lock_screen()
    start = time.time()
    while time.time() - start < seconds:
        lock_screen()
        time.sleep(1)


if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        error_message = f"Script crashed: {e}"
        print(error_message)
        send_notification(error_message)
        raise
