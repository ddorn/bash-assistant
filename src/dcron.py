from datetime import datetime
from pathlib import Path
import random
import re
import subprocess
from textwrap import dedent
from time import time, sleep
import json
import traceback
import utils
import typer

CALLS_FILE = utils.DATA / "dcron_calls.json"


class Dcron:

    def __init__(self):
        try:
            self.calls = json.loads(CALLS_FILE.read_text() or "{}")
        except FileNotFoundError:
            self.calls = {}

        self.registered = {}
        self.app = typer.Typer(
            callback=self.run,
            invoke_without_command=True,
            #    pretty_exceptions_enable=False,
            pretty_exceptions_show_locals=False,
            add_completion=False,
        )

    def every(self, minutes=1):
        """Call the decorated function roughly every `minutes` minutes.

        It is guaranteed that at least `minutes - 10s` will pass between two consecutive calls.
        """

        def wrapper(func):
            assert func.__name__ not in self.registered, f"Function {func.__name__} already exists."
            self.registered[func.__name__] = {
                "func": func,
                "minutes": minutes,
            }
            self.app.command()(func)
            return func

        return wrapper

    def run(self, ctx: typer.Context):
        """Called every minute by the cron job, runs every registered function if needed."""

        # Don't run if there's a typer command
        if ctx.invoked_subcommand:
            return

        errors = []

        now = time()
        for name, info in self.registered.items():
            last_call = self.calls.get(name, 0)
            minutes = info["minutes"]
            if now - last_call > minutes * 60 - 10:
                print(f"🚀 {name}")
                try:
                    info["func"]()
                except Exception as e:
                    errors.append(e)
                    print(f"❌ {name}: {e}")
                    traceback.print_exc()

                self.calls[name] = now
            else:
                print(f"⏳ {name} - {minutes * 60 - 10 - (now - last_call):.0f}s left")

        CALLS_FILE.write_text(json.dumps(self.calls))

        if errors:
            raise ExceptionGroup("Failed to run one or more dcron", errors)


# %%

dcron = Dcron()
every = dcron.every


@every(minutes=1)
def log_battery():
    logs = Path("~/logs/battery.csv").expanduser()
    logs.parent.mkdir(exist_ok=True)

    if not logs.exists():
        logs.write_text("DateTime,Battery Level,Battery Status,Estimated Time Remaining,Uptime")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info = subprocess.check_output(["acpi", "-b"], text=True).strip()
    level = re.search(r"(\d+)%", info).group(1)
    status = re.search(r"Charging|Discharging|Not charging", info).group(0)
    estimated = re.search(r"\d+:\d+:\d+", info)
    estimated = estimated.group(0) if estimated else ""

    uptime_since = subprocess.check_output(["uptime", "-s"], text=True).strip()
    uptime = datetime.now() - datetime.strptime(uptime_since, "%Y-%m-%d %H:%M:%S")
    uptime = int(uptime.total_seconds())

    # Append log to the file
    with logs.open("a") as f:
        f.write(f"{now},{level},{status},{estimated},{uptime}\n")


@every(minutes=7)
def go_to_sleep(
    notify: str = "20:00",
    shutdown: str = "22:30",
    end: str = "07:00",
    snooze_file: Path = Path("/tmp/no_sleep_today"),
    good_night_music: Path = utils.DATA / "goodnight.mp3",
):
    def is_in_order(a, b, c):
        """Check if a, b, c are 3 times in clockwise order on the clock."""
        return a < b < c or c < a < b or b < c < a

    def time_to_seconds(t):
        h, m = map(int, t.split(":"))
        return h * 3600 + m * 60

    notify = time_to_seconds(notify)
    shutdown = time_to_seconds(shutdown)
    end = time_to_seconds(end)
    now = datetime.now().hour * 3600 + datetime.now().minute * 60

    if not is_in_order(notify, shutdown, end):
        raise ValueError("Times should be in clockwise order.")

    if is_in_order(notify, now, shutdown):
        minutes = (shutdown - now) // 60
        utils.notify("Go to sleep!", f"Shutting down in {minutes} minutes.")
    elif is_in_order(shutdown, now, end):
        if snooze_file.exists() and (
            snooze_file.read_text().strip() == "SKIP" or random.random() < 0.8
        ):
            snooze_file.unlink()
            utils.notify(
                "You do you.",
                "Remember you are happier when you sleep enough 😘🧡💜",
                urgency="critical",
            )
        else:
            commands = f"""
            playerctl pause || true

            # Unmute
            pactl set-sink-mute @DEFAULT_SINK@ 0
            # Increase volume by 5%
            pactl set-sink-volume @DEFAULT_SINK@ +5%
            # pactl set-sink-volume @DEFAULT_SINK@ 100%

            # Play music
            cvlc --play-and-exit --no-loop --no-volume-save "{good_night_music}"

            # Hibernate
            # systemctl hibernate
            shutdown
            """
            subprocess.run(dedent(commands), shell=True)
    else:
        pass


# @every(minutes=1)
def no_code_at_home():
    home_networks = [
        # "Freebox-23D444",
    ]

    forbidden_processes = ["code", "nvim"]

    # Check if I'm at home, on home network. If so kill vscode
    wifi_network = subprocess.check_output(
        "nmcli connection show --active | grep wifi | awk '{print $1}'", shell=True, text=True
    ).strip()

    if wifi_network not in home_networks:
        return

    # if none running, abort
    running = subprocess.check_output("ps -A", shell=True, text=True)
    if not any(p in running for p in forbidden_processes):
        return

    # Warn and kill vscode & nvim in a minute
    utils.notify("No coding at home!", "You're at home, enjoy your time!", urgency="critical")
    # lock screen in red
    subprocess.run("swaylock -c FF0000 --config /none", shell=True)
    subprocess.run("sleep 50 && killall code nvim", shell=True)


# @every(minutes=1)
def keep_pucoti():
    # Launch pucoti if not running
    if subprocess.run("pgrep pucoti", shell=True).returncode:
        utils.notify("Pucoti not running", "Starting it now")
        # Run pucoti, detached from this script
        proc = subprocess.Popen("uv tool run pucoti", shell=True)
        # Wait for it to start
        sleep(5)
        if proc.poll():
            utils.notify("Pucoti failed to start", "Check logs")


# @every(minutes=1)
def screenshot():
    SCREENSHOTS = utils.DATA / "screenshots"
    SCREENSHOTS.mkdir(exist_ok=True)

    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.webp")
    path = SCREENSHOTS / filename

    subprocess.check_call(["grim", str(path)])


# @every(minutes=45)
def random_background():
    pass


# %%
if __name__ == "__main__":
    dcron.app()
