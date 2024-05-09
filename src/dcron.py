from datetime import datetime
from pathlib import Path
import re
import subprocess
from textwrap import dedent
from time import time
import json
import traceback
import utils
import typer

CALLS_FILE = utils.DATA / "dcron_calls.json"


class Dcron:

    def __init__(self):
        try:
            self.calls = json.loads(CALLS_FILE.read_text())
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
                    errors.append((name, e))
                    print(f"❌ {name}: {e}")
                    traceback.print_exc()

                self.calls[name] = now

        CALLS_FILE.write_text(json.dumps(self.calls))

        if errors:
            raise ExceptionGroup(errors)


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
    status = re.search(r"Charging|Discharging", info).group(0)
    estimated = re.search(r"\d+:\d+:\d+", info).group(0)

    uptime_since = subprocess.check_output(["uptime", "-s"], text=True).strip()
    uptime = datetime.now() - datetime.strptime(uptime_since, "%Y-%m-%d %H:%M:%S")
    uptime = int(uptime.total_seconds())

    # Append log to the file
    with logs.open("a") as f:
        f.write(f"{now},{level},{status},{estimated},{uptime}\n")


@every(minutes=7)
def go_to_sleep(
    notify: str = "20:00",
    shutdown: str = "22:00",
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
    now = time() % 86400

    if not is_in_order(notify, shutdown, end):
        raise ValueError("Times should be in clockwise order.")

    if is_in_order(notify, now, shutdown):
        minutes = (shutdown - now) // 60
        utils.notify("Go to sleep!", f"Shutting down in {minutes} minutes.")
    elif is_in_order(shutdown, now, end):
        if snooze_file.exists():
            snooze_file.unlink()
            utils.notify("You do you.", "Remember you are happier when you sleep enough 😘🧡💜")
        else:
            commands = f"""
            playerctl pause || true
            # Max volume & unmute
            pactl set-sink-volume @DEFAULT_SINK@ 100%
            pactl set-sink-mute @DEFAULT_SINK@ 0
            # Play music
            cvlc --play-and-exit --no-loop --no-volume-save "{good_night_music}"
            # Hibernate
            systemctl hibernate
            """
            subprocess.run(dedent(commands), shell=True)
    else:
        pass


@every(minutes=1)
def screenshot():
    pass


@every(minutes=45)
def random_background():
    pass


# %%
if __name__ == "__main__":
    dcron.app()
