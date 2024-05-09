from datetime import datetime
from pathlib import Path
import re
import subprocess
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


@every(minutes=1)
def screenshot():
    pass


@every(minutes=7)
def go_to_sleep():
    pass


@every(minutes=45)
def random_background():
    pass


# %%
if __name__ == "__main__":
    dcron.app()
