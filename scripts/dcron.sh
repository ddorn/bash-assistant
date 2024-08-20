#!/bin/bash

# This script is used to start the dcron python script.
# It is called every minute by systemd.

# Get the directory of the script
script_directory=$(dirname "$(realpath "$0")")
directory=$(dirname "$script_directory")

# Run the dcron.py script and capture the output
output=$("$directory"/.venv/bin/python "$directory"/src/dcron.py "$@" 2>&1)
code=$?

if [ $code -ne 0 ]; then
    notify-send "Failed to run dcron.py !!" "$output" --urgency=critical -a dcron
    exit $code
fi
