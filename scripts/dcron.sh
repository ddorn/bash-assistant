#!/bin/bash

# This script is used to start the dcron python script.
# It is called every minute by systemd.

# Get the directory of the script
script_directory=$(dirname "$(realpath "$0")")
directory=$(dirname "$script_directory")

if ! "$directory"/.venv/bin/python "$directory"/src/dcron.py "$@" ; then
    code=$?
    notify-send "Failed to run dcron.py !!" --urgency=critical -a dcron
    exit $code
fi
