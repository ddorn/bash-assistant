#!/bin/bash
set -e
directory=/home/diego/ai/bash-ai
$directory/.venv/bin/python $directory/bai/bai.py "$@"
