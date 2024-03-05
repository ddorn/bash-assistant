#!/bin/bash
set -e
cd /home/diego/ai/bash-ai
poetry run bai/bai.py "$@"
