#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m poc_automation demo --workspace .tmp/demo --iterations "${1:-2}"
