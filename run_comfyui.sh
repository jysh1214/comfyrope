#!/usr/bin/env bash
# Start ComfyUI from its own venv. Extra args pass through, e.g.:
#   ./run_comfyui.sh                       # http://127.0.0.1:8188
#   ./run_comfyui.sh --listen --port 8188  # reachable from the LAN
#   ./run_comfyui.sh --reserve-vram 2      # leave 2 GB for the desktop if the display stutters
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/ComfyUI"
exec .venv/bin/python main.py "$@"
