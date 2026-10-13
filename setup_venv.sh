#!/usr/bin/env bash
# Create .venv next to this script with every package img2img.py and download_models.py need, so they also run as
#   source .venv/bin/activate && python img2img.py ...
# The package lists live in each script's "# /// script" header; uv reads them from there, so this stays in sync.
# Re-run after a script gains a package. To rebuild from scratch: rm -rf .venv, then run this again.
# ComfyUI has its own venv (ComfyUI/.venv), managed by install_comfyui.py.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
PYTHON_VERSION=3.12
SCRIPTS=(img2img.py download_models.py)

command -v uv >/dev/null || { echo "error: uv is not on PATH (https://docs.astral.sh/uv/)" >&2; exit 1; }

if [ -x .venv/bin/python ]; then
    echo "[venv] keep $PWD/.venv ($(.venv/bin/python -V))"
else
    echo "[venv] create $PWD/.venv with Python $PYTHON_VERSION"
    uv venv --python "$PYTHON_VERSION" .venv
fi

echo "[packages] from the headers of ${SCRIPTS[*]}"
requirements=()
for s in "${SCRIPTS[@]}"; do requirements+=(-r "$s"); done
uv pip install --python .venv/bin/python "${requirements[@]}"

echo "[check] importing every package"
.venv/bin/python -c 'import websocket, PIL, pillow_heif, pillow_jxl, rawpy, huggingface_hub, tqdm; print("  ok")'

echo
echo "done. Run the scripts with this venv:"
echo "  source $PWD/.venv/bin/activate"
echo "  python img2img.py photo.jpg \"把天空換成星空\""
echo "or skip the venv entirely with ./img2img.py, which lets uv supply the packages."
