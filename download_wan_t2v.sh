#!/bin/bash
# Download Wan14BT2VFusioniX into the local ComfyUI models directory.
# Run again to continue an interrupted download.

cd "$(dirname "$0")"
exec uv run --with huggingface_hub python scripts/download_wan_t2v.py "$@"
