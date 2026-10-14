#!/usr/bin/env python3
"""Download Wan14BT2VFusioniX_fp16_.safetensors from Hugging Face.

Default destination matches workflows/selfhost/video_wan2.1_fusionx.json:

    ComfyUI/models/diffusion_models/wan-fusionx/Wan14BT2VFusioniX_fp16_.safetensors

Run:

    uv run --with huggingface_hub --with hf_xet python scripts/download_wan_t2v.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "vrgamedevgirl84/Wan14BT2VFusioniX"
FILENAME = "Wan14BT2VFusioniX_fp16_.safetensors"
DEFAULT_DIR = (
    Path.home()
    / "ComfyUI-Installs/ComfyUI/ComfyUI/models/diffusion_models/wan-fusionx"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Wan14BT2VFusioniX_fp16_.safetensors")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Directory to save the file (default: {DEFAULT_DIR})",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "缺少 huggingface_hub。请用下面这条命令运行：\n"
            "  uv run --with huggingface_hub --with hf_xet python scripts/download_wan_t2v.py",
            file=sys.stderr,
        )
        return 1

    destination = args.output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / FILENAME

    print(f"仓库: {REPO_ID}")
    print(f"文件: {FILENAME}")
    print(f"保存到: {target}")
    print("约 14.3 GB，中断后重新运行同一命令会续传。")

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=destination,
    )
    print(f"下载完成: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
