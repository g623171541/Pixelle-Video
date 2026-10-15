#!/usr/bin/env python3
"""Download Wan14BT2VFusioniX_fp16_.safetensors from Hugging Face.

Default destination matches workflows/selfhost/video_wan2.1_fusionx.json:

    ComfyUI/models/diffusion_models/wan-fusionx/Wan14BT2VFusioniX_fp16_.safetensors

Run from the repository root:

    ./download_wan_t2v.sh

The download goes to ``<filename>.incomplete`` beside the destination. Run the
same command again to continue. huggingface.co is used when it answers; otherwise
the script switches to hf-mirror.com. Set HF_ENDPOINT to choose the host yourself.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO_ID = "vrgamedevgirl84/Wan14BT2VFusioniX"
FILENAME = "Wan14BT2VFusioniX_fp16_.safetensors"
DEFAULT_DIR = (
    Path.home()
    / "ComfyUI-Installs/ComfyUI/ComfyUI/models/diffusion_models/wan-fusionx"
)
MIRROR = "https://hf-mirror.com"
MAX_STALLS = 30


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Wan14BT2VFusioniX_fp16_.safetensors")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Directory to save the file (default: {DEFAULT_DIR})",
    )
    args = parser.parse_args()

    get_hf_file_metadata, hf_hub_url, http_get = _import_hub()
    destination = args.output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / FILENAME
    partial = destination / f"{FILENAME}.incomplete"

    metadata = get_hf_file_metadata(hf_hub_url(repo_id=REPO_ID, filename=FILENAME))
    expected_size = metadata.size
    if not expected_size:
        print("服务器没有返回文件大小，停止。", file=sys.stderr)
        return 1
    etag = _sha256_etag(metadata.etag)

    print(f"仓库: {REPO_ID}", flush=True)
    print(f"文件: {FILENAME}", flush=True)
    print(f"保存到: {target}", flush=True)
    print(f"大小: {expected_size / (1024 ** 3):.2f} GiB", flush=True)
    if etag:
        print(f"SHA256: {etag}", flush=True)

    if target.is_file():
        return _accept_finished(target, expected_size, etag)

    stalls = 0
    seen = -1
    while True:
        resume_size = _resume_size(partial, expected_size)
        if resume_size == expected_size:
            break
        if resume_size == seen:
            stalls += 1
        else:
            stalls = 0
            seen = resume_size
        if stalls >= MAX_STALLS:
            print(
                "连续多次没有新数据，先停下来。再次运行同一条命令会从当前进度继续。",
                file=sys.stderr,
            )
            return 1

        metadata = get_hf_file_metadata(hf_hub_url(repo_id=REPO_ID, filename=FILENAME))
        if resume_size:
            print(f"从已下载的 {resume_size / (1024 ** 3):.2f} GiB 继续。", flush=True)
        try:
            with partial.open("r+b" if resume_size else "wb") as handle:
                if resume_size:
                    handle.seek(resume_size)
                http_get(
                    metadata.location,
                    handle,
                    resume_size=resume_size,
                    expected_size=expected_size,
                    displayed_filename=FILENAME,
                )
        except Exception as exc:
            print(f"下载中断，5 秒后换新链接继续: {exc}", flush=True)
            time.sleep(5)

    if _holes(partial):
        print("文件长度够了，但中间有没写上的空洞，未改名。", file=sys.stderr)
        return 1
    if etag and not _hash_matches(partial, etag):
        print("SHA256 和官方记录不一致，未改名。删掉 .incomplete 后可以重新下载。", file=sys.stderr)
        return 1
    partial.replace(target)
    print(f"下载完成: {target}", flush=True)
    return 0


def _import_hub():
    if not os.environ.get("HF_ENDPOINT") and not _reachable("https://huggingface.co"):
        os.environ["HF_ENDPOINT"] = MIRROR
        print(f"连不上 huggingface.co，改用 {MIRROR}", flush=True)
    elif os.environ.get("HF_ENDPOINT"):
        print(f"使用 HF_ENDPOINT={os.environ['HF_ENDPOINT']}", flush=True)
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        from huggingface_hub.file_download import http_get
    except ImportError:
        print(
            "缺少 huggingface_hub。请用下面这条命令运行：\n"
            "  uv run --with huggingface_hub python scripts/download_wan_t2v.py",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return get_hf_file_metadata, hf_hub_url, http_get


def _reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            return response.status < 500
    except Exception:
        return False


def _sha256_etag(etag: str | None) -> str | None:
    if etag and re.fullmatch(r"[0-9a-f]{64}", etag.lower()):
        return etag.lower()
    return None


def _accept_finished(path: Path, expected_size: int, etag: str | None) -> int:
    if path.stat().st_size != expected_size:
        print(f"目标文件大小不对，未覆盖: {path}", file=sys.stderr)
        return 1
    if _holes(path):
        print(f"文件长度正确，但中间有空洞，不能当完整文件用: {path}", file=sys.stderr)
        return 1
    if etag and not _hash_matches(path, etag):
        print(f"SHA256 和官方记录不一致: {path}", file=sys.stderr)
        return 1
    print(f"文件已存在: {path}", flush=True)
    return 0


def _resume_size(partial: Path, expected_size: int) -> int:
    if not partial.is_file():
        return 0
    holes = _holes(partial)
    if holes:
        cut = holes[0][0]
        print(f"未完成文件中间有空洞，截到 {cut / (1024 ** 3):.2f} GiB 后继续。", flush=True)
        os.truncate(partial, cut)
    size = partial.stat().st_size
    if size > expected_size:
        print("未完成文件比预期更大，重新下载。", flush=True)
        partial.unlink()
        return 0
    return size


def _holes(path: Path) -> list[tuple[int, int]]:
    """Return unwritten byte ranges in a sparse file."""
    size = path.stat().st_size
    holes: list[tuple[int, int]] = []
    fd = os.open(path, os.O_RDONLY)
    offset = 0
    try:
        while offset < size:
            try:
                data = os.lseek(fd, offset, os.SEEK_DATA)
            except OSError:
                holes.append((offset, size))
                break
            if data > offset:
                holes.append((offset, data))
            try:
                offset = os.lseek(fd, data, os.SEEK_HOLE)
            except OSError:
                break
    finally:
        os.close(fd)
    return [(start, end) for start, end in holes if end > start]


def _hash_matches(path: Path, expected: str) -> bool:
    print("正在校验 SHA256……", flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    actual = digest.hexdigest()
    print(f"本地 SHA256: {actual}", flush=True)
    return actual == expected


if __name__ == "__main__":
    raise SystemExit(main())
