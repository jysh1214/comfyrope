#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pillow>=11.3",            # reads 40+ formats, AVIF included
#     "pillow-heif>=1.0",        # HEIC/HEIF, e.g. iPhone photos
#     "pillow-jxl-plugin>=1.3",  # JPEG XL
#     "rawpy>=0.24",             # camera RAW: DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2, ...
# ]
# ///
"""The input converter as a command, for rope-queue's conversion stage: what img2img.py does to an input before
uploading it, as one call.

    comfyconvert.py FILE OUTDIR --json

A single-frame PNG, JPEG or WebP goes to ComfyUI as it is: {"as_is": true}. Anything else becomes an upright,
single-frame, 8-bit sRGB PNG in OUTDIR: {"file": "1_photo.png", "steps": "HEIF, converted to PNG", "libs": {...}},
the libraries being the decoders' versions, which rope records with the rendition. A file it cannot read is a message
on stderr and exit 1. Without --json it says the same in words."""
import argparse
import json
import sys
from pathlib import Path

from comfycore import Error, check_packages, decoder_versions, die, prepare_inputs

PACKAGES = {"PIL": "pillow", "pillow_heif": "pillow-heif", "pillow_jxl": "pillow-jxl-plugin", "rawpy": "rawpy"}


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", type=Path, help="the input image")
    p.add_argument("out", type=Path, help="where a converted PNG goes")
    p.add_argument("--json", action="store_true", help="answer in JSON, one line")
    a = p.parse_args(argv)
    if not a.file.is_file():
        die(f"no such file: {a.file}")
    check_packages(PACKAGES, Path(__file__))
    a.out.mkdir(parents=True, exist_ok=True)
    (upload, how), = prepare_inputs([a.file], a.out, [a.file.stem])
    answer = {"file": upload.name, "steps": how, "libs": decoder_versions()} if how else {"as_is": True}
    if a.json:
        print(json.dumps(answer, ensure_ascii=False))
    elif how:
        print(f"{a.file.name}: {how}, as {upload}")
    else:
        print(f"{a.file.name}: goes to ComfyUI as it is")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Error as e:
        sys.exit(f"error: {e}")
