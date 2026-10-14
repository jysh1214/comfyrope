#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "websocket-client>=1.8,<2",
#     "pillow>=11.3",            # reads 40+ formats, AVIF included
#     "pillow-heif>=1.0",        # HEIC/HEIF, e.g. iPhone photos
#     "pillow-jxl-plugin>=1.3",  # JPEG XL
#     "rawpy>=0.24",             # camera RAW: DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2, ...
# ]
# ///
"""Image to image from the command line: input images plus a prompt make a new image (the inputs are never
modified). Runs Qwen-Image-Edit-2511 or Qwen-Image-2.1 through ComfyUI's API.

    ./img2img.py photo.jpg "把背景換成黃昏的海邊"
    ./img2img.py person.png shirt.png "讓圖1的人穿上圖2的襯衫" -n 4
    ./img2img.py photo.jpg "turn this into a watercolor painting" -m 2.1
    ./img2img.py photo.jpg "..." --lora my_style.safetensors:0.8 -s 42 --into trips/kyoto --tag 夕陽 --tag 海邊

Give the input images first (2511: up to 3, 2.1: up to 10), then the prompt. The first image sets the
output size. With several images, refer to them in the prompt as 圖1 / 圖2 (or image 1 / image 2).

Single-frame PNG, JPEG and WebP files are uploaded as they are. Anything else is converted here to an
upright, single-frame sRGB PNG first: HEIC/HEIF, AVIF, JPEG XL, camera RAW, GIF, TIFF, BMP, PSD and the
other formats Pillow reads. Animations and multi-page files keep only their first frame.

The script talks to the ComfyUI at --server (default http://127.0.0.1:8188). If none is running it starts
one with run_comfyui.sh for this command and stops it afterwards; --keep-server leaves it running so the
next command skips the model load.

Results go into the rope image library (rope/, by default in ~/.local/share/rope), which has to be built
first, and so does every input, so a result can be made again after its inputs are moved or deleted. Each
result is named <time>_<first image>_<model>_s<seed>, for example 20261013-225301_photo_2511_s42, and rope
keeps with it the recipe, the API workflow that ComfyUI ran, and a repro.sh. --into puts the results in a rope
folder, and an input can be @ID, an image in rope. --replay ID makes an image again from that record and
compares the pixels. The PNG also embeds the API workflow ComfyUI ran; drop it on the ComfyUI page to see that
workflow as a node graph.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import mimetypes
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "run_comfyui.sh"
SERVER = "http://127.0.0.1:8188"
TIMEOUT = 1800
UPLOAD_SUBFOLDER = "img2img"  # uploads land in ComfyUI/input/img2img/
ROPE_BUILD = HERE / "rope" / "target" / "release" / "rope"  # the image library; see rope/README.md
ROPE_HOWTO = "cargo build --release --manifest-path rope/Cargo.toml"

# Model files as the ComfyUI templates name them (download_models.py fetches them).
EDIT_2511 = {
    "unet": "qwen_image_edit_2511_fp8mixed.safetensors",
    "clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "vae": "qwen_image_vae.safetensors",
}
LIGHTNING = {
    "4": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "8": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
}
Q21 = {
    "unet": "qwen_image_2.1_int8_convrot.safetensors",
    "clip": "qwen3vl_8b_int8_convrot.safetensors",
    "vae": "qwen_image_2.1_vae_bf16.safetensors",
    "pe": "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors",
}
FETCH = {  # model file -> the command that downloads it
    **dict.fromkeys([*EDIT_2511.values(), LIGHTNING["4"]], "uv run download_models.py 2511"),
    LIGHTNING["8"]: "uv run download_models.py 2511 --lightning 8",
    **dict.fromkeys([Q21["unet"], Q21["clip"], Q21["vae"]], "uv run download_models.py 2.1"),
    Q21["pe"]: "uv run download_models.py 2.1 --prompt-enhancer i2i",
}
MAX_IMAGES = {"2511": 3, "2.1": 10}  # 2.1: what its template exposes and its model card states
PE_TEMPLATE = "image_qwen_image_2_1_image_edit.json"  # ComfyUI's 2.1 edit template holds the enhancer's system prompt

# Single-frame files of these formats go up untouched. Checked against LoadImage in ComfyUI 0.39.2: it reads them
# correctly, EXIF orientation included, but turns any multi-frame file into a batch of images, pads HEIC to its
# coded size, and cannot read JPEG XL, camera RAW or SVG. Everything else is converted to PNG here first.
AS_IS = {"PNG", "JPEG", "WEBP"}
RAW_SUFFIXES = {".dng", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf", ".orf", ".rw2",
                ".pef", ".srw", ".x3f", ".3fr", ".iiq", ".rwl", ".mrw", ".erf", ".kdc", ".dcr", ".mos"}


def die(msg: str) -> NoReturn:
    raise SystemExit(f"error: {msg}")


def tilde(p: Path) -> str:
    home, s = str(Path.home()), str(p)
    return "~" + s[len(home):] if s == home or s.startswith(home + "/") else s


class Status:
    """A live status line on a terminal; plain lines (one per stage) when stdout is a file or pipe."""

    def __init__(self) -> None:
        self.tty = sys.stdout.isatty()
        self.stage = ""

    def show(self, stage: str, detail: str = "") -> None:
        if self.tty:
            sys.stdout.write(f"\r\x1b[K  {stage} {detail}".rstrip())
            sys.stdout.flush()
        elif stage != self.stage:
            print(f"  {stage}")
        self.stage = stage

    def line(self, text: str) -> None:
        if self.tty and self.stage:
            sys.stdout.write("\r\x1b[K")
        self.stage = ""
        print(text)


# ---------------------------------------------------------------- input images

def flatten(im):
    """First frame, upright, 8-bit sRGB RGB (or RGBA when the image has transparency)."""
    from PIL import ImageCms, ImageOps

    im.seek(0)
    out = ImageOps.exif_transpose(im)  # PNG output: bake the EXIF rotation into the pixels
    if out.mode in ("I;16", "I;16L", "I;16B", "I;16N", "I"):
        out = out.convert("I").point(lambda v: v / 256).convert("L")  # 16-bit grey to 8-bit
    elif out.mode == "F":
        out = out.point(lambda v: v * 255).convert("L")  # float images are usually 0..1
    mode = "RGBA" if "A" in out.getbands() or "transparency" in out.info else "RGB"
    icc = im.info.get("icc_profile")
    if icc and out.mode in ("RGB", "RGBA", "CMYK", "L"):
        try:  # e.g. iPhone HEIC is Display P3; ComfyUI treats every pixel as sRGB
            source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            return ImageCms.profileToProfile(out, source, ImageCms.createProfile("sRGB"), outputMode=mode)
        except (ImageCms.PyCMSError, OSError, ValueError, TypeError):
            pass  # an unusable profile is no reason to fail: fall back to a plain conversion
    return out.convert(mode)


def develop_raw(src: Path, dst: Path) -> None:
    import rawpy
    from PIL import Image

    try:
        with rawpy.imread(str(src)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)  # sRGB, with the camera's orientation applied
    except (rawpy.LibRawError, OSError) as e:
        die(f"cannot develop the camera RAW file {src}: {e}")
    Image.fromarray(rgb).save(dst)


def prepare_inputs(paths: list[Path], workdir: Path, stems: list[str]) -> list[tuple[Path, str]]:
    """The files to upload, each with how it was converted, or "" when it goes up as it is: single-frame
    PNG/JPEG/WebP as they are, everything else converted to PNG in workdir, named after its stem in `stems`."""
    import pillow_heif
    import pillow_jxl  # noqa: F401  (importing it registers JPEG XL with Pillow)
    from PIL import Image, UnidentifiedImageError

    pillow_heif.register_heif_opener()
    ready = []
    for n, (src, stem) in enumerate(zip(paths, stems), start=1):
        dst = workdir / f"{n}_{stem.replace('/', '_')}.png"
        label = f"{stem}{src.suffix}"
        if src.suffix.lower() in RAW_SUFFIXES:  # before Pillow, which would open a DNG's small TIFF preview
            develop_raw(src, dst)
            how = "camera RAW, developed with the camera's white balance and converted to PNG"
            print(f"{label}: {how}", file=sys.stderr)
            ready.append((dst, how))
            continue
        try:
            with Image.open(src) as im:
                kind, frames = im.format, getattr(im, "n_frames", 1)
                if kind in AS_IS and frames == 1:
                    ready.append((src, ""))
                    continue
                flatten(im).save(dst)
        except UnidentifiedImageError:
            die(f"cannot read {src}: not an image format this script knows. It reads what Pillow reads (PNG, JPEG, "
                "WebP, AVIF, GIF, BMP, TIFF, PSD, ...), plus HEIC/HEIF, JPEG XL and camera RAW; export vector "
                "formats such as SVG to PNG first")
        except (OSError, ValueError, SyntaxError) as e:  # truncated or damaged files
            die(f"cannot decode {src}: {e}")
        first = f", {frames} frames, kept the first" if frames > 1 else ""
        how = f"{kind}{first}, converted to PNG"
        print(f"{label}: {how}", file=sys.stderr)
        ready.append((dst, how))
    return ready


# ---------------------------------------------------------------- workflows (ComfyUI API format)

class Graph:
    """{"<id>": {"class_type": ..., "inputs": {...}}}; an input links to another node as [id, output_index]."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.labels: dict[str, str] = {}  # node id -> progress label
        self.image_out = ""  # PreviewImage node with the result
        self.text_out = ""   # PreviewAny node with the enhanced prompt (2.1 --enhance)

    def add(self, class_type: str, label: str = "", /, **inputs) -> str:
        nid = str(len(self.nodes) + 1)
        self.nodes[nid] = {"class_type": class_type, "inputs": inputs}
        if label:
            self.labels[nid] = label
        return nid


def link(nid: str, slot: int = 0) -> list:
    return [nid, slot]


def add_loras(g: Graph, model: str, loras: list[tuple[str, float]]) -> str:
    for name, strength in loras:
        model = g.add("LoraLoaderModelOnly", model=link(model), lora_name=name, strength_model=strength)
    return model


def build_2511(a: argparse.Namespace, images: list[str], seed: int) -> Graph:
    """ComfyUI template 'Qwen Image Edit 2511', flattened."""
    g = Graph()
    loads = [g.add("LoadImage", image=name) for name in images]
    first = g.add("FluxKontextImageScale", image=link(loads[0]))  # image 1 snapped to a trained size, about 1 MP
    unet = g.add("UNETLoader", "loading the diffusion model", unet_name=EDIT_2511["unet"], weight_dtype="default")
    clip = g.add("CLIPLoader", "loading the text encoder", clip_name=EDIT_2511["clip"], type="qwen_image", device="default")
    vae = g.add("VAELoader", vae_name=EDIT_2511["vae"])
    model = g.add("ModelSamplingAuraFlow", model=link(unet), shift=3.1)
    model = g.add("CFGNorm", model=link(model), strength=1.0)
    if a.lightning != "off":
        model = g.add("LoraLoaderModelOnly", model=link(model), lora_name=LIGHTNING[a.lightning], strength_model=1.0)
    model = add_loras(g, model, a.lora)
    refs = {"image1": link(first), **{f"image{i}": link(n) for i, n in enumerate(loads[1:], start=2)}}
    pos = g.add("TextEncodeQwenImageEditPlus", "encoding the prompt and images",
                clip=link(clip), vae=link(vae), prompt=a.prompt, **refs)
    neg = g.add("TextEncodeQwenImageEditPlus", clip=link(clip), vae=link(vae), prompt=a.negative, **refs)
    pos = g.add("FluxKontextMultiReferenceLatentMethod", conditioning=link(pos), reference_latents_method="index_timestep_zero")
    neg = g.add("FluxKontextMultiReferenceLatentMethod", conditioning=link(neg), reference_latents_method="index_timestep_zero")
    latent = g.add("VAEEncode", pixels=link(first), vae=link(vae))
    sampled = g.add("KSampler", "sampling", model=link(model), seed=seed, steps=a.steps, cfg=a.cfg,
                    sampler_name="euler", scheduler="simple", positive=link(pos), negative=link(neg),
                    latent_image=link(latent), denoise=1.0)
    decoded = g.add("VAEDecode", "decoding", samples=link(sampled), vae=link(vae))
    g.image_out = g.add("PreviewImage", images=link(decoded))
    return g


def build_21(a: argparse.Namespace, images: list[str], seed: int, system_prompt: str | None) -> Graph:
    """ComfyUI template 'Qwen Image 2.1: Image Edit', flattened; the prompt enhancer only with --enhance."""
    g = Graph()
    loads = [g.add("LoadImage", image=name) for name in images]
    unet = g.add("UNETLoader", "loading the diffusion model", unet_name=Q21["unet"], weight_dtype="default")
    model = add_loras(g, unet, a.lora)
    model = g.add("QwenImage21Cache", model=link(model), device="auto", dtype="default")
    clip = g.add("CLIPLoader", "loading the text encoder", clip_name=Q21["clip"], type="qwen_image", device="default")
    vae = g.add("VAELoader", vae_name=Q21["vae"])
    prompt: str | list = a.prompt
    if a.enhance:
        pe = g.add("CLIPLoader", "loading the prompt enhancer", clip_name=Q21["pe"], type="qwen_image", device="default")
        batch = g.add("BatchImagesNode", **{f"images.image{i}": link(n) for i, n in enumerate(loads)})
        sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
                    "repetition_penalty": 1.0, "seed": seed, "presence_penalty": 0.0}
        gen = g.add("TextGenerate", "enhancing the prompt", clip=link(pe), image=link(batch), prompt=a.prompt,
                    system_prompt=system_prompt, max_length=4096, thinking=False, use_default_template=True,
                    mtp="auto", sampling_mode="on", **{f"sampling_mode.{k}": v for k, v in sampling.items()})
        g.text_out = g.add("PreviewAny", source=link(gen))
        prompt = link(gen)
    enc = g.add("TextEncodeQwenImage21", "encoding the prompt and images", clip=link(clip), vae=link(vae),
                prompt=prompt, negative_prompt=a.negative, resolution=a.resolution,
                **{f"images.image_{i}": link(n) for i, n in enumerate(loads, start=1)})
    sampled = g.add("KSampler", "sampling", model=link(model), seed=seed, steps=a.steps, cfg=a.cfg,
                    sampler_name="euler", scheduler="simple", positive=link(enc, 0), negative=link(enc, 1),
                    latent_image=link(enc, 2), denoise=1.0)  # the encoder's latent has image 1's size
    decoded = g.add("VAEDecode", "decoding", samples=link(sampled), vae=link(vae))
    g.image_out = g.add("PreviewImage", images=link(decoded))
    return g


def build(a: argparse.Namespace, images: list[str], seed: int, system_prompt: str | None) -> Graph:
    return build_2511(a, images, seed) if a.model == "2511" else build_21(a, images, seed, system_prompt)


# ---------------------------------------------------------------- ComfyUI client

class ComfyError(Exception):
    def __init__(self, status: int, body: bytes) -> None:
        super().__init__(f"HTTP {status}: {body[:200]!r}")
        self.status, self.body = status, body


def upload_name(path: Path, stem: str = "", tag: str = "") -> str:
    """Server-side name of an input: content hash + an ASCII-safe form of `stem` (default: the file's), so a re-run
    reuses the same upload. A `tag` after the hash gives the same bytes a name ComfyUI has not cached results for."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem or path.stem).strip("_.") or "image"
    suffix = re.sub(r"[^A-Za-z0-9.]", "", path.suffix.lower())
    return f"{digest}_{tag + '_' if tag else ''}{stem}{suffix}"


def default_sigint() -> None:
    """Runs in the child before exec: a script started in the background has SIGINT ignored, and the child
    would inherit that, so stop() could not shut ComfyUI down cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)


class Comfy:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        parts = urllib.parse.urlsplit(self.url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            die(f"--server must look like http://127.0.0.1:8188, not {url!r}")
        self.host = parts.hostname
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.ws_url = f"{'wss' if parts.scheme == 'https' else 'ws'}://{parts.netloc}/ws"
        self.client_id = uuid.uuid4().hex
        self.proc: subprocess.Popen | None = None  # the ComfyUI this script started, if any
        self.log: Path | None = None

    # -- http
    def request(self, method: str, path: str, body: bytes | None = None, ctype: str | None = None,
                timeout: float = 30) -> bytes:
        req = urllib.request.Request(self.url + path, data=body, method=method)
        if ctype:
            req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise ComfyError(e.code, e.read()) from None

    def get_json(self, path: str, timeout: float = 30):
        return json.loads(self.request("GET", path, timeout=timeout))

    def post_json(self, path: str, obj, timeout: float = 30):
        raw = self.request("POST", path, json.dumps(obj).encode(), "application/json", timeout)
        return json.loads(raw) if raw.strip() else None

    def alive(self) -> bool:
        try:
            self.get_json("/system_stats", timeout=3)
            return True
        except (OSError, ValueError, ComfyError):
            return False

    def system_info(self) -> str:
        """Version, launch flags, torch and GPU of this ComfyUI, for repro.sh; empty if it does not tell."""
        try:
            stats = self.get_json("/system_stats")
            s, device = stats["system"], stats["devices"][0]["name"]
            return f"ComfyUI {s['comfyui_version']} ({shlex.join(s['argv'])}), torch {s['pytorch_version']}, {device}"
        except (OSError, ValueError, ComfyError, LookupError, TypeError):
            return ""

    # -- server process
    def start(self, extra_args: list[str]) -> None:
        if not LAUNCHER.exists():
            die(f"no ComfyUI answers at {self.url}, and {tilde(LAUNCHER)} is missing (run install_comfyui.py)")
        user_dir = HERE / "ComfyUI" / "user"
        self.log = (user_dir if user_dir.is_dir() else Path(tempfile.gettempdir())) / "img2img-server.log"
        cmd = [str(LAUNCHER), "--listen", self.host, "--port", str(self.port), *extra_args]
        print(f"no ComfyUI at {self.url}; starting one for this command (log: {tilde(self.log)})")
        with open(self.log, "wb") as log:
            # its own session: Ctrl-C reaches only this script, which then shuts ComfyUI down cleanly
            self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, preexec_fn=default_sigint)
        t0 = time.monotonic()
        while not self.alive():
            if self.proc.poll() is not None:
                die(f"ComfyUI exited with code {self.proc.returncode} while starting:\n{self.log_tail()}")
            if time.monotonic() - t0 > 300:
                self.stop()
                die(f"ComfyUI did not answer within 5 minutes:\n{self.log_tail()}")
            time.sleep(1)
        print(f"  ComfyUI is up after {time.monotonic() - t0:.0f}s")

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.send_signal(signal.SIGINT)  # handled like Ctrl-C: ComfyUI cleans up and exits
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def log_tail(self, n: int = 20) -> str:
        try:
            lines = self.log.read_text(errors="replace").splitlines()[-n:] if self.log else []
        except OSError:
            lines = []
        return "\n".join("  | " + line for line in lines)

    # -- jobs
    def upload(self, path: Path, name: str) -> str:
        boundary = uuid.uuid4().hex
        # no overwrite: the name holds the content hash, and ComfyUI skips writing a same-named file with the same
        # content, where overwrite would rewrite it while another job may be reading it
        fields = {"type": "input", "subfolder": UPLOAD_SUBFOLDER}
        body = b"".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                        for k, v in fields.items())
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                 f"Content-Type: {mimetypes.guess_type(name)[0] or 'application/octet-stream'}\r\n\r\n").encode()
        body += path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        r = json.loads(self.request("POST", "/upload/image", body, f"multipart/form-data; boundary={boundary}", 120))
        return f"{r['subfolder']}/{r['name']}" if r.get("subfolder") else r["name"]

    def enhancer_system_prompt(self) -> str:
        try:
            template = self.get_json(f"/templates/{PE_TEMPLATE}")
        except (OSError, ValueError, ComfyError) as e:
            die(f"--enhance reads its system prompt from ComfyUI's {PE_TEMPLATE}, which failed: {e}")
        for sub in template.get("definitions", {}).get("subgraphs", []):
            for node in sub.get("nodes", []):
                if node.get("type") == "PrimitiveStringMultiline" and "System Prompt" in (node.get("title") or ""):
                    return node["widgets_values"][0]
        die(f"ComfyUI's {PE_TEMPLATE} no longer has the enhancer's system prompt node; --enhance cannot run")

    def queue(self, graph: Graph) -> str:
        try:
            return self.post_json("/prompt", {"prompt": graph.nodes, "client_id": self.client_id})["prompt_id"]
        except ComfyError as e:
            die(explain_rejection(e.body))

    def history(self, prompt_id: str) -> dict | None:
        return self.get_json(f"/history/{prompt_id}").get(prompt_id)

    def fetch(self, ref: dict) -> bytes:
        query = urllib.parse.urlencode({"filename": ref["filename"], "subfolder": ref.get("subfolder", ""),
                                        "type": ref.get("type", "output")})
        return self.request("GET", f"/view?{query}", timeout=120)

    def cancel(self, prompt_ids: list[str]) -> None:
        try:
            self.post_json("/queue", {"delete": prompt_ids})
            for pid in prompt_ids:
                self.post_json("/interrupt", {"prompt_id": pid})  # a no-op unless that prompt is the running one
        except (OSError, ComfyError):
            pass


def explain_rejection(body: bytes) -> str:
    try:
        j = json.loads(body)
    except ValueError:
        return f"ComfyUI rejected the workflow: {body[:300].decode(errors='replace')}"
    err = j.get("error") or {}
    lines = [f"ComfyUI rejected the workflow: {err.get('message', '?')}"]
    for nid, node in (j.get("node_errors") or {}).items():
        for e in node.get("errors", []):
            lines.append(f"  {node.get('class_type', '?')} (node {nid}): {e.get('message')}: {e.get('details')}")
            value = (e.get("extra_info") or {}).get("received_value")
            if e.get("type") == "value_not_in_list" and isinstance(value, str):
                if value in FETCH:
                    lines.append(f"  fix: {FETCH[value]}")
                elif node.get("class_type") == "LoraLoaderModelOnly":
                    lines.append(f"  fix: put {value} in {tilde(HERE / 'models' / 'loras')}")
    if len(lines) == 1 and err.get("details"):
        lines.append(f"  {err['details']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- rope library

class LibraryError(Exception):
    pass


class Library:
    """The rope library this run keeps its inputs and results in (see rope/README.md). Every change goes through
    the rope command. This script only holds the library's run.lock, shared, from keeping its inputs to the end of
    the run, so that `rope purge` cannot delete an input that a result of this run will name."""

    def __init__(self, exe: str, flags: list[str]) -> None:
        self.exe, self.flags = exe, flags
        self.root = Path()
        self.run_lock = None

    @staticmethod
    def find(a: argparse.Namespace) -> Library:
        """The library: results have nowhere else to go, so a missing rope stops the run before any GPU work."""
        exe = os.environ.get("ROPE") or (str(ROPE_BUILD) if ROPE_BUILD.exists() else shutil.which("rope"))
        if not exe:
            die(f"rope is not built; build it with: {ROPE_HOWTO} (see rope/README.md)")
        lib = Library(exe, [f"--library={a.library}"] if a.library else [])
        try:
            lib.root = Path(lib.call("where")["library"])
        except LibraryError as e:
            die(str(e))
        return lib

    def call(self, *args: str):
        """Runs `rope --json ARGS` and returns its JSON."""
        cmd = [self.exe, *self.flags, "--json", *args]
        try:
            r = subprocess.run(cmd, capture_output=True, encoding="utf-8", timeout=600)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise LibraryError(f"cannot run {self.exe}: {e}") from None
        if r.returncode != 0:
            raise LibraryError(r.stderr.strip().removeprefix("rope: ") or f"{shlex.join(cmd)} failed")
        return json.loads(r.stdout) if r.stdout.strip() else None

    def hold(self) -> None:
        """Holds run.lock shared until this process ends."""
        import fcntl

        if self.run_lock is None:
            self.run_lock = open(self.root / "run.lock", "a")
            fcntl.flock(self.run_lock, fcntl.LOCK_SH)


def call_library(lib: Library, *args: str):
    try:
        return lib.call(*args)
    except LibraryError as e:
        die(str(e))


def iso(t: time.struct_time) -> str:
    """2026-10-14T10:19:19+08:00"""
    s = time.strftime("%Y-%m-%dT%H:%M:%S%z", t)
    return f"{s[:-2]}:{s[-2:]}"


def git(*args: str, cwd: Path = HERE) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def code_identity() -> dict:
    """Which img2img.py this is: its git blob id, which names this exact file content, plus the commit and
    whether the file differs from it."""
    data = Path(__file__).read_bytes()
    out = {"img2img_blob": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()}
    commit = git("rev-parse", "--short", "HEAD")
    if commit:
        out["commit"] = commit
        out["dirty"] = bool(git("status", "--porcelain", "--", Path(__file__).name))
    return out


def comfy_env(comfy: Comfy) -> dict:
    """What the image depended on besides the workflow: the server's /system_stats without the fields that change
    from minute to minute, and, when it is the ComfyUI in this repo, its commit, custom nodes and package versions."""
    try:
        stats = comfy.get_json("/system_stats")
    except (OSError, ValueError, ComfyError):
        return {}
    system = {k: v for k, v in (stats.get("system") or {}).items() if not k.startswith("ram_")}
    env = {"system": system, "devices": [{k: d.get(k) for k in ("name", "type", "index")} for d in stats.get("devices") or []]}
    local = HERE / "ComfyUI"
    match = re.search(r'__version__\s*=\s*"([^"]+)"', (local / "comfyui_version.py").read_text()) \
        if (local / "comfyui_version.py").exists() else None
    if match and match.group(1) == system.get("comfyui_version"):
        env["comfyui"] = {"commit": git("rev-parse", "HEAD", cwd=local),
                          "dirty": bool(git("status", "--porcelain", "--untracked-files=no", cwd=local))}
        wanted = {"torch", "torchvision", "av", "numpy", "safetensors", "transformers", "pillow", "comfy_kitchen",
                  "comfy_aimdo", "comfyui_workflow_templates", "comfyui_frontend_package"}
        packages = {}
        for info in local.glob(".venv/lib/python*/site-packages/*.dist-info"):
            name, _, version = info.name.removesuffix(".dist-info").partition("-")
            if name.lower().replace("-", "_") in wanted:
                packages[name.lower()] = version
        env["packages"] = dict(sorted(packages.items()))
        nodes = local / "custom_nodes"
        env["custom_nodes"] = sorted(p.name for p in nodes.iterdir() if p.is_dir() and p.name != "__pycache__") \
            if nodes.is_dir() else []
    return env


MODEL_INPUTS = {"UNETLoader": ("diffusion_models", "unet_name"), "CLIPLoader": ("text_encoders", "clip_name"),
                "VAELoader": ("vae", "vae_name"), "LoraLoaderModelOnly": ("loras", "lora_name")}


def model_files(nodes: dict) -> list[dict]:
    """Every model file a workflow loads, with its size and modification time: a cheap check that a file of the same
    name was not replaced, which a name alone cannot show (both Lightning LoRAs are 849608296 bytes)."""
    out = []
    for node in nodes.values():
        kind = MODEL_INPUTS.get(node["class_type"])
        if not kind or not isinstance(node["inputs"].get(kind[1]), str):
            continue
        entry = {"dir": kind[0], "name": node["inputs"][kind[1]]}
        if entry in [{k: o[k] for k in ("dir", "name")} for o in out]:
            continue
        for base in (HERE / "models", HERE / "ComfyUI" / "models"):
            path = base / kind[0] / entry["name"]
            if path.exists():
                st = path.stat()
                entry |= {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
                break
        out.append(entry)
    return out


def decoder_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in ("pillow", "pillow-heif", "pillow-jxl-plugin", "rawpy"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            pass
    return out


def recipe(a: argparse.Namespace, job: Job, flags: list[str], enhanced: str) -> dict:
    """The made field of a result's meta.json in rope."""
    inputs = [{k: v for k, v in (("id", i.get("id")), ("uploaded", i.get("uploaded")), ("upload", name)) if v}
              for i, name in zip(a.inputs, a.upload_names)]
    made = {"by": "img2img.py", "at": iso(a.when), "flags": flags, "prompt": a.prompt, "inputs": inputs,
            "enhanced_prompt": enhanced or None,
            "batch": {"index": job.index, "count": a.count, "first_seed": job.seed - job.index + 1},
            "server": a.server_info or None, "env": a.env, "models": a.models, "code": a.code}
    return {k: v for k, v in made.items() if v is not None}


def keep_result(a: argparse.Namespace, job: Job, data: bytes, enhanced: str, status: Status) -> str:
    """Keeps one result in the library and returns what to show for it on the status line."""
    def args(png: Path, made: Path, graph: Path) -> list[str]:
        out = ["add", str(png), f"--made={made}", f"--graph={graph}", f"--name={name}", f"--repo={HERE}"]
        if a.into:
            out += [f"--into={a.into}", "--file-inputs"]
        out += [f"--tag={t}" for t in a.tags]
        return out + ([f"--note={a.note}"] if a.note else [])

    name = run_name(a, job.seed)
    made = json.dumps(recipe(a, job, image_flags(a, job.seed), enhanced), ensure_ascii=False)
    r = send_result(a, name, data, made, job, args, status)
    if r is None:
        return "not in rope; see above"
    where = f"{r['short']}  {r['folder'] or '/'}"
    return where if r["created"] else f"{where}  (rope had these bytes already)"


def send_result(a: argparse.Namespace, name: str, data: bytes, made: str, job: Job, args, status: Status):
    """Hands a result to rope: the PNG, its recipe and its workflow, as files for `rope ARGS`. Returns rope's
    answer, or None when rope failed; the GPU work is done then, so the files go to rescue/ in the library (or
    the system's temp folder) with the command that brings them in later."""
    graph = json.dumps(job.graph.nodes, ensure_ascii=False)
    with tempfile.TemporaryDirectory(prefix="img2img-result-") as tmp:
        files = Path(tmp) / f"{name}.png", Path(tmp) / "made.json", Path(tmp) / "graph.json"
        files[0].write_bytes(data)
        files[1].write_text(made, "utf-8")
        files[2].write_text(graph, "utf-8")
        try:
            return a.lib.call(*args(*files))
        except LibraryError as e:
            error = e
    for base in (a.lib.root / "rescue", Path(tempfile.gettempdir()) / "img2img-rescue"):
        try:
            base.mkdir(parents=True, exist_ok=True)
            folder = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=base))
            files = folder / f"{name}.png", folder / "made.json", folder / "graph.json"
            files[0].write_bytes(data)
            files[1].write_text(made, "utf-8")
            files[2].write_text(graph, "utf-8")
            break
        except OSError:
            continue
    else:
        die(f"rope could not keep a result ({error}), and it could not be saved anywhere else either")
    command = shlex.join([a.lib.exe, *a.lib.flags, *args(*files)])
    script = folder / "bring-in.sh"
    script.write_text(f"#!/bin/sh\n# Hands this result to rope, which could not take it: {error}\nexec {command}\n", "utf-8")
    script.chmod(0o755)
    status.line(f"  rope could not keep this result: {error}\n  it is saved in {tilde(folder)}; once rope works, "
                f"bring it in with:\n    {tilde(script)}")
    return None


def keep_inputs(a: argparse.Namespace, workdir: Path) -> None:
    """Prepares the inputs for upload. With the library, each input goes into it first, read once, and from then
    on the library's copy is used: the library holds exactly the bytes ComfyUI reads, even if the original is moved
    or rewritten while the job runs. A converted input's PNG is kept with it too."""
    lib = None if a.dry_run else a.lib
    if lib is not None:
        lib.hold()
        kept = call_library(lib, "import", "--via=img2img.py", f"--repo={HERE}", "--", *map(str, a.images))
        a.inputs = [{"id": k["id"], "path": Path(k["path"])} for k in kept]
    else:
        a.inputs = [{"path": p} for p in a.images]
    prepared = prepare_inputs([i["path"] for i in a.inputs], workdir, a.names)
    a.uploads = [p for p, _ in prepared]
    a.upload_stems = [p.stem if how else name for (p, how), name in zip(prepared, a.names)]
    if lib is not None:
        for i, (path, how) in zip(a.inputs, prepared):
            if how:
                info = json.dumps({"steps": how, "libs": decoder_versions()})
                i["uploaded"] = call_library(lib, "rendition", i["id"], str(path), f"--info={info}")["uploaded"]


def resolve_images(a: argparse.Namespace) -> None:
    """Turns each @ID input into its file in the library, and sets a.names: the name of each input, used for the
    result's name and the upload names (a file in the library is named after its rope name, not its hash)."""
    a.names = []
    for n, img in enumerate(a.images):
        in_library = a.lib is not None and img.exists() and img.resolve().is_relative_to(a.lib.root)
        if str(img).startswith("@") and not img.exists() or in_library:
            if a.lib is None:
                die(f"{img} names an image in rope, which this run does not use")
            info = call_library(a.lib, "show", str(img))
            a.images[n] = Path(info["path"])
            a.names.append(info["name"] or info["short"])
        else:
            a.names.append(img.stem)


# ---------------------------------------------------------------- running jobs

@dataclass
class Job:
    index: int
    seed: int
    graph: Graph
    started: float = 0.0
    saved: list[str] = field(default_factory=list)  # where each result went, for the status line


def png_size(data: bytes) -> tuple[int, int] | None:
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    return None


def run_name(a: argparse.Namespace, seed: int) -> str:
    """<time>_<first image>_<model>_s<seed>: sorts by time, and the folders of one -n run share the time."""
    stem = re.sub(r"[^\w.-]+", "_", a.stem)[:40].strip("_.") or "image"  # no spaces or shell characters; keeps CJK
    return f"{time.strftime('%Y%m%d-%H%M%S', a.when)}_{stem}_{a.model}_s{seed}"


def exact(x: float) -> str:
    """The shortest text that reads back as the same float: 1, 0.8, 2.123456789."""
    return str(int(x)) if x.is_integer() else repr(x)


def image_flags(a: argparse.Namespace, seed: int) -> list[str]:
    """Every flag that changes the image, as argv items: the seed is this image's own and every default is spelled
    out, so later changes to the defaults cannot change the result. A new image-affecting flag has to go here too."""
    flags = [f"--model={a.model}"]
    if a.model == "2511":
        flags.append(f"--lightning={a.lightning}")
    else:
        flags.append(f"--resolution={a.resolution}")
        if a.enhance:
            flags.append("--enhance")
    flags += [f"--steps={a.steps}", f"--cfg={exact(a.cfg)}", f"--seed={seed}"]
    flags += [f"--lora={name}:{exact(strength)}" for name, strength in a.lora]
    if a.negative:
        flags.append(f"--negative={a.negative}")
    return flags


def collect(comfy: Comfy, pid: str, job: Job, entry: dict | None, a: argparse.Namespace, status: Status) -> None:
    for _ in range(50):  # execution_success can arrive a moment before the history entry is written
        entry = entry or comfy.history(pid)
        if entry:
            break
        time.sleep(0.2)
    else:
        die(f"ComfyUI finished prompt {pid} but has no history entry for it")
    state = entry.get("status") or {}
    if state.get("status_str") == "error":
        errors = [m[1] for m in state.get("messages", []) if m and m[0] == "execution_error"]
        e = errors[0] if errors else {}
        die(f"ComfyUI failed in {e.get('node_type', '?')}: {str(e.get('exception_message', '')).strip()}")
    outputs = entry.get("outputs") or {}
    enhanced = ""
    if job.graph.text_out:
        text = (outputs.get(job.graph.text_out) or {}).get("text") or []
        if text:
            enhanced = text[0]
            status.line(f"  enhanced prompt: {enhanced}")
    refs = (outputs.get(job.graph.image_out) or {}).get("images") or []
    if not refs:
        die("ComfyUI finished without returning an image")
    took = f", {time.monotonic() - job.started:.1f}s" if job.started else ""
    for ref in refs:
        data = comfy.fetch(ref)
        where = a.keep(a, job, data, enhanced, status)
        size = png_size(data)
        status.line(f"  -> {where}" + (f"  {size[0]}x{size[1]}" if size else "") + took)
        job.saved.append(where)


def wait(ws, comfy: Comfy, pending: dict[str, Job], total: int, a: argparse.Namespace, status: Status) -> None:
    import websocket

    deadline = time.monotonic() + a.timeout
    while pending:
        if time.monotonic() > deadline:
            die(f"gave up after {a.timeout}s; the job may still be running in ComfyUI")
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            for pid in list(pending):  # quiet for a while: make sure no finish message was missed
                entry = comfy.history(pid)
                if entry is not None:
                    collect(comfy, pid, pending.pop(pid), entry, a, status)
            continue
        except websocket.WebSocketConnectionClosedException:
            die("lost the connection to ComfyUI" + (f":\n{comfy.log_tail()}" if comfy.proc else ""))
        if not isinstance(raw, str):
            continue  # binary frames carry live previews
        msg = json.loads(raw)
        kind, data = msg.get("type"), msg.get("data") or {}
        pid = data.get("prompt_id")
        job = pending.get(pid)
        if job is None:
            continue
        node = data.get("node")
        if kind == "execution_start":
            job.started = time.monotonic()
            status.line(f"[{job.index}/{total}] seed {job.seed}")
        elif kind == "executing" and node in job.graph.labels:
            status.show(job.graph.labels[node])
        elif kind == "progress" and node in job.graph.labels:
            status.show(job.graph.labels[node], f"{data.get('value')}/{data.get('max')}")
        elif kind == "execution_error":
            status.line("")
            die(f"ComfyUI failed in {data.get('node_type')} (node {data.get('node_id')}): "
                f"{str(data.get('exception_message', '')).strip()}")
        elif kind == "execution_interrupted":
            die("the job was interrupted inside ComfyUI")
        elif kind == "execution_success" or (kind == "executing" and node is None):
            collect(comfy, pid, pending.pop(pid), None, a, status)


def describe(a: argparse.Namespace, base: int) -> str:
    bits = [f"model {a.model}"]
    if a.model == "2511":
        bits.append(f"Lightning {a.lightning}-step" if a.lightning != "off" else "Lightning off")
    else:
        bits.append(f"resolution {a.resolution or 'as input'}")
        if a.enhance:
            bits.append("prompt enhancer on")
    bits += [f"{a.steps} steps", f"cfg {a.cfg:g}", f"seed {base}" + (f"-{base + a.count - 1}" if a.count > 1 else "")]
    if a.lora:
        bits.append("LoRA " + ", ".join(f"{name}:{strength:g}" for name, strength in a.lora))
    return f"queued {a.count} job{'s' if a.count > 1 else ''}: " + ", ".join(bits)


def run_jobs(a: argparse.Namespace, comfy: Comfy) -> list[str]:
    import websocket

    status = Status()
    names = [comfy.upload(p, upload_name(p, s)) for p, s in zip(a.uploads, a.upload_stems)]
    system_prompt = comfy.enhancer_system_prompt() if a.enhance else None
    base = a.seed if a.seed is not None else random.randrange(2**32)
    a.when, a.server_info = time.localtime(), comfy.system_info()  # one time for all the folders of this run
    a.upload_names, a.keep = names, keep_result
    if a.lib is not None:
        a.env, a.code = comfy_env(comfy), code_identity()
    ws = websocket.create_connection(f"{comfy.ws_url}?clientId={comfy.client_id}", timeout=15)
    jobs: dict[str, Job] = {}
    try:
        for i in range(a.count):
            graph = build(a, names, base + i, system_prompt)
            a.models = model_files(graph.nodes)
            jobs[comfy.queue(graph)] = Job(i + 1, base + i, graph)
        print(describe(a, base))
        pending = dict(jobs)
        try:
            wait(ws, comfy, pending, len(jobs), a, status)
        except KeyboardInterrupt:
            status.line("\ncancelling: dropping queued jobs and interrupting the running one")
            comfy.cancel(list(pending))
            raise
    finally:
        ws.close()
    return [p for job in jobs.values() for p in job.saved]


# ---------------------------------------------------------------- cli

def parse_lora(text: str) -> tuple[str, float]:
    name, strength = text, 1.0
    head, sep, tail = text.rpartition(":")
    if sep:
        try:
            name, strength = head, float(tail)
        except ValueError:
            pass
    if not re.search(r"\.(safetensors|ckpt|pt|pth|bin)$", name):
        name += ".safetensors"
    return name, strength


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", type=Path, metavar="IMAGE",
                   help="input images, or @ID for an image in rope; the first one sets the output size")
    p.add_argument("prompt", help="the edit to make, in Chinese or English")
    p.add_argument("-m", "--model", choices=["2511", "2.1"], default="2511",
                   help="Qwen-Image-Edit-2511 (default) or Qwen-Image-2.1")
    p.add_argument("-n", "--count", type=int, default=1, help="number of variations, one seed each (default 1)")
    p.add_argument("-s", "--seed", type=int, help="seed of the first variation (default: random, and printed)")
    p.add_argument("--negative", default="", help="negative prompt; only has an effect when cfg is above 1")
    p.add_argument("--steps", type=int, help="sampling steps (default: 2511 uses 4 with Lightning, 40 without; 2.1 uses 25)")
    p.add_argument("--cfg", type=float, help="guidance scale (default: 2511 uses 1 with Lightning, 4 without; 2.1 uses 1)")
    p.add_argument("--lora", type=parse_lora, action="append", default=[], metavar="FILE[:STRENGTH]",
                   help="extra LoRA from models/loras, e.g. style.safetensors:0.8; repeatable")
    g = p.add_argument_group("Qwen-Image-Edit-2511 only")
    g.add_argument("--lightning", choices=["4", "8", "off"],
                   help="speed LoRA: 4-step (default), 8-step (download it first), or off for the full 40-step model")
    g = p.add_argument_group("Qwen-Image-2.1 only")
    g.add_argument("--resolution", type=int, help="reference images are resized to about RxR pixels (default 1024); "
                                                  "0 keeps their own size, which is slow for large photos")
    g.add_argument("--enhance", action="store_true",
                   help="let the 9B prompt enhancer rewrite the prompt first (loads another 9.5 GB model)")
    g = p.add_argument_group("rope library", "Results go into the rope image library (see rope/README.md), and so does "
                             "every input, so that a result can be made again after its inputs are moved or deleted. "
                             "rope has to be built first.")
    g.add_argument("--into", metavar="FOLDER", default=os.environ.get("IMG2IMG_INTO") or None,
                   help="the rope folder for the results, e.g. trips/kyoto; it must exist (rope mkdir -p ...). "
                        "Default: $IMG2IMG_INTO, else unfiled")
    g.add_argument("--tag", dest="tags", action="append", default=[], metavar="TAG", help="tag the results; repeatable")
    g.add_argument("--note", help="a note on each result")
    g.add_argument("--library", metavar="DIR", help="the rope library (default: $ROPE_LIBRARY, else ~/.local/share/rope)")
    g.add_argument("--replay", metavar="ID", help="make a rope image again from its record; see --replay ID --help")
    g = p.add_argument_group("ComfyUI")
    add_comfy_args(g)
    g.add_argument("--dry-run", action="store_true", help="print the API workflow JSON and stop; nothing is uploaded or run")
    a = p.parse_args(argv)

    for img in a.images:
        if not img.is_file() and not (str(img).startswith("@") and not img.exists()):
            p.error(f"no such image: {img}")
    if len(a.images) > MAX_IMAGES[a.model]:
        p.error(f"model {a.model} takes at most {MAX_IMAGES[a.model]} images")
    if a.count < 1:
        p.error("--count must be at least 1")
    if a.model == "2511":
        if a.resolution is not None or a.enhance:
            p.error("--resolution and --enhance belong to Qwen-Image-2.1 (-m 2.1)")
        a.lightning = a.lightning or "4"
        fast = a.lightning != "off"
        a.steps = a.steps or (int(a.lightning) if fast else 40)
        a.cfg = a.cfg if a.cfg is not None else (1.0 if fast else 4.0)
    else:
        if a.lightning is not None:
            p.error("--lightning belongs to Qwen-Image-Edit-2511")
        a.resolution = 1024 if a.resolution is None else a.resolution
        if not 0 <= a.resolution <= 4096:
            p.error("--resolution takes 0 to 4096")
        a.steps = a.steps or 25
        a.cfg = a.cfg if a.cfg is not None else 1.0
    a.comfy_args = shlex.split(a.comfy_args)
    return a


def add_comfy_args(g) -> None:
    g.add_argument("--server", default=SERVER, help=f"ComfyUI address (default {SERVER})")
    g.add_argument("--keep-server", action="store_true", help="when this command starts ComfyUI, leave it running")
    g.add_argument("--no-start", action="store_true", help="fail instead of starting ComfyUI when none is running")
    g.add_argument("--comfy-args", default="", metavar="ARGS",
                   help='extra flags for a ComfyUI this command starts, e.g. "--reserve-vram 2"')
    g.add_argument("--timeout", type=int, default=TIMEOUT, help=f"give up after this many seconds (default {TIMEOUT})")


def check_packages(needed: dict[str, str]) -> None:
    """Fail with a how-to instead of a ModuleNotFoundError when this Python lacks the script's packages."""
    import importlib.util

    missing = [package for module, package in needed.items() if importlib.util.find_spec(module) is None]
    if missing:
        die(f"this Python ({sys.executable}) lacks {', '.join(missing)}. Either run the script directly, "
            f"as ./{Path(__file__).name}, and uv supplies the packages; or run ./setup_venv.sh once and use "
            f"{tilde(HERE / '.venv' / 'bin' / 'python')}, e.g. after: source {tilde(HERE / '.venv' / 'bin' / 'activate')}")


def main(argv: list[str]) -> int:
    if any(arg == "--replay" or arg.startswith("--replay=") for arg in argv):
        return replay_main(argv)
    a = parse_args(argv)
    check_packages({"websocket": "websocket-client", "PIL": "pillow", "pillow_heif": "pillow-heif",
                    "pillow_jxl": "pillow-jxl-plugin", "rawpy": "rawpy"})
    sys.stdout.reconfigure(line_buffering=True)
    # a dry run keeps nothing, so it needs rope only to look up @ID inputs
    at_ids = any(str(i).startswith("@") and not i.exists() for i in a.images)
    a.lib = Library.find(a) if not a.dry_run or at_ids else None
    if a.into and not a.dry_run:
        call_library(a.lib, "where", f"--folder={a.into}")  # a missing folder fails now, not after the GPU work
    resolve_images(a)
    a.stem = a.names[0]
    with tempfile.TemporaryDirectory(prefix="img2img-") as workdir:
        keep_inputs(a, Path(workdir))
        if a.dry_run:
            names = [f"{UPLOAD_SUBFOLDER}/{upload_name(p, s)}" for p, s in zip(a.uploads, a.upload_stems)]
            system_prompt = "<system prompt from ComfyUI's Qwen-Image-2.1 edit template>" if a.enhance else None
            graph = build(a, names, a.seed if a.seed is not None else 0, system_prompt)
            print(json.dumps(graph.nodes, indent=2, ensure_ascii=False))
            return 0
        return run(a, run_jobs)


def run(a: argparse.Namespace, work) -> int:
    comfy = Comfy(a.server)
    started = False
    if not comfy.alive():
        if a.no_start:
            die(f"no ComfyUI answers at {a.server}; start it with {tilde(LAUNCHER)} or drop --no-start")
        comfy.start(a.comfy_args)
        started = True
    t0 = time.monotonic()
    try:
        saved = work(a, comfy)
    finally:
        if started and a.keep_server and comfy.proc.poll() is None:
            print(f"ComfyUI keeps running at {a.server} (pid {comfy.proc.pid}); stop it with: kill -INT {comfy.proc.pid}")
        elif started:
            print("stopping the ComfyUI this command started")
            comfy.stop()
    print(f"done: {len(saved)} image{'' if len(saved) == 1 else 's'} in {time.monotonic() - t0:.0f}s")
    return 0


# ---------------------------------------------------------------- replay

REPLAY_LABELS = {"UNETLoader": "loading the diffusion model", "CLIPLoader": "loading the text encoder",
                 "TextGenerate": "enhancing the prompt", "TextEncodeQwenImageEditPlus": "encoding the prompt and images",
                 "TextEncodeQwenImage21": "encoding the prompt and images", "KSampler": "sampling",
                 "VAEDecode": "decoding"}


def parse_replay_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description="Makes an image in the rope library again. It uploads the bytes its inputs had, sends the "
                    "workflow recorded with it, and compares the pixels of the new image with the old: an identical "
                    "result is only recorded, a different one is kept as a new image next to the old. Before sending, "
                    "it says what differs from when the image was made: the workflow today's img2img.py would build, "
                    "model files, ComfyUI and its packages.")
    p.add_argument("--replay", required=True, metavar="ID", help="the image: at least 6 hex digits of its id")
    p.add_argument("-s", "--seed", type=int, help="change only the sampling seed")
    p.add_argument("--rebuild", action="store_true",
                   help="send the workflow today's img2img.py builds from the recorded flags, not the recorded one")
    p.add_argument("--rerun-enhancer", action="store_true",
                   help="2.1 with --enhance: let the prompt enhancer write the prompt again, instead of reusing what "
                        "it wrote then (it samples, so the result will differ)")
    p.add_argument("--library", metavar="DIR", help="the rope library (default: $ROPE_LIBRARY, else ~/.local/share/rope)")
    p.add_argument("--dry-run", action="store_true", help="print the differences and the workflow, and stop")
    g = p.add_argument_group("ComfyUI")
    add_comfy_args(g)
    a = p.parse_args(argv)
    a.comfy_args = shlex.split(a.comfy_args)
    a.into, a.tags, a.note, a.count = None, [], None, 1
    return a


def replay_main(argv: list[str]) -> int:
    a = parse_replay_args(argv)
    check_packages({"websocket": "websocket-client"})
    sys.stdout.reconfigure(line_buffering=True)
    a.lib = Library.find(a)
    info = call_library(a.lib, "show", a.replay)
    made = info["meta"].get("made")
    if not made:
        die(f"{info['short']} has no recipe: rope did not see it being made, so it cannot be replayed")
    missing = [str(i["slot"]) for i in info["inputs"] if i.get("missing")]
    if missing:
        die(f"{info['short']} cannot be replayed: rope lacks the bytes of input image {', '.join(missing)}")
    a.info, a.made, a.prompt, a.flags = info, made, made.get("prompt", ""), made.get("flags", [])
    a.inputs = [{"id": i["id"], "uploaded": i.get("uploaded"), "path": Path(i["path"]), "recorded": i.get("upload")}
                for i in info["inputs"]]
    a.today = parse_args([*a.flags, "--", *(str(i["path"]) for i in a.inputs), a.prompt])
    a.seed_used = a.seed if a.seed is not None else a.today.seed
    a.recorded = info.get("graph")
    if a.recorded is None:
        print(f"{info['short']} has no recorded workflow, so the one today's img2img.py builds is sent")
    else:
        names = [i["recorded"] or "" for i in a.inputs]
        rebuilt = build(a.today, names, a.today.seed, enhancer_prompt(a.recorded)).nodes
        diff = graph_diff(a.recorded, rebuilt)
        if diff:
            sent = "today's (--rebuild)" if a.rebuild else "the recorded one; --rebuild sends today's"
            print(f"today's img2img.py builds a different workflow from these flags; sending {sent}:")
            for line in diff[:20]:
                print(f"  {line}")
            if len(diff) > 20:
                print(f"  ... and {len(diff) - 20} more")
    for line in model_changes(made.get("models") or [], model_files(a.recorded or {})):
        print(f"  model: {line}")
    if a.dry_run:
        nodes = replay_graph(a, [i["recorded"] or "" for i in a.inputs], "<system prompt from ComfyUI>")
        print(json.dumps(nodes, indent=2, ensure_ascii=False))
        return 0
    return run(a, replay_jobs)


def replay_graph(a: argparse.Namespace, names: list[str], system_prompt: str | None) -> dict:
    """The workflow to send: the recorded one with the inputs renamed to `names`, or today's build of the recorded
    flags; with the seed changed by -s, and the enhanced prompt reused unless --rerun-enhancer."""
    if a.rebuild or a.recorded is None:
        nodes = build(a.today, names, a.seed_used, system_prompt).nodes
    else:
        nodes = copy.deepcopy(a.recorded)
        renamed = {i["recorded"]: n for i, n in zip(a.inputs, names) if i["recorded"]}
        for node in nodes.values():
            if node["class_type"] == "LoadImage" and node["inputs"].get("image") in renamed:
                node["inputs"]["image"] = renamed[node["inputs"]["image"]]
        for node in nodes.values():
            if node["class_type"] == "KSampler":
                node["inputs"]["seed"] = a.seed_used
            if node["class_type"] == "TextGenerate" and "sampling_mode.seed" in node["inputs"]:
                node["inputs"]["sampling_mode.seed"] = a.seed_used
    enhanced = a.made.get("enhanced_prompt")
    if enhanced and not a.rerun_enhancer:
        pin_enhanced(nodes, enhanced)
    return nodes


def enhancer_prompt(nodes: dict) -> str | None:
    return next((n["inputs"].get("system_prompt") for n in nodes.values() if n["class_type"] == "TextGenerate"), None)


def pin_enhanced(nodes: dict, text: str) -> None:
    """Feeds the prompt the enhancer wrote straight to the text encoder and drops the enhancer: it samples, so
    running it again may write another prompt. PreviewAny passes its string on unchanged, so the encoder gets the
    same text it got the first time."""
    for gid in [nid for nid, n in nodes.items() if n["class_type"] == "TextGenerate"]:
        gen = nodes.pop(gid)
        for nid, n in list(nodes.items()):
            if n["class_type"] == "PreviewAny" and n["inputs"].get("source") == [gid, 0]:
                del nodes[nid]
                continue
            for k, v in n["inputs"].items():
                if v == [gid, 0]:
                    n["inputs"][k] = text
        for link in (gen["inputs"].get("clip"), gen["inputs"].get("image")):  # the enhancer's model and image batch
            if isinstance(link, list) and not any(link[0] == v[0] for n in nodes.values()
                                                  for v in n["inputs"].values() if isinstance(v, list)):
                nodes.pop(link[0], None)


def graph_diff(old: dict, new: dict) -> list[str]:
    """How two API workflows differ, node by node."""
    def show(v) -> str:
        s = json.dumps(v, ensure_ascii=False)
        return s if len(s) <= 60 else s[:57] + "..."

    out = []
    for nid in sorted(set(old) | set(new), key=lambda k: (len(k), k)):
        o, n = old.get(nid), new.get(nid)
        if o is None:
            out.append(f"+ node {nid} {n['class_type']}")
        elif n is None:
            out.append(f"- node {nid} {o['class_type']}")
        elif o["class_type"] != n["class_type"]:
            out.append(f"node {nid}: {o['class_type']} -> {n['class_type']}")
        else:
            for k in sorted(set(o["inputs"]) | set(n["inputs"])):
                if o["inputs"].get(k) != n["inputs"].get(k):
                    out.append(f"{o['class_type']}.{k}: {show(o['inputs'].get(k))} -> {show(n['inputs'].get(k))}")
    return out


def model_changes(then: list[dict], now: list[dict]) -> list[str]:
    out = []
    for old in then:
        new = next((m for m in now if m["dir"] == old["dir"] and m["name"] == old["name"]), None)
        if new is None:
            continue
        if "size" in old and "size" not in new:
            out.append(f"{old['name']} is missing now")
        elif (old.get("size"), old.get("mtime_ns")) != (new.get("size"), new.get("mtime_ns")):
            out.append(f"{old['name']} is not the same file as then (size or modification time differs)")
    return out


def env_changes(then: dict, now: dict, prefix: str = "") -> list[str]:
    out = []
    for k in sorted(set(then) | set(now)):
        o, n = then.get(k), now.get(k)
        if isinstance(o, dict) and isinstance(n, dict):
            out += env_changes(o, n, f"{prefix}{k}.")
        elif o != n and o is not None:
            out.append(f"{prefix}{k}: {o} -> {n}")
    return out


def replay_jobs(a: argparse.Namespace, comfy: Comfy) -> list[str]:
    import websocket

    status = Status()
    tag = "r" + time.strftime("%H%M%S")  # a fresh name: ComfyUI would answer the same graph from its cache
    names = []
    for i in a.inputs:
        stem = re.sub(r"^[0-9a-f]{12}_(r\d{6}_)?", "", Path(i["recorded"] or i["path"].name).stem)
        names.append(comfy.upload(i["path"], upload_name(i["path"], stem, tag)))
    a.when, a.server_info = time.localtime(), comfy.system_info()
    a.env, a.code, a.upload_names, a.keep = comfy_env(comfy), code_identity(), names, keep_replay
    env_id = a.made.get("env")
    if isinstance(env_id, str) and (a.lib.root / "envs" / f"{env_id}.json").exists():
        then = json.loads((a.lib.root / "envs" / f"{env_id}.json").read_text("utf-8"))
        for line in env_changes(then, a.env):
            print(f"  environment: {line}")
    system_prompt = comfy.enhancer_system_prompt() if a.today.enhance and (a.rebuild or a.recorded is None) else None
    nodes = replay_graph(a, names, system_prompt)
    a.models = model_files(nodes)
    graph = Graph()
    graph.nodes = nodes
    graph.labels = {nid: REPLAY_LABELS[n["class_type"]] for nid, n in nodes.items() if n["class_type"] in REPLAY_LABELS}
    graph.image_out = next(nid for nid, n in nodes.items() if n["class_type"] == "PreviewImage")
    graph.text_out = next((nid for nid, n in nodes.items() if n["class_type"] == "PreviewAny"), "")
    ws = websocket.create_connection(f"{comfy.ws_url}?clientId={comfy.client_id}", timeout=15)
    job = Job(1, a.seed_used, graph)
    try:
        pending = {comfy.queue(graph): job}
        print(f"replaying {a.info['short']} {a.info['name']}, seed {a.seed_used}")
        try:
            wait(ws, comfy, pending, 1, a, status)
        except KeyboardInterrupt:
            status.line("\ncancelling the replay")
            comfy.cancel(list(pending))
            raise
    finally:
        ws.close()
    return job.saved


def keep_replay(a: argparse.Namespace, job: Job, data: bytes, enhanced: str, status: Status) -> str:
    """Records a replay's result with the image it replays, and returns what came of it."""
    flags = image_flags(a.today, a.seed_used) if a.rebuild or a.recorded is None else \
        [f"--seed={a.seed_used}" if f.startswith("--seed=") else f for f in a.flags]
    made = json.dumps(recipe(a, job, flags, enhanced or a.made.get("enhanced_prompt") or ""), ensure_ascii=False)

    def args(png: Path, made: Path, graph: Path) -> list[str]:
        return ["replayed", a.info["id"], str(png), f"--made={made}", f"--graph={graph}", f"--repo={HERE}"]

    r = send_result(a, f"replay-{a.info['short']}", data, made, job, args, status)
    if r is None:
        return "not recorded in rope; see above"
    if r["result"] == "identical":
        return f"identical to {a.info['short']}, pixel for pixel"
    return (f"differs from {a.info['short']}: largest difference {r['max_diff']} of 255, in "
            f"{r['differing'] * 100:.2f}% of the pixels; kept as {r['id'][:12]}")


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
