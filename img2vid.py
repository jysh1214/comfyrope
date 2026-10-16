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
"""Image to video from the command line: a start image plus a prompt make a short clip with Wan 2.2, the 14B
image-to-video pair, through ComfyUI's API. A first version, built on comfycore.py like img2img.py.

    ./img2vid.py photo.jpg "the camera slowly pushes in while the leaves move in the wind"
    ./img2vid.py photo.jpg "..." -n 2 -s 42 --seconds 3 --size 480
    ./img2vid.py photo.jpg "..." --lightning off      # the full 20-step model at cfg 3.5: about five times slower
    ./img2vid.py photo.jpg "..." --dry-run            # only print the API workflow JSON

The clip keeps the start image's aspect ratio, scaled so that its long side is --size pixels (default 640), and
lasts --seconds at --fps (default 5 s at Wan's own 16 fps: 81 frames). The start image is read as img2img.py
reads its inputs (HEIC, RAW and the like are converted first). Prompts work best as a description of the
motion: what moves, and how the camera moves.

rope cannot keep videos yet, so each clip goes to --out (default videos/ in the repo), or to a folder under it
with --into, as <name>.mp4, next to <name>.made.json, its recipe, and <name>.graph.json, the workflow ComfyUI
ran: the files rope add takes, for when it knows videos. The start image does go into rope, so the recipe names it by id. Names are
<time>_<image>_<model>_s<seed>, as img2img.py's. ComfyUI keeps its own copy of each clip under
ComfyUI/output/img2vid/. The ComfyUI and --library options are img2img.py's, and so is the Python side:
import img2vid, build a Request per clip, and run() them in one go.

On the RTX 4090 at 640 px and 5 s, ComfyUI's own template notes about 1.5 minutes per clip with the 4-step
LoRAs (the first one longer: 30 GB of models to load) and about 9 minutes without.
"""
from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from comfycore import (FETCH, HERE, SERVER, TIMEOUT, UPLOAD_SUBFOLDER, Error, Graph, Inputs, Job, Library, Options,
                       Result, Run, Tool, add_comfy_args, add_loras, check_packages, exact, link, parse_lora,
                       prepare, recipe, rope_ref, run_name, run_requests, tilde, upload_name)

__all__ = ["OUT", "SERVER", "TIMEOUT", "Error", "Request", "Result", "run", "workflow"]

OUT = HERE / "videos"  # where the clips go until rope keeps videos

# Model files as ComfyUI's template names them (download_models.py wan fetches them).
WAN22 = {
    "high": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    "low": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "wan_2.1_vae.safetensors",
}
LIGHTX2V = {  # the 4-step LoRAs, one per model
    "high": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
    "low": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
}
FETCH.update(dict.fromkeys([*WAN22.values(), *LIGHTX2V.values()], "uv run download_models.py wan"))
MODELS = {"14b": "Wan 2.2 image-to-video, the A14B high-noise and low-noise pair"}
# Wan's own negative prompt, as the template has it: garish colours, overexposure, still frames, blur, text, ...
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
            "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
            "三条腿，背景人很多，倒着走")
PACKAGES = {"websocket": "websocket-client", "PIL": "pillow", "pillow_heif": "pillow-heif",
            "pillow_jxl": "pillow-jxl-plugin", "rawpy": "rawpy"}  # module -> package, as the header lists them


# ---------------------------------------------------------------- requests

@dataclass
class Request:
    """What to make: the start image, the prompt and the settings, one field per command-line option (see
    parse_args, or --help). A field left at None takes the option's default; settled() fills them in.
    comfycore.Settled names the fields the core needs of it."""

    images: list[Path | str]  # the start image, alone in a list as comfycore wants it, or "@ID" for one in rope
    prompt: str
    model: str = "14b"  # see MODELS
    count: int = 1  # -n: how many clips, one seed each
    seed: int | None = None  # -s: the first clip's seed; None picks one at random
    negative: str | None = None  # default NEGATIVE, Wan's own; has an effect only when cfg is above 1
    steps: int | None = None  # default 4 with Lightning, 20 without; the high-noise model takes the first half
    cfg: float | None = None  # default 1 with Lightning, 3.5 without
    shift: float | None = None  # ModelSamplingSD3's shift; default 5
    lora: list[tuple[str, float] | str] = field(default_factory=list)  # applied to both models; "style.safetensors:0.8"
    lightning: str | None = None  # "4", the lightx2v 4-step LoRAs (default), or "off"
    seconds: float | None = None  # default 5
    fps: int | None = None  # default 16, Wan's own rate; another value only changes the playback speed
    size: int | None = None  # the long side of the clip, in pixels; default 640
    into: str | None = None  # a folder under `out` for the clips, e.g. "trips/京都"; the rope folder once rope keeps videos
    tags: list[str] = field(default_factory=list)
    note: str | None = None

    def settled(self) -> Request:
        """A copy with the defaults filled in and the settings checked, which is what run() works from and what a
        Result reports. A bad setting is a ValueError. The original is left as it is."""
        r = copy.copy(self)
        r.images = [Path(i) for i in ([self.images] if isinstance(self.images, (str, Path)) else self.images)]
        r.model = str(self.model)
        r.tags = [str(t) for t in ([self.tags] if isinstance(self.tags, str) else self.tags)]
        loras = [self.lora] if isinstance(self.lora, str) else list(self.lora)
        r.lora = [parse_lora(l) if isinstance(l, str) else (str(l[0]), float(l[1])) for l in loras]
        r.into = self.into or None
        if r.into and (Path(r.into).is_absolute() or ".." in Path(r.into).parts):
            raise ValueError("into is a folder under out: a relative path, without ..")
        r.note = self.note or None
        if len(r.images) != 1:
            raise ValueError("exactly one start image is needed")
        if not r.images[0].is_file() and not rope_ref(r.images[0]):
            raise ValueError(f"no such image: {r.images[0]}")
        if r.model not in MODELS:
            raise ValueError(f"model must be {' or '.join(MODELS)}, not {r.model}")
        if r.count < 1:
            raise ValueError("count must be at least 1")
        r.lightning = "4" if self.lightning is None else str(self.lightning)
        if r.lightning not in ("4", "off"):
            raise ValueError(f"lightning must be 4 or off, not {r.lightning}")
        fast = r.lightning != "off"
        r.steps = self.steps or (4 if fast else 20)
        r.cfg = (1.0 if fast else 3.5) if self.cfg is None else float(self.cfg)
        r.shift = 5.0 if self.shift is None else float(self.shift)
        r.negative = NEGATIVE if self.negative is None else self.negative
        r.seconds = 5.0 if self.seconds is None else float(self.seconds)
        r.fps = 16 if self.fps is None else int(self.fps)
        r.size = 640 if self.size is None else int(self.size)
        if r.steps < 2:
            raise ValueError("steps must be at least 2: one for each model")
        if not 0 < r.seconds <= 60:
            raise ValueError("seconds takes more than 0 and at most 60")
        if not 1 <= r.fps <= 120:
            raise ValueError("fps takes 1 to 120")
        if not 64 <= r.size <= 2048:
            raise ValueError("size takes 64 to 2048")
        return r


def frames(r: Request) -> int:
    """How many frames a clip has: seconds * fps, as the nearest 4k+1, which is what Wan's latent holds; 81 for
    5 s at 16 fps."""
    return max(1, round(r.seconds * r.fps / 4)) * 4 + 1


def frame_size(image: Path, size: int) -> tuple[int, int]:
    """The clip's width and height: the start image's aspect ratio with the long side scaled to `size`, both
    sides rounded to a multiple of 16 (what Wan's latent needs). WanImageToVideo then scales the image to it."""
    from PIL import Image

    with Image.open(image) as im:
        w, h = im.size
        if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):  # turned by its EXIF orientation, as ComfyUI reads it
            w, h = h, w
    scale = size / max(w, h)
    return max(16, round(w * scale / 16) * 16), max(16, round(h * scale / 16) * 16)


# ---------------------------------------------------------------- workflows (ComfyUI API format)

def build(r: Request, inputs: Inputs, seed: int) -> Graph:
    """ComfyUI template 'Wan 2.2 14B Image to Video', flattened: the two models sample one after the other, the
    high-noise one the first half of the steps, with the 4-step LoRAs unless Lightning is off."""
    g = Graph()
    width, height = frame_size(inputs.uploads[0], r.size)
    image = g.add("LoadImage", image=inputs.uploaded[0])
    clip = g.add("CLIPLoader", "loading the text encoder", clip_name=WAN22["clip"], type="wan", device="default")
    vae = g.add("VAELoader", vae_name=WAN22["vae"])
    models = {}
    for noise in ("high", "low"):
        model = g.add("UNETLoader", f"loading the {noise}-noise model", unet_name=WAN22[noise], weight_dtype="default")
        if r.lightning != "off":
            model = g.add("LoraLoaderModelOnly", model=link(model), lora_name=LIGHTX2V[noise], strength_model=1.0)
        model = add_loras(g, model, r.lora)
        models[noise] = g.add("ModelSamplingSD3", model=link(model), shift=r.shift)
    pos = g.add("CLIPTextEncode", "encoding the prompt", clip=link(clip), text=r.prompt)
    neg = g.add("CLIPTextEncode", clip=link(clip), text=r.negative)
    wan = g.add("WanImageToVideo", "encoding the start image", positive=link(pos), negative=link(neg), vae=link(vae),
                width=width, height=height, length=frames(r), batch_size=1, start_image=link(image))
    split = (r.steps + 1) // 2  # the template's split_step: 2 of 4, 10 of 20
    sampling = {"steps": r.steps, "cfg": r.cfg, "sampler_name": "euler", "scheduler": "simple",
                "positive": link(wan, 0), "negative": link(wan, 1)}
    high = g.add("KSamplerAdvanced", "sampling with the high-noise model", model=link(models["high"]),
                 add_noise="enable", noise_seed=seed, latent_image=link(wan, 2), start_at_step=0, end_at_step=split,
                 return_with_leftover_noise="enable", **sampling)
    low = g.add("KSamplerAdvanced", "sampling with the low-noise model", model=link(models["low"]),
                add_noise="disable", noise_seed=0, latent_image=link(high), start_at_step=split, end_at_step=r.steps,
                return_with_leftover_noise="disable", **sampling)
    decoded = g.add("VAEDecode", "decoding", samples=link(low), vae=link(vae))
    video = g.add("CreateVideo", "encoding the video", images=link(decoded), fps=float(r.fps))
    g.image_out = g.add("SaveVideo", video=link(video), filename_prefix="img2vid/clip", format="mp4")  # codec auto: h264
    return g


# ---------------------------------------------------------------- running jobs

def video_flags(r: Request, seed: int) -> list[str]:
    """Every flag that changes the clip, as argv items: the seed is this clip's own and every default is spelled
    out, so later changes to the defaults cannot change the result. A new clip-affecting flag has to go here too."""
    flags = [f"--model={r.model}", f"--lightning={r.lightning}", f"--steps={r.steps}", f"--cfg={exact(r.cfg)}",
             f"--shift={exact(r.shift)}", f"--seconds={exact(r.seconds)}", f"--fps={r.fps}", f"--size={r.size}",
             f"--seed={seed}"]
    flags += [f"--lora={name}:{exact(strength)}" for name, strength in r.lora]
    return flags + [f"--negative={r.negative}"]


def describe(r: Request, base: int) -> str:
    """A request's settings in a line: model, steps, cfg, length, size, its seeds from `base`, LoRAs."""
    bits = [f"model {r.model}", "Lightning 4-step" if r.lightning != "off" else "Lightning off", f"{r.steps} steps",
            f"cfg {r.cfg:g}", f"shift {r.shift:g}", f"{frames(r)} frames ({r.seconds:g} s at {r.fps} fps)",
            f"{r.size} px", f"seed {base}" + (f"-{base + r.count - 1}" if r.count > 1 else "")]
    if r.lora:
        bits.append("LoRA " + ", ".join(f"{name}:{strength:g}" for name, strength in r.lora))
    return ", ".join(bits)


def keep_video(out: Path, run: Run, job: Job, data: bytes, text: str, suffix: str) -> str:
    """Writes a clip to `out`, or to `out`/into with Request.into, with its recipe and the workflow, as a Result on
    its job, and says where it went. A stopgap until rope keeps videos: the three files are what `rope add` takes.
    A name already taken in that folder (the same image, model and seed in one run, with other settings) gets -2,
    -3, ... rather than overwrite anything."""
    r = job.request
    dest = out / r.into if r.into else out
    dest.mkdir(parents=True, exist_ok=True)
    base = name = run_name(run.when, job.inputs.names[0], r.model, job.seed)
    n = 1
    while (dest / f"{name}{suffix}").exists():
        n += 1
        name = f"{base}-{n}"
    made = recipe(run, job, video_flags(r, job.seed), text)
    path = dest / f"{name}{suffix}"
    path.write_bytes(data)
    (dest / f"{name}.made.json").write_text(json.dumps(made, ensure_ascii=False, indent=1), "utf-8")
    (dest / f"{name}.graph.json").write_text(json.dumps(job.graph.nodes, ensure_ascii=False), "utf-8")
    wan = next(n["inputs"] for n in job.graph.nodes.values() if n["class_type"] == "WanImageToVideo")
    result = Result(r, job.index, job.seed, path=path, folder=r.into, created=True, width=wan["width"],
                    height=wan["height"], seconds=job.took)
    job.results.append(result)
    return tilde(path)


TOOL = Tool(Path(__file__), PACKAGES, build=lambda r, inputs, seed, context: build(r, inputs, seed),
            flags=video_flags, describe=describe, unit="clip")


def run(requests: Request | Iterable[Request], *, out: str | Path = OUT, server: str = SERVER,
        keep_server: bool = False, no_start: bool = False, comfy_args: str | Iterable[str] = (),
        timeout: int = TIMEOUT, fresh: bool = False, library: str | None = None) -> list[Result]:
    """Makes the clips of every request, all through one ComfyUI, and writes them to `out` (see the top of this
    file). Returns a Result per clip: the requests in order, and each request's clips in seed order; `path` is the
    clip, `width` and `height` its size. Otherwise as img2img.run(): every request is checked first, a failure
    raises Error and drops the jobs still queued, Ctrl-C does the same, and the keyword arguments are the ComfyUI
    and --library options with the same defaults."""
    flags = shlex.split(comfy_args) if isinstance(comfy_args, str) else list(comfy_args)
    opts = Options(server, keep_server, no_start, flags, timeout, fresh, library)
    wanted = [requests] if isinstance(requests, Request) else requests
    return run_requests(TOOL, wanted, opts, keep=partial(keep_video, Path(out)))


# ---------------------------------------------------------------- cli

def parse_args(argv: list[str]) -> argparse.Namespace:
    """The command line, with the Request it asks for as .request and the rest as .options, .out and .dry_run."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path, help="the start image, or @ID for an image in rope")
    p.add_argument("prompt", help="the motion to make, in Chinese or English")
    p.add_argument("-m", "--model", choices=list(MODELS), default="14b", help="Wan 2.2 14B image-to-video (the only one yet)")
    p.add_argument("-n", "--count", type=int, default=1, help="number of clips, one seed each (default 1)")
    p.add_argument("-s", "--seed", type=int, help="seed of the first clip (default: random, and printed)")
    p.add_argument("--negative", help="negative prompt (default: Wan's own); only has an effect when cfg is above 1")
    p.add_argument("--steps", type=int, help="sampling steps, half for each model (default 4 with Lightning, 20 without)")
    p.add_argument("--cfg", type=float, help="guidance scale (default 1 with Lightning, 3.5 without)")
    p.add_argument("--shift", type=float, help="sampling shift (default 5)")
    p.add_argument("--lora", type=parse_lora, action="append", default=[], metavar="FILE[:STRENGTH]",
                   help="extra LoRA from models/loras, applied to both models, e.g. style.safetensors:0.8; repeatable")
    p.add_argument("--lightning", choices=["4", "off"],
                   help="the lightx2v 4-step LoRAs (default), or off for the full 20-step model")
    p.add_argument("--seconds", type=float, help="length of the clip (default 5)")
    p.add_argument("--fps", type=int, help="frames per second (default 16, Wan's own; another value only changes the speed)")
    p.add_argument("--size", type=int, help="the long side of the clip in pixels (default 640); the start image's aspect ratio is kept")
    g = p.add_argument_group("where the clips go")
    g.add_argument("--out", type=Path, default=OUT, metavar="DIR",
                   help=f"folder for the clips, their recipes and workflows (default {tilde(OUT)})")
    g.add_argument("--into", metavar="FOLDER",
                   help="a folder under --out for these clips, e.g. trips/京都; the rope folder once rope keeps videos")
    g.add_argument("--library", metavar="DIR", help="the rope library the start image goes into (default: $ROPE_LIBRARY, else ~/.local/share/rope)")
    g = p.add_argument_group("ComfyUI")
    add_comfy_args(g)
    g.add_argument("--dry-run", action="store_true", help="print the API workflow JSON and stop; nothing is uploaded or run")
    a = p.parse_args(argv)
    try:
        a.request = Request([a.image], a.prompt, model=a.model, count=a.count, seed=a.seed, negative=a.negative,
                            steps=a.steps, cfg=a.cfg, shift=a.shift, lora=a.lora, lightning=a.lightning,
                            seconds=a.seconds, fps=a.fps, size=a.size, into=a.into).settled()
    except ValueError as e:
        p.error(str(e))
    a.options = Options(a.server, a.keep_server, a.no_start, shlex.split(a.comfy_args), a.timeout, a.fresh, a.library)
    return a


def workflow(request: Request, seed: int = 0, *, fresh: bool = False, library: str | None = None) -> dict:
    """The API workflow run() would send for this request and seed, which --dry-run prints: nothing is uploaded,
    run or kept, and rope is only asked where an @ID input is."""
    r = request.settled()
    check_packages(TOOL.packages, TOOL.file)
    lib = Library.find(library) if any(rope_ref(i) for i in r.images) else None
    with tempfile.TemporaryDirectory(prefix="img2vid-") as workdir:
        inputs = prepare([r], Path(workdir), lib, keep=False)[0]
        tag = "f000000" if fresh else ""  # the real tag is random
        inputs.uploaded = [f"{UPLOAD_SUBFOLDER}/{upload_name(p, s, tag)}" for p, s in zip(inputs.uploads, inputs.stems)]
        return build(r, inputs, seed).nodes


def main(argv: list[str]) -> int:
    a = parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    if a.dry_run:
        seed = a.request.seed if a.request.seed is not None else 0
        print(json.dumps(workflow(a.request, seed, fresh=a.fresh, library=a.library), indent=2, ensure_ascii=False))
        return 0
    run(a.request, out=a.out, **vars(a.options))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Error as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
