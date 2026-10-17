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

The command hands its images to rope-queue (rope/README.md), which owns ComfyUI, runs the jobs in order and
puts every result into rope, and it comes back at once; rope-queue has to be running. --no-queue does the
ComfyUI work in this command instead: it talks to the ComfyUI at --server (default http://127.0.0.1:8188),
starts one with run_comfyui.sh for the command if none answers, and --keep-server leaves that one running.

Results go into the rope image library (rope/, by default in ~/.local/share/rope), which has to be built
first, and so does every input, so a result can be made again after its inputs are moved or deleted. Each
result is named <time>_<first image>_<model>_s<seed>, for example 20261013-225301_photo_2511_s42, and rope
keeps with it the recipe, the API workflow that ComfyUI ran, and a repro.sh. --into puts the results in a rope
folder, made if there is none, and an input can be @ID, an image in rope. --replay ID makes an image again from that record and
compares the pixels. The PNG also embeds the API workflow ComfyUI ran; drop it on the ComfyUI page to see that
workflow as a node graph.

From Python, import it and give run() a Request per combination to try: each becomes a group of jobs in
rope-queue, and every result lands in rope with its recipe, so the combinations can be compared in rope-gui:

    import img2img
    skin = "qwen-edit-skin-1.1.safetensors"
    requests = [img2img.Request(["photo.jpg"], "make the subjects skin details more prominent and natural",
                                seed=42, lora=[(skin, s)], into="tests/skin", tags=[f"skin {s}"])
                for s in (0.8, 1.0, 1.2)]
    for s in img2img.run(requests):
        print(s.group, s.jobs, s.request.lora)

A Request has a field per option above; run() takes fresh and library as keyword arguments.
The Python that imports it needs the packages listed at the top, e.g. .venv from setup_venv.sh.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from comfycore import (FETCH, SERVER, TIMEOUT, UPLOAD_SUBFOLDER, Comfy, ComfyError, Error, Graph, Library, Options,
                       Replay, Tool, add_comfy_args, add_loras, check_packages, die, direct_only, exact, link,
                       parse_lora, prepare, rope_ref, run_requests, upload_name)
from comfyqueue import Submitted, job_main, replay_main, submit

__all__ = ["SERVER", "TIMEOUT", "Error", "Request", "Submitted", "run", "workflow"]

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
FETCH.update({  # this tool's model files -> the command that downloads each
    **dict.fromkeys([*EDIT_2511.values(), LIGHTNING["4"]], "uv run download_models.py 2511"),
    LIGHTNING["8"]: "uv run download_models.py 2511 --lightning 8",
    **dict.fromkeys([Q21["unet"], Q21["clip"], Q21["vae"]], "uv run download_models.py 2.1"),
    Q21["pe"]: "uv run download_models.py 2.1 --prompt-enhancer i2i",
    "qwen-edit-skin-1.1.safetensors": "uv run download_models.py 2511 --skin-loras edit-skin",
    "Skin_Fix_rank64.safetensors": "uv run download_models.py 2511 --skin-loras skin-fix  # needs $CIVITAI_TOKEN",
})
MAX_IMAGES = {"2511": 3, "2.1": 10}  # 2.1: what its template exposes and its model card states
PE_TEMPLATE = "image_qwen_image_2_1_image_edit.json"  # ComfyUI's 2.1 edit template holds the enhancer's system prompt
PACKAGES = {"websocket": "websocket-client", "PIL": "pillow", "pillow_heif": "pillow-heif",
            "pillow_jxl": "pillow-jxl-plugin", "rawpy": "rawpy"}  # module -> package, as the header lists them


# ---------------------------------------------------------------- requests

@dataclass
class Request:
    """What to make: the input images, the prompt and the settings, one field per command-line option (see
    parse_args, or --help). A field left at None takes the option's default; settled() fills them in. comfycore.Settled names the fields the
    core needs of it."""

    images: list[Path | str]  # the inputs, or "@ID" for an image in rope; the first one sets the output size
    prompt: str
    model: str = "2511"  # "2511" (Qwen-Image-Edit-2511) or "2.1" (Qwen-Image-2.1)
    count: int = 1  # -n: how many images, one seed each
    seed: int | None = None  # -s: the first image's seed; None picks one at random
    negative: str = ""  # only has an effect when cfg is above 1
    steps: int | None = None
    cfg: float | None = None
    lora: list[tuple[str, float] | str] = field(default_factory=list)  # ("style.safetensors", 0.8), or "style.safetensors:0.8"
    lightning: str | None = None  # 2511 only: "4", "8" or "off"; default "4"
    resolution: int | None = None  # 2.1 only; default 1024
    enhance: bool = False  # 2.1 only: the prompt enhancer
    into: str | None = None  # the rope folder; default $IMG2IMG_INTO, else unfiled
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
        r.into = self.into or os.environ.get("IMG2IMG_INTO") or None
        r.note = self.note or None
        if not r.images:
            raise ValueError("an input image is needed")
        for img in r.images:
            if not img.is_file() and not rope_ref(img):
                raise ValueError(f"no such image: {img}")
        if r.model not in MAX_IMAGES:
            raise ValueError(f"model must be {' or '.join(MAX_IMAGES)}, not {r.model}")
        if len(r.images) > MAX_IMAGES[r.model]:
            raise ValueError(f"model {r.model} takes at most {MAX_IMAGES[r.model]} images")
        if r.count < 1:
            raise ValueError("count must be at least 1")
        if r.model == "2511":
            if r.resolution is not None or r.enhance:
                raise ValueError("resolution and enhance are for Qwen-Image-2.1 only (model 2.1)")
            r.lightning = "4" if self.lightning is None else str(self.lightning)
            if r.lightning not in (*LIGHTNING, "off"):
                raise ValueError(f"lightning must be 4, 8 or off, not {r.lightning}")
            fast = r.lightning != "off"
            r.steps = self.steps or (int(r.lightning) if fast else 40)
            r.cfg = (1.0 if fast else 4.0) if self.cfg is None else float(self.cfg)
        else:
            if self.lightning is not None:
                raise ValueError("lightning is for Qwen-Image-Edit-2511 only (model 2511)")
            r.resolution = 1024 if self.resolution is None else self.resolution
            if not 0 <= r.resolution <= 4096:
                raise ValueError("resolution takes 0 to 4096")
            r.steps = self.steps or 25
            r.cfg = 1.0 if self.cfg is None else float(self.cfg)
        return r


# ---------------------------------------------------------------- workflows (ComfyUI API format)

def build_2511(r: Request, images: list[str], seed: int) -> Graph:
    """ComfyUI template 'Qwen Image Edit 2511', flattened."""
    g = Graph()
    loads = [g.add("LoadImage", image=name) for name in images]
    first = g.add("FluxKontextImageScale", image=link(loads[0]))  # image 1 snapped to a trained size, about 1 MP
    unet = g.add("UNETLoader", "loading the diffusion model", unet_name=EDIT_2511["unet"], weight_dtype="default")
    clip = g.add("CLIPLoader", "loading the text encoder", clip_name=EDIT_2511["clip"], type="qwen_image", device="default")
    vae = g.add("VAELoader", vae_name=EDIT_2511["vae"])
    model = g.add("ModelSamplingAuraFlow", model=link(unet), shift=3.1)
    model = g.add("CFGNorm", model=link(model), strength=1.0)
    if r.lightning != "off":
        model = g.add("LoraLoaderModelOnly", model=link(model), lora_name=LIGHTNING[r.lightning], strength_model=1.0)
    model = add_loras(g, model, r.lora)
    refs = {"image1": link(first), **{f"image{i}": link(n) for i, n in enumerate(loads[1:], start=2)}}
    pos = g.add("TextEncodeQwenImageEditPlus", "encoding the prompt and images",
                clip=link(clip), vae=link(vae), prompt=r.prompt, **refs)
    neg = g.add("TextEncodeQwenImageEditPlus", clip=link(clip), vae=link(vae), prompt=r.negative, **refs)
    pos = g.add("FluxKontextMultiReferenceLatentMethod", conditioning=link(pos), reference_latents_method="index_timestep_zero")
    neg = g.add("FluxKontextMultiReferenceLatentMethod", conditioning=link(neg), reference_latents_method="index_timestep_zero")
    latent = g.add("VAEEncode", pixels=link(first), vae=link(vae))
    sampled = g.add("KSampler", "sampling", model=link(model), seed=seed, steps=r.steps, cfg=r.cfg,
                    sampler_name="euler", scheduler="simple", positive=link(pos), negative=link(neg),
                    latent_image=link(latent), denoise=1.0)
    decoded = g.add("VAEDecode", "decoding", samples=link(sampled), vae=link(vae))
    g.image_out = g.add("PreviewImage", images=link(decoded))
    return g


def build_21(r: Request, images: list[str], seed: int, system_prompt: str | None) -> Graph:
    """ComfyUI template 'Qwen Image 2.1: Image Edit', flattened; the prompt enhancer only with --enhance."""
    g = Graph()
    loads = [g.add("LoadImage", image=name) for name in images]
    unet = g.add("UNETLoader", "loading the diffusion model", unet_name=Q21["unet"], weight_dtype="default")
    model = add_loras(g, unet, r.lora)
    model = g.add("QwenImage21Cache", model=link(model), device="auto", dtype="default")
    clip = g.add("CLIPLoader", "loading the text encoder", clip_name=Q21["clip"], type="qwen_image", device="default")
    vae = g.add("VAELoader", vae_name=Q21["vae"])
    prompt: str | list = r.prompt
    if r.enhance:
        pe = g.add("CLIPLoader", "loading the prompt enhancer", clip_name=Q21["pe"], type="qwen_image", device="default")
        batch = g.add("BatchImagesNode", **{f"images.image{i}": link(n) for i, n in enumerate(loads)})
        sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
                    "repetition_penalty": 1.0, "seed": seed, "presence_penalty": 0.0}
        gen = g.add("TextGenerate", "enhancing the prompt", clip=link(pe), image=link(batch), prompt=r.prompt,
                    system_prompt=system_prompt, max_length=4096, thinking=False, use_default_template=True,
                    mtp="auto", sampling_mode="on", **{f"sampling_mode.{k}": v for k, v in sampling.items()})
        g.text_out = g.add("PreviewAny", source=link(gen))
        prompt = link(gen)
    enc = g.add("TextEncodeQwenImage21", "encoding the prompt and images", clip=link(clip), vae=link(vae),
                prompt=prompt, negative_prompt=r.negative, resolution=r.resolution,
                **{f"images.image_{i}": link(n) for i, n in enumerate(loads, start=1)})
    sampled = g.add("KSampler", "sampling", model=link(model), seed=seed, steps=r.steps, cfg=r.cfg,
                    sampler_name="euler", scheduler="simple", positive=link(enc, 0), negative=link(enc, 1),
                    latent_image=link(enc, 2), denoise=1.0)  # the encoder's latent has image 1's size
    decoded = g.add("VAEDecode", "decoding", samples=link(sampled), vae=link(vae))
    g.image_out = g.add("PreviewImage", images=link(decoded))
    return g


def build(r: Request, images: list[str], seed: int, system_prompt: str | None) -> Graph:
    """The API workflow for a settled request, with `images` as ComfyUI names its uploads."""
    return build_2511(r, images, seed) if r.model == "2511" else build_21(r, images, seed, system_prompt)


def enhancer_system_prompt(comfy: Comfy) -> str:
    """The prompt enhancer's system prompt, as ComfyUI's Qwen-Image-2.1 edit template gives it to the enhancer."""
    try:
        template = comfy.get_json(f"/templates/{PE_TEMPLATE}")
    except (OSError, ValueError, ComfyError) as e:
        die(f"--enhance reads its system prompt from ComfyUI's {PE_TEMPLATE}, which failed: {e}")
    for sub in template.get("definitions", {}).get("subgraphs", []):
        for node in sub.get("nodes", []):
            if node.get("type") == "PrimitiveStringMultiline" and "System Prompt" in (node.get("title") or ""):
                return node["widgets_values"][0]
    die(f"ComfyUI's {PE_TEMPLATE} no longer has the enhancer's system prompt node; --enhance cannot run")


# ---------------------------------------------------------------- running jobs

def image_flags(r: Request, seed: int) -> list[str]:
    """Every flag that changes the image, as argv items: the seed is this image's own and every default is spelled
    out, so later changes to the defaults cannot change the result. A new image-affecting flag has to go here too."""
    flags = [f"--model={r.model}"]
    if r.model == "2511":
        flags.append(f"--lightning={r.lightning}")
    else:
        flags.append(f"--resolution={r.resolution}")
        if r.enhance:
            flags.append("--enhance")
    flags += [f"--steps={r.steps}", f"--cfg={exact(r.cfg)}", f"--seed={seed}"]
    flags += [f"--lora={name}:{exact(strength)}" for name, strength in r.lora]
    if r.negative:
        flags.append(f"--negative={r.negative}")
    return flags


def describe(r: Request, base: int) -> str:
    """A request's settings in a line: model, steps, cfg, its seeds from `base`, LoRAs."""
    bits = [f"model {r.model}"]
    if r.model == "2511":
        bits.append(f"Lightning {r.lightning}-step" if r.lightning != "off" else "Lightning off")
    else:
        bits.append(f"resolution {r.resolution or 'as input'}")
        if r.enhance:
            bits.append("prompt enhancer on")
    bits += [f"{r.steps} steps", f"cfg {r.cfg:g}", f"seed {base}" + (f"-{base + r.count - 1}" if r.count > 1 else "")]
    if r.lora:
        bits.append("LoRA " + ", ".join(f"{name}:{strength:g}" for name, strength in r.lora))
    return ", ".join(bits)


def context(comfy: Comfy, requests: list[Request]) -> str | None:
    """What every workflow of a run shares: the enhancer's system prompt, when any request uses the enhancer."""
    return enhancer_system_prompt(comfy) if any(r.enhance for r in requests) else None



def run(requests: Request | Iterable[Request], *, fresh: bool = False, library: str | None = None) -> list[Submitted]:
    """Hands the images of every request to rope-queue, one group of `count` jobs per request, and returns a Submitted
    per request, in order: its group and job ids, or why it was skipped. rope-queue runs the jobs on its ComfyUI and
    puts every image into rope with its recipe; nothing here waits for them, and what becomes of them is rope-queue's
    to show (its screen, `rope-queue ls`, rope-gui).

    Every request is checked first (a bad one is a ValueError), so nothing is sent until all of them are good. A
    request whose input the converter cannot read, or whose model or LoRA file ComfyUI lacks, is printed with its fix
    and skipped; the others go on. Like the command line, it prints what it does, one line per request, and raises
    Error when rope-queue is not built or not running. The keyword arguments:
      fresh    compute again even when ComfyUI has cached the result of an identical workflow
      library  the rope library; default $ROPE_LIBRARY, else ~/.local/share/rope
    """
    return submit(TOOL, [requests] if isinstance(requests, Request) else requests, fresh=fresh, library=library)


# ---------------------------------------------------------------- cli

def parse_args(argv: list[str]) -> argparse.Namespace:
    """The command line, with the Request it asks for as .request and the rest as .options and .dry_run."""
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
    g.add_argument("--into", metavar="FOLDER",
                   help="the rope folder for the results, e.g. trips/kyoto, made if there is none. "
                        "Default: $IMG2IMG_INTO, else unfiled")
    g.add_argument("--tag", dest="tags", action="append", default=[], metavar="TAG", help="tag the results; repeatable")
    g.add_argument("--note", help="a note on each result")
    g.add_argument("--library", metavar="DIR", help="the rope library (default: $ROPE_LIBRARY, else ~/.local/share/rope)")
    g.add_argument("--replay", metavar="ID", help="make a rope image again from its record; see --replay ID --help")
    g = p.add_argument_group("ComfyUI")
    add_comfy_args(g)
    g.add_argument("--dry-run", action="store_true", help="print the API workflow JSON and stop; nothing is uploaded or run")
    a = p.parse_args(argv)
    direct_only(p, a)
    try:
        a.request = Request(a.images, a.prompt, model=a.model, count=a.count, seed=a.seed, negative=a.negative,
                            steps=a.steps, cfg=a.cfg, lora=a.lora, lightning=a.lightning, resolution=a.resolution,
                            enhance=a.enhance, into=a.into, tags=a.tags, note=a.note).settled()
    except ValueError as e:
        p.error(str(e))
    a.options = Options(a.server, a.keep_server, a.no_start, shlex.split(a.comfy_args), a.timeout, a.fresh, a.library)
    return a


def workflow(request: Request, seed: int = 0, *, fresh: bool = False, library: str | None = None) -> dict:
    """The API workflow run() would send for this request and seed, which --dry-run prints: nothing is uploaded,
    run or kept, and rope is only asked where @ID inputs are."""
    r = request.settled()
    check_packages(TOOL.packages, TOOL.file)
    lib = Library.find(library) if any(rope_ref(i) for i in r.images) else None
    with tempfile.TemporaryDirectory(prefix="img2img-") as workdir:
        inputs = prepare([r], Path(workdir), lib, keep=False)[0]
        tag = "f000000" if fresh else ""  # the real tag is random
        names = [f"{UPLOAD_SUBFOLDER}/{upload_name(p, s, tag)}" for p, s in zip(inputs.uploads, inputs.stems)]
        system_prompt = "<system prompt from ComfyUI's Qwen-Image-2.1 edit template>" if r.enhance else None
        return build(r, names, seed, system_prompt).nodes


def main(argv: list[str]) -> int:
    if argv[:1] == ["--job"] and len(argv) == 2:  # rope-queue's child: one job, see comfyqueue.job_main
        return job_main(TOOL, Path(argv[1]))
    if any(arg == "--replay" or arg.startswith("--replay=") for arg in argv):
        return replay_main(TOOL, argv)
    a = parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    if a.dry_run:
        seed = a.request.seed if a.request.seed is not None else 0
        print(json.dumps(workflow(a.request, seed, fresh=a.fresh, library=a.library), indent=2, ensure_ascii=False))
        return 0
    if a.no_queue:  # the direct path: this command runs the jobs itself
        run_requests(TOOL, [a.request], a.options)
        return 0
    submitted = submit(TOOL, [a.request], fresh=a.fresh, library=a.library)
    return 1 if any(s.skipped for s in submitted) else 0


# ---------------------------------------------------------------- replay

# What the status line says while a node of a recorded workflow runs, by node class.
REPLAY_LABELS = {"UNETLoader": "loading the diffusion model", "CLIPLoader": "loading the text encoder",
                 "TextGenerate": "enhancing the prompt", "TextEncodeQwenImageEditPlus": "encoding the prompt and images",
                 "TextEncodeQwenImage21": "encoding the prompt and images", "KSampler": "sampling",
                 "VAEDecode": "decoding"}


def reseed(nodes: dict, seed: int) -> None:
    """Puts `seed` into a recorded workflow: the sampler's, and the prompt enhancer's."""
    for node in nodes.values():
        if node["class_type"] == "KSampler":
            node["inputs"]["seed"] = seed
        if node["class_type"] == "TextGenerate" and "sampling_mode.seed" in node["inputs"]:
            node["inputs"]["sampling_mode.seed"] = seed


def replay_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rerun-enhancer", action="store_true",
                   help="2.1 with --enhance: let the prompt enhancer write the prompt again, instead of reusing what "
                        "it wrote then (it samples, so the result will differ)")


def reuse_enhanced(nodes: dict, made: dict, a: argparse.Namespace) -> None:
    """Feeds the workflow the prompt the enhancer wrote then, unless --rerun-enhancer."""
    enhanced = made.get("enhanced_prompt")
    if enhanced and not a.rerun_enhancer:
        pin_enhanced(nodes, enhanced)


def enhancer_prompt(nodes: dict) -> str | None:
    """The enhancer's system prompt in a recorded workflow: build()'s context, without asking ComfyUI for it."""
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
        for ref in (gen["inputs"].get("clip"), gen["inputs"].get("image")):  # the enhancer's model and image batch
            if isinstance(ref, list) and not any(ref[0] == v[0] for n in nodes.values()
                                                  for v in n["inputs"].values() if isinstance(v, list)):
                nodes.pop(ref[0], None)


REPLAY = Replay(parse=lambda argv: parse_args(argv).request, reseed=reseed, labels=REPLAY_LABELS, args=replay_args,
                context_of=enhancer_prompt, context_dry="<system prompt from ComfyUI>", adjust=reuse_enhanced)
TOOL = Tool(Path(__file__), PACKAGES, build=lambda r, inputs, seed, context: build(r, inputs.uploaded, seed, context),
            flags=image_flags, describe=describe, context=context, replay=REPLAY)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Error as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
