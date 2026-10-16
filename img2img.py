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
folder, made if there is none, and an input can be @ID, an image in rope. --replay ID makes an image again from that record and
compares the pixels. The PNG also embeds the API workflow ComfyUI ran; drop it on the ComfyUI page to see that
workflow as a node graph.

From Python, import it and give run() a Request per combination to try: they all go through one ComfyUI, and
every result lands in rope with its recipe, so the combinations can be compared in rope-gui:

    import img2img
    skin = "qwen-edit-skin-1.1.safetensors"
    requests = [img2img.Request(["photo.jpg"], "make the subjects skin details more prominent and natural",
                                seed=42, lora=[(skin, s)], into="tests/skin", tags=[f"skin {s}"])
                for s in (0.8, 1.0, 1.2)]
    for r in img2img.run(requests):
        print(r.short, r.folder, r.request.lora)

A Request has a field per option above, and run() takes the ComfyUI and --library options as keyword arguments.
The Python that imports it needs the packages listed at the top, e.g. .venv from setup_venv.sh.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import shlex
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from comfycore import (FETCH, HERE, SERVER, TIMEOUT, UPLOAD_SUBFOLDER, Comfy, ComfyError, Error, Graph, Inputs, Job,
                       Library, ModelHashes, Options, Result, Run, Status, add_comfy_args, add_loras, check_packages,
                       code_identity, comfy_env, die, exact, fresh_tag, iso, link, model_files, parse_lora, png_size,
                       prepare, rope_ref, run_name, send_result, start_prehash, upload_name, wait, with_comfy)

__all__ = ["SERVER", "TIMEOUT", "Error", "Request", "Result", "run", "workflow"]

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

def recipe(run: Run, job: Job, flags: list[str], enhanced: str) -> dict:
    """The made field of a result's meta.json in rope."""
    inputs = [{k: v for k, v in (("id", i.get("id")), ("uploaded", i.get("uploaded")), ("upload", name)) if v}
              for i, name in zip(job.inputs.records, job.inputs.uploaded)]
    made = {"by": "img2img.py", "at": iso(run.when), "flags": flags, "prompt": job.request.prompt, "inputs": inputs,
            "enhanced_prompt": enhanced or None,
            "batch": {"index": job.index, "count": job.request.count, "first_seed": job.seed - job.index + 1},
            "server": run.server_info or None, "env": run.env, "models": run.models(job.graph.nodes), "code": run.code}
    return {k: v for k, v in made.items() if v is not None}


def keep_result(run: Run, job: Job, data: bytes, enhanced: str) -> str:
    """Keeps one result in the library, as a Result on its job, and returns what to show for it on the status line."""
    r = job.request

    def args(png: Path, made: Path, graph: Path) -> list[str]:
        out = ["add", str(png), f"--made={made}", f"--graph={graph}", f"--name={name}", f"--repo={HERE}"]
        if r.into:
            out += [f"--into={r.into}", "--file-inputs"]
        out += [f"--tag={t}" for t in r.tags]
        return out + ([f"--note={r.note}"] if r.note else [])

    name = run_name(run.when, job.inputs.names[0], r.model, job.seed)
    made = json.dumps(recipe(run, job, image_flags(r, job.seed), enhanced), ensure_ascii=False)
    result = Result(r, job.index, job.seed, seconds=job.took, enhanced_prompt=enhanced)
    result.width, result.height = png_size(data) or (0, 0)
    answer, rescued = send_result(run, name, data, made, job, args)
    job.results.append(result)
    if answer is None:
        result.path = rescued
        return "not in rope; see above"
    result.id, result.short, result.path = answer["id"], answer["short"], Path(answer["path"])
    result.folder, result.created = answer["folder"], answer["created"]
    where = f"{result.short}  {result.folder or '/'}"
    return where if result.created else f"{where}  (rope had these bytes already)"


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


def run_jobs(run: Run, requests: list[Request], all_inputs: list[Inputs]) -> list[Result]:
    """Uploads the inputs, queues every image of every request, follows them, and keeps each result as it comes.
    Ctrl-C, or a failure in one job, drops what is still queued and interrupts the running job."""
    import websocket

    comfy, status = run.comfy, run.status
    tag = fresh_tag("f") if run.options.fresh else ""
    uploaded: dict[Path, str] = {}  # a file several requests share goes up once
    for inputs in all_inputs:
        for path, stem in zip(inputs.uploads, inputs.stems):
            if path not in uploaded:
                uploaded[path] = comfy.upload(path, upload_name(path, stem, tag))
        inputs.uploaded = [uploaded[path] for path in inputs.uploads]
    system_prompt = enhancer_system_prompt(comfy) if any(r.enhance for r in requests) else None
    run.when, run.server_info = time.localtime(), comfy.system_info()  # one time for all the results of this run
    run.env, run.code = comfy_env(comfy), code_identity(Path(__file__))
    total = sum(r.count for r in requests)
    jobs: list[Job] = []
    bases: list[int] = []
    for n, (r, inputs) in enumerate(zip(requests, all_inputs), start=1):
        base = r.seed if r.seed is not None else random.randrange(2**32)
        bases.append(base)
        which = f"#{n} " if len(requests) > 1 else ""
        for i in range(r.count):
            graph = build(r, inputs.uploaded, base + i, system_prompt)
            jobs.append(Job(r, inputs, i + 1, base + i, graph, f"[{len(jobs) + 1}/{total}] {which}seed {base + i}"))
    run.prehash = start_prehash(run.hashes, [job.graph.nodes for job in jobs])
    ws = websocket.create_connection(f"{comfy.ws_url}?clientId={comfy.client_id}", timeout=15)
    pending: dict[str, Job] = {}
    try:
        for job in jobs:
            pending[comfy.queue(job.graph)] = job
        if len(requests) == 1:
            status.line(f"queued {total} job{'s' if total > 1 else ''}: {describe(requests[0], bases[0])}")
        else:
            status.line(f"queued {total} jobs for {len(requests)} requests:")
            for n, (r, inputs, base) in enumerate(zip(requests, all_inputs, bases), start=1):
                tags = f", tags {', '.join(r.tags)}" if r.tags else ""
                status.line(f"  #{n} {inputs.names[0]}: {describe(r, base)}{tags}")
        wait(run, ws, pending, total)
    except BaseException as e:
        if pending:
            what = "cancelling" if isinstance(e, KeyboardInterrupt) else "stopping"
            status.line(f"\n{what}: dropping the queued jobs and interrupting the running one")
            comfy.cancel(list(pending))
        raise
    finally:
        ws.close()
    return [result for job in jobs for result in job.results]


def run(requests: Request | Iterable[Request], *, server: str = SERVER, keep_server: bool = False,
        no_start: bool = False, comfy_args: str | Iterable[str] = (), timeout: int = TIMEOUT, fresh: bool = False,
        library: str | None = None) -> list[Result]:
    """Makes the images of every request, all through one ComfyUI, and keeps them in rope. Returns a Result per
    image: the requests in order, and each request's images in seed order.

    Every request is checked first (a bad one is a ValueError), so nothing runs until all of them are good. Like
    the command line, it prints what it does. A failure raises Error and drops the jobs still queued; Ctrl-C does
    the same; what was made before stays in rope. The keyword arguments are the command line's ComfyUI and
    --library options, with the same defaults:
      server       the ComfyUI; if none answers there, one is started for this run and stopped afterwards
      keep_server  leave a ComfyUI this run started running, so that the next run skips the model load
      no_start     raise instead of starting a ComfyUI
      comfy_args   extra flags for a ComfyUI this run starts, e.g. "--reserve-vram 2"
      timeout      give up when ComfyUI has said nothing about the jobs for this many seconds
      fresh        compute again even when ComfyUI has cached the result of an identical workflow
      library      the rope library; default $ROPE_LIBRARY, else ~/.local/share/rope
    """
    flags = shlex.split(comfy_args) if isinstance(comfy_args, str) else list(comfy_args)
    opts = Options(server, keep_server, no_start, flags, timeout, fresh, library)
    wanted = [requests] if isinstance(requests, Request) else list(requests)
    settled = []
    for n, r in enumerate(wanted, start=1):
        try:
            settled.append(r.settled())
        except ValueError as e:
            raise ValueError(f"request {n} of {len(wanted)}: {e}" if len(wanted) > 1 else str(e)) from None
    if not settled:
        return []
    check_packages(PACKAGES, Path(__file__))
    lib = Library.find(opts.library)
    for into in dict.fromkeys(r.into for r in settled if r.into):
        # made now, so that a name rope refuses fails before the GPU work, and a typo shows up right here
        if lib.call("mkdir", "-p", into)["created"]:
            print(f"new rope folder {into}")
    hashes = ModelHashes(lib.root / ".cache" / "model-sha256.json")
    with tempfile.TemporaryDirectory(prefix="img2img-") as workdir:
        all_inputs = prepare(settled, Path(workdir), lib)
        return with_comfy(opts, lambda comfy: run_jobs(Run(comfy, lib, opts, Status(), keep_result, hashes),
                                                       settled, all_inputs))


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
    try:
        a.request = Request(a.images, a.prompt, model=a.model, count=a.count, seed=a.seed, negative=a.negative,
                            steps=a.steps, cfg=a.cfg, lora=a.lora, lightning=a.lightning, resolution=a.resolution,
                            enhance=a.enhance, into=a.into, tags=a.tags, note=a.note).settled()
    except ValueError as e:
        p.error(str(e))
    a.options = Options(a.server, a.keep_server, a.no_start, shlex.split(a.comfy_args), a.timeout, a.fresh, a.library)
    return a


PACKAGES = {"websocket": "websocket-client", "PIL": "pillow", "pillow_heif": "pillow-heif",
            "pillow_jxl": "pillow-jxl-plugin", "rawpy": "rawpy"}  # module -> package, as the header lists them


def workflow(request: Request, seed: int = 0, *, fresh: bool = False, library: str | None = None) -> dict:
    """The API workflow run() would send for this request and seed, which --dry-run prints: nothing is uploaded,
    run or kept, and rope is only asked where @ID inputs are."""
    r = request.settled()
    check_packages(PACKAGES, Path(__file__))
    lib = Library.find(library) if any(rope_ref(i) for i in r.images) else None
    with tempfile.TemporaryDirectory(prefix="img2img-") as workdir:
        inputs = prepare([r], Path(workdir), lib, keep=False)[0]
        tag = "f000000" if fresh else ""  # the real tag is random
        names = [f"{UPLOAD_SUBFOLDER}/{upload_name(p, s, tag)}" for p, s in zip(inputs.uploads, inputs.stems)]
        system_prompt = "<system prompt from ComfyUI's Qwen-Image-2.1 edit template>" if r.enhance else None
        return build(r, names, seed, system_prompt).nodes


def main(argv: list[str]) -> int:
    if any(arg == "--replay" or arg.startswith("--replay=") for arg in argv):
        return replay_main(argv)
    a = parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    if a.dry_run:
        seed = a.request.seed if a.request.seed is not None else 0
        print(json.dumps(workflow(a.request, seed, fresh=a.fresh, library=a.library), indent=2, ensure_ascii=False))
        return 0
    run(a.request, **vars(a.options))
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
    a.options = Options(a.server, a.keep_server, a.no_start, shlex.split(a.comfy_args), a.timeout, a.fresh, a.library)
    return a


def replay_main(argv: list[str]) -> int:
    a = parse_replay_args(argv)
    check_packages({"websocket": "websocket-client"}, Path(__file__))
    sys.stdout.reconfigure(line_buffering=True)
    a.lib = Library.find(a.library)
    info = a.lib.call("show", a.replay)
    made = info["meta"].get("made")
    if not made:
        die(f"{info['short']} has no recipe: rope did not see it being made, so it cannot be replayed")
    missing = [str(i["slot"]) for i in info["inputs"] if i.get("missing")]
    if missing:
        die(f"{info['short']} cannot be replayed: rope lacks the bytes of input image {', '.join(missing)}")
    a.info, a.made, a.prompt, a.flags = info, made, made.get("prompt", ""), made.get("flags", [])
    a.inputs = [{"id": i["id"], "uploaded": i.get("uploaded"), "path": Path(i["path"]), "recorded": i.get("upload")}
                for i in info["inputs"]]
    a.lib.using([info["id"], *(i["id"] for i in a.inputs)])  # purge leaves these alone until this run ends
    a.today = parse_args([*a.flags, "--", *(str(i["path"]) for i in a.inputs), a.prompt]).request
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
    hashes = ModelHashes(a.lib.root / ".cache" / "model-sha256.json")
    for line in model_changes(made.get("models") or [], model_files(a.recorded or {}, hashes)):
        print(f"  model: {line}")
    if a.dry_run:
        nodes = replay_graph(a, [i["recorded"] or "" for i in a.inputs], "<system prompt from ComfyUI>")
        print(json.dumps(nodes, indent=2, ensure_ascii=False))
        return 0
    with_comfy(a.options, lambda comfy: replay_jobs(a, comfy))
    return 0


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
        for ref in (gen["inputs"].get("clip"), gen["inputs"].get("image")):  # the enhancer's model and image batch
            if isinstance(ref, list) and not any(ref[0] == v[0] for n in nodes.values()
                                                  for v in n["inputs"].values() if isinstance(v, list)):
                nodes.pop(ref[0], None)


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
    """How the model files differ from when an image was made: by content where both sides have a hash, else (an
    older record) by size and modification time."""
    out = []
    for old in then:
        new = next((m for m in now if m["dir"] == old["dir"] and m["name"] == old["name"]), None)
        if new is None:
            continue
        if "size" in old and "size" not in new:
            out.append(f"{old['name']} is missing now")
        elif "sha256" in old and "sha256" in new:
            if old["sha256"] != new["sha256"]:
                out.append(f"{old['name']} has other content than then (sha256 {old['sha256'][:12]} -> {new['sha256'][:12]})")
        elif (old.get("size"), old.get("mtime_ns")) != (new.get("size"), new.get("mtime_ns")):
            out.append(f"{old['name']} may not be the same file as then (size or modification time differs)")
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

    run = Run(comfy, a.lib, a.options, Status(), lambda run, job, data, enhanced: keep_replay(a, run, job, data, enhanced),
              ModelHashes(a.lib.root / ".cache" / "model-sha256.json"))
    status = run.status
    tag = fresh_tag("r")
    names = []
    for i in a.inputs:
        stem = re.sub(r"^[0-9a-f]{12}_([rf][0-9a-f]{6}_)?", "", Path(i["recorded"] or i["path"].name).stem)
        names.append(comfy.upload(i["path"], upload_name(i["path"], stem, tag)))
    run.server_info, run.env, run.code = comfy.system_info(), comfy_env(comfy), code_identity(Path(__file__))
    env_id = a.made.get("env")
    if isinstance(env_id, str) and (a.lib.root / "envs" / f"{env_id}.json").exists():
        then = json.loads((a.lib.root / "envs" / f"{env_id}.json").read_text("utf-8"))
        for line in env_changes(then, run.env):
            print(f"  environment: {line}")
    system_prompt = enhancer_system_prompt(comfy) if a.today.enhance and (a.rebuild or a.recorded is None) else None
    nodes = replay_graph(a, names, system_prompt)
    run.prehash = start_prehash(run.hashes, [nodes])
    graph = Graph()
    graph.nodes = nodes
    graph.labels = {nid: REPLAY_LABELS[n["class_type"]] for nid, n in nodes.items() if n["class_type"] in REPLAY_LABELS}
    graph.image_out = next(nid for nid, n in nodes.items() if n["class_type"] == "PreviewImage")
    graph.text_out = next((nid for nid, n in nodes.items() if n["class_type"] == "PreviewAny"), "")
    inputs = Inputs([a.info["name"]], a.inputs, [i["path"] for i in a.inputs], [], names)
    job = Job(a.today, inputs, 1, a.seed_used, graph, f"[1/1] seed {a.seed_used}")
    ws = websocket.create_connection(f"{comfy.ws_url}?clientId={comfy.client_id}", timeout=15)
    try:
        pending = {comfy.queue(graph): job}
        print(f"replaying {a.info['short']} {a.info['name']}, seed {a.seed_used}")
        try:
            wait(run, ws, pending, 1)
        except KeyboardInterrupt:
            status.line("\ncancelling the replay")
            comfy.cancel(list(pending))
            raise
    finally:
        ws.close()
    return job.saved


def keep_replay(a: argparse.Namespace, run: Run, job: Job, data: bytes, enhanced: str) -> str:
    """Records a replay's result with the image it replays, and returns what came of it."""
    flags = image_flags(a.today, a.seed_used) if a.rebuild or a.recorded is None else \
        [f"--seed={a.seed_used}" if f.startswith("--seed=") else f for f in a.flags]
    made = json.dumps(recipe(run, job, flags, enhanced or a.made.get("enhanced_prompt") or ""), ensure_ascii=False)

    def args(png: Path, made: Path, graph: Path) -> list[str]:
        return ["replayed", a.info["id"], str(png), f"--made={made}", f"--graph={graph}", f"--repo={HERE}"]

    r, _ = send_result(run, f"replay-{a.info['short']}", data, made, job, args)
    if r is None:
        return "not recorded in rope; see above"
    if r["result"] == "identical":
        return f"identical to {a.info['short']}, pixel for pixel"
    return (f"differs from {a.info['short']}: largest difference {r['max_diff']} of 255, in "
            f"{r['differing'] * 100:.2f}% of the pixels; kept as {r['id'][:12]}")


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Error as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
