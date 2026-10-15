#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=1.0,<3"]
# ///
"""Download Qwen-Image-2.1 / Qwen-Image-Edit-2511 model files (ComfyUI repacks) from Hugging Face, and
optional skin LoRAs for 2511 (one of them from Civitai, which needs an API key).

Files land in a ComfyUI-style layout under --dest:
    <dest>/diffusion_models/   <dest>/text_encoders/   <dest>/vae/   <dest>/loras/
Point ComfyUI at <dest> through extra_model_paths.yaml, or pass --dest <ComfyUI>/models.

Examples:
    ./download_models.py --list                         # every known file + size
    ./download_models.py 2.1 --dry-run                  # plan only, nothing written
    ./download_models.py 2511 --lightning both          # fp8 model + 4-step and 8-step LoRA
    ./download_models.py all --prompt-enhancer i2i      # both models, plus 2.1's i2i prompt enhancer
    ./download_models.py all --limit-rate 5M            # cap at 5 MiB/s (k/M/G suffix, 1024-based)
    ./download_models.py 2.1 --q21-precision bf16 --only diffusion
    ./download_models.py 2511 --skin-loras edit-skin skin-fix --only lora   # skin LoRAs; skin-fix needs $CIVITAI_TOKEN

Downloads resume on re-run; a file already present with the right size is skipped.
Without --limit-rate the hf_xet fast path is used; with it, a plain HTTP stream with
Range resume and a byte-rate cap (the only way to throttle: hf_xet has no rate option).
If the fast path errors out (CDN hiccup, dead IPv6 route inside the Rust client) the same
plain-HTTP stream takes over for that file, uncapped.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

# hf_xet (bundled with huggingface_hub) streams big files in parallel chunks; this asks it for more.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def force_ipv4() -> None:
    """Resolve hostnames to IPv4 only (Python-side requests: metadata, resolve URLs, throttled path).

    huggingface.co publishes AAAA records; on a box whose IPv6 is advertised but not routed every
    request first waits out a dead v6 handshake (tens of seconds) before falling back to v4.
    The HF CDN hosts are IPv4-only, so the bulk transfer never needed v6 in the first place.
    """
    real_getaddrinfo = socket.getaddrinfo

    def v4_only(host, port, family=0, type=0, proto=0, flags=0):
        if family in (0, socket.AF_UNSPEC):
            family = socket.AF_INET
        return real_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = v4_only

Q21 = "Comfy-Org/Qwen-Image-2.1"
EDIT = "Comfy-Org/Qwen-Image-Edit_ComfyUI"
QI = "Comfy-Org/Qwen-Image_ComfyUI"
L2511 = "lightx2v/Qwen-Image-Edit-2511-Lightning"
SKIN = "tlennon-ie/qwen-edit-skin"

KIND_DIR = {
    "diffusion": "diffusion_models",
    "text_encoder": "text_encoders",
    "vae": "vae",
    "lora": "loras",
}

MODEL_ALIASES = {
    "2.1": "qwen-image-2.1",
    "qwen-image-2.1": "qwen-image-2.1",
    "2511": "qwen-image-edit-2511",
    "edit": "qwen-image-edit-2511",
    "qwen-image-edit-2511": "qwen-image-edit-2511",
    "all": "all",
}

CHUNK = 256 * 1024  # bytes per read in the throttled path
RETRIES = 8  # network retries per file in the plain-HTTP path (each resumes from the .part size)
FAST_ATTEMPTS = 2  # hf_xet attempts per file before falling back to the plain-HTTP path


@dataclass(frozen=True)
class F:
    repo: str
    path: str  # path inside the repo
    kind: str  # key of KIND_DIR
    approx_gb: float  # for --list only; the real size is fetched from the Hub
    name: str | None = None  # the local file name, when it should differ from the repo's
    url: str | None = None  # a direct download (Civitai) instead of the Hub; size and sha256 are then given here
    size: int | None = None
    sha256: str | None = None

    @property
    def basename(self) -> str:
        return self.name or Path(self.path).name

    @property
    def on_hub(self) -> bool:
        return self.url is None

    @property
    def key(self) -> tuple[str, str]:
        return (self.repo, self.path)


# Skin LoRAs for Qwen-Image-Edit (2509/2511); see README.md, "Skin LoRAs".
#   edit-skin: qwen-edit-skin 1.1 by TL_, the 2750-step file Civitai's v1.1 serves, under a shorter name
#   skin-fix:  Skin Fix Qwen 2.0 by OgreLemonSoup, against the plastic look of the Lightning LoRA; Civitai only
SKIN_LORAS = {
    "edit-skin": F(SKIN, "qwen-edit-skin_1.1_000002750.safetensors", "lora", 0.30, name="qwen-edit-skin-1.1.safetensors"),
    "skin-fix": F("civitai.com/models/2031296", "Skin_Fix_rank64.safetensors", "lora", 0.59,
                  url="https://civitai.com/api/download/models/2369080?fileId=2259910", size=589940496,
                  sha256="ce1ae6a9201526d85bffae50aaa76a98160d7c48aa1214e06029024e06a6ab8b"),
}
CIVITAI_KEYS = "https://civitai.com/user/account"  # where an API key is made (API Keys, at the bottom)


# ---------------------------------------------------------------- manifests

def plan_qwen_image_21(a: argparse.Namespace) -> list[F]:
    """Qwen-Image-2.1 (7B DiT + Qwen3-VL 8B encoder). Files the official ComfyUI templates reference."""
    int8 = a.q21_precision == "int8"
    dit = "qwen_image_2.1_turbo" if a.q21_variant == "turbo" else "qwen_image_2.1"
    files = [
        F(Q21, f"diffusion_models/{dit}_{'int8_convrot' if int8 else 'bf16'}.safetensors",
          "diffusion", 7.26 if int8 else 14.23),
        F(Q21, f"text_encoders/qwen3vl_8b_{'int8_convrot' if int8 else 'bf16'}.safetensors",
          "text_encoder", 9.35 if int8 else 17.53),
        F(Q21, "vae/qwen_image_2.1_vae_bf16.safetensors", "vae", 0.68),
    ]
    if a.turbo_lora:
        files.append(F(Q21, "loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors", "lora", 0.91))
    for pe in a.prompt_enhancer:  # optional Qwen3.5-9B prompt rewriter used by the templates
        files.append(F(Q21, f"text_encoders/qwen3.5_9b_qwen_image_2.1_pe_{pe}.int8_convrot.safetensors",
                       "text_encoder", 9.47))
    return files


def plan_edit_2511(a: argparse.Namespace) -> list[F]:
    """Qwen-Image-Edit-2511 (20B). Same workflow as 2509; text encoder + VAE come from the Qwen-Image repack."""
    dit = {
        "fp8": ("qwen_image_edit_2511_fp8mixed", 20.53),
        "int8": ("qwen_image_edit_2511_int8_convrot", 20.50),
        "bf16": ("qwen_image_edit_2511_bf16", 40.86),
    }[a.edit_precision]
    files = [
        F(EDIT, f"split_files/diffusion_models/{dit[0]}.safetensors", "diffusion", dit[1]),
        F(QI, "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoder", 9.38),
        F(QI, "split_files/vae/qwen_image_vae.safetensors", "vae", 0.25),
    ]
    steps = {"none": [], "4": ["4"], "8": ["8"], "both": ["4", "8"]}[a.lightning]
    for s in steps:
        files.append(F(L2511, f"Qwen-Image-Edit-2511-Lightning-{s}steps-V1.0-bf16.safetensors", "lora", 0.85))
    files += [SKIN_LORAS[s] for s in a.skin_loras]
    return files


def full_catalog() -> list[F]:
    """Everything this script knows how to fetch (for --list)."""
    out: list[F] = []
    for prec in ("int8", "bf16"):
        for variant in ("base", "turbo"):
            ns = argparse.Namespace(q21_precision=prec, q21_variant=variant, turbo_lora=True,
                                    prompt_enhancer=["i2i", "t2i"])
            out += plan_qwen_image_21(ns)
    for prec in ("fp8", "int8", "bf16"):
        out += plan_edit_2511(argparse.Namespace(edit_precision=prec, lightning="both", skin_loras=list(SKIN_LORAS)))
    return dedupe(out)


def dedupe(files: list[F]) -> list[F]:
    seen: set[tuple[str, str]] = set()
    out: list[F] = []
    for f in files:
        if (f.repo, f.path) not in seen:
            seen.add((f.repo, f.path))
            out.append(f)
    return out


# ---------------------------------------------------------------- helpers

def gb(n: float | int | None) -> str:
    return "   ?  " if n is None else f"{n / 1e9:6.2f}"


def parse_rate(text: str) -> int:
    """'5M' -> 5 MiB/s in bytes; k/M/G suffixes are 1024-based like curl's --limit-rate."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kKmMgG]?)[bB]?\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"bad rate {text!r}; examples: 500k, 5M, 1.5G")
    mult = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}[m.group(2).lower()]
    rate = int(float(m.group(1)) * mult)
    if rate < 16 * 1024:
        raise argparse.ArgumentTypeError("rate below 16k is not useful")
    return rate


def real_sizes(files: list[F], token: str | None) -> dict[tuple[str, str], int | None]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    sizes: dict[tuple[str, str], int | None] = {f.key: f.size for f in files if not f.on_hub}
    by_repo: dict[str, list[str]] = {}
    for f in files:
        if f.on_hub:
            by_repo.setdefault(f.repo, []).append(f.path)
    for repo, paths in by_repo.items():
        found = {info.path: getattr(info, "size", None) for info in api.get_paths_info(repo, paths=paths)}
        for p in paths:
            if p not in found:
                sys.exit(f"error: {repo}/{p} does not exist on the Hub (repo layout changed?)")
            sizes[(repo, p)] = found[p]
    return sizes


def final_path(dest: Path, f: F) -> Path:
    return dest / KIND_DIR[f.kind] / f.basename


def is_complete(path: Path, expected: int | None) -> bool:
    return path.exists() and (expected is None or path.stat().st_size == expected)


def sha256_of(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- download: fast path (hf_xet)

def fetch_fast(f: F, dest: Path, staging: Path, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    final = final_path(dest, f)
    final.parent.mkdir(parents=True, exist_ok=True)
    # local_dir mode writes a real file (resumable via staging/.cache) instead of a symlink into ~/.cache.
    got = Path(hf_hub_download(repo_id=f.repo, filename=f.path,
                               local_dir=staging / f.repo.replace("/", "__"), token=token))
    if final.exists():
        final.unlink()
    os.replace(got, final)  # same filesystem: atomic rename, no second copy
    return final


# ---------------------------------------------------------------- download: throttled path (plain HTTP)

class _DropAuthOnCrossHostRedirect(urllib.request.HTTPRedirectHandler):
    """The Hub 302s to a signed CDN URL; forwarding Authorization there gets a 400, so strip it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urlsplit(newurl).netloc != urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


_OPENER = urllib.request.build_opener(_DropAuthOnCrossHostRedirect())


class _Throttle:
    """Cumulative byte-rate cap: sleep whenever we are ahead of rate × elapsed. Credit from stalls is capped at 1 s."""

    def __init__(self, rate: int | None) -> None:
        self.rate = rate  # None: no cap
        self.start = time.monotonic()
        self.sent = 0

    def account(self, n: int) -> None:
        if self.rate is None:
            return
        self.sent += n
        now = time.monotonic()
        ahead = self.sent / self.rate - (now - self.start)
        if ahead > 0:
            time.sleep(ahead)
        elif ahead < -1.0:  # fell behind by more than a second (stall): forget the surplus credit
            self.start = now - self.sent / self.rate - 1.0


def fetch_throttled(f: F, dest: Path, rate: int | None, token: str | None, expected: int | None) -> Path:
    """Plain HTTP stream to <final>.part with Range resume; rate caps bytes/s, None means uncapped. `token` is
    the one for where `f` lives: the Hub, or Civitai for a file with a url."""
    from huggingface_hub import hf_hub_url
    from tqdm import tqdm

    url = f.url or hf_hub_url(f.repo, f.path)
    final = final_path(dest, f)
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_name(final.name + ".part")
    headers = {"User-Agent": "download_models.py"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    attempt = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if expected is not None and have > expected:
            part.unlink()
            have = 0
        if expected is not None and have == expected:
            break
        req = urllib.request.Request(url, headers={**headers, **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with _OPENER.open(req, timeout=60) as resp:
                status = resp.status
                if have and status == 200:  # server ignored the Range: start over
                    have = 0
                    part.unlink(missing_ok=True)
                total = expected
                if total is None:
                    cl = resp.headers.get("Content-Length")
                    total = (have + int(cl)) if cl else None
                throttle = _Throttle(rate)
                with open(part, "ab" if have else "wb") as out, tqdm(
                    total=total, initial=have, unit="B", unit_scale=True, unit_divisor=1024,
                    desc=f.basename[:40], leave=False, dynamic_ncols=True,
                ) as bar:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        bar.update(len(chunk))
                        throttle.account(len(chunk))
            # a clean EOF: loop once more to re-check the size (or finish when the size is unknown)
            if expected is None:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403) and not f.on_hub:
                raise SystemExit(f"error: {f.repo} refused the download (HTTP {e.code}): Civitai needs an API key; "
                                 f"make one at {CIVITAI_KEYS} and pass --civitai-token or set $CIVITAI_TOKEN") from e
            attempt += 1
            if attempt > RETRIES:
                raise SystemExit(f"error: giving up on {f.basename} after {RETRIES} retries: {e}") from e
            wait = min(60, 2 ** attempt)
            print(f"  network error ({e}); retry {attempt}/{RETRIES} in {wait}s, resuming from {part.stat().st_size if part.exists() else 0} bytes")
            time.sleep(wait)

    got = part.stat().st_size
    if expected is not None and got != expected:
        raise SystemExit(f"error: {f.basename} ended at {got} bytes, expected {expected}; re-run to resume")
    if f.sha256 and (got_sha := sha256_of(part)) != f.sha256:
        part.unlink()
        raise SystemExit(f"error: {f.basename} does not match its SHA-256 (got {got_sha[:12]}…, expected "
                         f"{f.sha256[:12]}…); the download was discarded, re-run to try again")
    if final.exists():
        final.unlink()
    os.replace(part, final)
    return final


# ---------------------------------------------------------------- cli

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", metavar="MODEL",
                   help="2.1 | 2511 | all  (also: qwen-image-2.1, qwen-image-edit-2511, edit)")
    p.add_argument("--dest", type=Path, default=Path(__file__).resolve().parent / "models",
                   help="ComfyUI-style models root (default: ./models next to this script)")
    p.add_argument("--only", nargs="+", choices=list(KIND_DIR), metavar="KIND",
                   help="restrict to these kinds: diffusion text_encoder vae lora")
    p.add_argument("--limit-rate", type=parse_rate, metavar="RATE",
                   help="cap download speed, e.g. 500k, 5M, 1.5G (bytes/s, 1024-based); uses the plain-HTTP path")
    p.add_argument("--dry-run", action="store_true", help="print the plan and sizes, download nothing")
    p.add_argument("--list", action="store_true", help="print every file this script knows about, then exit")
    p.add_argument("--force", action="store_true", help="ignore the free-disk check")
    p.add_argument("--ipv6", action="store_true",
                   help="allow IPv6 (default: IPv4 only; a half-working v6 stack stalls every HF request)")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="Hugging Face token (public repos need none; default: $HF_TOKEN)")
    p.add_argument("--civitai-token", default=os.environ.get("CIVITAI_TOKEN"), metavar="KEY",
                   help=f"Civitai API key, needed for the files that come from Civitai (default: $CIVITAI_TOKEN); "
                        f"make one at {CIVITAI_KEYS}")

    g = p.add_argument_group("Qwen-Image-2.1")
    g.add_argument("--q21-precision", choices=["int8", "bf16"], default="int8",
                   help="DiT + Qwen3-VL encoder precision (default int8: 7.3 + 9.4 GB; bf16: 14.2 + 17.5 GB)")
    g.add_argument("--q21-variant", choices=["base", "turbo"], default="base",
                   help="base model (templates' default) or the distilled turbo model")
    g.add_argument("--turbo-lora", action="store_true", help="also fetch the 0.9 GB turbo LoRA for the base model")
    g.add_argument("--prompt-enhancer", nargs="*", choices=["i2i", "t2i"], default=[],
                   help="optional Qwen3.5-9B prompt-enhancer encoders (9.5 GB each) the templates can use")

    g = p.add_argument_group("Qwen-Image-Edit-2511")
    g.add_argument("--edit-precision", choices=["fp8", "int8", "bf16"], default="fp8",
                   help="fp8mixed (default, 20.5 GB) | int8_convrot (20.5 GB) | bf16 (40.9 GB, needs offload)")
    g.add_argument("--lightning", choices=["none", "4", "8", "both"], default="4",
                   help="Lightning LoRA steps to fetch (0.85 GB each; default 4)")
    g.add_argument("--skin-loras", nargs="*", choices=list(SKIN_LORAS), default=[], metavar="NAME",
                   help="skin LoRAs for 2511: edit-skin (qwen-edit-skin 1.1, 0.3 GB, Hugging Face), skin-fix "
                        "(Skin Fix Qwen 2.0, 0.6 GB, Civitai: needs --civitai-token); see README.md, Skin LoRAs")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    a = parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up live when stdout is a log file
    if not a.ipv6:
        force_ipv4()

    if a.list:
        files = full_catalog()
        print(f"{'approx GB':>9}  {'kind':<12} repo / path")
        for f in files:
            print(f"{f.approx_gb:9.2f}  {f.kind:<12} {f.repo}/{f.path}")
        return 0

    if not a.models:
        sys.exit("error: name at least one model (2.1, 2511, all) or use --list / --help")
    import importlib.util
    missing = [pkg for pkg in ("huggingface_hub", "tqdm") if importlib.util.find_spec(pkg) is None]
    if missing:
        venv = Path(__file__).resolve().parent / ".venv" / "bin"
        sys.exit(f"error: this Python ({sys.executable}) lacks {', '.join(missing)}. Either run the script directly, "
                 f"as ./download_models.py or uv run download_models.py, and uv supplies the packages; or run "
                 f"./setup_venv.sh once and use {venv}/python, e.g. after: source {venv}/activate")
    chosen: set[str] = set()
    for m in a.models:
        key = MODEL_ALIASES.get(m.lower())
        if key is None:
            sys.exit(f"error: unknown model {m!r}; choose from 2.1, 2511, all")
        chosen.update({"qwen-image-2.1", "qwen-image-edit-2511"} if key == "all" else {key})

    files: list[F] = []
    if "qwen-image-2.1" in chosen:
        files += plan_qwen_image_21(a)
    if "qwen-image-edit-2511" in chosen:
        files += plan_edit_2511(a)
    files = dedupe(files)
    if a.only:
        files = [f for f in files if f.kind in a.only]
    if not files:
        sys.exit("error: nothing selected")

    dest: Path = a.dest.expanduser()
    staging = dest / ".hf-staging"
    sizes = real_sizes(files, a.token)

    todo: list[F] = []
    print(f"dest: {dest}" + (f"   rate cap: {a.limit_rate / 1024 ** 2:.2f} MiB/s" if a.limit_rate else "") + "\n")
    print(f"{'GB':>6}  {'status':<8} {'target':<16} source")
    for f in files:
        size = sizes[(f.repo, f.path)]
        done = is_complete(final_path(dest, f), size)
        if not done:
            todo.append(f)
        print(f"{gb(size)}  {'present' if done else 'download':<8} {KIND_DIR[f.kind] + '/':<16} {f.repo}/{f.path}")
    need = sum(sizes[(f.repo, f.path)] or 0 for f in todo)
    print(f"\n{len(todo)} of {len(files)} files to download, {need / 1e9:.1f} GB")
    if a.limit_rate and need:
        print(f"at {a.limit_rate / 1024 ** 2:.2f} MiB/s that is about {need / a.limit_rate / 60:.0f} min")

    if a.dry_run:
        return 0
    if not todo:
        print("nothing to do")
        return 0
    civitai = [f for f in todo if not f.on_hub]
    if civitai and not a.civitai_token:
        names = ", ".join(f.basename for f in civitai)
        sys.exit(f"error: {names} comes from Civitai, which needs an API key for downloads: make one at {CIVITAI_KEYS} "
                 f"(API Keys, at the bottom) and pass --civitai-token or set $CIVITAI_TOKEN")

    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if need > free and not a.force:
        sys.exit(f"error: need {need / 1e9:.1f} GB but only {free / 1e9:.1f} GB free at {dest} (--force to try anyway)")

    for i, f in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {f.repo}/{f.path}")
        t0 = time.monotonic()
        if not f.on_hub:  # Civitai: no xet, always the plain-HTTP path, capped only with --limit-rate
            out = fetch_throttled(f, dest, a.limit_rate, a.civitai_token, sizes[(f.repo, f.path)])
        elif a.limit_rate:
            out = fetch_throttled(f, dest, a.limit_rate, a.token, sizes[(f.repo, f.path)])
        else:
            out = None
            for attempt in range(1, FAST_ATTEMPTS + 1):
                try:
                    out = fetch_fast(f, dest, staging, a.token)
                    break
                except Exception as e:  # noqa: BLE001 - hf_xet surfaces many error types; any of them means "try again"
                    msg = str(e).splitlines()[0][:160]
                    print(f"  fast path failed ({type(e).__name__}: {msg}); attempt {attempt}/{FAST_ATTEMPTS}")
            if out is None:
                print("  falling back to the plain-HTTP path (IPv4, Range-resumable)")
                out = fetch_throttled(f, dest, None, a.token, sizes[(f.repo, f.path)])
        dt = time.monotonic() - t0
        print(f"  -> {out}  ({out.stat().st_size / 1e9:.2f} GB in {dt:.0f}s)")

    # staging keeps only tiny resume metadata once every file has moved out
    if staging.exists() and not any(p.is_file() and p.suffix == ".safetensors" for p in staging.rglob("*")):
        shutil.rmtree(staging, ignore_errors=True)
    print("\nall done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\ninterrupted; re-run the same command to resume", file=sys.stderr)
        sys.exit(130)
