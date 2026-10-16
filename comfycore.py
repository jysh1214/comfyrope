"""What the scripts that make things with ComfyUI share; img2img.py is one, and keeps only what is about its
models. Nothing in here knows a model. It talks to ComfyUI, starting one when none answers; brings the inputs
into the rope library and converts those ComfyUI cannot read; follows the queued workflows; and hands every
result to rope with its recipe: the flags, the inputs, the model files' hashes, and the ComfyUI it ran on.

Not a script, so it has no package header: the script that imports it lists the packages (websocket-client and
the image decoders) in its own header, and setup_venv.sh installs them from there. A tool on top of it defines
its Request (Settled says what the core needs of it), builds its workflows as a Graph, and keeps each result
through Run.keep; img2img.py shows all of it."""
from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import os
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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn, Protocol

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "run_comfyui.sh"
SERVER = "http://127.0.0.1:8188"
TIMEOUT = 1800
UPLOAD_SUBFOLDER = "img2img"  # uploads land in ComfyUI/input/img2img/
ROPE_BUILD = HERE / "rope" / "target" / "release" / "rope"  # the image library; see rope/README.md
ROPE_HOWTO = "cargo build --release --manifest-path rope/Cargo.toml"
# Model file -> the command that downloads it, for the message when ComfyUI lacks one; each tool adds its own.
FETCH: dict[str, str] = {}


class Error(Exception):
    """What stops a run; the message says what went wrong and, where there is one, the fix. The command line
    prints it as `error: ...` and exits with 1."""


def die(msg: str) -> NoReturn:
    raise Error(msg)


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


# ---------------------------------------------------------------- requests and results

def parse_lora(text: str) -> tuple[str, float]:
    """"style.safetensors:0.8" -> ("style.safetensors", 0.8); the strength defaults to 1, the suffix to .safetensors."""
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


def rope_ref(img: Path) -> bool:
    """@ID names an image in rope, unless a file of that name exists."""
    return str(img).startswith("@") and not img.exists()


class Settled(Protocol):
    """What this module needs of a tool's request once settled: the fields every tool's Request has (img2img.py's
    has more). A tool's settled() returns a copy with the defaults filled in and the settings checked, raising
    ValueError for a bad one; that copy is what the tool runs from and what a Result reports."""

    images: list[Path]  # the inputs, each a file or "@ID", an image in rope; the first one names the result
    prompt: str
    model: str  # names the result, with the first input and the seed: <time>_<input>_<model>_s<seed>
    count: int  # how many results, one seed each
    seed: int | None  # the first one's seed; None picks one at random
    into: str | None  # the rope folder for the results; None for unfiled
    tags: list[str]
    note: str | None

    def settled(self) -> Settled: ...


@dataclass
class Options:
    """How to reach ComfyUI and rope: run()'s keyword arguments, the command line's ComfyUI and --library options."""

    server: str = SERVER
    keep_server: bool = False
    no_start: bool = False
    comfy_args: list[str] = field(default_factory=list)
    timeout: int = TIMEOUT
    fresh: bool = False
    library: str | None = None


@dataclass
class Result:
    """One image run() made, and where it went."""

    request: Settled  # settled: the settings as run, every default filled in
    index: int  # which of the request's `count` images, from 1
    seed: int
    id: str = ""  # rope's id, the PNG's SHA-256; "" when rope could not keep it (then `path` is in rescue/)
    short: str = ""  # the id as rope shows it, its first 12 digits
    path: Path | None = None  # the PNG: in rope, or in rescue/ when rope could not keep it
    folder: str | None = None  # its rope folder; None when unfiled
    created: bool = False  # False: rope had these bytes already (ComfyUI answered from its cache)
    width: int = 0
    height: int = 0
    seconds: float = 0.0  # from the job's start in ComfyUI to its result
    enhanced_prompt: str = ""  # what the 2.1 prompt enhancer wrote, with enhance=True


# ---------------------------------------------------------------- input images

# Single-frame files of these formats go up untouched. Checked against LoadImage in ComfyUI 0.39.2: it reads them
# correctly, EXIF orientation included, but turns any multi-frame file into a batch of images, pads HEIC to its
# coded size, and cannot read JPEG XL, camera RAW or SVG. Everything else is converted to PNG here first.
AS_IS = {"PNG", "JPEG", "WEBP"}
RAW_SUFFIXES = {".dng", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf", ".orf", ".rw2",
                ".pef", ".srw", ".x3f", ".3fr", ".iiq", ".rwl", ".mrw", ".erf", ".kdc", ".dcr", ".mos"}


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
        self.image_out = ""  # the node whose output is the result, e.g. PreviewImage
        self.text_out = ""   # PreviewAny node with a text to show and record, e.g. 2.1's enhanced prompt; "" for none

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


# ---------------------------------------------------------------- ComfyUI client

class ComfyError(Error):
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


def fresh_tag(kind: str) -> str:
    """A tag for upload names that ComfyUI has not seen: it caches node results by their inputs, the image names
    included, and answers an identical workflow with the old result instead of computing it again."""
    return kind + uuid.uuid4().hex[:6]


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

class LibraryError(Error):
    """rope refused or failed; the message is rope's own."""


class Library:
    """The rope library this run keeps its inputs and results in (see rope/README.md). Every change goes through
    the rope command. A run takes only two locks: run.lock, shared, while it brings its inputs in, and then
    its own runs/<pid>.json, which names those inputs, until it ends. `rope purge` leaves the images named in a
    locked runs file alone, since a result of that run will be made from them, and deletes anything else."""

    run_file = None  # this process's runs/<pid>.json, open and locked; one per process, whatever made the Library

    def __init__(self, exe: str, flags: list[str]) -> None:
        self.exe, self.flags = exe, flags
        self.root = Path()
        self.run_lock = None

    @staticmethod
    def find(library: str | None = None) -> Library:
        """The library (--library, else $ROPE_LIBRARY, else ~/.local/share/rope): results have nowhere else to go,
        so a missing rope stops the run before any GPU work."""
        exe = os.environ.get("ROPE") or (str(ROPE_BUILD) if ROPE_BUILD.exists() else shutil.which("rope"))
        if not exe:
            die(f"rope is not built; build it with: {ROPE_HOWTO} (see rope/README.md)")
        lib = Library(exe, [f"--library={library}"] if library else [])
        lib.root = Path(lib.call("where")["library"])
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
        """Holds run.lock shared, until `using` records what this run uses: purge waits for that, so an input
        cannot go between being brought in and being recorded."""
        import fcntl

        if self.run_lock is None:
            self.run_lock = open(self.root / "run.lock", "a")
            fcntl.flock(self.run_lock, fcntl.LOCK_SH)

    def using(self, ids: list[str]) -> None:
        """Records the images this run uses in runs/<pid>.json, locked until this process ends, and lets go of
        run.lock. The file goes when the process ends normally; after a crash, rope removes it on seeing that
        nothing holds its lock."""
        import atexit
        import fcntl

        self.hold()
        if Library.run_file is None:
            (self.root / "runs").mkdir(exist_ok=True)
            f = open(self.root / "runs" / f"{os.getpid()}.json", "a+")  # not "w": a file left by a dead run is reused
            fcntl.flock(f, fcntl.LOCK_EX)
            Library.run_file = f
            atexit.register(Library.done)
        f = Library.run_file
        f.seek(0)
        f.truncate()
        command = shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]])
        record = {"pid": os.getpid(), "started": iso(time.localtime()), "inputs": ids, "command": command}
        f.write(json.dumps(record, ensure_ascii=False))
        f.flush()
        self.run_lock.close()
        self.run_lock = None

    @staticmethod
    def done() -> None:
        """Removes this process's runs file; its lock goes with the file handle."""
        if Library.run_file is not None:
            try:
                os.unlink(Library.run_file.name)
            except OSError:
                pass
            Library.run_file.close()
            Library.run_file = None


# ---------------------------------------------------------------- recipes

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


def code_identity(script: Path) -> dict:
    """Which code made the result: the git blob id of the script and of comfycore.py, which names that exact file
    content (img2img_blob, comfycore_blob), plus the commit and whether either file differs from it."""
    files = [script, Path(__file__).resolve()]
    out = {}
    for f in files:
        data = f.read_bytes()
        out[f"{f.stem}_blob"] = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    commit = git("rev-parse", "--short", "HEAD")
    if commit:
        out["commit"] = commit
        out["dirty"] = bool(git("status", "--porcelain", "--", *(f.name for f in files)))
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


MODEL_ROOTS = (HERE / "models", HERE / "ComfyUI" / "models")  # where ComfyUI finds them, in its order


class ModelHashes:
    """SHA-256 of model files, cached in the library by file identity (device, inode, size, modification time):
    each file is read once, about 30 s for the default 58 GB, and again only when it changes. A file moved within
    its file system, say with the whole repo, keeps its entry."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.known = self.load()

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def key(st: os.stat_result) -> str:
        return f"{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"

    def cached(self, st: os.stat_result) -> str | None:
        return self.known.get(self.key(st))

    def get(self, path: Path, st: os.stat_result) -> str:
        key = self.key(st)
        if key not in self.known:
            with open(path, "rb") as f:
                self.known[key] = hashlib.file_digest(f, "sha256").hexdigest()
            self.save()
        return self.known[key]

    def save(self) -> None:
        """Merges with what another run may have saved meanwhile, and replaces the file in one step."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            merged = self.load() | self.known
            tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(merged, indent=1), "utf-8")
            tmp.replace(self.path)
        except OSError:
            pass  # only a cache: the hash is computed again next time


def model_paths(nodes: dict) -> list[tuple[dict, Path | None]]:
    """Every model file a workflow loads, once each: its record ({"dir", "name"}) and where it is, if anywhere."""
    out = []
    for node in nodes.values():
        kind = MODEL_INPUTS.get(node["class_type"])
        if not kind or not isinstance(node["inputs"].get(kind[1]), str):
            continue
        entry = {"dir": kind[0], "name": node["inputs"][kind[1]]}
        if entry in [e for e, _ in out]:
            continue
        found = next((p for p in (root / kind[0] / entry["name"] for root in MODEL_ROOTS) if p.exists()), None)
        out.append((entry, found))
    return out


def model_files(nodes: dict, hashes: ModelHashes) -> list[dict]:
    """Every model file a workflow loads, with its SHA-256, size and modification time. The name alone says little:
    a file can be replaced under the same name, a LoRA retrained, a download repeated."""
    out = []
    for entry, path in model_paths(nodes):
        if path is not None:
            st = path.stat()
            entry |= {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
            try:
                entry["sha256"] = hashes.get(path, st)
            except OSError:
                pass
        out.append(entry)
    return out


def start_prehash(hashes: ModelHashes, graphs: list[dict]) -> Background:
    """Hashes the model files of these workflows in the background, so that reading a new model file overlaps the
    GPU work. Says first which files have to be read, since that takes a while the first time."""
    todo: dict[Path, os.stat_result] = {}
    for nodes in graphs:
        for _, path in model_paths(nodes):
            if path is not None and path not in todo and hashes.cached(path.stat()) is None:
                todo[path] = path.stat()
    if todo:
        size = sum(st.st_size for st in todo.values()) / 1e9
        print(f"hashing {len(todo)} model file(s), {size:.1f} GB, for the recipe; once, while the job runs")

    def read() -> None:
        for path, st in todo.items():
            try:
                hashes.get(path, st)
            except OSError:
                pass  # model_files() records the file without a hash

    return Background(read)


class Background:
    """A call running in a daemon thread: Ctrl-C or an error ends the script without waiting for it."""

    def __init__(self, fn, *args) -> None:
        import threading

        self.done = threading.Event()
        self.value, self.error = None, None
        threading.Thread(target=self.run, args=(fn, *args), daemon=True).start()

    def run(self, fn, *args) -> None:
        try:
            self.value = fn(*args)
        except Exception as e:  # handed to whoever waits for the result
            self.error = e
        finally:
            self.done.set()

    def result(self):
        self.done.wait()
        if self.error is not None:
            raise self.error
        return self.value


def decoder_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in ("pillow", "pillow-heif", "pillow-jxl-plugin", "rawpy"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            pass
    return out


def send_result(run: Run, name: str, data: bytes, made: str, job: Job, args) -> tuple[dict | None, Path | None]:
    """Hands a result to rope: the PNG, its recipe and its workflow, as files for `rope ARGS`. Returns rope's
    answer; or, when rope failed, None and the PNG where it went instead: the GPU work is done then, so the files
    go to rescue/ in the library (or the system's temp folder) with the command that brings them in later."""
    lib = run.lib
    graph = json.dumps(job.graph.nodes, ensure_ascii=False)
    with tempfile.TemporaryDirectory(prefix="img2img-result-") as tmp:
        files = Path(tmp) / f"{name}.png", Path(tmp) / "made.json", Path(tmp) / "graph.json"
        files[0].write_bytes(data)
        files[1].write_text(made, "utf-8")
        files[2].write_text(graph, "utf-8")
        try:
            return lib.call(*args(*files)), None
        except LibraryError as e:
            error = e
    for base in (lib.root / "rescue", Path(tempfile.gettempdir()) / "img2img-rescue"):
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
    command = shlex.join([lib.exe, *lib.flags, *args(*files)])
    script = folder / "bring-in.sh"
    script.write_text(f"#!/bin/sh\n# Hands this result to rope, which could not take it: {error}\nexec {command}\n", "utf-8")
    script.chmod(0o755)
    run.status.line(f"  rope could not keep this result: {error}\n  it is saved in {tilde(folder)}; once rope works, "
                    f"bring it in with:\n    {tilde(script)}")
    return None, files[0]


# ---------------------------------------------------------------- running jobs

@dataclass
class Inputs:
    """One request's inputs, brought into rope and ready for ComfyUI."""

    names: list[str]  # each input's name: the file's stem, or its rope name; the first one names the result
    records: list[dict]  # per input, rope's "id" and "path" (and "uploaded", the id of its PNG conversion)
    uploads: list[Path]  # the file to upload for each input: the input itself, or its PNG conversion
    stems: list[str]  # the name part of each upload's server-side name
    uploaded: list[str] = field(default_factory=list)  # the server-side names, once uploaded


def resolve_input(lib: Library | None, img: Path) -> tuple[Path, str]:
    """An input's file and name: the file itself and its stem; or, for @ID and for a file inside the library,
    rope's copy and the image's rope name (not its hash), which names the result and the upload."""
    in_library = lib is not None and img.exists() and img.resolve().is_relative_to(lib.root)
    if rope_ref(img) or in_library:
        if lib is None:
            die(f"{img} names an image in rope, which this run does not use")
        info = lib.call("show", str(img))
        return Path(info["path"]), info["name"] or info["short"]
    return img, img.stem


def prepare(requests: list[Settled], workdir: Path, lib: Library | None, keep: bool = True) -> list[Inputs]:
    """The inputs of every request, ready for upload; a file several requests share is handled once. With `keep`,
    each input goes into the library first, read once, and from then on the library's copy is used: the library
    holds exactly the bytes ComfyUI reads, even if the original is moved or rewritten while the job runs. A
    converted input's PNG is kept with it too. Without `keep` (a dry run), rope only says where @ID inputs are."""
    wanted = [[resolve_input(lib, img) for img in r.images] for r in requests]
    files: dict[Path, str] = {}  # each distinct file, with its name
    for pairs in wanted:
        for path, name in pairs:
            files.setdefault(path, name)
    records: dict[Path, dict] = {path: {"path": path} for path in files}
    keep = keep and lib is not None
    if keep:
        lib.hold()
        kept = lib.call("import", "--via=img2img.py", f"--repo={HERE}", "--", *map(str, files))
        records = {path: {"id": k["id"], "path": Path(k["path"])} for path, k in zip(files, kept)}
        lib.using([k["id"] for k in kept])
    prepared = dict(zip(files, prepare_inputs([records[p]["path"] for p in files], workdir, list(files.values()))))
    if keep:
        for path, (upload, how) in prepared.items():
            if how:
                info = json.dumps({"steps": how, "libs": decoder_versions()})
                records[path]["uploaded"] = lib.call("rendition", records[path]["id"], str(upload), f"--info={info}")["uploaded"]
    return [Inputs(names=[name for _, name in pairs], records=[records[path] for path, _ in pairs],
                   uploads=[prepared[path][0] for path, _ in pairs],
                   stems=[prepared[path][0].stem if prepared[path][1] else name for path, name in pairs])
            for pairs in wanted]


@dataclass
class Job:
    """One workflow in ComfyUI's queue: image `index` of a request, with its own seed."""

    request: Settled
    inputs: Inputs
    index: int  # which of the request's `count` images, from 1
    seed: int
    graph: Graph
    label: str = ""  # the status line when it starts: "[3/6] seed 42", with "#2" before the seed among several requests
    started: float = 0.0
    took: float = 0.0  # seconds from its start in ComfyUI to its result
    saved: list[str] = field(default_factory=list)  # where each result went, for the status line
    results: list[Result] = field(default_factory=list)


@dataclass
class Run:
    """What the jobs of one run share: the ComfyUI, the library, and what every recipe records."""

    comfy: Comfy
    lib: Library
    options: Options
    status: Status
    keep: Callable[[Run, Job, bytes, str], str]  # keeps a result and says where it went: img2img.py's keep_result, or its keep_replay
    hashes: ModelHashes
    when: time.struct_time = field(default_factory=time.localtime)
    server_info: str = ""
    env: dict = field(default_factory=dict)
    code: dict = field(default_factory=dict)
    prehash: Background | None = None  # hashing every job's model files, started before the GPU work

    def models(self, nodes: dict) -> list[dict]:
        """The model files of a workflow with their hashes, waiting for the hashes if they are still being computed."""
        if self.prehash is not None:
            if not self.prehash.done.is_set():
                self.status.line("  waiting for the model files' hashes")
            self.prehash.result()
        return model_files(nodes, self.hashes)


def png_size(data: bytes) -> tuple[int, int] | None:
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    return None


def run_name(when: time.struct_time, stem: str, model: str, seed: int) -> str:
    """<time>_<first image>_<model>_s<seed>: sorts by time, and the images of one run share the time."""
    stem = re.sub(r"[^\w.-]+", "_", stem)[:40].strip("_.") or "image"  # no spaces or shell characters; keeps CJK
    return f"{time.strftime('%Y%m%d-%H%M%S', when)}_{stem}_{model}_s{seed}"


def exact(x: float) -> str:
    """The shortest text that reads back as the same float: 1, 0.8, 2.123456789."""
    return str(int(x)) if x.is_integer() else repr(x)


def collect(run: Run, pid: str, job: Job, entry: dict | None) -> None:
    comfy, status = run.comfy, run.status
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
    job.took = time.monotonic() - job.started if job.started else 0.0
    took = f", {job.took:.1f}s" if job.started else ""
    for ref in refs:
        data = comfy.fetch(ref)
        where = run.keep(run, job, data, enhanced)
        size = png_size(data)
        status.line(f"  -> {where}" + (f"  {size[0]}x{size[1]}" if size else "") + took)
        job.saved.append(where)


def wait(run: Run, ws, pending: dict[str, Job], total: int) -> None:
    import websocket

    comfy, status, timeout = run.comfy, run.status, run.options.timeout
    deadline = time.monotonic() + timeout  # moved on by every word about one of the jobs: a batch has no total limit
    while pending:
        if time.monotonic() > deadline:
            die(f"no word from ComfyUI about the job for {timeout}s; it may still be running there")
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            for pid in list(pending):  # quiet for a while: make sure no finish message was missed
                entry = comfy.history(pid)
                if entry is not None:
                    collect(run, pid, pending.pop(pid), entry)
                    deadline = time.monotonic() + timeout
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
        deadline = time.monotonic() + timeout
        node = data.get("node")
        if kind == "execution_start":
            job.started = time.monotonic()
            status.line(job.label)
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
            collect(run, pid, pending.pop(pid), None)


def with_comfy(opts: Options, work: Callable[[Comfy], list]) -> list:
    """Runs `work` against the ComfyUI at opts.server, starting one for the duration if none answers there, and
    reports how many images it made."""
    comfy = Comfy(opts.server)
    started = False
    if not comfy.alive():
        if opts.no_start:
            die(f"no ComfyUI answers at {opts.server}; start it with {tilde(LAUNCHER)} or drop --no-start")
        comfy.start(opts.comfy_args)
        started = True
    t0 = time.monotonic()
    try:
        made = work(comfy)
    finally:
        if started and opts.keep_server and comfy.proc.poll() is None:
            print(f"ComfyUI keeps running at {opts.server} (pid {comfy.proc.pid}); stop it with: kill -INT {comfy.proc.pid}")
        elif started:
            print("stopping the ComfyUI this run started")
            comfy.stop()
    print(f"done: {len(made)} image{'' if len(made) == 1 else 's'} in {time.monotonic() - t0:.0f}s")
    return made


# ---------------------------------------------------------------- cli

def add_comfy_args(g) -> None:
    g.add_argument("--server", default=SERVER, help=f"ComfyUI address (default {SERVER})")
    g.add_argument("--keep-server", action="store_true", help="when this command starts ComfyUI, leave it running")
    g.add_argument("--no-start", action="store_true", help="fail instead of starting ComfyUI when none is running")
    g.add_argument("--comfy-args", default="", metavar="ARGS",
                   help='extra flags for a ComfyUI this command starts, e.g. "--reserve-vram 2"')
    g.add_argument("--timeout", type=int, default=TIMEOUT,
                   help=f"give up when ComfyUI has said nothing about the jobs for this many seconds (default {TIMEOUT})")
    g.add_argument("--fresh", action="store_true",
                   help="compute the image again even when ComfyUI has the result of an identical workflow cached "
                        "(the GPU's results differ slightly from run to run); the loaded models are kept. A replay "
                        "is always fresh")


def check_packages(needed: dict[str, str], script: Path) -> None:
    """Fail with a how-to instead of a ModuleNotFoundError when this Python lacks the script's packages."""
    import importlib.util

    missing = [package for module, package in needed.items() if importlib.util.find_spec(module) is None]
    if missing:
        die(f"this Python ({sys.executable}) lacks {', '.join(missing)}. Either run the script directly, "
            f"as ./{script.name}, and uv supplies the packages; or run ./setup_venv.sh once and use "
            f"{tilde(HERE / '.venv' / 'bin' / 'python')}, e.g. after: source {tilde(HERE / '.venv' / 'bin' / 'activate')}")
