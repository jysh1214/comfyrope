#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Install or update ComfyUI in a uv-managed venv, wired to the models folder next to this script.

Steps (each is skipped when already in place, so re-running is safe):
  1. comfyui    git clone at a release tag (default: newest vX.Y.Z), or move an existing clone to --ref
  2. venv       <dir>/.venv created by uv (default Python 3.12)
  3. packages   requirements.txt + pip via `uv pip install`, PyTorch from the --cuda index (default cu130)
  4. config     <dir>/extra_model_paths.yaml pointing ComfyUI at --models
  5. launcher   run_comfyui.sh next to this script
  6. self-test  main.py --cpu --quick-test-for-ci with the GPU hidden (CUDA_VISIBLE_DEVICES=""), using
                throwaway temp/user folders so a ComfyUI already running from this checkout is not disturbed

torchaudio is left out on purpose: ComfyUI does not import it, and without it ComfyUI-Manager's
"PIPFixer" never runs its torch rollback, which installs a nightly torch for versions it does not know
(its map stops at 2.7.0). The manager logs "PyTorch is not installed" because of this; that line is harmless.

Options you leave out keep what an existing install already has (ComfyUI version, Python, CUDA build);
options you pass are enforced. Nothing touches the GPU unless you pass --gpu-test.

Examples:
    ./install_comfyui.py --dry-run        # show the plan, change nothing
    ./install_comfyui.py                  # fresh install, or re-verify an existing one
    ./install_comfyui.py --ref latest     # update ComfyUI to the newest release
    ./install_comfyui.py --ref v0.39.2    # pin a release (--ref master for the dev branch)
    ./install_comfyui.py --manager        # add the built-in ComfyUI-Manager
    ./install_comfyui.py --gpu-test       # finish with a small CUDA matmul (uses the GPU)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

REPO_URL = "https://github.com/Comfy-Org/ComfyUI.git"
HERE = Path(__file__).resolve().parent
DEFAULT_PYTHON = "3.12"  # ComfyUI also supports 3.13; 3.12 has the widest custom-node wheel coverage
DEFAULT_CUDA = "cu130"
# --cuda value (uv's --torch-backend) -> lowest CUDA version the NVIDIA driver must support
CUDA_BACKENDS: dict[str, tuple[int, int] | None] = {
    "cu130": (13, 0), "cu129": (12, 9), "cu128": (12, 8), "cu126": (12, 6), "cpu": None,
}
RELEASE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
MIN_FREE_GB = 10  # a CUDA venv is ~7 GB when uv cannot hardlink from its cache
MODEL_KINDS = ("diffusion_models", "text_encoders", "vae", "loras")  # the folders download_models.py fills

DRY_RUN = False


# ---------------------------------------------------------------- output and subprocess helpers

def die(msg: str) -> NoReturn:
    sys.exit(f"error: {msg}")


def step(name: str, msg: str) -> None:
    print(f"\n[{name}] {msg}")


def note(msg: str) -> None:
    print(f"  {msg}")


def tilde(p: Path) -> str:
    home, s = str(Path.home()), str(p)
    return "~" + s[len(home):] if s == home or s.startswith(home + "/") else s


def run(cmd: list, *, cwd: Path | None = None) -> None:
    """State-changing command; output streams to the terminal. Under --dry-run it is only printed."""
    print("  $ " + shlex.join(str(c) for c in cmd))
    if not DRY_RUN:
        subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def query(cmd: list, *, cwd: Path | None = None, timeout: float = 60, check: bool = True) -> str | None:
    """Read-only command (also runs under --dry-run). Returns stripped stdout, or None on failure if check=False."""
    argv = [str(c) for c in cmd]
    r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        if check:
            raise subprocess.CalledProcessError(r.returncode, argv, r.stdout, r.stderr)
        return None
    return r.stdout.strip()


# ---------------------------------------------------------------- git

@dataclass
class RemoteRefs:
    tags: dict[str, str] = field(default_factory=dict)      # tag -> commit sha
    branches: dict[str, str] = field(default_factory=dict)  # branch -> commit sha

    @property
    def releases(self) -> list[str]:
        """vX.Y.Z tags, oldest first."""
        rel = [t for t in self.tags if RELEASE_TAG.match(t)]
        return sorted(rel, key=lambda t: tuple(int(x) for x in RELEASE_TAG.match(t).groups()))


def remote_refs(remote: str, cwd: Path | None = None) -> RemoteRefs:
    out = query(["git", "ls-remote", "--tags", "--heads", remote], cwd=cwd, timeout=120)
    refs = RemoteRefs()
    for line in out.splitlines():
        sha, _, ref = line.partition("\t")
        if ref.startswith("refs/heads/"):
            refs.branches[ref.removeprefix("refs/heads/")] = sha
        elif ref.startswith("refs/tags/"):
            name = ref.removeprefix("refs/tags/")
            if name.endswith("^{}"):  # peeled line: the commit an annotated tag points at
                refs.tags[name[:-3]] = sha
            else:
                refs.tags.setdefault(name, sha)
    return refs


def resolve_ref(ref: str, refs: RemoteRefs) -> tuple[str, str, str]:
    """Map --ref to (name, 'tag' or 'branch', commit sha)."""
    if ref == "latest":
        if not refs.releases:
            die("the remote has no vX.Y.Z release tags")
        ref = refs.releases[-1]
    if ref in refs.tags:
        return ref, "tag", refs.tags[ref]
    if ref in refs.branches:
        return ref, "branch", refs.branches[ref]
    newest = refs.releases[-1] if refs.releases else "vX.Y.Z"
    die(f"{ref!r} is neither a tag nor a branch on the remote (try latest, master or {newest})")


@dataclass
class Checkout:
    commit: str
    tag: str | None
    branch: str | None
    version: str | None
    dirty: list[str]

    def describe(self) -> str:
        where = self.tag or (f"branch {self.branch}" if self.branch else "a detached commit")
        return f"{where} ({self.commit[:10]})"


def inspect_checkout(d: Path) -> Checkout:
    vfile = d / "comfyui_version.py"
    m = re.search(r'__version__\s*=\s*"([^"]+)"', vfile.read_text()) if vfile.exists() else None
    return Checkout(
        commit=query(["git", "rev-parse", "HEAD"], cwd=d),
        tag=query(["git", "describe", "--tags", "--exact-match", "HEAD"], cwd=d, check=False) or None,
        branch=query(["git", "symbolic-ref", "--short", "-q", "HEAD"], cwd=d, check=False) or None,
        version=m.group(1) if m else None,
        dirty=query(["git", "status", "--porcelain", "--untracked-files=no"], cwd=d).splitlines(),
    )


def comfyui_pids(d: Path) -> list[int]:
    """PIDs of `python main.py` processes whose working directory is d, i.e. ComfyUI running from this tree."""
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            if any(arg.endswith(b"main.py") for arg in argv) and Path(os.readlink(proc / "cwd")) == d:
                pids.append(int(proc.name))
        except OSError:
            continue
    return pids


def refuse_if_running(d: Path, action: str) -> None:
    pids = comfyui_pids(d)
    if pids:
        die(f"ComfyUI is running from {d} (pid {', '.join(map(str, pids))}); stop it before {action}")


# ---------------------------------------------------------------- venv and GPU facts

FACTS = r"""
import importlib.metadata as m, json, platform, sys
v = {"python": platform.python_version()}
for n in sys.argv[1:]:
    try:
        v[n] = m.version(n)
    except m.PackageNotFoundError:
        v[n] = None
print(json.dumps(v))
"""


def venv_facts(py: Path, *dists: str) -> dict[str, str | None] | None:
    """Python version and installed distribution versions in the venv, from package metadata
    (no torch import, so no CUDA). None when the venv interpreter does not run."""
    if not py.exists():
        return None
    out = query([py, "-I", "-c", FACTS, *dists], check=False)
    return json.loads(out) if out else None


def torch_build(version: str | None) -> str | None:
    """'2.14.1+cu130' -> 'cu130'; a PyPI wheel without a local tag -> 'pypi'; not installed -> None."""
    if version is None:
        return None
    return version.split("+", 1)[1] if "+" in version else "pypi"


def pick_backend(requested: str | None, build: str | None) -> str | None:
    """uv --torch-backend value for this run, or None to leave the installed torch alone."""
    if requested:
        return requested
    if build is None:  # no torch yet
        return DEFAULT_CUDA
    if build in CUDA_BACKENDS:  # keep the build that is installed
        return build
    return None  # PyPI or an index this script does not know


def driver_cuda() -> tuple[int, int] | None:
    """Highest CUDA version the NVIDIA driver supports, from the nvidia-smi banner (an NVML query, no CUDA context).
    Drivers up to 5xx print 'CUDA Version: X.Y', 6xx prints 'CUDA UMD Version: X.Y'."""
    if shutil.which("nvidia-smi") is None:
        return None
    m = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", query(["nvidia-smi"], timeout=30, check=False) or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def same_python(have: str | None, want: str) -> bool:
    return have is not None and (have == want or have.startswith(want + "."))


# ---------------------------------------------------------------- generated files

def yaml_text(models: Path) -> str:
    base = f"{models}/"
    if re.search(r"[\s:#'\"{}\[\],&*!|>%@`]", base):
        base = json.dumps(base)
    origin = (f"Models downloaded by {tilde(HERE / 'download_models.py')} live outside the ComfyUI tree."
              if models == HERE / "models" else f"Models live in {models}, outside the ComfyUI tree.")
    return (
        f"# {origin}\n"
        "# is_default makes these folders the first listed and the default download target.\n"
        "i2i:\n"
        f"    base_path: {base}\n"
        "    is_default: true\n"
        + "".join(f"    {k}: {k}/\n" for k in MODEL_KINDS)
    )


def launcher_text(d: Path, manager: bool) -> str:
    cd = '"$(dirname "$(readlink -f "$0")")/ComfyUI"' if d == HERE / "ComfyUI" else shlex.quote(str(d))
    flags = "--enable-manager " if manager else ""
    return (
        "#!/usr/bin/env bash\n"
        "# Start ComfyUI from its own venv. Extra args pass through, e.g.:\n"
        "#   ./run_comfyui.sh                       # http://127.0.0.1:8188\n"
        "#   ./run_comfyui.sh --listen --port 8188  # reachable from the LAN\n"
        "#   ./run_comfyui.sh --reserve-vram 2      # leave 2 GB for the desktop if the display stutters\n"
        "set -euo pipefail\n"
        f"cd {cd}\n"
        f'exec .venv/bin/python main.py {flags}"$@"\n'
    )


def write_file(path: Path, text: str, label: str, rewrite: bool, executable: bool = False) -> bool:
    """Write a generated file, but never over an edited copy unless --rewrite-config.
    Returns True when the file ends up with the generated content."""
    if path.exists():
        if path.read_text() == text:
            step(label, f"{tilde(path)} is up to date")
            return True
        if not rewrite:
            step(label, f"{tilde(path)} has other content; left unchanged "
                        "(--rewrite-config replaces it and keeps a .bak)")
            return False
        step(label, f"rewrite {tilde(path)} (previous copy kept as {path.name}.bak)")
        if not DRY_RUN:
            shutil.copy2(path, path.with_name(path.name + ".bak"))
    else:
        step(label, f"write {tilde(path)}")
    if not DRY_RUN:
        path.write_text(text)
        if executable:
            path.chmod(0o755)
    return True


# ---------------------------------------------------------------- steps

def preflight() -> None:
    for tool, hint in (("git", "install git first"), ("uv", "https://docs.astral.sh/uv/getting-started/installation/")):
        if shutil.which(tool) is None:
            die(f"{tool} is not on PATH ({hint})")
    if "--torch-backend" not in (query(["uv", "pip", "install", "--help"], check=False) or ""):
        die("this uv is too old for --torch-backend; update uv (`uv self update` or your package manager)")


def check_driver(backend: str | None) -> None:
    need = CUDA_BACKENDS.get(backend) if backend else None
    if not need:
        return
    have = driver_cuda()
    if have is None:
        step("driver", f"could not read the driver's CUDA version from nvidia-smi; assuming it can run {backend}")
    elif have < need:
        die(f"the NVIDIA driver supports CUDA up to {have[0]}.{have[1]} but {backend} needs {need[0]}.{need[1]}; "
            "update the driver or pick a lower --cuda")
    else:
        step("driver", f"NVIDIA driver supports CUDA {have[0]}.{have[1]}; {backend} needs {need[0]}.{need[1]}")


def sync_source(a: argparse.Namespace) -> bool:
    """Clone, or move the checkout to --ref. Returns True when the checkout changes (or would, under --dry-run)."""
    d = a.dir
    if not (d / ".git").exists():
        if d.exists() and any(d.iterdir()):
            die(f"{d} exists but is not a git clone; pick another --dir or move it away")
        name, _, sha = resolve_ref(a.ref or "latest", remote_refs(REPO_URL))
        step("comfyui", f"clone {REPO_URL} at {name} ({sha[:10]}) into {tilde(d)}")
        run(["git", "-c", "advice.detachedHead=false", "clone", "--branch", name, REPO_URL, d])
        return True
    if not (d / "main.py").is_file() or not (d / "comfy").is_dir():
        die(f"{d} is a git clone but does not look like ComfyUI (no main.py or comfy/)")
    co = inspect_checkout(d)
    if a.ref is None:
        step("comfyui", f"keep {co.describe()}")
        try:
            newest = remote_refs("origin", cwd=d).releases[-1:]
        except subprocess.SubprocessError:
            newest = []
            note("could not reach the remote to look for a newer release")
        if newest and newest[0] != co.tag:
            note(f"newest release is {newest[0]}; --ref latest moves to it")
        return False
    name, kind, sha = resolve_ref(a.ref, remote_refs("origin", cwd=d))
    if co.commit == sha and (kind == "tag" or co.branch == name):
        step("comfyui", f"already at {name} ({sha[:10]})")
        return False
    if co.dirty:
        die(f"{d} has local changes to tracked files; commit or `git stash` them first:\n  " + "\n  ".join(co.dirty))
    refuse_if_running(d, "switching versions")
    step("comfyui", f"{co.describe()} -> {name} ({sha[:10]})")
    run(["git", "fetch", "--tags", "origin"], cwd=d)
    if kind == "tag":
        run(["git", "checkout", "--quiet", "--detach", f"refs/tags/{name}"], cwd=d)
    else:
        run(["git", "checkout", "--quiet", name], cwd=d)
        run(["git", "merge", "--ff-only", "--quiet", f"origin/{name}"], cwd=d)
    return True


def sync_venv(a: argparse.Namespace, facts: dict | None, build: str | None) -> bool:
    """Create or recreate <dir>/.venv as needed. Returns True when the existing venv is kept."""
    venv = a.dir / ".venv"
    have_py = facts["python"] if facts else None
    reasons = []
    if facts:
        if a.python and not same_python(have_py, a.python):
            reasons.append(f"Python {have_py} -> {a.python}")
        if a.cuda and build and build != a.cuda:
            reasons.append(f"torch build {build} -> {a.cuda}")
        if not reasons:
            step("venv", f"keep {tilde(venv)} (Python {have_py})")
            return True
    elif venv.exists():
        reasons.append("its interpreter does not run")
    if reasons:
        refuse_if_running(a.dir, "recreating its venv")
        step("venv", f"recreate {tilde(venv)}: {'; '.join(reasons)}")
    else:
        step("venv", f"create {tilde(venv)}")
    probe = a.dir
    while not probe.exists():
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    if free_gb < MIN_FREE_GB:
        note(f"warning: only {free_gb:.0f} GB free on {probe}; a CUDA venv can take ~7 GB")
    version = a.python or (".".join(have_py.split(".")[:2]) if have_py else DEFAULT_PYTHON)
    run(["uv", "venv", "--clear", "--python", version, venv])
    return False


def sync_packages(a: argparse.Namespace, backend: str | None, manager: bool, preview: bool) -> None:
    d, py = a.dir, a.dir / ".venv" / "bin" / "python"
    cmd: list = ["uv", "pip", "install", "--python", py]
    if backend:
        cmd += ["--torch-backend", backend]
    cmd += ["-r", d / "requirements.txt"]
    if manager:
        mreq = d / "manager_requirements.txt"
        if d.exists() and not mreq.exists():
            die(f"{mreq} is missing: this ComfyUI version predates the built-in manager")
        cmd += ["-r", mreq]
    cmd.append("pip")  # custom-node installers often run `python -m pip`, and uv venvs ship without it
    if a.torch == "latest":
        cmd += ["--upgrade-package", "torch", "--upgrade-package", "torchvision"]
    elif a.torch:
        cmd.append(f"torch=={a.torch}")
    what = "requirements.txt" + (" + manager_requirements.txt" if manager else "") + " + pip"
    step("packages", f"sync {what}" + (f"; PyTorch from the {backend} index" if backend else ""))
    run(cmd)
    if preview:  # checkout and venv stay as they are, so uv can tell exactly what would change
        r = subprocess.run([str(c) for c in cmd] + ["--dry-run"], capture_output=True, text=True)
        for line in (r.stderr + r.stdout).strip().splitlines():
            note("  " + line)


def self_test(a: argparse.Namespace, models: Path | None, manager: bool) -> None:
    py = a.dir / ".venv" / "bin" / "python"
    cmd = [py, "main.py", "--cpu", "--quick-test-for-ci"]
    if manager:  # test the setup the launcher actually starts
        cmd.append("--enable-manager")

    def isolated(scratch: str) -> list[str]:
        # Every ComfyUI start wipes its temp folder and opens its user folder and database. Point all three
        # elsewhere, so a ComfyUI running from this checkout right now keeps its previews and settings.
        return ["--temp-directory", scratch, "--user-directory", f"{scratch}/user", "--database-url", "sqlite:///:memory:"]

    step("self-test", "start ComfyUI once on the CPU with the GPU hidden and throwaway temp/user folders, then exit")
    print("  $ CUDA_VISIBLE_DEVICES= " + shlex.join([str(c) for c in cmd] + isolated("<scratch>")))
    if DRY_RUN:
        return
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="comfyui-selftest-") as scratch:
        os.mkdir(f"{scratch}/user")
        try:
            r = subprocess.run([str(c) for c in cmd] + isolated(scratch), cwd=a.dir,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True,
                               timeout=600)
        except subprocess.TimeoutExpired:
            die("the self-test did not finish within 10 minutes")
    log = ANSI.sub("", r.stdout + r.stderr)
    log_path = a.dir / "install-selftest.log"  # *.log is in ComfyUI's .gitignore
    log_path.write_text(log)
    if r.returncode != 0:
        print("\n".join("  | " + line for line in log.splitlines()[-30:]))
        die(f"the self-test exited with {r.returncode}; full log: {log_path}")
    version = re.search(r"ComfyUI version: (\S+)", log)
    note(f"ok in {time.monotonic() - t0:.0f}s: ComfyUI {version.group(1) if version else '?'} starts "
         f"(log: {tilde(log_path)})")
    if models is not None:
        found = {kind for kind, path in re.findall(r"Adding extra search path (\S+) (\S+)", log)
                 if Path(path).parent == models}
        missing = [k for k in MODEL_KINDS if k not in found]
        if missing:
            note(f"warning: ComfyUI did not add {', '.join(missing)} under {tilde(models)}; check extra_model_paths.yaml")
        else:
            note(f"model folders from {tilde(models)}: {', '.join(MODEL_KINDS)}")
    for line in log.splitlines():
        if "IMPORT FAILED" in line or "did not import correctly" in line:
            note(f"warning: {line.strip()}")


GPU_TEST = r"""
import time, torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
p = torch.cuda.get_device_properties(0)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
x @ x
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    x @ x
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 20
print(f"  {p.name}, {p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}: "
      f"fp16 4096x4096 matmul {dt * 1e3:.2f} ms ({2 * 4096**3 / dt / 1e12:.0f} TFLOPS)")
"""


def gpu_test(a: argparse.Namespace) -> None:
    py = a.dir / ".venv" / "bin" / "python"
    step("gpu-test", "open a CUDA context and time an fp16 matmul (the display shares this GPU)")
    print(f"  $ {shlex.quote(str(py))} -c <matmul test>")
    if not DRY_RUN:
        subprocess.run([str(py), "-c", GPU_TEST], check=True)


def summary(a: argparse.Namespace, launcher: Path, launcher_ok: bool, t0: float) -> None:
    if DRY_RUN:
        print("\ndry run finished; nothing was changed")
        return
    f = venv_facts(a.dir / ".venv" / "bin" / "python",
                   "torch", "torchvision", "comfyui-frontend-package", "comfyui_manager") or {}
    co = inspect_checkout(a.dir)
    manager = f.get("comfyui_manager")
    direct = f"cd {tilde(a.dir)} && .venv/bin/python main.py" + (" --enable-manager" if manager else "")
    rows = [
        ("ComfyUI", f"{co.tag or co.branch or co.commit[:10]} (version {co.version}) in {tilde(a.dir)}"),
        ("Python", f"{f.get('python')} in {tilde(a.dir / '.venv')}"),
        ("torch", f"{f.get('torch')}, torchvision {f.get('torchvision')}"),
        ("frontend", f"comfyui-frontend-package {f.get('comfyui-frontend-package')}"),
        ("manager", f"comfyui_manager {manager}" if manager else "not installed (--manager adds it)"),
        ("models", "not managed (--no-model-paths)" if a.no_model_paths
         else f"{tilde(a.models)} via extra_model_paths.yaml"),
        ("start", f"{tilde(launcher)}  ->  http://127.0.0.1:8188" if launcher_ok else direct),
    ]
    print(f"\ndone in {time.monotonic() - t0:.0f}s")
    for k, v in rows:
        print(f"  {k:<9} {v}")


# ---------------------------------------------------------------- cli

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=HERE / "ComfyUI",
                   help="ComfyUI checkout (default: ./ComfyUI next to this script)")
    p.add_argument("--ref", help="ComfyUI version: latest (newest vX.Y.Z), a tag like v0.39.2, or a branch like "
                                 "master (default: latest for a fresh clone, otherwise keep the checkout)")
    p.add_argument("--python", metavar="VER", help=f"venv Python, e.g. 3.12 or 3.13 (default: {DEFAULT_PYTHON} for a "
                                                   "new venv, otherwise keep; a different version recreates .venv)")
    p.add_argument("--cuda", choices=list(CUDA_BACKENDS),
                   help=f"PyTorch build, passed to uv --torch-backend (default: {DEFAULT_CUDA} for a new venv, "
                        "otherwise keep; a different build recreates .venv)")
    p.add_argument("--torch", metavar="VER", help="pin torch, e.g. 2.14.1, or 'latest' to upgrade torch and "
                                                  "torchvision (default: keep what is installed)")
    p.add_argument("--models", type=Path, default=HERE / "models",
                   help="models root written to extra_model_paths.yaml (default: ./models next to this script)")
    p.add_argument("--no-model-paths", action="store_true", help="leave extra_model_paths.yaml alone")
    p.add_argument("--manager", action="store_true",
                   help="install the built-in ComfyUI-Manager; the launcher then passes --enable-manager")
    p.add_argument("--rewrite-config", action="store_true",
                   help="replace extra_model_paths.yaml and run_comfyui.sh even if edited (keeps a .bak)")
    p.add_argument("--no-test", action="store_true", help="skip the CPU-only startup self-test")
    p.add_argument("--gpu-test", action="store_true", help="finish with a small CUDA matmul; this one uses the GPU")
    p.add_argument("--dry-run", action="store_true", help="print the plan and commands, change nothing")
    a = p.parse_args(argv)
    a.dir = a.dir.expanduser().resolve()
    a.models = a.models.expanduser().resolve()
    if a.python and not re.fullmatch(r"3\.\d+(\.\d+)?", a.python):
        p.error("--python takes a version like 3.12 or 3.12.13")
    if a.torch and a.torch != "latest" and not re.fullmatch(r"\d+\.\d+(\.\d+)?", a.torch):
        p.error("--torch takes a version like 2.14.1, or latest")
    return a


def main(argv: list[str]) -> int:
    global DRY_RUN
    a = parse_args(argv)
    DRY_RUN = a.dry_run
    sys.stdout.reconfigure(line_buffering=True)
    t0 = time.monotonic()
    preflight()
    print(f"ComfyUI in {tilde(a.dir)}" + ("  (dry run: commands are printed, not executed)" if DRY_RUN else ""))

    # An existing venv decides the defaults for --python and --cuda.
    facts = venv_facts(a.dir / ".venv" / "bin" / "python", "torch", "comfyui_manager")
    build = torch_build(facts["torch"]) if facts else None
    backend = pick_backend(a.cuda, build)
    manager = a.manager or bool(facts and facts["comfyui_manager"])

    check_driver(backend)
    source_changed = sync_source(a)
    venv_kept = sync_venv(a, facts, build)
    sync_packages(a, backend, manager, preview=DRY_RUN and venv_kept and not source_changed)
    if not DRY_RUN:
        torch_now = (venv_facts(a.dir / ".venv" / "bin" / "python", "torch") or {}).get("torch")
        if backend and torch_build(torch_now) != backend:
            note(f"warning: torch is {torch_now}, not a {backend} build")

    models = None
    if not a.no_model_paths:
        models = a.models
        write_file(a.dir / "extra_model_paths.yaml", yaml_text(models), "config", a.rewrite_config)
        if not models.exists():
            note(f"{tilde(models)} does not exist yet; download_models.py creates it")
    launcher = HERE / "run_comfyui.sh"
    launcher_ok = write_file(launcher, launcher_text(a.dir, manager), "launcher", a.rewrite_config, executable=True)

    if not a.no_test:
        self_test(a, models, manager)
    if a.gpu_test:
        gpu_test(a)
    summary(a, launcher, launcher_ok, t0)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\ninterrupted; re-run the same command to continue", file=sys.stderr)
        sys.exit(130)
    except subprocess.CalledProcessError as e:
        cmd = e.cmd if isinstance(e.cmd, str) else shlex.join(str(c) for c in e.cmd)
        last = (e.stderr or "").strip().splitlines()[-1:]
        sys.exit(f"error: exit {e.returncode} from: {cmd}" + (f"\n  {last[0]}" if last else "")
                 + "\nfix the cause and re-run; finished steps are skipped")
    except subprocess.TimeoutExpired as e:
        sys.exit(f"error: timed out after {e.timeout:.0f}s: {shlex.join(str(c) for c in e.cmd)}")
