"""What the tools do with rope-queue, the work queue in rope/ (rope/rope-queue/DESIGN.md, "Python side"). A command
hands its requests to the running daemon and comes back at once: submit(), one group of jobs per request. The daemon
owns ComfyUI, brings the inputs and the results into rope, and runs each job as a child of the tool, `img2img.py
--job DIR`: job_main(), which makes exactly one image or clip, writes it into the job's out/ with its recipe, and
speaks JSON events on stdout. A tool knows its requests; comfycore.py knows how a result is made and recorded; this
module only connects the two to the daemon. In neither role does it change the library: submitting reads rope, a job
reads the library's files, and the daemon writes.

Not a script. img2img.py and img2vid.py import it; the Python that imports them has the packages."""
from __future__ import annotations

import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from comfycore import (FETCH, HERE, ROPE_HOWTO, UPLOAD_SUBFOLDER, Comfy, Error, Inputs, Job, Library, LibraryError,
                       ModelHashes, Options, Run, Settled, Status, Tool, Unreachable, check_packages, code_identity,
                       comfy_env, convertible, die, fresh_tag, model_paths, parse_replay_args, recipe, replay_jobs,
                       replay_prepare, replay_recipe, replay_run, replay_setup, resolve_input, run_name, settle,
                       start_prehash, tilde, upload_name, wait)

QUEUE_BUILD = HERE / "rope" / "target" / "release" / "rope-queue"


# ---------------------------------------------------------------- the daemon

def find_queue() -> str:
    """The rope-queue command: $ROPE_QUEUE, else the workspace's release build, else one on PATH. rope's build makes it."""
    exe = os.environ.get("ROPE_QUEUE") or (str(QUEUE_BUILD) if QUEUE_BUILD.exists() else shutil.which("rope-queue"))
    if not exe:
        die(f"rope-queue is not built; build it with: {ROPE_HOWTO} (see rope/README.md)")
    return exe


def queue_call(exe: str, *args: str, stdin: str | None = None, library: str | None = None):
    """Runs `rope-queue --json ARGS` and returns its exit code and its answer. An exit without an answer is an Error
    with rope-queue's message; with one (`add` exits 1 when a request was skipped) the caller reads the answer."""
    cmd = [exe, "--json", *([f"--library={library}"] if library else []), *args]
    try:
        r = subprocess.run(cmd, input=stdin, capture_output=True, encoding="utf-8", timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Error(f"cannot run {exe}: {e}") from None
    answer = json.loads(r.stdout) if r.stdout.strip() else None
    if answer is None and r.returncode != 0:
        raise Error(r.stderr.strip().removeprefix("rope-queue: ") or f"{shlex.join(cmd)} failed")
    return r.returncode, answer


def daemon(exe: str, lib: Library, library: str | None) -> dict:
    """What the running daemon says of itself (`rope-queue status`). Without one the command fails and says how to
    start it; so it does when the daemon serves another library than this command's: the socket is one per user."""
    try:
        _, status = queue_call(exe, "status", library=library)
    except Error as e:
        if "not running" not in str(e):
            raise
        flag = f" --library {shlex.quote(str(lib.root))}" if library else ""
        die(f"rope-queue is not running; start it in another terminal with: rope-queue{flag} --repo {tilde(HERE)}")
    theirs = Path(status.get("library") or "")
    if theirs.resolve() != lib.root.resolve():
        die(f"the running rope-queue serves {tilde(theirs)}, not {tilde(lib.root)}: stop it and start one for this "
            "library, or drop --library")
    return status


def origin() -> dict:
    """What the user typed, and where: shown on rope-queue's screen and kept with every job."""
    return {"argv": [Path(sys.argv[0]).name, *sys.argv[1:]], "cwd": os.getcwd()}


# ---------------------------------------------------------------- submitting

@dataclass
class Submitted:
    """What became of one request: its group of jobs in rope-queue, or why it was skipped. run() returns one per
    request, in order."""

    request: Settled  # settled: every default filled in
    first_seed: int  # the seed of its first job; the others count up from it
    group: int | None = None  # the group's id in rope-queue, once queued
    jobs: tuple[int, int] | None = None  # the first and the last job id; they are consecutive
    skipped: list[str] = field(default_factory=list)  # why it was not queued, each reason with its fix

    @property
    def count(self) -> int:
        """How many jobs were queued for it."""
        return self.jobs[1] - self.jobs[0] + 1 if self.jobs else 0


def submit(tool: Tool, requests: Iterable[Settled], *, fresh: bool = False, library: str | None = None) -> list[Submitted]:
    """Hands every request of a tool to rope-queue, one group of `count` jobs per request, and returns a Submitted per
    request. Every request is settled first, so a bad field (a ValueError) stops the batch before anything is sent; a
    request that fails a check (problems) is printed with its reasons and skipped, and the others go on. The command
    fails (Error) when rope-queue is not built or not running. It prints one line per request and comes back at
    once: what then happens to the jobs is rope-queue's to show."""
    settled = settle(requests)
    if not settled:
        return []
    check_packages(tool.packages, tool.file)  # the decoders, for the format check
    exe = find_queue()
    lib = Library.find(library)
    daemon(exe, lib, library)
    out: list[Submitted] = []
    sending: list[tuple[Submitted, dict]] = []
    for r in settled:
        s = Submitted(r, r.seed if r.seed is not None else random.randrange(2**32))
        out.append(s)
        s.skipped = problems(tool, r, lib)
        if s.skipped:
            continue
        sending.append((s, {"flags": [tool.flags(r, s.first_seed + i) for i in range(r.count)], "prompt": r.prompt,
                            "images": [str(i) for i in r.images], "into": r.into, "tags": r.tags, "note": r.note,
                            "fresh": fresh, "replay": None, "first_seed": s.first_seed}))
    if sending:
        submission = {"tool": tool.name, "requests": [req for _, req in sending], "submitted": origin()}
        _, answer = queue_call(exe, "add", stdin=json.dumps(submission, ensure_ascii=False), library=library)
        for (s, _), g in zip(sending, answer["groups"]):
            if g.get("skipped"):
                s.skipped = [g["skipped"]]
            else:
                s.group, s.jobs = g["group"], (g["jobs"][0], g["jobs"][1])
    report(tool, out)
    return out


def problems(tool: Tool, r: Settled, lib: Library) -> list[str]:
    """What would make this request fail only when its turn comes, caught now instead: an input the converter cannot
    read, an @ID rope does not have, a model or LoRA file ComfyUI lacks (each with its fix)."""
    out: list[str] = []
    files: list[tuple[Path, str]] = []
    for img in r.images:
        try:
            path, name = resolve_input(lib, img)
        except LibraryError as e:
            out.append(str(e))
            continue
        why = convertible(path)
        if why:
            out.append(why)
        files.append((path, name))
    if out:
        return out  # the workflow is built from the inputs (img2vid.py reads the start image's size)
    inputs = Inputs(names=[n for _, n in files], records=[], uploads=[p for p, _ in files], stems=[n for _, n in files],
                    uploaded=[f"{UPLOAD_SUBFOLDER}/{n}{p.suffix}" for p, n in files])
    try:
        nodes = tool.build(r, inputs, r.seed or 0, tool.replay.context_dry if tool.replay else None).nodes
    except Exception as e:  # the tool could not build it from the input as it is (a RAW it does not open, say)
        print(f"  (the model files could not be checked: {e})", file=sys.stderr)
        return []
    for entry, path in model_paths(nodes):
        if path is None:
            fix = FETCH.get(entry["name"]) or f"put it in {tilde(HERE / 'models' / entry['dir'])}"
            out.append(f"ComfyUI lacks models/{entry['dir']}/{entry['name']}; fix: {fix}")
    return out


def report(tool: Tool, submitted: list[Submitted]) -> None:
    """One line per request: its job ids and settings, or why it was skipped."""
    for n, s in enumerate(submitted, start=1):
        r = s.request
        which = f"request {n}, " if len(submitted) > 1 else ""
        first = str(r.images[0]) if r.images else ""
        name = first if first.startswith("@") else Path(first).name
        if s.skipped:
            print(f"skipped {which}{name}:")
            for why in s.skipped:
                print(f"  {why}")
        else:
            ids = f"#{s.jobs[0]}" if s.jobs[0] == s.jobs[1] else f"#{s.jobs[0]}–#{s.jobs[1]}"
            tags = f", tags {', '.join(r.tags)}" if r.tags else ""
            print(f"queued {ids} (group {s.group}) {name}: {tool.describe(r, s.first_seed)}{tags} → {r.into or 'unfiled'}")


def replay_main(tool: Tool, argv: list[str]) -> int:
    """A tool's --replay: the item's record, today's request and what differs are worked out here, as on the direct
    path (comfycore.replay_prepare); then one job in rope-queue makes the item again and compares. --dry-run prints
    the workflow and --no-queue makes it here, both as before."""
    a = replay_prepare(tool, argv)
    if a.dry_run or a.no_queue:
        return replay_run(tool, a)
    exe = find_queue()
    daemon(exe, a.lib, a.library)
    request = {"flags": [replay_flags(argv)], "prompt": a.prompt, "images": [f"@{i['id']}" for i in a.inputs],
               "into": None, "tags": [], "note": None, "fresh": True, "replay": f"@{a.info['id']}",
               "first_seed": a.seed_used}
    submission = {"tool": tool.name, "requests": [request], "submitted": origin()}
    _, answer = queue_call(exe, "add", stdin=json.dumps(submission, ensure_ascii=False), library=a.library)
    g = answer["groups"][0]
    if g.get("skipped"):
        print(f"skipped: {g['skipped']}")
        return 1
    print(f"queued #{g['jobs'][0]} (group {g['group']}): replay of {a.info['short']} {a.info['name']}, seed {a.seed_used}")
    return 0


def replay_flags(argv: list[str]) -> list[str]:
    """A replay job's flags: the command line without --replay ID and --library DIR, which the job has from its record
    and its environment. What is left (-s, --rebuild, the tool's own options) the job parses as this command did."""
    out = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
        elif arg in ("--replay", "--library"):
            skip = True
        elif not arg.startswith(("--replay=", "--library=")):
            out.append(arg)
    return out


# ---------------------------------------------------------------- --job DIR: one job, as rope-queue's child

def job_main(tool: Tool, jobdir: Path) -> int:
    """rope-queue's child, `TOOL --job DIR`: makes the one image or clip of the job in DIR, against the ComfyUI the
    daemon names in $COMFYUI_SERVER as $COMFYUI_CLIENT_ID, writes it into DIR/out/ with its recipe and workflow, and
    speaks the events of DESIGN.md on stdout; everything else printed goes to stderr, the job's log. It changes
    nothing in the library: the daemon imports what it delivers. Exit codes: 0 done; 1 failed, an `error` event
    saying why; 75 ComfyUI unreachable, the job goes back in line; 130 interrupted by the daemon."""
    events = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr  # every print() lands in the log; only emit() speaks to the daemon

    def emit(event: dict) -> None:
        events.write(json.dumps(event, ensure_ascii=False) + "\n")

    pending: list[str] = []  # the prompt in ComfyUI, cancelled when the daemon interrupts
    comfy = None
    try:
        job = json.loads((jobdir / "job.json").read_text("utf-8"))
        root = Path(os.environ.get("ROPE_LIBRARY") or die("ROPE_LIBRARY is not set: --job is rope-queue's to run"))
        server = os.environ.get("COMFYUI_SERVER") or die("COMFYUI_SERVER is not set: --job is rope-queue's to run")
        comfy = Comfy(server if "://" in server else f"http://{server}")
        comfy.client_id = os.environ.get("COMFYUI_CLIENT_ID") or comfy.client_id
        out = jobdir / "out"
        out.mkdir(exist_ok=True)
        make = replay_job if job.get("replay") else run_job
        done = make(tool, job, root, comfy, out, Status(events=emit), emit, pending)
        emit({"event": "end", "seconds": done.took})
        return 0
    except KeyboardInterrupt:
        if comfy is not None and pending:
            comfy.cancel(pending)
        print("interrupted", file=sys.stderr)
        return 130
    except (Unreachable, urllib.error.URLError, ConnectionError) as e:
        print(f"ComfyUI does not answer: {e}", file=sys.stderr)
        return 75
    except Error as e:
        emit({"event": "error", "message": str(e)})
        return 1
    except Exception as e:  # a bug: the job fails, and the traceback is in its log
        traceback.print_exc()
        emit({"event": "error", "message": f"{type(e).__name__}: {e}"})
        return 1


def run_job(tool: Tool, job: dict, root: Path, comfy: Comfy, out: Path, status: Status, emit, pending: list[str]) -> Job:
    """A job from a submission: the request rebuilt from the job's pinned flags and its inputs' library files, one
    workflow queued and followed, its result kept in out/."""
    if tool.replay is None:
        die(f"{tool.name} cannot rebuild a request from its flags")
    files: list[Path] = []
    uploads: list[Path] = []
    for i in job["inputs"]:
        item, meta = item_meta(root, i["id"])
        files.append(item / meta["file"])
        rendition = i.get("rendition")
        uploads.append(item / "uploaded" / meta["uploaded"][rendition]["file"] if rendition else files[-1])
    r = tool.replay.parse([*job["flags"], "--", *map(str, files), job["prompt"]])
    if r.seed is None:
        die("the job's flags name no seed")
    group = job["group"]
    names = [i["name"] for i in job["inputs"]]
    records = [{"id": i["id"], "path": f, "uploaded": i.get("rendition")} for i, f in zip(job["inputs"], files)]
    inputs = Inputs(names, records, uploads, names)
    tag = fresh_tag("f") if job.get("fresh") else ""
    inputs.uploaded = [comfy.upload(p, upload_name(p, stem, tag)) for p, stem in zip(uploads, names)]
    context = tool.context(comfy, [r])
    graph = tool.build(r, inputs, r.seed, context)
    run = Run(tool, comfy, Library.at(root), Options(), status, keep_out(out, emit, job["flags"]),
              ModelHashes(root / ".cache" / "model-sha256.json"), when=local_time(job["created"]))
    run.server_info, run.env, run.code = comfy.system_info(), comfy_env(comfy), code_identity(tool.file)
    first_seed = group.get("first_seed")
    batch = {"index": group["index"], "count": group["count"],
             "first_seed": first_seed if first_seed is not None else r.seed - group["index"] + 1}
    j = Job(r, inputs, group["index"], r.seed, graph, f"job #{job['id']}, seed {r.seed}", batch=batch)
    run.prehash = start_prehash(run.hashes, [graph.nodes])
    return follow(run, j, emit, pending)


def follow(run: Run, job: Job, emit, pending: list[str]) -> Job:
    """Queues the job's workflow, tells the daemon its prompt id, and follows it to its result."""
    import websocket

    comfy = run.comfy
    ws = websocket.create_connection(f"{comfy.ws_url}?clientId={comfy.client_id}", timeout=15)  # before queueing:
    try:  # the first messages about the prompt come at once
        pid = comfy.queue(job.graph)
        pending.append(pid)
        emit({"event": "comfy", "client_id": comfy.client_id, "prompt_id": pid})
        wait(run, ws, {pid: job}, 1)
    finally:
        ws.close()
    pending.clear()
    return job


def keep_out(out: Path, emit, flags: list[str]) -> Callable[[Run, Job, bytes, str, str], str]:
    """A Run.keep for a job: the result goes into out/ with its recipe (the job's flags as recorded) for the daemon
    to import."""
    def keep(run: Run, job: Job, data: bytes, text: str, suffix: str) -> str:
        name = run_name(run.when, job.inputs.names[0], job.request.model, job.seed)
        return deliver(out, emit, name, suffix, data, recipe(run, job, flags, text), job.graph.nodes, "add", None, text)
    return keep


def deliver(out: Path, emit, name: str, suffix: str, data: bytes, made: dict, graph: dict, kind: str,
            replay_of: str | None, text: str) -> str:
    """Writes a result into out/ (the file, made.json, graph.json, result.json) and reports it to the daemon."""
    file = out / f"{name}{suffix}"
    file.write_bytes(data)
    (out / "made.json").write_text(json.dumps(made, ensure_ascii=False), "utf-8")
    (out / "graph.json").write_text(json.dumps(graph, ensure_ascii=False), "utf-8")
    result = {"file": f"out/{file.name}", "made": "out/made.json", "graph": "out/graph.json", "name": name,
              "kind": kind, "replay_of": replay_of, "text": text or None}
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False), "utf-8")  # for a daemon that comes back
    emit({"event": "result", **result})
    return f"out/{file.name}"


def replay_job(tool: Tool, job: dict, root: Path, comfy: Comfy, out: Path, status: Status, emit, pending: list[str]) -> Job:
    """A replay job: the replayed item's record read from its files, the replay's options from the job's flags, and
    the replay as on the direct path, its result kept in out/ for the daemon to compare and record."""
    a = parse_replay_args(tool, ["--replay", job["replay"], *job["flags"]])
    a.lib = Library.at(root)
    replay_setup(tool, a, record(root, job["replay"]))

    def keep(run: Run, j: Job, data: bytes, enhanced: str, suffix: str) -> str:
        return deliver(out, emit, f"replay-{a.info['short']}", suffix, data, replay_recipe(tool, a, run, j, enhanced),
                       j.graph.nodes, "replayed", a.info["id"], enhanced)

    def queued(pid: str) -> None:
        pending.append(pid)
        emit({"event": "comfy", "client_id": comfy.client_id, "prompt_id": pid})

    j = replay_jobs(tool, a, comfy, status, keep, queued)
    pending.clear()
    return j


def item_meta(root: Path, sha: str) -> tuple[Path, dict]:
    """An item's directory in the library and its meta.json, read as files."""
    item = root / "items" / sha[:2] / sha
    try:
        return item, json.loads((item / "meta.json").read_text("utf-8"))
    except (OSError, ValueError) as e:
        die(f"cannot read the record of {sha[:12]} in the library: {e}")


def record(root: Path, sha: str) -> dict:
    """An item's record as `rope show --json` answers it, read from its files: id, name and path, `meta`, the inputs
    with the file each one's upload came from, and the recorded workflow."""
    item, meta = item_meta(root, sha)
    inputs = []
    for n, inp in enumerate((meta.get("made") or {}).get("inputs") or [], start=1):
        iid = inp.get("id")
        idir = root / "items" / iid[:2] / iid if iid else None
        if idir is None or not (idir / "meta.json").is_file():
            inputs.append({"slot": n, "id": iid, "missing": True, "upload": inp.get("upload")})
            continue
        imeta = json.loads((idir / "meta.json").read_text("utf-8"))
        up = inp.get("uploaded")
        rendition = ((imeta.get("uploaded") or {}).get(up) or {}).get("file") if up else None
        path = idir / "uploaded" / rendition if rendition else idir / imeta["file"]
        inputs.append({"slot": n, "id": iid, "short": iid[:12], "name": imeta.get("name"), "path": str(path),
                       "upload": inp.get("upload"), "uploaded": up})
    graph = item / "graph.json"
    return {"id": sha, "short": sha[:12], "name": meta.get("name"), "path": str(item / meta["file"]), "meta": meta,
            "inputs": inputs, "graph": json.loads(graph.read_text("utf-8")) if graph.is_file() else None}


def local_time(iso: str) -> time.struct_time:
    """A job's `created`, 2026-10-17T14:30:22+08:00, as the local struct_time run_name() takes."""
    try:
        return time.localtime(datetime.fromisoformat(iso).timestamp())  # localtime(): with the zone, for iso()
    except ValueError:
        return time.localtime()
