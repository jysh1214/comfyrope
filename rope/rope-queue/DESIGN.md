# rope-queue: a work queue for ComfyUI jobs

Written 2026-10-17. rope-core's `queue.rs` and the `rope-queue` crate follow it; the Python side (`--job`, submitting, `comfyconvert.py`) is next. Where the code settled a detail the draft left open, this file says what the code does.

## What it is for

Today each `img2img.py` or `img2vid.py` command blocks until its images are made, and whoever started ComfyUI stops it at the end. The user wants to keep issuing commands, have them wait their turn, and have images and clips mixed in one line.

rope-queue is that line. It is the program the user starts instead of `./run_comfyui.sh`:

- It starts ComfyUI and stops it when it ends. ComfyUI's log goes to a file.
- It takes jobs from the Python tools, runs them one at a time in order, and puts every result into rope.
- Its screen shows the queue: each job's id, state and progress.

A **job is one image or one clip**: one ComfyUI prompt. A command with `-n 50` becomes 50 jobs; a batch template becomes one job per image; a replay is a job. The jobs of one command form a **group**, which is what the screen shows collapsed and what can be cancelled in one go.

## Decisions already made

| Decision | Choice |
|---|---|
| Who owns ComfyUI | rope-queue. ComfyUI lives as long as rope-queue does; no idle stop, but VRAM is handed back when the queue is paused or has run dry. A ComfyUI already on the port is an error, never adopted |
| Order | FIFO by job id. No automatic grouping by model |
| Ids | Up by one for every job, never reused, kept across restarts |
| Job size | One image or one clip |
| Where the queue lives | In the rope library, `queue/`, owned by rope-core like the rest of the data |
| Who writes the library | rope-queue, in process through rope-core. In queue mode the Python tools write no library file |
| Who knows requests | The Python tools: validation, flags, workflows, recipes. rope-queue stores and runs; it knows no model |
| Who converts inputs | A conversion stage in rope-queue that calls the Python converter, the way rope calls ffmpeg for posters |
| How the user submits | Through the Python tools (`./img2img.py …`, `run_*.py`). When rope-queue is not running, the command fails and says to start it |
| Bad requests | Reported and skipped at submission, with the fix; the rest of the submission is queued. A failed job never holds the queue up |
| Stopping | Ctrl-C interrupts the running job at once; it goes back to the head of the queue with its id |
| Python environment | The repo's `.venv`, never uv's. Checked when rope-queue starts |
| rope-gui | Shows the queue with live progress. Submitting from the GUI comes later |

## Pieces

```
rope/
├── rope-core/src/queue.rs   # the data: job files, states, ids; reading them; purge sees queued inputs
├── rope-queue/              # NEW leaf crate: the daemon and its client commands, one binary
├── rope/                    # CLI: unchanged in behavior; the replay comparison moves into rope-core
└── rope-gui/                # a Queue page; progress over the socket
comfyrope/
├── comfycore.py             # submits to the daemon; job mode: runs one job for it
├── comfyconvert.py          # NEW: the input converter as a command, for the conversion stage
└── img2img.py, img2vid.py   # unchanged in what they know; gain --job, --wait, --no-queue
```

rope-core and the rope CLI gain no dependency: the CLI is started by Python for every library call and has to stay small. Everything that talks to ComfyUI or draws a terminal screen is in rope-queue:

| Dependency | For | Note |
|---|---|---|
| rope-core, serde, serde_json, anyhow, libc | the data, `prctl`, signals | all in the workspace already |
| ratatui, crossterm | the screen | CJK width is handled |
| ureq with default features off | `GET /system_stats`, `/queue`; `POST /queue`, `/interrupt`, `/free` | a handful of crates, no TLS; the websocket stays in the Python child |

## Data

```
~/.local/share/rope/queue/
├── queue.json                # {"format": 1, "next_job": 173, "next_group": 42}
├── lock                      # held (flock) by the daemon while it runs: "a daemon is alive", as runs/ does it
├── jobs/
│   └── 000123/
│       ├── job.json          # the job, below
│       ├── log               # the child's output, stdout and stderr
│       └── out/              # what the child made: the result file, made.json, graph.json; removed after import
└── comfyui/
    └── 20261017-143022.log   # one ComfyUI log per start; the last 10 are kept
```

- Every file is written with `atomic::write`. The daemon holds `queue/lock` for its whole life; the few writes made while no daemon runs (`clear`) take that same lock, which succeeds exactly when no daemon is alive. The library's `.lock` and `run.lock` are taken only inside rope-core calls and never held while waiting on a child.
- `journal.jsonl` gets a line when a group is submitted and when a job reaches `done`, `failed` or `cancelled`: `op: "queue"`, an empty `ids`, and the job and state in fields of their own. The `ids` field must hold only image hashes, since rope-gui loads each of them. Progress never goes into the journal: the GUI rereads the catalog on every journal line.
- The socket is `$XDG_RUNTIME_DIR/rope-queue.sock`, mode 0600. It is on tmpfs, so a reboot leaves no stale socket behind. One daemon per machine, serving the default library or `--library`.
- `queue/` is state, not content: the README's rsync recipe excludes it. A restored old `queue/` would otherwise bring finished jobs back as queued.
- `queue/rope-queue.log` is what the daemon said, one line per event with the time: the screen's notes, kept. `queue/convert/<id>/` is scratch space for a conversion in progress, removed when it is done.

### job.json

```json
{
  "format": 1,
  "id": 123,
  "group": {"id": 40, "index": 3, "count": 50, "first_seed": 42},
  "state": "ready",
  "created": "2026-10-17T14:30:22+08:00", "started": null, "ended": null,
  "tool": "img2img.py",
  "flags": ["--model=2511", "--lightning=4", "--steps=4", "--cfg=1", "--seed=44",
            "--lora=qwen-edit-skin-1.1.safetensors:1"],
  "prompt": "make the subjects skin details more prominent and natural",
  "inputs": [{"id": "9f4ad8be8504…", "name": "photo", "rendition": null}],
  "into": "緣/v-咖啡廳_2511_skin", "tags": [], "note": null, "fresh": false, "replay": null, "retry_of": null,
  "submitted": {"argv": ["./img2img.py", "photo.jpg", "…", "-n", "50", "--into", "…"],
                "cwd": "/home/alex", "at": "2026-10-17T14:30:22+08:00"},
  "attempts": 0, "exit": null, "error": null,
  "comfy": null,
  "result": null
}
```

- `flags` are **pinned**: every default spelled out and the seed this job's own, as `tool.flags()` already writes them into recipes. Followed by `--`, the inputs and the prompt, they are exactly what the tool's `--replay` parses today, so the tool rebuilds its Request with its own parser. rope-queue never interprets a flag.
- `inputs` are rope ids: the files were imported at submission, so a file changed or moved while the job waits changes nothing. `name` names the result (the file's stem, or the rope name for `@ID`). `rendition` is the id of the converted PNG when the input needed converting.
- `group.index`, `count` and `first_seed` go into the recipe's `batch` field, so repro.sh still says "as 3 of 50 (seeds 42-91)". The result's name takes the group's `created` time, so one command's images sort together, as today.
- `submitted` is what the user typed: shown on the screen, and what "retry the group" runs again.
- `comfy` is reported by the running child, `{"client_id", "prompt_id"}`, so the daemon can clear a dead child's prompt. `result` is `{"id", "short", "folder", "created", "seconds"}` once imported; for a replay, `{"replay_of", "compared", "result"}` as `rope replayed` answers today.

### States

```
queued ──► converting ──► ready ──► running ──► importing ──► done
   │            │            │          │            │
   └────────────┴────────────┴──────────┴────────────┴──► failed
cancelled: from queued, converting or ready it is dropped; from running, interrupted
```

| State | Meaning |
|---|---|
| queued | accepted, inputs imported |
| converting | an input is being converted; once per input, shared by every job that uses it |
| ready | can go to ComfyUI |
| running | a child is working on it |
| importing | the child has delivered the result; the daemon is putting it into rope |
| done | in rope; `result` says where |
| failed | `error` says why; `retry` makes a new job with the same flags |
| cancelled | by the user |

Only the daemon changes a state. `retry` gives a new id, so FIFO stays a sort by id. A job interrupted by stopping the daemon goes back to `ready` with its id, so it is first in line next time.

## The daemon

### Start

1. Connect to the socket. If a daemon answers, say so with its pid and exit.
2. Find the repo: `--repo`, else the library's `REPO` file, which every tool run writes and repro.sh already reads. Check the venv: `<repo>/.venv/bin/python` must exist and import `websocket`, `PIL`, `pillow_heif`, `pillow_jxl` and `rawpy`; otherwise exit with "run ./setup_venv.sh". Never fall back to uv: its environment may need the network and can be pruned, and the daemon runs unattended.
3. Check the port: if a ComfyUI answers at `--server` (default 8188), exit with "a ComfyUI is already running (pid N); stop it first". The lifetime rule only holds for a ComfyUI the daemon started, and a second ComfyUI on the card that drives the display is never wanted.
4. Start ComfyUI: `run_comfyui.sh --listen 127.0.0.1 --port 8188 --reserve-vram 2`, plus `--comfy-args`. In its own session, with `PR_SET_PDEATHSIG=SIGINT` set in `pre_exec`; the signal is tied to the thread that spawned, so this happens on the main thread. Output to `queue/comfyui/<time>.log`; the screen shows only the path. Wait up to 5 minutes for `/system_stats`; on failure, show the log's tail.
5. Take `queue/lock`. Recover: a job in `importing` or `running` with a result in `out/` is imported; any other `running` job goes back to `ready`; a `converting` job starts its conversion again.
6. Open the socket, start the loop and the screen.

### Loop

- The lowest-id `ready` job runs. One child at a time. Starting the next job's upload while the current one samples, so the GPU never idles between jobs, is a later step; nothing in the data forbids it.
- The child is `<repo>/.venv/bin/python -E -u <repo>/<tool> --job <jobdir>`, cwd the repo. Its environment is built, not inherited: `PATH`, `HOME`, `LANG`, `XDG_*`, `ROPE_LIBRARY`, `COMFYUI_SERVER`, `COMFYUI_CLIENT_ID` (new for every attempt). No `VIRTUAL_ENV`, no `PYTHONPATH`, and `-E` drops any `PYTHON*` that got through. stdin is `/dev/null`.
- stdout carries events (below), stderr the tool's human output. Both go to the job's `log`; the events also update the job and the screen.
- On a `result` event the daemon imports in process: `store::add` with a `NewImage` holding `made`, `graph`, `name`, `via` the tool, `folder` the job's `into`, its tags and note, and `file_inputs`, so unfiled inputs follow the result into the folder as today. A replay runs the comparison that is `replayed()` in the CLI now, moved into rope-core. `out/` is removed after a successful import. If rope cannot take the result (disk full, a broken catalog), the job becomes `failed` with its files kept in `out/`, and the queue **pauses**: the next job would fail the same way. The retry key on such a job imports again without running anything.
- Exit codes: 0 done. 1 failed, `error` being the last `error` event or stderr line. 75 ComfyUI unreachable: the job goes back to `ready` and the daemon checks ComfyUI. 130 interrupted by the daemon's own SIGINT: back to `ready`. The daemon trusts its own checks over the code: a child that died with 1 while ComfyUI is gone counts as 75, and a job that comes back with 75 twice while ComfyUI answers the daemon fails instead, since then the tool is wrong, not ComfyUI. A child killed by a signal records the signal as a negative exit, as Python does.
- A failed job never holds the queue up: the next `ready` job runs. The queue pauses itself in two cases only, because every following job would fail the same way: rope cannot take a result (below), or ComfyUI has died twice (below).
- ComfyUI health: its process is polled every tick and `/system_stats` every 5 s. When it has died, the running job goes back to `ready` and ComfyUI is started again, once. A second death within 10 minutes pauses the queue and shows the log's tail: a restart loop on the card that drives the display is not acceptable.
- A child that ends without an `end` event (killed, out of memory) leaves its prompt in ComfyUI. The daemon reads `GET /queue`, deletes the entries whose `client_id` is the child's, and interrupts the running one if it is among them.
- A child that says nothing for `--job-timeout` (default 30 minutes) gets SIGINT, then SIGKILL after 30 s, and fails with "no progress".

### Stop

Ctrl-C, SIGTERM, SIGHUP (closing the terminal tab) and `rope-queue stop` all do the same, at once:

1. SIGINT to the running child, which cancels its prompt in ComfyUI as today. The job goes back to `ready` with its id, so it is first in line next time. A result the child had already delivered is imported first.
2. ComfyUI: SIGINT, wait 30 s, SIGKILL. Remove the socket, release the lock.

To stop without interrupting anything, `pause` first and stop when the running job is done. Handling SIGHUP makes closing a Ghostty tab the same as Ctrl-C, instead of leaving ComfyUI holding the GPU. PDEATHSIG covers a daemon that is killed outright.

### Pause and reload

- `pause`: finish the running job, start none; `resume` goes on. ComfyUI stays up.
- VRAM is handed back when the queue is paused, and when it has been empty for a minute (a grace period, so a burst of commands does not reload between them): `POST /free` with `unload_models` and `free_memory`. ComfyUI moves the models out of VRAM but keeps them in RAM, so the next job brings them back over PCIe in seconds, not from NVMe. `free_memory` also drops ComfyUI's cache of finished results, so a job run again after a release is computed again. Whether the memory then really reaches the desktop under comfy-aimdo's dynamic VRAM is checked with `nvidia-smi` once this runs.
- `reload`: after the running job ends, the daemon `exec`s its own binary again, on the main thread, handing over the ComfyUI pid and log in `ROPE_QUEUE_ADOPT=<pid>:<log>`. ComfyUI stays the same child and its models stay loaded. This is how a rebuilt rope-queue is picked up without a ComfyUI restart; a binary rebuilt underneath the running one is found at its plain path.

## Submitting

Python does everything that needs a Request: `settled()`, the seed base (drawn now, not at run time), every default spelled out through `tool.flags()`, `into` with `$IMG2IMG_INTO` resolved. It also checks what would otherwise fail hours later: the model and LoRA files of the workflow with `model_paths()`, and that every input is a format the converter reads (the decoders are loaded anyway). A request that fails a check is printed with its fix, the download command for a missing file, and **skipped; the other requests of the submission go on**. Then:

```
rope-queue --json add   # on stdin:
{"tool": "img2img.py",
 "requests": [{"flags": [["--model=2511", "--seed=42", ...], ["--model=2511", "--seed=43", ...]],
               "prompt": "...", "images": ["photo.jpg", "@9f4ad8"], "first_seed": 42,
               "into": "...", "tags": [], "note": null, "fresh": false, "replay": null}],
 "submitted": {"argv": [...], "cwd": "..."}}
```

`flags` holds one list per job, the job's own seed inside, so the request's `count` is its length. `images` are relative to `cwd` unless absolute; `@ID` names an image in rope. `replay` names the image a replay runs again. The answer is `{"ok": true, "groups": [{"group": 40, "jobs": [123, 172]}, {"skipped": "why"}, ...]}`, one entry per request in order. `add --wait` then stays subscribed and answers every job of the submission once all have finished.

The daemon, under the queue lock:

1. Imports each image through `store::add`, `via` the tool, as `prepare()` does today; resolves `@ID`. Bytes rope already has cost nothing.
2. Allocates a group id and `count` job ids per request, consecutive; writes the job dirs; one journal line for the group.
3. Replies with a group and its job ids per request, `{"group": 40, "jobs": [123, 172]}`, or the reason a request was skipped: an `@ID` rope does not have, an `into` that is not a folder name rope accepts. The tool prints `queued #123–#172 (group 40): model 2511, Lightning 4-step, seeds 42–91 → 緣/v-咖啡廳_2511_skin`, one line per request, and the skipped ones with their reasons; it exits with 1 when any was skipped.
4. Starts the conversion stage for every distinct input that has no rendition yet.

Only two things fail the whole submission: the socket does not answer ("rope-queue is not running; start it first"), or the tool is not in the repo.

`--wait` keeps the connection open, subscribed to the group, until every job is terminal, and then returns the results. This keeps the contract of `img2img.run()` returning a `Result` per image, and lets a script feed one step's image to the next.

## Conversion stage

Single-frame PNG, JPEG and WebP go to ComfyUI as they are; everything else becomes an upright, single-frame, 8-bit sRGB PNG, exactly as `prepare_inputs()` does today. The stage is rope-queue's; the pixels are Python's:

- For each distinct input id without a rendition, a worker thread runs `<repo>/.venv/bin/python -E -u comfyconvert.py <library file> <out dir> --json`. The answer is `{"as_is": true}`, or `{"file": "….png", "steps": "HEIF, converted to PNG", "libs": {...}}`.
- The daemon calls `store::add_rendition` and records the id in every job that uses the input. Jobs move to `ready` as their inputs are done. A rendition kept with the image from an earlier conversion is used again without running the converter. Two conversions run at a time.
- A format the converter does not know is caught at submission (above) with "convert it to PNG first": that is the user's job, as ffmpeg is for clips. A conversion that still fails here (a damaged file) fails the jobs that use the input, with the converter's message, and the queue goes on.
- A RAW used by 50 jobs is developed once; today every run converts it again. The stage runs while the GPU is busy with earlier jobs.

`comfyconvert.py` is `prepare_inputs()` as a command. The decoders stay the Python ones, so the libraries recorded in renditions stay what they have been.

## Events from the child

One JSON object per stdout line:

| Event | Fields | For |
|---|---|---|
| `stage` | `text` | the status line: "loading the diffusion model", "sampling" |
| `progress` | `value`, `max` | the bar |
| `comfy` | `client_id`, `prompt_id` | clearing the prompt if the child dies |
| `result` | `file`, `made`, `graph`, `name`, `kind` (`add` or `replayed`), `replay_of`, `text` (the enhanced prompt) | the import |
| `error` | `message` | the job's `error` |
| `end` | `seconds` | the job's timing |

Anything else on stdout is a protocol error and fails the job. The child also writes the `result` object to `out/result.json`: a result delivered to a daemon that then died is imported when the next daemon starts, and `retry` on a job that failed at import imports it again.

## Screen

ratatui, redrawn on events:

```
rope-queue   ComfyUI 0.39.2 up 2h13m (pid 41022)   log ~/.local/share/rope/queue/comfyui/20261017-143022.log
─────────────────────────────────────────────────────────────────────────────────────────────────────
 group  jobs        tool     into                   progress               now
 ▸ #40  #123–#172   img2img  緣/v-咖啡廳_2511_skin   12/50 done, 1 failed   #135 sampling 3/4   0:06
 ▸ #41  #173–#177   img2vid  CB_L4_shift5_fps16     0/5                    queued
─────────────────────────────────────────────────────────────────────────────────────────────────────
 #135  -> 3fa9c1e27b0d  緣/v-咖啡廳_2511_skin  1024x1024, 6.8s            the selected job's last log lines
 ↑↓ select   Enter expand   c cancel   r retry   l log   p pause   q stop (interrupts the running job)
```

Groups are collapsed by default; Enter expands one into its jobs. `l` opens the job's log in `$PAGER` with the screen suspended. When stdout is not a terminal, the daemon prints one line per state change instead.

## rope-gui

A Queue page beside Prompts: groups and jobs read through `rope_core::queue`, refreshed on journal lines like everything else; a thread on the socket for progress, reconnecting when the daemon comes back; cancel and retry over the socket. Thumbnails come free, since inputs and results are rope ids. Submitting from a selected image with a saved prompt is the natural next step; it needs the tool's validation, so the GUI would run `img2img.py …` itself. Later.

## Python side

To be worked out next, together with `run_*.py`. The shape is fixed by the above:

- `--job DIR`: build the Request from `job.json` (`flags`, the inputs' library files, `prompt`), make exactly one image or clip against `$COMFYUI_SERVER` with `$COMFYUI_CLIENT_ID`, never start or stop ComfyUI, write the result with `made.json` and `graph.json` into `out/`, speak events on stdout, exit with the codes above. No rope calls. The recipe is built as today: flags, prompt, inputs (ids from `job.json`), ComfyUI's environment, model hashes (the cache stays in the library's `.cache/`), code identity.
- Submitting is the default. `--no-queue` runs directly as today, for debugging. `--wait` submits and waits.
- `comfyconvert.py`.
- Fixes found on the way, wanted with or without the queue: `--replay` cancels its prompt on any exception, not only Ctrl-C; the direct path masks SIGINT around rope calls, so a result in flight is never lost; `wait()` asks `/queue` before giving up on a silent job.

## Client commands

The same binary with a subcommand is a client of the running daemon, over the socket: `add [--wait]` (what the Python tools call), `ls`, `log ID`, `cancel 123 g40` and `retry 123 g40` (plain numbers are jobs, `g40` is group 40), `pause`, `resume`, `stop`, `reload`, `clear`. Every one takes `--json`, as the rope CLI does. `ls`, `log` and `clear` work without a daemon too, reading the files, so jobs left behind by a stopped daemon can be seen and cleaned up.

The daemon itself takes `--library DIR`, `--repo DIR` (default: where a tool run last recorded it), `--server HOST:PORT` (default 127.0.0.1:8188), `--reserve-vram GB` (default 2; 0 leaves ComfyUI's default), `--job-timeout MIN` (default 30), `--converter FILE` (default `comfyconvert.py` in the repo), `--no-screen`, and after `--` anything for ComfyUI. Without a terminal it prints one line per change instead of drawing the screen.

## Open points

None. Deferred by decision: keeping the GPU busy between jobs, the group layer on the screen, submitting from rope-gui, and the Python side (next).

## Size

Estimated from the code each part replaces or extends; not measured.

| Part | Lines |
|---|---|
| rope-core: `queue.rs` with tests, the purge hook, the replay comparison moved in | ~500 |
| rope-queue: daemon loop, ComfyUI management, import, socket and protocol, conversion stage, screen, client commands | ~1800 |
| rope-gui: the Queue page | ~500 |
| comfycore: job mode, submission, events; `comfyconvert.py` | ~400 |
