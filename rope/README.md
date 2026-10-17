# rope: an image library that remembers where each image came from and how it was made

rope keeps images, and the clips made from them, by the hash of their content. Each carries its own metadata:

- **sources**: the paths its file was seen at; there can be several.
- **recipe**: for an image a tool made, the command, the workflow sent to ComfyUI, and the input images, in order.
- **note**: free text.

Folders and tags exist only in rope; files on disk are never moved. `img2img.py` and `img2vid.py` put their inputs and results into rope on every run, so a result can be made again after its inputs are moved or deleted.

Three crates in one Cargo workspace:

| Crate | What it is |
|---|---|
| `rope` | The CLI: import, folders, tags, notes, lineage, repro commands, the trash |
| `rope-gui` | The GUI for browsing and organizing, written with egui after [twelf](https://github.com/jysh1214/twelf) |
| `rope-core` | The library both use. `img2img.py` writes through the `rope` CLI |

## Building and installing

Run these from the repo root. rope needs Rust 1.89 or newer, for the standard library's file locks. The GUI also needs libheif (for HEIC and AVIF) and the Noto Sans CJK fonts (for Chinese text); both are installed on this machine.

### Build only

```bash
cargo build --release --manifest-path rope/Cargo.toml
rope/target/release/rope --help
rope/target/release/rope-gui
```

The binaries are in `rope/target/release/`, where `img2img.py` finds them without any change to `PATH`.

### Install into ~/.cargo/bin

After this, `rope` and `rope-gui` work from anywhere:

```bash
cargo install --locked --path rope/rope
cargo install --locked --path rope/rope-gui
```

- `--path` takes one package at a time, hence two lines.
- `--locked` builds the versions pinned in `rope/Cargo.lock`, the ones the repo was tested with.
- After a `cargo build`, adding `--target-dir rope/target` reuses what is already compiled.
- After a code change, run the same commands again to replace the installed binaries. `cargo uninstall rope rope-gui` removes them.

### Which rope img2img.py runs

`img2img.py` looks for `rope` in this order:

1. `$ROPE`
2. `rope/target/release/rope`
3. `PATH`, e.g. `~/.cargo/bin/rope`

When both a build and an installed copy exist, the build in the repo wins. After a code change, rebuild and reinstall together, or `img2img.py` and the `rope` you type run different versions.

## Where the data is

By default in `~/.local/share/rope/` (following `$XDG_DATA_HOME`): outside the repo and out of git, so `git clean -xdf` or a fresh clone leaves it alone.

To keep it elsewhere, make it with `rope init DIR`, then point at it with `--library DIR` or `$ROPE_LIBRARY`. The default location is created on first use. A location named explicitly must be initialized first: when a data disk is not mounted, commands fail instead of starting a second, empty library and splitting the images between the two.

```
~/.local/share/rope/
├── rope.json            # marks the library root
├── catalog.json         # the folder tree, each image's folder, tags, the trash
├── prompts.json         # prompts kept to use again, each with a title
├── journal.jsonl        # one line per change, only ever appended; the GUI follows it
├── REPO                 # where the comfyrope repo was last seen, for repro.sh
├── envs/<id>.json       # environment snapshots, shared by every image made in the same environment
├── items/3f/<sha256>/   # one folder per image, named after the SHA-256 of its file
│   ├── 3fa9c1e27b0d.png # the image (or clip): 12 hex digits of the hash plus its suffix; read-only, never rewritten
│   ├── poster.png       # for a clip: its first frame, for thumbnails; made with ffmpeg
│   ├── meta.json        # name, sources, recipe, upstream images; for a clip its length, rate and codec
│   ├── graph.json       # the API workflow sent to ComfyUI
│   ├── repro.sh         # generated from meta.json; no absolute paths
│   ├── note.txt
│   └── uploaded/        # for HEIC, RAW and the like: the PNG actually uploaded in its place
├── tmp/                 # items being assembled, renamed into items/ when complete
├── runs/<pid>.json      # the images each img2img.py run in progress uses; the run keeps it locked
├── queue/               # rope-queue's jobs: queue.json counts ids, jobs/<id>/ holds each job, comfyui/ its logs
├── rescue/              # results img2img.py could not hand to rope, waiting for rope add
└── .cache/thumbs/       # GUI thumbnails; safe to delete
```

- `meta.json`, `catalog.json` and `note.txt` are plain text: grep them, or edit them by hand.
- Clips (mp4, m4v, mov, webm, mkv) are kept like images. Their size, length and poster frame come from the `ffprobe` and `ffmpeg` on the system; without them a clip is kept all the same, and `rope check --fix` asks again once they are there.
- Image files in `items/` never move. Organizing changes only `catalog.json`, so a copied path stays valid.

An image's id is the SHA-256 of its file:

- Commands take a prefix of at least 6 hex digits, with or without `@`.
- They also take a file path: rope hashes the file to find the image.
- The same bytes are stored once. Importing them again from another place only adds a source.

## CLI

| Command | What it does |
|---|---|
| `rope mkdir [-p] trips/京都` | Make a folder (in rope only) |
| `rope mv trips/京都 travel/京都` | Rename or move a folder |
| `rope mv @3fa9c1 @9b2e0f portraits` | Move images into a folder; `/` is unfiled |
| `rope rmdir [-r] F` | Remove an empty folder; with `-r`, a folder with everything in it: its subfolders go, and its images move to the trash (restored, they come back unfiled) |
| `rope import PATH... [--into F] [--tag T]... [--note T]` | Import files, or every image and clip in a folder recursively, all into F, made if there is none; `--tag` repeats |
| `rope import --tree DIR... [--into F]` | Make each disk folder a rope folder of the same name, with its subfolders, in F or at the top |
| `rope ls [F] [-r] [-l] [--tag T] [--fav] [--trash]` | List a folder; without F, the unfiled images. Favorites are marked ★ |
| `rope tree` | The folder tree with image counts, and every tag |
| `rope show ID [--downstream]` | Everything about an image: path, sources, recipe, upstream, how many images were made from it (`--downstream` lists them), note |
| `rope path ID` | The absolute path of the image's file in the library |
| `rope name ID TEXT` | Change the display name; the file is not touched |
| `rope note ID [--set T \| --append T \| -e]` | Print or change the note; `-e` edits it in `$EDITOR` |
| `rope tag add\|rm T ID...`, `rope tag rename OLD NEW`, `rope tags` | Manage tags |
| `rope fav ID... [--off]` | Mark images as favorites, or unmark them; rope-gui shows favorites first |
| `rope prompt add TITLE [TEXT \| --from ID]`, `rope prompt ls`, `rope prompt show NAME`, `rope prompt edit NAME [--title T] [--text T] [-e]`, `rope prompt mv NAME N`, `rope prompt rm NAME` | Keep prompts to use again. `show` prints the text, so `"$(rope prompt show 黃昏海邊)"` puts it in a command; `--from` takes the prompt an image was made with; `mv` puts it at position N; NAME is a title or an id |
| `rope lineage ID [--up \| --down]` | The tree of images above or below |
| `rope repro ID [--command]` | Print the command that makes the image again; `--command` prints the full command of the tool that made it (`img2img.py`, `img2vid.py`), to change and run as a variation |
| `rope rm ID...`, `rope restore ID...` | Move images to the trash, or back out |
| `rope purge ID... \| --trash [--break-lineage]` | Delete for good; see [Deleting](#deleting) |
| `rope check [--hash] [--fix]` | Check the library; `--hash` verifies every file against its hash; `--fix` repairs the catalog, writes outdated `repro.sh` again and makes a clip's missing poster frame and length |
| `rope log [ID]` | The latest changes |

Every command takes `--json` for scripts.

## GUI

```bash
rope-gui                 # or: rope-gui 3fa9c1, to open on that image
rope-gui --foreground    # stay attached to the terminal, e.g. to watch its messages
```

The window runs on its own: the command returns at once, and closing the terminal or pressing Ctrl-C there leaves the window open. What it prints goes to `.cache/rope-gui.log` in the library. A library that cannot be opened is still reported in the terminal.

**Left:**

- Prompts, the page of prompts kept to use again (below); then All, Unfiled, and rope's own folder tree, with image counts.
- Right-click a folder to copy its path (as commands take it: `trips/京都`, or `'trips/day 1'` when it has a space), make a subfolder, rename it, or delete it with everything in it (its images move to the trash; it says how many first); `+` makes a top-level folder. A new folder's name can be a path: `a/b` makes `b` inside `a`, and `a` too if it is missing.
- Ctrl-click folders to pick several; the pick shows as selected. Right-click one of them and choose Delete N folders…, or press Delete, to delete them all with everything in them; it says how many folders and images first. A plain click on the left, a click on an image, or Esc lets the pick go.
- Below the tree: the tags, then the trash. Right-click the trash and choose Clean to empty it; it asks first, since that deletes for good.

**Middle:**

- Thumbnails of the folder chosen on the left, without the images in its subfolders: favorites first, then the newest first.
- A star in a thumbnail's top-right corner marks it as a favorite: gold when it is one, an outline on hover otherwise. Click it, or press L.
- A clip shows its first frame, with a play mark and its length in the bottom-left corner.
- Click to select, Ctrl-click to add, Shift-click for a range. Double-click, Enter or Space opens the large view (below).
- Right-click to copy the path or the repro command, view large, open in the default image viewer, move to a folder, add a tag, add to or remove from the favorites, or trash.
- Drag a thumbnail onto a folder on the left to move it there; the selection drags together, with a ghost of the first image and the count at the pointer. Unfiled takes drops too, and so do a tag, which adds it, and the trash. A closed folder opens after half a second under the pointer, the list scrolls when the pointer is near its top or bottom, and Esc lets go.
- **Move to…** (in that menu, on M, and as the Folder button on the right) opens a dialog over the folders: type part of a path and the list narrows to the folders that have it, the ones whose own name starts with it first; ↑ ↓ pick, Enter moves, or click one. With nothing typed, the folders moved to lately come first. A path no folder has yet is made on Enter, with the folders on the way to it.

**Right**, the selected image:

- A preview (click it for the large view), the favorite star, and the name, which you can edit; for a clip, its length, rate and codec.
- Copy path, Copy repro, Copy command and Delete buttons.
- Its folder (click it to move the image), tags and note.
- Made from: its upstream images, in the order of the prompt's 圖1, 圖2; click one to go to it. Made from this: how many images were made from this one; Show lists them in the grid, as Made from 〈name〉 on the left until you choose another view.
- The recipe, replays, and sources; a source that no longer exists is marked gone.

**Queue**, the button at the right end of the top bar, with how many jobs are still to do and a dot while one runs: what rope-queue is doing. The jobs come from the library's `queue/`, so the page shows them whether or not the daemon runs; with the daemon running, the status line says so, with ComfyUI's version and state, and the page follows the daemon over its socket: the stage and progress of the running job as it goes, the queue pausing. Each command is a group, in the order the queue runs them, titled with its input and prompt, where its results go and its tags (what tells a batch's groups apart), with the counts, the time, the command line and a progress bar under that; then one row per job: the input, its name, the job's number, its state, its seed, then what came of it, with a thumbnail of the result, the error of a failed job, or the stage of the running one, and how long it took. Click a row or a group header to see it on the right: the whole command, with Copy, the inputs, the prompt, the pinned flags, folder, tags and note, the times, the result with Open in library, and Log for what the job printed, following along while it runs. Cancel, Retry, and on a group Cancel group and Retry failed and cancelled are there and in the right-click menu; Delete cancels the selection, Esc lets it go; double-click a done job to open its result in the library. Pause in the status line lets the running job finish and holds the next; Resume goes on; Cancel all, beside them, empties the queue: every job that is not finished is cancelled, the running one interrupted, and Retry brings any of them back. Finished jobs stay on the page until Clear finished, at the right of the status line, which forgets the done, failed and cancelled ones; it works without the daemon too. Stopping or reloading the daemon, and submitting, stay in the terminal. When the daemon on the socket serves another library, the page says which and its controls refuse.

**Prompts**, chosen on the left: prompts worth keeping, each with a title. The list is in the middle; click one to edit it on the right, where Save writes it, Copy puts its text on the clipboard and Delete… removes it. New prompt adds one at once, titled Draft (then Draft 2, …) with no text yet, and puts the cursor in the text, to paste into now or later; Save prompt under an image's recipe adds that image's prompt under the image's name. Drag a card up or down to put it elsewhere in the list. The search box filters them. On this page Delete, Ctrl+C and Esc act on the prompt being edited. `rope prompt` reaches the same list from the terminal.

**Keys**, when no text field has focus (Backspace works wherever Delete does: an Apple keyboard's delete key sends it):

| Key | Action |
|---|---|
| Arrow keys | Move the selection |
| Enter, Space | View the selected image large |
| L | Mark the selection as favorites, or unmark it when all of it is |
| M | Move the selection to a folder: the Move to… dialog |
| Ctrl+A | Select all |
| Delete, Backspace | Move to the trash (in the trash: delete for good); with folders picked, delete those |
| Ctrl+C | Copy the path |
| Esc | Clear the selection, and the folder pick |
| F5 | Reload |

**The large view** shows one image as large as the window allows, on a dark background, with a bar for stepping, zooming, the details panel and the default image viewer. It decodes images up to 4096 pixels on a side, and keeps only a few of them on the GPU, which ComfyUI needs more. A clip plays as soon as it opens, round and round: the `ffmpeg` on the system decodes it into memory (a clip whose frames would take more than about 768 MB is decoded smaller), Space pauses and plays, and a bar under the clip has Play/Pause, a progress bar across the width, which also shows how much of the clip is decoded yet, and the position; click or drag the progress bar to go to a frame. Open externally hands a clip to the default video player.

| Key or mouse | Action |
|---|---|
| ← → (or ↑ ↓) | Previous / next image of the view. Images that arrive while looking (an `img2img.py` run finishing, an import) join in, where the view puts them |
| Space | Next image; for a clip, pause or play |
| Ctrl+wheel | Zoom around the cursor |
| Drag | Pan, when zoomed |
| Double-click, 1 | 100%: one image pixel per screen pixel |
| F | Fit to the window |
| + − | Zoom in / out |
| I | Show or hide the details panel |
| L | Mark or unmark the image as a favorite; the order of stepping stays as it was when the view opened |
| M | Move the image to a folder: the Move to… dialog |
| Delete, Backspace | Move to the trash and go on to the next image (Undo in the status bar) |
| Ctrl+C | Copy the path |
| Esc | Back to the grid |

**Importing:** Import… and Import folder… open the desktop's file dialog, which gives real paths.

- Import… puts the images into the folder being shown; on a tag's page they get that tag.
- Import folder… makes the disk folder a rope folder of the same name at the top, whatever is being shown, with its subfolders as rope subfolders; hidden folders and folders without images are skipped. Importing the same folder again merges into it. The CLI does the same with `rope import --tree`.
- **There is no drag-and-drop from the file manager:** winit 0.30, which eframe uses, does not implement file drops on Wayland.

**Also:**

- **Live updates.** Changes made by the CLI or `img2img.py` show up at once, through the journal.
- **Clipboard.** Copying uses the native Wayland clipboard. This machine has no clipboard manager, so paste before closing the GUI.

## With img2img.py

`img2img.py` writes its results into rope; it does not run without it:

```bash
./img2img.py ~/Pictures/IMG_1234.HEIC "把背景換成黃昏的海邊" -n 4 --into trips/京都 --tag 夕陽 --tag 海邊 --note "first try"
#   new rope folder trips/京都
#   -> 3fa9c1e27b0d  trips/京都  1024x1024, 4.1s
./img2img.py @3fa9c1 "把天空換成星空" --into trips/京都     # edit a result further; its upstream is recorded
./img2img.py @3fa9c1 "$(rope prompt show 星空)"             # a prompt kept with rope prompt add, or on the GUI's Prompts page
./img2img.py --replay 3fa9c1                                # make it again and compare the pixels
```

On every run:

1. Before anything is uploaded, each input goes into rope, and from then on the run uses rope's copy. rope therefore holds exactly the bytes ComfyUI reads, even if the original is moved or rewritten while the job runs.
2. A HEIC, RAW or similar input is converted to a PNG in a temporary folder that is deleted when the run ends. rope keeps that PNG with the input image, in `uploaded/`.
3. Each result goes into rope with its recipe:
   - the flags, the prompt and the ordered inputs;
   - the workflow that was sent;
   - the rewrite of 2.1's prompt enhancer;
   - the ComfyUI version and commit, package versions, and the GPU;
   - the SHA-256 of every model file and LoRA, and the version of `img2img.py`. A model file is read once for its hash: the result is cached in `.cache/model-sha256.json` by file identity (device, inode, size, modification time), and the hashing runs alongside the GPU work. The first run reads the default 2511 set, about 31 GB.
4. With `--into`, unfiled inputs go into that folder too. The folder is made before the run if there is none.

| Option | Description |
|---|---|
| `--into FOLDER` | The rope folder for the results, made with its parents if there is none; the script says so, which is how a typo shows. Default: `$IMG2IMG_INTO`, else unfiled |
| `--tag T`, `--note T` | Tag the results or add a note. For several tags, repeat it: `--tag 夕陽 --tag 海邊` (a comma does not split) |
| `@ID` | An image in rope as an input |

When things go wrong:

- **rope is not built, or the library cannot be opened:** `img2img.py` stops before uploading or running anything.
- **rope cannot take a result** (a full disk, say): the GPU work is not lost. The PNG goes to `rescue/` in the library (or the system's temp folder when even that fails) with its `made.json` and `graph.json`, next to a `bring-in.sh` that hands it to rope once rope works again. `rope check` lists what is still waiting in `rescue/`.
- **Several runs at once:** each run records the images it uses in `runs/<pid>.json` and keeps that file locked while it lasts, so `rope purge` leaves those alone and deletes anything else. A file left by a crashed run is removed once nothing holds its lock.

## rope-queue

`rope-queue` is the work queue for ComfyUI jobs, a binary of its own in this workspace (`cargo build` makes it too). Started alone it is the daemon: it starts ComfyUI in place of `./run_comfyui.sh`, takes jobs from `img2img.py` and `img2vid.py`, runs them one at a time in order and puts every result into the library, with a screen that shows the queue. Ctrl-C interrupts the running job, which goes back to the head of the line, and stops ComfyUI with the daemon. With a subcommand it talks to the running daemon: `status`, `ls`, `log ID`, `cancel 123 g40`, `retry 123 g40` (plain numbers are jobs, `g40` is group 40), `pause`, `resume`, `stop`, `reload`, `clear`, each with `--json`; `ls`, `log` and `clear` read the files when no daemon runs, so jobs a stopped daemon left behind can be seen and cleaned up. Job and group ids count up across commands; a `clear` that leaves the queue empty starts them from 1 again.

`img2img.py` and `img2vid.py` hand every command to it: they check the request, print `queued #123–#172 (group 40) …` and return; what becomes of the jobs the screen, `ls` and `log` show, and so does rope-gui's Queue page, which can also cancel, retry, pause and clear. The daemon runs each job as `img2img.py --job DIR` and converts inputs with `comfyconvert.py`, both with the repo's `.venv` (setup_venv.sh), and finds the repo by `--repo` the first time, then by the library's `REPO` file. `--no-queue` runs a command's ComfyUI work in the command itself, as before the queue.

Its data lives in the library under `queue/`: one directory per job with its `job.json` and `log`, the id counters in `queue.json`, and ComfyUI's logs under `comfyui/`. `rope purge` leaves the inputs of unfinished jobs alone, as it leaves those of a running `img2img.py`.

The daemon takes `--library DIR`, `--repo DIR`, `--server HOST:PORT` (default 127.0.0.1:8188; it refuses when a ComfyUI already answers there), `--reserve-vram GB` (default 2, left to the desktop; 0 for ComfyUI's default), `--job-timeout MIN` (default 30: a job silent that long is killed), `--converter FILE`, `--no-screen` (one line per change instead of the screen, which is also what it does without a terminal), and after `--` anything for ComfyUI. The job file is `rope_core::queue::Job`, and what goes over the socket and what a job's child says on stdout are the types in `rope-core/src/queue/protocol.rs`.

## Reproducing an image

Copy repro and `rope repro ID` both give `img2img.py --replay ID`, or `img2vid.py --replay ID` for a clip: the tool that made the item, by its recipe. Each item's `repro.sh` runs the same command. It finds the library from its own location and the repo from the library's `REPO` file, so moving the inputs, the repo or the library does not break it.

`--replay` does this, in order:

1. It takes the bytes that were uploaded back then from rope, and sends the workflow recorded then. Changes to constants hard-coded in `img2img.py` (`shift=3.1`, the sampler, …) therefore do not change the result. Before sending, it lists how the workflow today's `img2img.py` would build differs; `--rebuild` sends that one instead.
2. It also lists what else differs from then: ComfyUI and package versions, and model files whose content is not what it was (by SHA-256, so a download of the same file again is not reported).
3. For 2.1 with `--enhance`, it reuses the prompt the enhancer wrote then and skips the enhancer, which samples and would most likely write a different prompt. `--rerun-enhancer` runs it again.
4. It uploads the inputs under new names. Otherwise a ComfyUI process that has seen the exact same workflow answers from its cache and recomputes nothing.
5. For an image it compares the decoded pixels, not the files; for a clip, whose encoder's output rope does not decode, the bytes:
   - Identical: recorded as `identical` in the original's replays.
   - Different: kept as a new image or clip next to the original; for an image with the largest difference and the share of differing pixels.

`-s N` changes only the seed. `--dry-run` prints the differences and the workflow without using the GPU.

**What rope guarantees is the same workflow on the same bytes.** Whether the pixels come out bit for bit the same also depends on the GPU and the software versions, and only a real run tells. 2.1 is the likelier to differ: its KV cache takes a different computation path depending on how much VRAM is free at the time.

## Deleting

1. **Delete moves an image to the trash.** The Delete button, the Delete or Backspace key and `rope rm` all do only that; undo it with Undo or `rope restore`.
2. **Emptying the trash deletes for good:** `rope purge --trash`, or Delete for good… on the GUI's trash page.
   - An image that other images were made from keeps only its file (and `uploaded/`). Its name, sources, recipe, note, folder and tags go, but the images made from it can still be replayed.
   - It goes for good when the last image made from it is purged.
   - `--break-lineage` deletes the bytes too; the images made from it can then no longer be replayed.
3. **While `img2img.py` runs, purge leaves that run's inputs alone** and says so; everything else is deleted as usual. Purge them again when the run is done.

rope deletes only what is in the library: never the originals, or ComfyUI's `input/`.

## Backups

The library holds the only copy of inputs whose originals are gone, so back it up. Image files never change once written, which keeps incremental backups cheap:

```bash
rsync -a --exclude .cache/ --exclude queue/ ~/.local/share/rope/ /backup/rope/
rope check --hash            # check every file still matches its hash
```

## Caveats

- **Do not edit files in `items/`.** Images are read-only, but their folders are writable, so an editor that saves by renaming can still replace one. Copy an image out before editing it. `rope check --hash` finds files that changed.
- **Only the text files are for hand editing.** When `meta.json`, `note.txt` or `catalog.json` is broken, rope reports it instead of overwriting it, and keeps the previous `catalog.json` as `catalog.json.bak`.
- **Every write takes the same lock,** so the CLI, the GUI and several `img2img.py` runs can work at once.
