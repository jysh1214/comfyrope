# rope: an image library that remembers where each image came from and how it was made

rope keeps images by the hash of their content. Each image carries its own metadata:

- **sources**: the paths its file was seen at; there can be several.
- **recipe**: for an image a tool made, the command, the workflow sent to ComfyUI, and the input images, in order.
- **note**: free text.

Folders and tags exist only in rope; files on disk are never moved. `img2img.py` puts its inputs and results into rope on every run, so a result can be made again after its inputs are moved or deleted.

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
├── journal.jsonl        # one line per change, only ever appended; the GUI follows it
├── REPO                 # where the img2img repo was last seen, for repro.sh
├── envs/<id>.json       # environment snapshots, shared by every image made in the same environment
├── items/3f/<sha256>/   # one folder per image, named after the SHA-256 of its file
│   ├── 3fa9c1e27b0d.png # the image: 12 hex digits of the hash plus its suffix; read-only, never rewritten
│   ├── meta.json        # name, sources, recipe, upstream images
│   ├── graph.json       # the API workflow sent to ComfyUI
│   ├── repro.sh         # generated from meta.json; no absolute paths
│   ├── note.txt
│   └── uploaded/        # for HEIC, RAW and the like: the PNG actually uploaded in its place
├── tmp/                 # items being assembled, renamed into items/ when complete
├── rescue/              # results img2img.py could not hand to rope, waiting for rope add
└── .cache/thumbs/       # GUI thumbnails; safe to delete
```

- `meta.json`, `catalog.json` and `note.txt` are plain text: grep them, or edit them by hand.
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
| `rope import PATH... [--into F] [--tag T]... [--note T]` | Import files, or every image in a folder recursively, all into F, made if there is none; `--tag` repeats |
| `rope import --tree DIR... [--into F]` | Make each disk folder a rope folder of the same name, with its subfolders, in F or at the top |
| `rope ls [F] [-r] [-l] [--tag T] [--fav] [--trash]` | List a folder; without F, the unfiled images. Favorites are marked ★ |
| `rope tree` | The folder tree with image counts, and every tag |
| `rope show ID` | Everything about an image: path, sources, recipe, upstream and downstream, note |
| `rope path ID` | The absolute path of the image's file in the library |
| `rope name ID TEXT` | Change the display name; the file is not touched |
| `rope note ID [--set T \| --append T \| -e]` | Print or change the note; `-e` edits it in `$EDITOR` |
| `rope tag add\|rm T ID...`, `rope tag rename OLD NEW`, `rope tags` | Manage tags |
| `rope fav ID... [--off]` | Mark images as favorites, or unmark them; rope-gui shows favorites first |
| `rope lineage ID [--up \| --down]` | The tree of images above or below |
| `rope repro ID [--command]` | Print the command that makes the image again; `--command` prints the full `img2img.py` command, to change and run as a variation |
| `rope rm ID...`, `rope restore ID...` | Move images to the trash, or back out |
| `rope purge ID... \| --trash [--break-lineage]` | Delete for good; see [Deleting](#deleting) |
| `rope check [--hash] [--fix]` | Check the library; `--hash` verifies every file against its hash |
| `rope log [ID]` | The latest changes |

Every command takes `--json` for scripts.

## GUI

```bash
rope-gui                 # or: rope-gui 3fa9c1, to open on that image
rope-gui --foreground    # stay attached to the terminal, e.g. to watch its messages
```

The window runs on its own: the command returns at once, and closing the terminal or pressing Ctrl-C there leaves the window open. What it prints goes to `.cache/rope-gui.log` in the library. A library that cannot be opened is still reported in the terminal.

**Left:**

- All, Unfiled, and rope's own folder tree, with image counts.
- Right-click a folder to copy its path (as commands take it: `trips/京都`, or `'trips/day 1'` when it has a space), make a subfolder, rename it, or delete it with everything in it (its images move to the trash; it says how many first); `+` makes a top-level folder. A new folder's name can be a path: `a/b` makes `b` inside `a`, and `a` too if it is missing.
- Below the tree: the tags, then the trash. Right-click the trash and choose Clean to empty it; it asks first, since that deletes for good.

**Middle:**

- Thumbnails of the folder chosen on the left, without the images in its subfolders: favorites first, then the newest first.
- A star in a thumbnail's top-right corner marks it as a favorite: gold when it is one, an outline on hover otherwise. Click it, or press L.
- Click to select, Ctrl-click to add, Shift-click for a range. Double-click, Enter or Space opens the large view (below).
- Right-click to copy the path or the repro command, view large, open in the default image viewer, move to a folder, add a tag, add to or remove from the favorites, or trash.

**Right**, the selected image:

- A preview (click it for the large view), the favorite star, and the name, which you can edit.
- Copy path, Copy repro, Copy command and Delete buttons.
- Its folder, tags and note.
- Made from: its upstream images, in the order of the prompt's 圖1, 圖2. Made from this: its downstream images. Click a thumbnail to go to that image.
- The recipe, replays, and sources; a source that no longer exists is marked gone.

**Keys**, when no text field has focus (Backspace works wherever Delete does: an Apple keyboard's delete key sends it):

| Key | Action |
|---|---|
| Arrow keys | Move the selection |
| Enter, Space | View the selected image large |
| L | Mark the selection as favorites, or unmark it when all of it is |
| Ctrl+A | Select all |
| Delete, Backspace | Move to the trash (in the trash: delete for good) |
| Ctrl+C | Copy the path |
| Esc | Clear the selection |
| F5 | Reload |

**The large view** shows one image as large as the window allows, on a dark background, with a bar for stepping, zooming, the details panel and the default image viewer. It decodes images up to 4096 pixels on a side, and keeps only a few of them on the GPU, which ComfyUI needs more.

| Key or mouse | Action |
|---|---|
| ← → (or ↑ ↓, Space) | Previous / next image of the view. Images that arrive while looking (an `img2img.py` run finishing, an import) join in, where the view puts them |
| Ctrl+wheel | Zoom around the cursor |
| Drag | Pan, when zoomed |
| Double-click, 1 | 100%: one image pixel per screen pixel |
| F | Fit to the window |
| + − | Zoom in / out |
| I | Show or hide the details panel |
| L | Mark or unmark the image as a favorite; the order of stepping stays as it was when the view opened |
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
- **Several runs at once:** each run holds `run.lock` while it lasts, so `rope purge` cannot delete an input that a running job uses.

## Reproducing an image

Copy repro and `rope repro ID` both give `img2img.py --replay ID`. Each image's `repro.sh` runs the same command. It finds the library from its own location and img2img from the library's `REPO` file, so moving the inputs, the repo or the library does not break it.

`--replay` does this, in order:

1. It takes the bytes that were uploaded back then from rope, and sends the workflow recorded then. Changes to constants hard-coded in `img2img.py` (`shift=3.1`, the sampler, …) therefore do not change the result. Before sending, it lists how the workflow today's `img2img.py` would build differs; `--rebuild` sends that one instead.
2. It also lists what else differs from then: ComfyUI and package versions, and model files whose content is not what it was (by SHA-256, so a download of the same file again is not reported).
3. For 2.1 with `--enhance`, it reuses the prompt the enhancer wrote then and skips the enhancer, which samples and would most likely write a different prompt. `--rerun-enhancer` runs it again.
4. It uploads the inputs under new names. Otherwise a ComfyUI process that has seen the exact same workflow answers from its cache and recomputes nothing.
5. It compares the decoded pixels, not the files:
   - Identical: recorded as `identical` in the original's replays.
   - Different: kept as a new image next to the original, with the largest difference and the share of differing pixels.

`-s N` changes only the seed. `--dry-run` prints the differences and the workflow without using the GPU.

**What rope guarantees is the same workflow on the same bytes.** Whether the pixels come out bit for bit the same also depends on the GPU and the software versions, and only a real run tells. 2.1 is the likelier to differ: its KV cache takes a different computation path depending on how much VRAM is free at the time.

## Deleting

1. **Delete moves an image to the trash.** The Delete button, the Delete or Backspace key and `rope rm` all do only that; undo it with Undo or `rope restore`.
2. **Emptying the trash deletes for good:** `rope purge --trash`, or Delete for good… on the GUI's trash page.
   - An image that other images were made from keeps only its file (and `uploaded/`). Its name, sources, recipe, note, folder and tags go, but the images made from it can still be replayed.
   - It goes for good when the last image made from it is purged.
   - `--break-lineage` deletes the bytes too; the images made from it can then no longer be replayed.
3. **Purge refuses while `img2img.py` runs**, so it cannot delete an input in use.

rope deletes only what is in the library: never the originals, or ComfyUI's `input/`.

## Backups

The library holds the only copy of inputs whose originals are gone, so back it up. Image files never change once written, which keeps incremental backups cheap:

```bash
rsync -a --exclude .cache/ ~/.local/share/rope/ /backup/rope/
rope check --hash            # check every file still matches its hash
```

## Caveats

- **Do not edit files in `items/`.** Images are read-only, but their folders are writable, so an editor that saves by renaming can still replace one. Copy an image out before editing it. `rope check --hash` finds files that changed.
- **Only the text files are for hand editing.** When `meta.json`, `note.txt` or `catalog.json` is broken, rope reports it instead of overwriting it, and keeps the previous `catalog.json` as `catalog.json.bak`.
- **Every write takes the same lock,** so the CLI, the GUI and several `img2img.py` runs can work at once.
