# comfyrope

Edit images and make short clips with ComfyUI on this machine, from the command line:

- `img2img.py` — images + prompt → a new image (Qwen-Image-Edit-2511, or Qwen-Image-2.1)
- `img2vid.py` — one image + prompt → a short clip (Wan 2.2)
- `rope` / `rope-gui` — the library every input and result goes into, with the recipe that made it
- `rope-queue` — the queue that starts ComfyUI and runs the jobs one by one

Run every command from this folder. Steps 1 to 5 are the setup, once; 6 onwards is daily use.

Needs: an NVIDIA driver, `git`, `uv`, Rust (`cargo`), `ffmpeg`, and for the GUI libheif and the Noto Sans CJK fonts. All of them are on this machine.

## 1. Install ComfyUI

```bash
./install_comfyui.py
```

Clones ComfyUI into `ComfyUI/`, makes its venv (`ComfyUI/.venv`), and writes `run_comfyui.sh` and `ComfyUI/extra_model_paths.yaml`, which points ComfyUI at `models/` next to it. Safe to rerun: it skips what is done.

Later: `./install_comfyui.py --ref latest` upgrades ComfyUI. `--dry-run` shows the plan. `--help` for the rest.

## 2. Download the models

```bash
uv run download_models.py all     # the image models for img2img.py: 2511 and 2.1, about 48 GB
uv run download_models.py wan     # Wan 2.2 for img2vid.py, about 38 GB
```

Downloads from Hugging Face into `models/`. Ctrl-C and run the same command again to continue; files already there are skipped.

Optional, when you want them:

```bash
uv run download_models.py 2511 --lightning 8                       # the 8-step Lightning LoRA (img2img.py --lightning 8)
uv run download_models.py 2511 --skin-loras edit-skin skin-fix     # skin LoRAs; skin-fix needs $CIVITAI_TOKEN
uv run download_models.py 2.1 --prompt-enhancer i2i                # 2.1's prompt enhancer (img2img.py --enhance)
uv run download_models.py --list                                   # every file it knows, with sizes
uv run download_models.py all --dry-run                            # only the plan
```

## 3. Build rope

```bash
cargo build --release --manifest-path rope/Cargo.toml
```

Makes `rope`, `rope-gui` and `rope-queue` in `rope/target/release/`. To run them from anywhere:

```bash
cargo install --locked --path rope/rope
cargo install --locked --path rope/rope-gui
cargo install --locked --path rope/rope-queue
```

The library itself is made on first use, at `~/.local/share/rope`. Nothing to configure.

## 4. Build the Python environment

```bash
./setup_venv.sh
```

Makes `.venv` with the packages the scripts need. rope-queue runs every job with `.venv/bin/python`, so this step is required. Rerun it after pulling changes; to rebuild from scratch, `rm -rf .venv` first.

## 5. Start ComfyUI through rope-queue

```bash
rope-queue --repo .      # or rope/target/release/rope-queue --repo .
```

Keep it running in its own pane. It starts ComfyUI, shows the queue, runs the jobs in order and puts every result into rope. Ctrl-C stops the running job and ComfyUI with it. `--repo .` is needed the first time only; after that it remembers.

It refuses to start while another ComfyUI answers on port 8188: stop that one first. The GPU also drives the desktop, so the screen may stall while a model loads; 2 GB of VRAM are left to the desktop by default (`--reserve-vram`).

From another pane:

```bash
rope-queue status            # is it running, is ComfyUI up
rope-queue ls                # the jobs
rope-queue log 123           # what job 123 printed
rope-queue cancel 123        # a job;  cancel g40  a whole group
rope-queue retry 123         # run a failed or cancelled job again
rope-queue pause             # finish the running job, start none;  resume  goes on
rope-queue clear             # forget the finished jobs
rope-queue stop
```

## 6. Edit an image

```bash
./img2img.py photo.jpg "把背景換成黃昏的海邊"
./img2img.py person.png shirt.png "讓圖1的人穿上圖2的襯衫" -n 4      # several images: 圖1, 圖2 in order; 4 results
./img2img.py photo.jpg "turn this into a watercolor painting" -m 2.1
./img2img.py photo.jpg "..." -s 42 --into trips/京都 --tag 夕陽         # fixed seed, into a rope folder, tagged
./img2img.py @3fa9c1 "把天空換成星空"                                   # an image already in rope as the input
```

The command checks the request, prints `queued #123–#126 (group 40) …` and returns at once. rope-queue makes the images; watch them on its screen or in rope-gui (step 8). The first job after a start loads about 30 GB of models and takes a minute or two; the next ones start right away.

Images first, the prompt last. The first image sets the output size. Prompts can be Chinese or English and work best as instructions: "replace … with …", "remove …", "add …". PNG, JPEG and WebP go as they are; HEIC, AVIF, RAW, GIF, TIFF and the like are converted first.

| Option | Meaning |
|---|---|
| `-m 2511\|2.1` | The model (default 2511) |
| `-n N` | Make N images, seeds going up by one |
| `-s SEED` | The first seed (default: random). The same inputs, settings and seed give the same image again, from ComfyUI's cache; `--fresh` computes it anew |
| `--lightning 4\|8\|off` | 2511: the 4-step LoRA (default), the 8-step one, or the full model at 40 steps (much slower, better) |
| `--lora FILE[:STRENGTH]` | A LoRA from `models/loras`; repeatable |
| `--into FOLDER`, `--tag TAG`, `--note TEXT` | Where the results go in rope, their tags (repeat `--tag`), a note |
| `--dry-run` | Only print the workflow JSON |

`./img2img.py --help` lists them all (`--steps`, `--cfg`, `--negative`, `--resolution` and `--enhance` for 2.1, …).

A missing model file is reported with the download command, and the request is skipped. Results are named `<time>_<input>_<model>_s<seed>`.

## 7. Make a clip

```bash
./img2vid.py photo.jpg "the camera slowly pushes in while the leaves move in the wind"
./img2vid.py photo.jpg "..." -n 2 -s 42 --seconds 3 --size 480      # two clips of 3 s, long side 480 px
./img2vid.py photo.jpg "..." --lightning off                        # the full 20-step model, about five times slower
```

Default: 5 seconds at 16 fps, long side 640 px, the 4-step LoRAs; about 1.5 minutes per clip on the 4090. Describe the motion: what moves, how the camera moves. `--into`, `--tag`, `--note`, `-n`, `-s` work as for images. `./img2vid.py --help` for the rest.

## 8. Browse the results

```bash
rope-gui                 # or rope/target/release/rope-gui
```

Folders and tags on the left, thumbnails in the middle, the selected image on the right with its recipe. Double-click for the large view; clips play. The **Queue** button at the top right shows what rope-queue is doing, with cancel, retry, pause and Clear finished.

The same from the terminal: `rope ls trips/京都`, `rope show 3fa9c1`, `rope tree`, `rope mv`, `rope tag`, `rope rm`. `rope --help`. Everything about the library, the GUI, deleting and backups is in [rope/README.md](rope/README.md).

## 9. Make an image again

```bash
./img2img.py --replay 3fa9c1          # the recorded workflow on the recorded inputs; compares the pixels
./img2img.py --replay 3fa9c1 -s 7     # the same recipe with another seed
```

rope-gui's Copy repro button and `rope repro 3fa9c1` give this command. `img2vid.py --replay` does the same for a clip.

## 10. Many settings in one go

Copy the template, fill in its three numbered parts (the base request, the settings to vary, the combinations), and run it:

```bash
cp run_img2img.py my_run.py
.venv/bin/python my_run.py --list    # check every request, queue nothing
.venv/bin/python my_run.py           # queue them; each combination into its own rope folder
```

`run_img2vid.py` is the same for clips. In Python directly:

```python
import dataclasses, img2img

base = img2img.Request(["photo.jpg"], "make the skin look natural", seed=42, into="tests/skin")
requests = [base, dataclasses.replace(base, lora=[("qwen-edit-skin-1.1.safetensors", 1.0)], tags=["skin"])]
for s in img2img.run(requests):
    print(s.group, s.jobs, s.skipped)
```

`run()` returns as soon as the jobs are queued. A request with a typo raises before anything is sent; one whose model file is missing is printed with the fix and skipped.

## Without the queue

```bash
./run_comfyui.sh                          # ComfyUI with its UI at http://127.0.0.1:8188
./img2img.py photo.jpg "..." --no-queue   # do the work in this command, not in rope-queue
```

Only one ComfyUI runs at a time, so stop rope-queue before `run_comfyui.sh`. `--no-queue` uses whichever ComfyUI is running (rope-queue's or this one), or starts one itself and stops it at the end. In the UI, Workflow → Browse Templates has the Qwen and Wan templates; they use the same model files.
