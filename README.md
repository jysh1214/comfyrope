# img2img: local image-to-image with Qwen-Image

Runs Qwen-Image-Edit-2511 and Qwen-Image-2.1 through ComfyUI on this machine (RTX 4090, 24 GB).

| Tool | What it does |
|---|---|
| `setup_venv.sh` | Builds `.venv`, the Python environment for the scripts below |
| `install_comfyui.py` | Installs or updates ComfyUI with its own venv |
| `download_models.py` | Downloads the model files from Hugging Face: the image models, optional skin LoRAs, and Wan 2.2 for video |
| `run_comfyui.sh` | Starts ComfyUI |
| `img2img.py` | Makes a new image from input images and a prompt, through ComfyUI's API. Also a Python module, for running many combinations in one go |
| `rope/` | An image library (CLI and GUI, in Rust). It keeps every input and result of `img2img.py` with its recipe and lineage, so a result can be made again after its inputs are moved or lost. See [rope/README.md](rope/README.md) |

Run every command from the repo root. Each script finds `ComfyUI/`, `models/` and `.venv/` next to itself, so the repo can live anywhere and be moved without reinstalling.

## Quick start

```bash
./install_comfyui.py                                    # ComfyUI, with its own venv
uv run download_models.py all                           # the model files
cargo build --release --manifest-path rope/Cargo.toml   # the rope image library
./run_comfyui.sh                                        # in another pane: keeps the models loaded between runs
./img2img.py photo.jpg "把背景換成黃昏的海邊"             # "turn the background into a beach at dusk"
rope/target/release/rope-gui                            # browse the results
```

## Layout

```
.
├── setup_venv.sh        # builds .venv with the packages img2img.py and download_models.py need
├── .venv/               # Python 3.12 venv for the scripts, separate from ComfyUI's
├── install_comfyui.py   # installs or updates ComfyUI (Python standard library only)
├── download_models.py   # downloads models from Hugging Face (run with uv, nothing to install first)
├── run_comfyui.sh       # starts ComfyUI (written by install_comfyui.py)
├── img2img.py           # images + prompt -> new image, through ComfyUI's HTTP API; import it from Python for batches
├── comfycore.py         # what img2img.py is built on: the ComfyUI client, rope, the inputs, running jobs; knows no model
├── run.py               # template for a batch of settings to compare, on img2img.py's Python API
├── rope/                # Rust source of the image library (rope, rope-gui); its data is in ~/.local/share/rope
├── models/              # the downloaded models; ComfyUI reads them through extra_model_paths.yaml
│   ├── diffusion_models/
│   ├── text_encoders/
│   ├── vae/
│   └── loras/
└── ComfyUI/             # git clone, tag v0.39.2
    ├── .venv/           # Python 3.12 venv made by uv (torch 2.14.1+cu130, with pip)
    └── extra_model_paths.yaml
```

`models/` sits outside ComfyUI, so reinstalling or upgrading ComfyUI leaves the models alone.

## Python environment: setup_venv.sh

The packages `img2img.py` and `download_models.py` need are listed in the `# /// script` block at the top of each file. There are two ways to run them; either works:

| Way | Command | Preparation |
|---|---|---|
| Let uv handle it | `./img2img.py …` | None. uv reads the package list at the top of the file and builds and caches an environment for it |
| Use the project venv | `source .venv/bin/activate`, then `python img2img.py …` | Run `./setup_venv.sh` once |

```bash
./setup_venv.sh                 # builds .venv and installs the packages; a rerun only adds what is missing
source .venv/bin/activate
python img2img.py photo.jpg "把天空換成星空"
```

`setup_venv.sh` uses uv to make a Python 3.12 `.venv`, installs the package lists from the two scripts, and checks each package imports. When a script gains a package, run it again; to rebuild from scratch, delete `.venv` first. A script of your own that does `import img2img` (see [From Python](#from-python)) needs the same packages, so run it with `.venv/bin/python` too.

`python img2img.py` without activating `.venv` runs the system Python or ComfyUI's venv. Neither has these packages, so the script lists what is missing and how to run it instead. ComfyUI's own venv is `ComfyUI/.venv`, managed by `install_comfyui.py` and separate from `.venv`.

## Installing and updating ComfyUI: install_comfyui.py

Uses only the Python standard library and needs `git` and `uv`. Every step checks the current state first and skips what is already done, so a rerun is safe:

```bash
./install_comfyui.py --dry-run       # print the plan and the commands, change nothing
./install_comfyui.py                 # fresh install; on an existing one, check everything again
./install_comfyui.py --ref latest    # upgrade ComfyUI to the latest release
./install_comfyui.py --ref v0.39.2   # pin a release (--ref master follows the development branch)
./install_comfyui.py --manager       # also install the bundled ComfyUI-Manager
```

### What it does

| Step | What happens |
|---|---|
| driver | Reads the CUDA version the driver supports from `nvidia-smi`, and stops if it is older than `--cuda` needs |
| comfyui | Clones the chosen tag of Comfy-Org/ComfyUI; an existing clone changes version only when `--ref` is given |
| venv | Makes a Python 3.12 venv in `ComfyUI/.venv` with uv |
| packages | `uv pip install --torch-backend cu130 -r requirements.txt pip`: torch from PyTorch's cu130 index, the rest from PyPI |
| config | Writes `ComfyUI/extra_model_paths.yaml` pointing at `models/` through the relative path `../models/`, so moving the repo needs no change |
| launcher | Writes `run_comfyui.sh` |
| self-test | Runs `main.py --cpu --quick-test-for-ci` with the GPU hidden (`CUDA_VISIBLE_DEVICES=`) to check that ComfyUI starts and finds all four model folders. Its temp and user folders are throwaway, so it is safe to run while ComfyUI is running |

### Options

| Option | Description |
|---|---|
| `--dir DIR` | Where ComfyUI is; default `./ComfyUI` |
| `--ref REF` | `latest`, a tag or a branch. Without it, a new install takes the latest release and an existing one stays as it is |
| `--python VER` | The venv's Python, e.g. `3.13`. A different one from the current venv rebuilds it |
| `--cuda BUILD` | PyTorch build: `cu130` (default), `cu129`, `cu128`, `cu126`, `cpu`. A different one from the current torch rebuilds the venv |
| `--torch VER` | Pin torch, e.g. `2.14.1`; `latest` upgrades to the newest on that index |
| `--models DIR` | The model root written into `extra_model_paths.yaml`; default `./models` |
| `--no-model-paths` | Leave `extra_model_paths.yaml` alone |
| `--manager` | Install `manager_requirements.txt`; the launcher adds `--enable-manager` |
| `--rewrite-config` | Overwrite an edited `extra_model_paths.yaml` and `run_comfyui.sh`, keeping the old file as `.bak` |
| `--no-test` | Skip the self-test |
| `--gpu-test` | Finish with an fp16 matmul on the GPU to check CUDA really works. **Uses the GPU** |
| `--dry-run` | Only print the plan; when neither the ComfyUI version nor the venv changes, also show uv's preview of package changes |

### Behavior

- **Options not given keep the current state; options given are enforced.** With a Python 3.12 venv, leaving out `--python` keeps it and `--python 3.13` rebuilds it. Rebuilding is quick, since large packages such as torch are hardlinked from uv's cache.
- **Your changes are not overwritten.** It refuses to switch versions when tracked ComfyUI files are modified; `git stash` them first. An `extra_model_paths.yaml` or `run_comfyui.sh` that differs from the generated one is kept unless `--rewrite-config` is given.
- **A running ComfyUI is not touched.** Before switching versions or rebuilding the venv, it looks for a `main.py` started from that folder and asks you to stop it first.
- **The GPU is not touched by default.** The driver check only reads `nvidia-smi` output, and the self-test runs with the GPU hidden. Only `--gpu-test` makes a CUDA context.
- **The venv has pip.** uv venvs come without pip, but many custom-node installers call `python -m pip`.
- **torchaudio is left out on purpose.** ComfyUI does not import it. Without it, Manager prints `[ERROR] [ComfyUI-Manager] PyTorch is not installed` at startup; that is a false alarm and can be ignored.
  Do not install torchaudio to silence it: Manager's torch version table stops at 2.7.0, and with all three packages present it installs a **nightly** torch whenever a custom node changes the torch version.
- The self-test's full output is in `ComfyUI/install-selftest.log`, which ComfyUI's `.gitignore` ignores.

## Downloading models: download_models.py

Run it directly; the first run makes an environment with just huggingface_hub (PEP 723 inline metadata):

```bash
uv run download_models.py --list                    # every known file and its size
uv run download_models.py all --dry-run             # only the plan (real sizes, free space); writes nothing
uv run download_models.py all                       # the default set for 2.1 and 2511
uv run download_models.py 2511                      # only Edit-2511
uv run download_models.py 2.1 --prompt-enhancer i2i # 2.1 plus the i2i prompt enhancer
uv run download_models.py all --limit-rate 5M       # limit the speed to 5 MiB/s
uv run download_models.py 2511 --skin-loras edit-skin skin-fix   # the skin LoRAs; skin-fix needs $CIVITAI_TOKEN
uv run download_models.py wan                       # Wan 2.2 image-to-video: the 14B pair in fp8, with the 4-step LoRAs
```

MODEL is `2.1`, `2511`, `wan` or `all` (aliases: `qwen-image-2.1`, `qwen-image-edit-2511`, `edit`, `wan2.2`). `all` is the two image models, the ones `img2img.py` runs; `wan` is asked for by name.

### The default set

| MODEL | File | Size |
|---|---|---|
| 2.1 | `diffusion_models/qwen_image_2.1_int8_convrot` | 7.3 GB |
| 2.1 | `text_encoders/qwen3vl_8b_int8_convrot` | 9.4 GB |
| 2.1 | `vae/qwen_image_2.1_vae_bf16` | 0.7 GB |
| 2511 | `diffusion_models/qwen_image_edit_2511_fp8mixed` | 20.5 GB |
| 2511 | `text_encoders/qwen_2.5_vl_7b_fp8_scaled` | 9.4 GB |
| 2511 | `vae/qwen_image_vae` | 0.25 GB |
| 2511 | `loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16` | 0.85 GB |

### Skin LoRAs

Not in the default set; `--skin-loras NAME…` adds them for 2511. How to use them is under [Skin LoRAs](#skin-loras-1) in the `img2img.py` section.

| NAME | File | Size | From |
|---|---|---|---|
| `edit-skin` | `loras/qwen-edit-skin-1.1.safetensors` | 0.3 GB | Hugging Face, [tlennon-ie/qwen-edit-skin](https://huggingface.co/tlennon-ie/qwen-edit-skin): the 2750-step file of version 1.1, the one Civitai serves as v1.1, saved under a shorter name |
| `skin-fix` | `loras/Skin_Fix_rank64.safetensors` | 0.6 GB | Civitai, [Skin Fix Qwen v2.0](https://civitai.com/models/2031296/skin-fix-qwen). Civitai refuses downloads without an API key: make one at https://civitai.com/user/account (API Keys, at the bottom) and pass `--civitai-token` or set `$CIVITAI_TOKEN`. The file is checked against its SHA-256 |

### Wan 2.2

Alibaba's open video models (Apache 2.0), as repacked by Comfy-Org for ComfyUI's own Wan 2.2 templates ([ComfyUI's Wan 2.2 guide](https://docs.comfy.org/tutorials/video/wan/wan2_2)). `img2img.py` does not run them yet; the files land in the same `models/` layout, so ComfyUI finds them. `--wan-task` picks the models, several at once if wanted:

| TASK | Files | Size |
|---|---|---|
| `i2v` (default) | `diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled` + `..._low_noise_...`, `vae/wan_2.1_vae`, `loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise` + `..._low_noise` | 14.3 + 14.3 + 0.25 + 1.2 + 1.2 GB |
| `t2v` | the same for `t2v` (LoRAs `v1.1`) | 31.3 GB |
| `ti2v` | `diffusion_models/wan2.2_ti2v_5B_fp16`, `vae/wan2.2_vae` | 10 + 1.4 GB |
| all | `text_encoders/umt5_xxl_fp8_e4m3fn_scaled` | 6.7 GB |

The 14B pair is a high-noise and a low-noise model that the workflow runs one after the other; `--wan-precision fp16` doubles them to 28.6 GB each. The 4-step LoRAs are what the Lightning LoRAs are for 2511; `--wan-lightning none` leaves them out. The 5B model is one file and runs within 8 GB of VRAM; it has no 4-step LoRA in the repack.

### Options

| Option | Description |
|---|---|
| `--dest DIR` | Model root; default `./models`, next to the script |
| `--only KIND…` | Only these kinds: `diffusion` `text_encoder` `vae` `lora` |
| `--limit-rate RATE` | Speed limit: `500k` / `5M` / `1.5G`, in powers of 1024. Switches to the plain-HTTP path |
| `--dry-run` | Only print the plan |
| `--list` | List every known file |
| `--force` | Skip the free disk space check |
| `--ipv6` | Allow IPv6. By default only IPv4 is used: huggingface.co has IPv6 addresses, and when local IPv6 is configured but broken, every request waits for the IPv6 attempt to time out before falling back |
| `--token` | Hugging Face token, not needed for public repos; defaults to `$HF_TOKEN` |
| `--civitai-token KEY` | Civitai API key, needed for the files that come from Civitai; defaults to `$CIVITAI_TOKEN` |
| `--q21-precision int8\|bf16` | Precision of the 2.1 DiT and Qwen3-VL encoder; bf16 is 14.2 + 17.5 GB |
| `--q21-variant base\|turbo` | 2.1 base (the template's default) or the distilled turbo model |
| `--turbo-lora` | Also fetch the turbo LoRA for 2.1 base (0.9 GB) |
| `--prompt-enhancer i2i t2i` | 2.1's Qwen3.5-9B prompt enhancers, 9.5 GB each |
| `--edit-precision fp8\|int8\|bf16` | The 2511 model: fp8mixed (default) / int8_convrot / bf16 (40.9 GB, needs offloading) |
| `--skin-loras NAME…` | Skin LoRAs for 2511: `edit-skin`, `skin-fix` (see above) |
| `--lightning none\|4\|8\|both` | 2511's Lightning LoRAs; by default only the 4-step one |
| `--wan-task TASK…` | Wan 2.2 models: `i2v` (default), `t2v`, `ti2v` (see above) |
| `--wan-precision fp8\|fp16` | The Wan 14B pair: fp8_scaled (default, 14.3 GB each) or fp16 (28.6 GB each) |
| `--wan-lightning none\|4` | The lightx2v 4-step LoRAs for the Wan 14B pair; fetched by default |

### Behavior

- **Interruptible and resumable.** After Ctrl-C or a dropped connection, rerun the same command to continue. Files of the right size are skipped and shown as `present`.
- **Two download paths.** Without `--limit-rate` it uses huggingface_hub's multi-connection hf_xet path, the fastest. With `--limit-rate`, or after hf_xet fails twice in a row, it switches to its own plain-HTTP path: it writes `<file>.part`, resumes with a `Range` header, and retries network errors up to 8 times (backing off up to 60 seconds).
- **Size check.** It asks the Hugging Face API for the real size before downloading and compares afterwards; on a mismatch it keeps the `.part` file and asks for a rerun.
- **Existing files are judged by size only**, not by hash. To fetch a file again, delete it and rerun.
- Progress lines are flushed one by one, so `nohup uv run download_models.py all > dl.log 2>&1 &` with `tail -f dl.log` works.

### Adding files

The file lists are `plan_qwen_image_21()`, `plan_edit_2511()` and `plan_wan_22()` in the script, one `F(repo, path, kind, approx_gb)` per file. `kind` picks the subfolder (`diffusion` → `diffusion_models/`, and so on); `name=` saves the file under another name. A file that is not on Hugging Face takes `url=`, `size=` and `sha256=` instead, as the Civitai entry in `SKIN_LORAS` does; it is fetched over plain HTTP and checked against the hash. Check with `--list` afterwards.

## Starting ComfyUI: run_comfyui.sh

```bash
./run_comfyui.sh                    # http://127.0.0.1:8188
./run_comfyui.sh --reserve-vram 2   # leave 2 GB more VRAM to niri when the desktop stutters
./run_comfyui.sh --listen           # let other machines on the LAN connect
```

`install_comfyui.py` writes this file. It only `cd`s into `ComfyUI/` and starts `main.py` with `.venv/bin/python`, adding `--enable-manager` when Manager is installed. Other arguments go to `main.py` unchanged; `./run_comfyui.sh --help` lists them.

### Notes

- **The GPU also drives the desktop.** The screen may stall during the tens of seconds a model loads; if the whole desktop goes black, try `--reserve-vram 2`.
- **VRAM needs no tuning.** ComfyUI 0.39 turns on dynamic VRAM (comfy-aimdo) by default, so 2511's 20.5 GB model and 9.4 GB text encoder are offloaded to RAM as needed. Do not add `--lowvram`.
- **The first run of a workflow reads about 30 GB from NVMe into RAM**, so a minute or two of waiting is normal; later runs of the same workflow with another prompt do not reload.
- The model folders come from `ComfyUI/extra_model_paths.yaml`. Its `is_default: true` puts the repo's `models/` first, and makes it where the frontend saves downloaded models.

### Templates

In the UI, open Workflow → Browse Templates on the left and search for `Qwen`:

| Template | Files it needs |
|---|---|
| Qwen Image Edit 2511 - Material Replacement | The four default 2511 files; runs as it is |
| Qwen Image Edit 2511 Int8: Image Edit | First `download_models.py 2511 --edit-precision int8` |
| Qwen Image 2.1: Image Edit | The three 2.1 files **plus** the `--prompt-enhancer i2i` file |
| Qwen Image 2.1: Text to Image | The three 2.1 files plus `--prompt-enhancer t2i` |

The 2.1 templates load the prompt enhancer in a CLIPLoader inside a subgraph. ComfyUI checks the file names of every connected node before running, so even with `refine_prompt` off, a missing enhancer file fails with `Value not in list`.

## Editing images: img2img.py

Give it images and a prompt and it makes a new image, with no UI. It flattens ComfyUI's built-in templates into API workflows and runs them through ComfyUI's HTTP API:

```bash
./img2img.py photo.jpg "把背景換成黃昏的海邊"
./img2img.py person.png shirt.png "讓圖1的人穿上圖2的襯衫" -n 4     # 4 images, one seed each
./img2img.py photo.jpg "..." --lightning 8                            # the 8-step LoRA, 8 steps; download it first
./img2img.py photo.jpg "..." --lightning off                          # the full model: 40 steps, CFG 4, much slower
./img2img.py photo.jpg "..." --lightning off --steps 20               # the full model at 20 steps, the template's value
./img2img.py photo.jpg "turn this into a watercolor painting" -m 2.1
./img2img.py photo.jpg "..." --lora my_style.safetensors:0.8 -s 42
./img2img.py photo.jpg "make the subjects skin details more prominent and natural" --lora qwen-edit-skin-1.1.safetensors   # real skin texture
./img2img.py photo.jpg "..." --lightning 8 --lora Skin_Fix_rank64.safetensors:0.8   # less plastic skin with the Lightning LoRA
./img2img.py photo.jpg "..." --dry-run                                # only print the API workflow JSON
```

With `.venv` activated, `python img2img.py` works the same as `./img2img.py`. To try many settings in one go, [import it from Python](#from-python).

`img2img.py` holds only what is about its models: the requests, the workflows and the replay. The ComfyUI client, rope, the input conversion and the running of jobs are `comfycore.py` beside it, which knows no model, so another tool can be built on it the same way. `img2img.py` needs it next to itself.

Images come first and the prompt last; the first image sets the output size. With several images, refer to them in the prompt as 圖1, 圖2 or image 1, image 2, in command-line order. Prompts can be Chinese or English and work best as instructions that describe only the change: "replace … with …", "remove …", "add …".

### How it works

1. Each input is kept in rope first (see below), then checked: single-frame PNG, JPEG and WebP files are used as they are; anything else is converted to PNG here.
2. It checks for a ComfyUI at `--server` (default http://127.0.0.1:8188). If there is none, it starts one with `run_comfyui.sh` and stops it when the command ends.
3. It uploads the inputs to ComfyUI's `input/img2img/`, named after a hash of their content, so a rerun with the same image reuses the same upload.
4. It builds the workflow for `-m`, queues it, and follows loading, encoding and sampling over the websocket.
5. Each result goes into rope with its recipe.

### Where results go

Results go into the rope library (`~/.local/share/rope`), so build rope first; without it, `img2img.py` stops before doing anything. Organize the results with rope folders and tags, and browse them with `rope-gui`:

```bash
./img2img.py photo.jpg "把背景換成黃昏的海邊" -n 4 --into trips/京都 --tag 夕陽 --tag 海邊   # the folder is made if there is none
./img2img.py @3fa9c1 "把天空換成星空" --into trips/京都   # an image in rope as the input; its lineage is recorded
```

Every input goes into rope too: the original file, and for HEIC, RAW and the like, the PNG actually uploaded. rope also records the workflow that was sent, the 2.1 prompt enhancer's rewrite, the ComfyUI and package versions, and the SHA-256 of each model file. So a result can be made again after its inputs or the repo move, or `img2img.py` changes. [rope/README.md](rope/README.md) covers the library, the GUI and how to build or install rope.

Results are named `<start time>_<first input's name>_<model>_s<seed>`, for example `20261013-225301_photo_2511_s3418827001`:

- The time comes first, so names sort by time. The images of one `-n N` run share its start time and differ only in the seed.
- Spaces and symbols in the input's name become `_`; Chinese stays; anything past 40 characters is cut.

If rope cannot take a result (a full disk, say), the GPU work is not lost: the PNG, its recipe and its workflow are saved in `rescue/` in the library, next to a `bring-in.sh` that hands them to rope once it works again. `rope check` lists what is still waiting there.

### Making an image again

Replay an image's record:

```bash
./img2img.py --replay 3fa9c1          # make it again and compare the pixels
./img2img.py --replay 3fa9c1 -s 7     # the same recipe with seed 7
./img2img.py --replay 3fa9c1 --dry-run
```

`rope-gui`'s Copy repro button, `rope repro ID` and the `repro.sh` that rope keeps with each image all run this same command. Before sending, the replay says what differs from when the image was made. It sends the recorded workflow on the recorded input bytes, then compares the pixels: an identical result is only noted, and a different one is kept next to the original. [rope/README.md](rope/README.md#reproducing-an-image) has the details.

### Input formats

| Input | Handling |
|---|---|
| Single-frame PNG, JPEG, WebP | Uploaded as they are |
| HEIC/HEIF, AVIF, JPEG XL, BMP, TIFF, PSD and the other formats Pillow reads | Converted to PNG first |
| Camera RAW, e.g. DNG, CR2, CR3, NEF, ARW, RAF, ORF, RW2 | Developed with the camera's white balance, then converted to PNG |
| Animations and multi-page files, e.g. GIF, animated WebP, APNG | The first frame, converted to PNG |
| SVG and other vector formats | Not readable; export to PNG first |

Conversion applies the EXIF orientation and turns images with an ICC profile into sRGB, e.g. an iPhone's Display P3. The original file is never modified.

### Model defaults

| | 2511 (default) | 2.1 |
|---|---|---|
| Model | Qwen-Image-Edit-2511 fp8mixed | Qwen-Image-2.1 int8 |
| Input images | Up to 3 | Up to 10 |
| Steps and CFG | With the Lightning 4-step LoRA: 4 steps, CFG 1 | 25 steps, CFG 1 |
| Output size | The first image scaled to about 1 MP | The first image scaled to about the area of 1024×1024 |
| Its own options | `--lightning 4\|8\|off` | `--resolution N`, `--enhance` |

### Skin LoRAs

Qwen-Image-Edit with the Lightning LoRA tends to give smooth, plastic skin. Two LoRAs help; `uv run download_models.py 2511 --skin-loras edit-skin skin-fix` fetches them (see [Skin LoRAs](#skin-loras) under `download_models.py`), and `--lora` applies them, together or alone:

| LoRA | What it does | How to use it |
|---|---|---|
| `qwen-edit-skin-1.1.safetensors` | Adds skin detail back: pores, texture, small imperfections. Trained on smoothed images as input and the originals as target, on Qwen-Image-Edit-2509; reported to work on 2511 | Strength 1 to 1.5 (its author says 2 is too much). Put its trigger in the prompt: `make the subjects skin details more prominent and natural`. At full strength it can leave blotches on plain backgrounds |
| `Skin_Fix_rank64.safetensors` | Takes the plastic look off skin made with the Lightning LoRA. Its author says it does not make skin realistic, only less plastic | Strength 0.7 to 1.2, with the 8-step Lightning LoRA (`--lightning 8`); its author uses Lightning at 0.75. No trigger word. Trained for Qwen-Image, so results on Edit-2511 vary |

To compare, run the same image and seed with and without, into one rope folder, and step through them in the large view:

```bash
./img2img.py photo.jpg "make the subjects skin details more prominent and natural" -s 42 --into tests/skin --tag plain
./img2img.py photo.jpg "make the subjects skin details more prominent and natural" -s 42 --lora qwen-edit-skin-1.1.safetensors --into tests/skin --tag edit-skin
./img2img.py photo.jpg "make the subjects skin details more prominent and natural" -s 42 --lightning 8 --lora qwen-edit-skin-1.1.safetensors --lora Skin_Fix_rank64.safetensors:0.8 --into tests/skin --tag both
```

A whole grid of strengths and Lightning settings goes in one run [from Python](#from-python); the example there is this comparison.

### Options

Generation:

| Option | Description |
|---|---|
| `-m 2511\|2.1` | The model; default 2511 |
| `-n N` | Make N images, with the seed going up by 1 for each |
| `-s SEED` | The first image's seed; random when not given. Each image's seed is in its name and recipe |
| `--negative TEXT` | Negative prompt; only has an effect when CFG is above 1 |
| `--steps N`, `--cfg X` | Override the defaults above |
| `--lora FILE[:STRENGTH]` | Apply a LoRA from `models/loras`; repeatable; strength defaults to 1.0 |
| `--lightning off` | 2511 runs the full model: 40 steps, CFG 4. Better quality, about 20 times the sampling work of Lightning |
| `--lightning 8` | 2511 uses the 8-step LoRA; download it first with `uv run download_models.py 2511 --lightning 8` |
| `--resolution N` | 2.1 scales the reference images to about N×N pixels; default 1024; 0 keeps their size, which is slow for large photos |
| `--enhance` | 2.1 first lets its 9B prompt enhancer rewrite the prompt and prints the result; loads another 9.5 GB model |
| `--dry-run` | Only print the API workflow JSON: nothing is uploaded, run or kept in rope |

The rope library:

| Option | Description |
|---|---|
| `--into FOLDER` | The rope folder for the results, made with its parents if there is none (the script says so, which is how a typo shows). Default: `$IMG2IMG_INTO`, else unfiled |
| `--tag TAG` | Tag the results. For several tags, repeat it: `--tag 夕陽 --tag 海邊` (a comma does not split) |
| `--note TEXT` | A note on each result |
| `@ID` as an input | An image in rope, named by at least 6 hex digits of its id |
| `--replay ID` | Make a rope image again from its record; `--replay ID --help` lists its own options |
| `--library DIR` | Where the library is. Default: `$ROPE_LIBRARY`, else `~/.local/share/rope` |

ComfyUI:

| Option | Description |
|---|---|
| `--server URL` | ComfyUI's address |
| `--keep-server` | Leave a ComfyUI this command started running, so the next command skips the model load |
| `--no-start` | Fail when no ComfyUI is running instead of starting one |
| `--comfy-args "..."` | Extra flags for a ComfyUI this command starts, e.g. `"--reserve-vram 2"` |
| `--timeout N` | Give up when ComfyUI has said nothing about the jobs for N seconds; default 1800. A batch has no total limit: every progress message starts the count again |
| `--fresh` | Compute the image again even when ComfyUI has cached the result of an identical workflow (same inputs, settings and seed); the loaded models are kept. GPU results differ slightly from run to run, so a fresh run is a new image |

### From Python

`img2img.py` is also a module. `import img2img`, build a `Request` per combination to try, and `run()` makes them all in one go: through one ComfyUI (started once if none is running), each input uploaded once, and every result in rope with its recipe, so the combinations can be compared side by side in `rope-gui`. A `Request` has a field per option above, with the same names and defaults, and `dataclasses.replace` makes variants of one:

```python
# the skin LoRAs at a few strengths, with the 4- and 8-step Lightning LoRA, on one seed
import dataclasses, itertools
import img2img

base = img2img.Request(["photo.jpg"], "make the subjects skin details more prominent and natural", seed=42, into="tests/skin")
skin, fix = "qwen-edit-skin-1.1.safetensors", "Skin_Fix_rank64.safetensors"
requests = [dataclasses.replace(base, tags=["plain"])]
for lightning, strength in itertools.product(["4", "8"], [0.8, 1.0, 1.2]):
    requests.append(dataclasses.replace(base, lightning=lightning, lora=[(skin, strength)], tags=[f"L{lightning}", f"skin {strength}"]))
requests.append(dataclasses.replace(base, lightning="8", lora=[(skin, 1.0), (fix, 0.8)], tags=["L8", "skin 1", "fix 0.8"]))

for r in img2img.run(requests):
    print(r.short, r.folder, r.request.lightning, r.request.lora, r.request.tags)
```

`run.py` is this as a template: a `BASE` request, the settings to vary, and a `batch()` that yields the combinations, each in a rope folder named after it. Copy it, fill in the three numbered parts, and run it from the repo root with `.venv/bin/python run.py` (see [setup_venv.sh](#python-environment-setup_venvsh)); `--list` prints the batch after checking every request, without making anything. From elsewhere, put the repo on `PYTHONPATH`.

| Name | What it is |
|---|---|
| `Request(images, prompt, ...)` | `images`: paths or `@ID`s (a single one may be a plain string), `prompt`, then the options as fields: `model`, `count`, `seed`, `negative`, `steps`, `cfg`, `lora` (`[("style.safetensors", 0.8)]`, or `["style.safetensors:0.8"]` as on the command line), `lightning`, `resolution`, `enhance`, `into`, `tags`, `note`. `request.settled()` is a copy with the defaults filled in |
| `run(requests, **options)` | Makes every request's images and returns a `Result` per image, the requests in order. The keyword arguments are the ComfyUI options and `library`, with the same names and defaults: `server`, `keep_server`, `no_start`, `comfy_args`, `timeout`, `fresh`, `library` |
| `Result` | `id` and `short` (rope's id), `path`, `folder`, `created` (False when rope had these bytes already), `width`, `height`, `seconds`, `enhanced_prompt`, `seed`, `index` (which of the request's `count` images), and `request`, the settled Request it came from |
| `workflow(request, seed=0)` | The API workflow `run()` would send, as `--dry-run` prints it |
| `Error` | What a failed run raises. A bad `Request` is a `ValueError` instead, raised before anything runs |

- Every request is checked before anything runs, so one typo fails the whole batch at once, before any input goes into rope or any GPU work.
- It prints what the command line prints. A failure in one job stops the run and drops the jobs still queued; what was made before stays in rope. Ctrl-C does the same.
- Fix the seed when comparing settings: a request without one draws its own random seed.
- Identical requests (same inputs, settings and seed) are answered from ComfyUI's cache, as on the command line; `fresh=True` computes them again.

### Tips

- **Keep ComfyUI running between runs.** Start it with `run_comfyui.sh` in another zellij pane and `img2img.py` uses it, loading the models only once. A command that starts ComfyUI itself reloads about 30 GB of models every time.
- **The same command twice gives the same image at once.** ComfyUI caches results: an identical workflow (same inputs, settings and seed) is answered from the cache, not computed again, and rope takes the bytes it already has. Change the seed for a new image, or add `--fresh` to compute the same settings again.
- **Ctrl-C cancels.** Queued jobs are dropped and the running one is interrupted; a ComfyUI the command started is stopped too.
- **Result PNGs carry their workflow.** Drop one on the ComfyUI page to see the workflow as a node graph, a good way into the UI.
- **A missing model file comes with the fix.** For example, without the 8-step LoRA, ComfyUI rejects the workflow before running it, and the script prints the `download_models.py` command that fetches it.

### Further reading

- [Qwen-Image-Edit-2511 ComfyUI example](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511): the official guide to the same template.
- [Qwen-Image-2.1 ComfyUI example](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1): the 2.1 templates and the prompt enhancer.
- [ComfyUI Server Routes](https://docs.comfy.org/development/comfyui-server/comms_routes): `/prompt`, `/upload/image`, websocket events and the rest of the API.
- `ComfyUI/script_examples/`: official API examples; `websockets_api_example.py` works the way `img2img.py` does.
- [Qwen-Image-Edit-2511 model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511): what the model can do and how to write prompts.

## The image library: rope

rope keeps images by content hash. Each image has the paths its original was seen at, the recipe that made it (with its ordered input images), and a note. Folders and tags exist only in rope. It has a CLI, `rope`, and a GUI, `rope-gui`, written in Rust; their data lives in `~/.local/share/rope`, outside the repo and out of git.

```bash
cargo build --release --manifest-path rope/Cargo.toml   # binaries in rope/target/release/
cargo install --locked --path ~/.cargo/bin
cargo install --locked --path ~/.cargo/bin
```

[rope/README.md](rope/README.md) covers the CLI, the GUI, how replays work, deleting, and backups.
