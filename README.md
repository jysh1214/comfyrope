# comfyrope: local image editing and image-to-video with ComfyUI

Runs Qwen-Image-Edit-2511 and Qwen-Image-2.1 for images and Wan 2.2 for short clips, through ComfyUI on this machine (RTX 4090, 24 GB), from the command line or from Python. Every image goes into rope, its own library, with the recipe that made it.

| Tool | What it does |
|---|---|
| `setup_venv.sh` | Builds `.venv`, the Python environment for the scripts below |
| `install_comfyui.py` | Installs or updates ComfyUI with its own venv |
| `download_models.py` | Downloads the model files from Hugging Face: the image models, optional skin LoRAs, and Wan 2.2 and MiniMax H3 for video |
| `run_comfyui.sh` | Starts ComfyUI |
| `img2img.py` | Makes a new image from input images and a prompt with Qwen-Image, through ComfyUI's API. Also a Python module, for running many combinations in one go |
| `img2vid.py` | Makes a short clip from a start image and a prompt with Wan 2.2, the same way; the clips go into rope too |
| `comfycore.py` | What both scripts are built on: the ComfyUI client, rope, the input conversion and the running of jobs. Not a command |
| `comfyqueue.py` | What both do with rope-queue: queueing a command's jobs, and running one job as the queue's child. Not a command |
| `comfyconvert.py` | The input conversion as a command, which rope-queue runs for inputs ComfyUI cannot read |
| `rope/` | An image library (CLI and GUI, in Rust) and `rope-queue`, the work queue that owns ComfyUI and runs the tools' jobs in order. The library keeps every input and result of `img2img.py` and `img2vid.py` with its recipe and lineage, so a result can be made again after its inputs are moved or lost. See [rope/README.md](rope/README.md) |

Run every command from the repo root. Each script finds `ComfyUI/`, `models/` and `.venv/` next to itself, so the repo can live anywhere and be moved without reinstalling.

## Quick start

```bash
./install_comfyui.py                                    # ComfyUI, with its own venv
uv run download_models.py all wan                       # the model files: the image models, and Wan 2.2 for clips
cargo build --release --manifest-path rope/Cargo.toml   # the rope image library
rope/target/release/rope-queue --repo .                 # in another pane: starts ComfyUI and runs the jobs the tools queue
./img2img.py photo.jpg "把背景換成黃昏的海邊"             # "turn the background into a beach at dusk"
./img2vid.py photo.jpg "the leaves move in the wind"    # a 5-second clip, into rope as well
rope/target/release/rope-gui                            # browse the results
```

## Layout

```
.
├── setup_venv.sh        # builds .venv with the packages img2img.py, img2vid.py and download_models.py need
├── .venv/               # Python 3.12 venv for the scripts, separate from ComfyUI's
├── install_comfyui.py   # installs or updates ComfyUI (Python standard library only)
├── download_models.py   # downloads models from Hugging Face (run with uv, nothing to install first)
├── run_comfyui.sh       # starts ComfyUI (written by install_comfyui.py)
├── img2img.py           # images + prompt -> new image, through ComfyUI's HTTP API; import it from Python for batches
├── img2vid.py           # start image + prompt -> short clip with Wan 2.2, the same way, into rope
├── comfycore.py         # what both are built on: the ComfyUI client, rope, the inputs, running jobs; knows no model
├── comfyqueue.py        # what both do with rope-queue: queueing a command's jobs, and running one job as its child
├── comfyconvert.py      # the input conversion as a command, for rope-queue's conversion stage
├── run_img2img.py       # template for a batch of settings to compare, on img2img.py's Python API
├── run_img2vid.py       # the same for clips, on img2vid.py's
├── rope/                # Rust source of the image library (rope, rope-gui) and the work queue (rope-queue); data in ~/.local/share/rope
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

The packages `img2img.py`, `img2vid.py` and `download_models.py` need are listed in the `# /// script` block at the top of each file. There are two ways to run them; either works:

| Way | Command | Preparation |
|---|---|---|
| Let uv handle it | `./img2img.py …` | None. uv reads the package list at the top of the file and builds and caches an environment for it |
| Use the project venv | `source .venv/bin/activate`, then `python img2img.py …` | Run `./setup_venv.sh` once |

```bash
./setup_venv.sh                 # builds .venv and installs the packages; a rerun only adds what is missing
source .venv/bin/activate
python img2img.py photo.jpg "把天空換成星空"
```

`setup_venv.sh` uses uv to make a Python 3.12 `.venv`, installs the package lists from the three scripts, and checks each package imports. When a script gains a package, run it again; to rebuild from scratch, delete `.venv` first. A script of your own that does `import img2img` (see [From Python](#from-python)) needs the same packages, so run it with `.venv/bin/python` too.

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
uv run download_models.py h3                        # MiniMax H3: FastVideo's 8-step FastH3 in int8, with the encoder and VAEs
uv run download_models.py h3 --h3-model fasth3 fl2va   # and H3 itself in int8, with its 8-step turbo LoRA
```

MODEL is `2.1`, `2511`, `wan`, `h3` or `all` (aliases: `qwen-image-2.1`, `qwen-image-edit-2511`, `edit`, `wan2.2`, `minimax`). `all` is the two image models, the ones `img2img.py` runs; the video models, `wan` for `img2vid.py` and `h3`, are asked for by name, alone or with the others.

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

Alibaba's open video models (Apache 2.0), as repacked by Comfy-Org for ComfyUI's own Wan 2.2 templates ([ComfyUI's Wan 2.2 guide](https://docs.comfy.org/tutorials/video/wan/wan2_2)). `img2vid.py` runs the `i2v` pair; `t2v` and `ti2v` are for ComfyUI's own templates for now. The files land in the same `models/` layout, so ComfyUI finds them. `--wan-task` picks the models, several at once if wanted:

| TASK | Files | Size |
|---|---|---|
| `i2v` (default) | `diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled` + `..._low_noise_...`, `vae/wan_2.1_vae`, `loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise` + `..._low_noise` | 14.3 + 14.3 + 0.25 + 1.2 + 1.2 GB |
| `t2v` | the same for `t2v` (LoRAs `v1.1`) | 31.3 GB |
| `ti2v` | `diffusion_models/wan2.2_ti2v_5B_fp16`, `vae/wan2.2_vae` | 10 + 1.4 GB |
| all | `text_encoders/umt5_xxl_fp8_e4m3fn_scaled` | 6.7 GB |

The 14B pair is a high-noise and a low-noise model that the workflow runs one after the other; `--wan-precision fp16` doubles them to 28.6 GB each. The 4-step LoRAs are what the Lightning LoRAs are for 2511; `--wan-lightning none` leaves them out. The 5B model is one file and runs within 8 GB of VRAM; it has no 4-step LoRA in the repack.

### MiniMax H3

MiniMax's open video model, 33B parameters, which makes video with stereo audio at 768p and 24 fps, 4 to 15 seconds long, from text or from a first and a last frame (its [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE): free, commercial use included under 20 M USD a year, outside the EU, UK, South Korea and USA; the outputs may not train other models). ComfyUI runs it natively, through its own templates ([ComfyUI's FastH3 guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-fastvideo)); `img2vid.py` does not run it yet. `--h3-model` picks the diffusion models, both if wanted; the pruned int8 files are the ones a 24 GB card holds:

| MODEL | Files | Size |
|---|---|---|
| `fasth3` (default) | `diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot`: FastVideo's 8-step distillation of H3, text-to-video and image-to-video, fixed at 8 steps | 22.1 GB |
| `fl2va` | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot`: H3 itself, with `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16` for 8 steps (`--h3-turbo`) | 21.0 + 2.0 GB |
| both | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq` (`--h3-encoder`), `vae/minimax_h3_video_vae_int8_convrot`, `vae/minimax_h3_audio_vae_fp32` | 15.7 + 2.8 + 0.6 GB |

The text encoder is Qwen3-VL-32B. The templates load its nvfp4 file; an RTX 40 card has no FP4 units, so ComfyUI computes it dequantized, slower but within memory. `--h3-encoder int8` is the 27.1 GB file with native int8 compute, `bf16` the 51.5 GB one. On an RTX 4090, FastVideo measured a 5-second FastH3 clip at about 55 s at 480p and 155 s at 768p; others measured H3 itself with the turbo LoRA at about 2 minutes for 5 seconds at 1280×736, out of memory at 15 seconds at 1367×768 and at 20 seconds at any size.

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
| `--h3-model MODEL…` | MiniMax H3 diffusion models: `fasth3` (default), `fl2va` (see above) |
| `--h3-encoder nvfp4\|int8\|bf16` | H3's Qwen3-VL-32B text encoder: nvfp4_awq (default, 15.7 GB), int8_convrot (27.1 GB), bf16 (51.5 GB) |
| `--h3-turbo none\|4\|8` | The turbo LoRA for `fl2va`: 8-step (default, what the template loads), 4-step (768p only), or none for the full 20-step model |

### Behavior

- **Interruptible and resumable.** After Ctrl-C or a dropped connection, rerun the same command to continue. Files of the right size are skipped and shown as `present`.
- **Two download paths.** Without `--limit-rate` it uses huggingface_hub's multi-connection hf_xet path, the fastest. With `--limit-rate`, or after hf_xet fails twice in a row, it switches to its own plain-HTTP path: it writes `<file>.part`, resumes with a `Range` header, and retries network errors up to 8 times (backing off up to 60 seconds).
- **Size check.** It asks the Hugging Face API for the real size before downloading and compares afterwards; on a mismatch it keeps the `.part` file and asks for a rerun.
- **Existing files are judged by size only**, not by hash. To fetch a file again, delete it and rerun.
- Progress lines are flushed one by one, so `nohup uv run download_models.py all > dl.log 2>&1 &` with `tail -f dl.log` works.

### Adding files

The file lists are `plan_qwen_image_21()`, `plan_edit_2511()`, `plan_wan_22()` and `plan_minimax_h3()` in the script, one `F(repo, path, kind, approx_gb)` per file. `kind` picks the subfolder (`diffusion` → `diffusion_models/`, and so on); `name=` saves the file under another name. A file that is not on Hugging Face takes `url=`, `size=` and `sha256=` instead, as the Civitai entry in `SKIN_LORAS` does; it is fetched over plain HTTP and checked against the hash. Check with `--list` afterwards.

## Starting ComfyUI: run_comfyui.sh

```bash
./run_comfyui.sh                    # http://127.0.0.1:8188
./run_comfyui.sh --reserve-vram 2   # leave 2 GB more VRAM to niri when the desktop stutters
./run_comfyui.sh --listen           # let other machines on the LAN connect
```

`install_comfyui.py` writes this file. It only `cd`s into `ComfyUI/` and starts `main.py` with `.venv/bin/python`, adding `--enable-manager` when Manager is installed. Other arguments go to `main.py` unchanged; `./run_comfyui.sh --help` lists them.

`rope-queue` starts ComfyUI through this file itself, so with the queue running there is nothing to start by hand (its `--reserve-vram` and anything after `--` go to ComfyUI). A `--no-queue` command uses a ComfyUI started here, or starts its own.

### Notes

- **The GPU also drives the desktop.** The screen may stall during the tens of seconds a model loads; if the whole desktop goes black, try `--reserve-vram 2`.
- **VRAM needs no tuning.** ComfyUI 0.39 turns on dynamic VRAM (comfy-aimdo) by default, so 2511's 20.5 GB model and 9.4 GB text encoder are offloaded to RAM as needed. Do not add `--lowvram`.
- **The first run of a workflow reads about 30 GB from NVMe into RAM**, so a minute or two of waiting is normal; later runs of the same workflow with another prompt do not reload.
- The model folders come from `ComfyUI/extra_model_paths.yaml`. Its `is_default: true` puts the repo's `models/` first, and makes it where the frontend saves downloaded models.

### Templates

In the UI, open Workflow → Browse Templates on the left and search for `Qwen`, `Wan` or `MiniMax`:

| Template | Files it needs |
|---|---|
| Qwen Image Edit 2511 - Material Replacement | The four default 2511 files; runs as it is |
| Qwen Image Edit 2511 Int8: Image Edit | First `download_models.py 2511 --edit-precision int8` |
| Qwen Image 2.1: Image Edit | The three 2.1 files **plus** the `--prompt-enhancer i2i` file |
| Qwen Image 2.1: Text to Image | The three 2.1 files plus `--prompt-enhancer t2i` |
| Wan 2.2 14B Image to Video | The `download_models.py wan` files; runs as it is. `img2vid.py` runs this template without the UI |
| Wan 2.2 14B Text to Video | `download_models.py wan --wan-task t2v` |
| Wan 2.2 5B Video Generation | `download_models.py wan --wan-task ti2v` |
| FastVideo FastH3: Image to Video | The `download_models.py h3` files; runs as it is |
| MiniMax H3: Image to Video | `download_models.py h3 --h3-model fl2va` |

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

Every command hands its images to `rope-queue`, the work queue in `rope/` ([rope/README.md](rope/README.md#rope-queue)): it prints what it queued and returns at once, and rope-queue, which owns ComfyUI, makes the images in order and puts each into rope. Start it once, in another pane, and keep issuing commands. `--no-queue` does the work in the command itself, as before the queue; the ComfyUI options apply only then.

With `.venv` activated, `python img2img.py` works the same as `./img2img.py`. To try many settings in one go, [import it from Python](#from-python).

`img2img.py` holds only what is about its models: the requests, the workflows and the replay. The ComfyUI client, rope, the input conversion and the running of jobs are `comfycore.py` beside it, which knows no model, and what both tools do with rope-queue is `comfyqueue.py`; `img2vid.py` is built on them the same way. Both need the two next to themselves.

Images come first and the prompt last; the first image sets the output size. With several images, refer to them in the prompt as 圖1, 圖2 or image 1, image 2, in command-line order. Prompts can be Chinese or English and work best as instructions that describe only the change: "replace … with …", "remove …", "add …".

### How it works

1. The request is checked: the settings, that every input is a format the converter reads, and that the model and LoRA files of its workflow are there. What fails is reported with its fix, and the request is skipped.
2. It is handed to rope-queue, which keeps the inputs in rope (see below) and queues one job per image with its settings pinned. The command prints `queued #123–#172 (group 40) …` and returns.
3. rope-queue converts inputs ComfyUI cannot read to PNG (single-frame PNG, JPEG and WebP go as they are), and runs the jobs in order, each as `img2img.py --job`: the inputs are uploaded to ComfyUI's `input/img2img/`, named after a hash of their content, so a rerun with the same image reuses the same upload; the workflow for `-m` is built and queued, and loading, encoding and sampling are followed over the websocket.
4. Each result goes into rope with its recipe.

With `--no-queue`, the command does steps 3 and 4 itself, against the ComfyUI at `--server` (default http://127.0.0.1:8188); if none answers there, it starts one with `run_comfyui.sh` and stops it when the command ends.

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

`rope-gui`'s Copy repro button, `rope repro ID` and the `repro.sh` that rope keeps with each image all run this same command. Before sending, the replay says what differs from when the image was made. It sends the recorded workflow on the recorded input bytes, then compares the pixels: an identical result is only noted, and a different one is kept next to the original. [rope/README.md](rope/README.md#reproducing-an-image) has the details. `img2vid.py --replay ID` does the same for a clip, comparing the bytes.

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
| `--no-queue` | Do the ComfyUI work in this command instead of handing the jobs to rope-queue; the four options below apply only then |
| `--server URL` | ComfyUI's address |
| `--keep-server` | Leave a ComfyUI this command started running, so the next command skips the model load |
| `--no-start` | Fail when no ComfyUI is running instead of starting one |
| `--comfy-args "..."` | Extra flags for a ComfyUI this command starts, e.g. `"--reserve-vram 2"` |
| `--timeout N` | Give up when ComfyUI has said nothing about the jobs for N seconds; default 1800. A batch has no total limit: every progress message starts the count again |
| `--fresh` | Compute the image again even when ComfyUI has cached the result of an identical workflow (same inputs, settings and seed); the loaded models are kept. GPU results differ slightly from run to run, so a fresh run is a new image. With or without the queue |

### From Python

`img2img.py` is also a module. `import img2img`, build a `Request` per combination to try, and `run()` hands them all to rope-queue: one group of jobs per request, and every result in rope with its recipe, so the combinations can be compared side by side in `rope-gui`. `run()` returns as soon as the jobs are queued; rope-queue's screen, `rope-queue ls` and `rope-gui` show what becomes of them. A `Request` has a field per option above, with the same names and defaults, and `dataclasses.replace` makes variants of one:

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

for s in img2img.run(requests):
    print(s.group, s.jobs, s.request.lightning, s.request.lora, s.request.tags)
```

`run_img2img.py` is this as a template: a `BASE` request, the settings to vary, and a `batch()` that yields the combinations, each in a rope folder named after it. Copy it, fill in the three numbered parts, and run it from the repo root with `.venv/bin/python run_img2img.py` (see [setup_venv.sh](#python-environment-setup_venvsh)); `--list` prints the batch after checking every request, without queueing anything. From elsewhere, put the repo on `PYTHONPATH`.

| Name | What it is |
|---|---|
| `Request(images, prompt, ...)` | `images`: paths or `@ID`s (a single one may be a plain string), `prompt`, then the options as fields: `model`, `count`, `seed`, `negative`, `steps`, `cfg`, `lora` (`[("style.safetensors", 0.8)]`, or `["style.safetensors:0.8"]` as on the command line), `lightning`, `resolution`, `enhance`, `into`, `tags`, `note`. `request.settled()` is a copy with the defaults filled in |
| `run(requests, **options)` | Queues every request's images in rope-queue and returns a `Submitted` per request, in order. The keyword arguments are `fresh` and `library`, as on the command line |
| `Submitted` | `group` (its id in rope-queue) and `jobs` (the first and the last job id) once queued, with `count` and `first_seed`; `skipped`, the reasons it was not queued, each with its fix; and `request`, the settled Request it came from |
| `workflow(request, seed=0)` | The API workflow a job would send, as `--dry-run` prints it |
| `Error` | What a failed submission raises: rope-queue not built, or not running. A bad `Request` is a `ValueError` instead, raised before anything is sent |

- Every request is checked before anything is sent, so one typo fails the whole batch at once. A request whose input the converter cannot read, or whose model or LoRA file ComfyUI lacks, is printed with its fix and skipped; the others are queued.
- It prints what the command line prints: one `queued` line per request. What then happens to the jobs is rope-queue's to show, and a failed job never stops the others.
- Fix the seed when comparing settings: a request without one draws its own random seed when it is queued.
- Identical requests (same inputs, settings and seed) are answered from ComfyUI's cache, as on the command line; `fresh=True` computes them again.
- There is no direct path from Python: a batch needs rope-queue running. `--no-queue` is the command line's.

### Tips

- **rope-queue keeps ComfyUI running between jobs.** Start it once in another zellij pane and keep issuing commands; the models load once. With `--no-queue`, start ComfyUI with `run_comfyui.sh` in a pane first, or the command starts one itself and reloads about 30 GB of models every time.
- **The same command twice gives the same image at once.** ComfyUI caches results: an identical workflow (same inputs, settings and seed) is answered from the cache, not computed again, and rope takes the bytes it already has. Change the seed for a new image, or add `--fresh` to compute the same settings again.
- **Ctrl-C in rope-queue** interrupts the running job, which goes back to the head of the line, and stops ComfyUI with the daemon; `rope-queue cancel 123` or `cancel g40` drops a job or a whole group. With `--no-queue`, Ctrl-C in the command drops its queued jobs and interrupts the running one; a ComfyUI the command started is stopped too.
- **Result PNGs carry their workflow.** Drop one on the ComfyUI page to see the workflow as a node graph, a good way into the UI.
- **A missing model file comes with the fix.** For example, without the 8-step LoRA, the command finds the file missing before queueing anything, prints the `download_models.py` command that fetches it, and skips the request.

### Further reading

- [Qwen-Image-Edit-2511 ComfyUI example](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511): the official guide to the same template.
- [Qwen-Image-2.1 ComfyUI example](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1): the 2.1 templates and the prompt enhancer.
- [ComfyUI Server Routes](https://docs.comfy.org/development/comfyui-server/comms_routes): `/prompt`, `/upload/image`, websocket events and the rest of the API.
- `ComfyUI/script_examples/`: official API examples; `websockets_api_example.py` works the way `img2img.py` does.
- [Qwen-Image-Edit-2511 model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511): what the model can do and how to write prompts.

## Making a clip from an image: img2vid.py

Give it a start image and a prompt and it makes a short clip with Wan 2.2, the 14B image-to-video pair, through ComfyUI's API, the way `img2img.py` makes images: the same ComfyUI handling, the same input conversion, the same recipe. It is built on `comfycore.py` and flattens ComfyUI's Wan 2.2 14B Image to Video template. Download the models first with `uv run download_models.py wan` (the 14B pair in fp8 with the 4-step LoRAs, 38 GB).

```bash
./img2vid.py photo.jpg "the camera slowly pushes in while the leaves move in the wind"
./img2vid.py photo.jpg "..." -n 2 -s 42 --seconds 3 --size 480     # two clips of 3 s, long side 480 px
./img2vid.py photo.jpg "..." --lightning off                       # the full 20-step model at cfg 3.5: about five times slower
./img2vid.py photo.jpg "..." --dry-run                             # only print the API workflow JSON
```

The clip keeps the start image's aspect ratio, scaled so that its long side is `--size` pixels (default 640; both sides are rounded to a multiple of 16), and lasts `--seconds` at `--fps` (default 5 s at Wan's own 16 fps: 81 frames). Prompts work best as a description of the motion: what moves, and how the camera moves.

**Where clips go.** Into rope, like images: each clip with its recipe and the workflow ComfyUI ran, its first frame as its thumbnail, and its start image as its upstream, so `--into`, `--tag` and `--note` work as for `img2img.py`, and `rope-gui` plays it (see [rope/README.md](rope/README.md)). Names are `<time>_<image>_<model>_s<seed>`, as `img2img.py`'s. ComfyUI keeps its own copy of each clip under `ComfyUI/output/img2vid/`.

**Speed.** On the RTX 4090 at 640 px and 5 s, ComfyUI's own template notes about 1.5 minutes per clip with the 4-step LoRAs (the first one longer, with 30 GB of models to load; rope-queue keeps them loaded between jobs) and about 9 minutes without.

### Options

| Option | Meaning |
|---|---|
| `-m 14b` | The model: Wan 2.2 image-to-video, the A14B high-noise and low-noise pair; the only one yet |
| `-n N`, `-s SEED` | How many clips, one seed each, and the first seed (default: random, and printed) |
| `--seconds S`, `--fps N` | Length (default 5) and frame rate (default 16, Wan's own; another value only changes the playback speed). The frame count is the nearest 4k+1 |
| `--size N` | The long side of the clip in pixels (default 640) |
| `--lightning 4\|off` | The lightx2v 4-step LoRAs (default: 4 steps, cfg 1) or the full model (20 steps, cfg 3.5) |
| `--steps N`, `--cfg X`, `--shift X` | Sampling steps (the high-noise model takes the first half), guidance scale, and the sampling shift (default 5) |
| `--lora FILE[:STRENGTH]` | An extra LoRA from `models/loras`, applied to both models; repeatable |
| `--negative TEXT` | The negative prompt (default: Wan's own, in Chinese); has an effect only when cfg is above 1 |
| `--into FOLDER`, `--tag TAG`, `--note TEXT` | The rope folder for the clips (made if there is none; default: unfiled), tags and a note, as `img2img.py`'s |
| `--replay ID` | Make a rope clip again from its record, as `img2img.py --replay` does for an image, except that the new clip is compared byte for byte; `--replay ID --help` lists its own options |
| `--library DIR`, `--no-queue` and the ComfyUI options | As `img2img.py`'s |

From Python it works like `img2img`: `import img2vid`, a `Request` per clip (its fields are the options above), and `run(requests, **options)` queues them in rope-queue and returns a `Submitted` per request, as `img2img.run()` does. `run_img2vid.py` is the batch template for clips, as `run_img2img.py` is for images: a `BASE` request, the settings to vary (Lightning, shift, fps, LoRAs) and a `batch()`; each combination's clips go into a rope folder named after it, so `--list` and a run work the same way.

**Not there yet.** Only the 14B image-to-video pair runs; the 5B model and text-to-video are left to ComfyUI's templates.

## The image library: rope

rope keeps images and clips by content hash. Each has the paths its original was seen at, the recipe that made it (with its ordered input images), and a note. Folders and tags exist only in rope. It has a CLI, `rope`, and a GUI, `rope-gui`, written in Rust; their data lives in `~/.local/share/rope`, outside the repo and out of git. A clip's first frame is its thumbnail, and the GUI plays it.

```bash
cargo build --release --manifest-path rope/Cargo.toml   # binaries in rope/target/release/
cargo install --locked --path rope/rope                 # or install both into ~/.cargo/bin
cargo install --locked --path rope/rope-gui
```

[rope/README.md](rope/README.md) covers the CLI, the GUI, how replays work, deleting, and backups.
