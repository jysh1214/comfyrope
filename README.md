# i2i：Qwen-Image 本地 Image-to-Image 環境

在這台機器（RTX 4090 24 GB）用 ComfyUI 跑 Qwen-Image-Edit-2511 與 Qwen-Image-2.1。
五個 script：`setup_venv.sh` 建立工具用的 venv，`install_comfyui.py` 安裝或更新 ComfyUI，`download_models.py` 抓模型，`run_comfyui.sh` 啟動 ComfyUI，`img2img.py` 用圖片加 prompt 產生新圖。

## 目錄結構

```
.
├── setup_venv.sh        # 建立 .venv，裝好 img2img.py 和 download_models.py 要用的套件
├── .venv/               # 工具用的 Python 3.12 venv，跟 ComfyUI 的 venv 分開
├── install_comfyui.py   # 安裝或更新 ComfyUI（只用 Python 標準函式庫）
├── download_models.py   # 從 Hugging Face 下載模型（uv 跑，不用先裝套件）
├── run_comfyui.sh       # 啟動 ComfyUI（由 install_comfyui.py 生成）
├── img2img.py           # 圖片加 prompt 產生新圖，透過 ComfyUI 的 HTTP API
├── outputs/             # img2img.py 的結果，一張圖一個資料夾，附 repro.sh
├── models/              # 下載結果，ComfyUI 透過 extra_model_paths.yaml 讀這裡
│   ├── diffusion_models/
│   ├── text_encoders/
│   ├── vae/
│   └── loras/
└── ComfyUI/             # git clone，tag v0.39.2
    ├── .venv/           # uv 建的 Python 3.12 venv（torch 2.14.1+cu130，附 pip）
    └── extra_model_paths.yaml
```

`models/` 放在 ComfyUI 外面，重裝或換版 ComfyUI 不會動到模型。

## Python 環境：setup_venv.sh

`img2img.py` 和 `download_models.py` 要用的套件，寫在各自檔案開頭的 `# /// script` 區塊裡。執行方式有兩種，擇一即可：

| 方式 | 指令 | 需要的準備 |
|---|---|---|
| 讓 uv 自動處理 | `./img2img.py …` | 不用準備。uv 讀檔案開頭的套件清單，自動建立並快取專用環境 |
| 用專案的 venv | `source .venv/bin/activate` 之後 `python img2img.py …` | 先跑一次 `./setup_venv.sh` |

```bash
cd ~/i2i
./setup_venv.sh                 # 建立 ~/i2i/.venv 並裝好套件，重跑只會補缺的
source .venv/bin/activate
python img2img.py photo.jpg "把天空換成星空"
```

`setup_venv.sh` 用 uv 建立 Python 3.12 的 `.venv`，直接讀兩支 script 開頭的套件清單來安裝，最後逐一 import 確認。
之後 script 多了套件，重跑一次就會補上；要整個重建，先刪掉 `.venv` 再跑。

用 `python img2img.py` 卻沒有先 activate `.venv` 時，跑的會是系統的 Python 或 ComfyUI 的 venv。兩者都沒有這些套件，script 會列出缺哪些、該怎麼執行。
ComfyUI 自己的 venv 在 `ComfyUI/.venv`，由 `install_comfyui.py` 管理，跟這個 `.venv` 是分開的。

## 安裝與更新 ComfyUI：install_comfyui.py

只用 Python 標準函式庫，需要 `git` 和 `uv`。每一步都先檢查現況，已經就緒就跳過，所以重跑是安全的：

```bash
cd ~/i2i
./install_comfyui.py --dry-run       # 只印計畫和指令，什麼都不改
./install_comfyui.py                 # 全新安裝；已裝過就重新檢查一遍
./install_comfyui.py --ref latest    # 把 ComfyUI 升到最新 release
./install_comfyui.py --ref v0.39.2   # 釘在某個 release（--ref master 追開發分支）
./install_comfyui.py --manager       # 加裝內建的 ComfyUI-Manager
```

### 它做了什麼

| 步驟 | 內容 |
|---|---|
| driver | 從 `nvidia-smi` 讀 driver 支援的 CUDA 版本，低於 `--cuda` 所需就停下 |
| comfyui | 從 Comfy-Org/ComfyUI clone 指定的 tag；已有 clone 時，只有給了 `--ref` 才切版本 |
| venv | 用 uv 在 `ComfyUI/.venv` 建 Python 3.12 venv |
| packages | `uv pip install --torch-backend cu130 -r requirements.txt pip`：torch 從 PyTorch 的 cu130 index 抓，其餘從 PyPI |
| config | 寫 `ComfyUI/extra_model_paths.yaml`，指向 `~/i2i/models` |
| launcher | 寫 `run_comfyui.sh` |
| self-test | 用 `CUDA_VISIBLE_DEVICES=` 把 GPU 藏起來，跑 `main.py --cpu --quick-test-for-ci`，確認能啟動、四個模型目錄都有載入。temp 和 user 資料夾用用完即丟的暫存目錄，所以 ComfyUI 正在跑時也能放心重跑 |

### 參數

| 參數 | 說明 |
|---|---|
| `--dir DIR` | ComfyUI 的位置，預設 `./ComfyUI` |
| `--ref REF` | `latest`、某個 tag 或分支名。不給時，新裝用最新 release，已裝的保持不動 |
| `--python VER` | venv 的 Python，例如 `3.13`。和現有 venv 不同就重建 venv |
| `--cuda BUILD` | PyTorch build：`cu130`（預設）、`cu129`、`cu128`、`cu126`、`cpu`。和現有 torch 不同就重建 venv |
| `--torch VER` | 釘 torch 版本，例如 `2.14.1`；`latest` 表示升到該 index 上的最新版 |
| `--models DIR` | 寫進 `extra_model_paths.yaml` 的模型根目錄，預設 `./models` |
| `--no-model-paths` | 不碰 `extra_model_paths.yaml` |
| `--manager` | 安裝 `manager_requirements.txt`，launcher 會加上 `--enable-manager` |
| `--rewrite-config` | 覆寫被改過的 `extra_model_paths.yaml` 和 `run_comfyui.sh`，舊檔留成 `.bak` |
| `--no-test` | 跳過 self-test |
| `--gpu-test` | 最後在 GPU 上跑一個 fp16 matmul，確認 CUDA 真的能用。**會用到 GPU** |
| `--dry-run` | 只印計畫；ComfyUI 版本和 venv 都不變時，還會附上 uv 預覽的套件差異 |

### 行為

- **沒給的選項沿用現況，給了的選項強制套用。** 例如現有 venv 是 Python 3.12，不給 `--python` 就保留，給 `--python 3.13` 就重建。重建很快，因為 torch 這類大檔會從 uv cache hardlink 過來。
- **不覆蓋你的修改。** ComfyUI 的 tracked 檔被改過時拒絕切版本，請先 `git stash`。`extra_model_paths.yaml` 和 `run_comfyui.sh` 的內容和生成版本不同時保持原樣，除非加 `--rewrite-config`。
- **ComfyUI 正在跑時不動它。** 要切版本或重建 venv 時，script 會先找有沒有從該目錄啟動的 `main.py`，找到就要求你先關掉。
- **預設完全不碰 GPU。** driver 檢查只讀取 `nvidia-smi` 的輸出，self-test 在 GPU 被藏起來的狀態下跑。只有 `--gpu-test` 會建立 CUDA context。
- **venv 附 pip。** uv 建的 venv 預設沒有 pip，但不少 custom node 的安裝腳本會呼叫 `python -m pip`。
- **刻意不裝 torchaudio。** ComfyUI 沒有 import 它。少了它，Manager 啟動時會印一行 `[ERROR] [ComfyUI-Manager] PyTorch is not installed`，這是誤報，可以忽略。
  不要為了消掉這行去裝 torchaudio：Manager 的 torch 版本表只到 2.7.0，三個套件都在時，只要有 custom node 改動 torch 版本，它就會自動裝一個 **nightly** torch 回來。
- self-test 的完整輸出存在 `ComfyUI/install-selftest.log`，這個檔被 ComfyUI 的 `.gitignore` 忽略。

## 下載模型：download_models.py

直接執行，第一次會自動建一個只含 huggingface_hub 的環境（PEP 723 inline metadata）：

```bash
cd ~/i2i
uv run download_models.py --list                    # 列出所有已知檔案與大小
uv run download_models.py all --dry-run             # 只印計畫（含實際大小、剩餘空間），不寫檔
uv run download_models.py all                       # 下載 2.1 + 2511 的預設組合
uv run download_models.py 2511                      # 只要 Edit-2511
uv run download_models.py 2.1 --prompt-enhancer i2i # 2.1 加上 i2i prompt enhancer
uv run download_models.py all --limit-rate 5M       # 限速 5 MiB/s
```

MODEL 可寫 `2.1`、`2511`、`all`（別名 `qwen-image-2.1`、`qwen-image-edit-2511`、`edit`）。

### 預設會抓什麼

| MODEL | 檔案 | 大小 |
|---|---|---|
| 2.1 | `diffusion_models/qwen_image_2.1_int8_convrot` | 7.3 GB |
| 2.1 | `text_encoders/qwen3vl_8b_int8_convrot` | 9.4 GB |
| 2.1 | `vae/qwen_image_2.1_vae_bf16` | 0.7 GB |
| 2511 | `diffusion_models/qwen_image_edit_2511_fp8mixed` | 20.5 GB |
| 2511 | `text_encoders/qwen_2.5_vl_7b_fp8_scaled` | 9.4 GB |
| 2511 | `vae/qwen_image_vae` | 0.25 GB |
| 2511 | `loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16` | 0.85 GB |

### 參數

| 參數 | 說明 |
|---|---|
| `--dest DIR` | 模型根目錄，預設 `./models`（與 script 同層） |
| `--only KIND…` | 只抓某幾類：`diffusion` `text_encoder` `vae` `lora` |
| `--limit-rate RATE` | 限速，`500k` / `5M` / `1.5G`，1024 進位。會改走 plain-HTTP 路徑 |
| `--dry-run` | 只印計畫 |
| `--list` | 列出所有已知檔案 |
| `--force` | 略過剩餘磁碟空間檢查 |
| `--ipv6` | 允許 IPv6。預設只走 IPv4，原因見「故障排除」 |
| `--token` | HF token，公開 repo 不需要；預設讀 `$HF_TOKEN` |
| `--q21-precision int8\|bf16` | 2.1 的 DiT 與 Qwen3-VL encoder 精度。bf16 是 14.2 + 17.5 GB |
| `--q21-variant base\|turbo` | 2.1 用 base（template 預設）或蒸餾過的 turbo 模型 |
| `--turbo-lora` | 多抓 2.1 base 用的 turbo LoRA（0.9 GB） |
| `--prompt-enhancer i2i t2i` | 2.1 的 Qwen3.5-9B prompt enhancer，各 9.5 GB |
| `--edit-precision fp8\|int8\|bf16` | 2511 的模型：fp8mixed（預設）/ int8_convrot / bf16（40.9 GB，要 offload） |
| `--lightning none\|4\|8\|both` | 2511 的 Lightning LoRA，預設只抓 4-step |

### 行為

- **可中斷、可續傳。** Ctrl-C 或斷線後重跑同一個指令就會接著抓。大小正確的檔案直接跳過，顯示 `present`。
- **兩條下載路徑。** 沒有 `--limit-rate` 時走 huggingface_hub 的 hf_xet 多連線路徑（最快）。有 `--limit-rate`，或 hf_xet 連續失敗兩次，就改走 script 自己的 plain-HTTP 路徑：寫到 `<檔名>.part`、用 `Range` header 續傳、網路錯誤最多重試 8 次（backoff 到 60 秒）。
- **大小驗證。** 下載前用 HF API 查實際大小，下載後比對；不符會留下 `.part` 並要求重跑。
- **跳過已有檔案的判斷只看大小**，不算 hash。要重抓某個檔，刪掉它再跑。
- 進度列會逐行 flush，所以可以 `nohup uv run download_models.py all > dl.log 2>&1 &` 再 `tail -f dl.log`。

### 要加新檔案

模型清單寫在 script 裡的 `plan_qwen_image_21()` 與 `plan_edit_2511()`，每個檔一個 `F(repo, path, kind, approx_gb)`。
`kind` 決定落在哪個子目錄（`diffusion` → `diffusion_models/`，依此類推）。加完跑 `--list` 確認。

## 啟動 ComfyUI：run_comfyui.sh

```bash
~/i2i/run_comfyui.sh                    # http://127.0.0.1:8188
~/i2i/run_comfyui.sh --reserve-vram 2   # 桌面卡頓時多留 2 GB VRAM 給 niri
~/i2i/run_comfyui.sh --listen           # 讓區網其他機器也能連
```

這個檔由 `install_comfyui.py` 生成，只做兩件事：`cd` 進 `ComfyUI/`，再用 `.venv/bin/python main.py` 啟動。裝了 Manager 時會多帶 `--enable-manager`。其他參數原樣轉給 `main.py`，完整清單看 `~/i2i/run_comfyui.sh --help`。

### 注意事項

- **GPU 和桌面共用。** 載入模型那幾十秒畫面可能頓一下；若整個桌面黑掉，先試 `--reserve-vram 2`。
- **VRAM 不用特別調。** ComfyUI 0.39 預設開 dynamic VRAM（comfy-aimdo），2511 的 20.5 GB 模型加 9.4 GB text encoder 會自動 offload 到 RAM，不要加 `--lowvram`。
- **第一次跑 workflow 會從 NVMe 讀約 30 GB 進 RAM**，等一兩分鐘是正常的；之後同一個 workflow 換 prompt 不用重載。
- 模型目錄來自 `ComfyUI/extra_model_paths.yaml`，裡面 `is_default: true` 讓 `~/i2i/models` 排在最前面，也是前端下載模型時的預設落點。

### 要用的 template

在 UI 左側 Workflow → Browse Templates，搜 `Qwen`：

| Template | 需要的檔案 |
|---|---|
| Qwen Image Edit 2511 - Material Replacement | 2511 的預設四個檔，開了就能跑 |
| Qwen Image Edit 2511 Int8: Image Edit | 要先 `download_models.py 2511 --edit-precision int8` |
| Qwen Image 2.1: Image Edit | 2.1 的三個檔，**加上** `--prompt-enhancer i2i` 那個檔（已下載） |
| Qwen Image 2.1: Text to Image | 2.1 的三個檔，加上 `--prompt-enhancer t2i`（未下載） |

2.1 的 template 把 prompt enhancer 寫死在 subgraph 的一個 CLIPLoader 裡；ComfyUI 送出 prompt 前會驗證所有連線節點的檔名，
所以就算 `refine_prompt` 關著，檔案不在就會報 `Value not in list`。

## 圖片加 prompt 產生新圖：img2img.py

給圖片和 prompt 就出圖，不必開 UI。背後是把 ComfyUI 內建的 template 攤平成 API workflow，再經由 HTTP API 交給 ComfyUI 執行：

```bash
cd ~/i2i
./img2img.py photo.jpg "把背景換成黃昏的海邊"
./img2img.py person.png shirt.png "讓圖1的人穿上圖2的襯衫" -n 4     # 4 個 seed 各出一張
./img2img.py photo.jpg "..." --lightning 8                            # 改用 8-step LoRA，自動跑 8 步；要先下載，見下方參數表
./img2img.py photo.jpg "..." --lightning off                          # 不用 Lightning，跑完整模型：40 步、CFG 4，慢很多
./img2img.py photo.jpg "..." --lightning off --steps 20               # 完整模型只跑 20 步，是 ComfyUI template 的參考值
./img2img.py photo.jpg "turn this into a watercolor painting" -m 2.1
./img2img.py photo.jpg "..." --lora my_style.safetensors:0.8 -s 42 -o result.png
./img2img.py photo.jpg "..." --dry-run                                # 只印出 API workflow JSON
```

已經 activate `.venv` 的話，把 `./img2img.py` 換成 `python img2img.py` 也一樣。

圖片放前面、prompt 放最後，第一張圖決定輸出尺寸。多張圖時，在 prompt 裡用「圖 1」「圖 2」或 image 1、image 2 指稱，順序就是命令列上的順序。
prompt 用中文或英文都可以，寫成指令句效果最好，例如「把……換成……」「移除……」「加上……」，只描述要改的部分。

### 它怎麼運作

1. 檢查圖片格式。單張的 PNG、JPEG、WebP 照原樣使用，其他格式先在本機轉成 PNG，細節見下一節。
2. 檢查 `--server`（預設 http://127.0.0.1:8188）有沒有 ComfyUI。沒有就用 `run_comfyui.sh` 啟動一個，這次指令結束時關掉。
3. 把圖片上傳到 ComfyUI 的 `input/img2img/`。檔名前綴是內容的 hash，同一張圖重跑會沿用同一個上傳檔。
4. 依 `-m` 組出 workflow 送進佇列，並從 websocket 接收載入、編碼、取樣的進度。
5. 每張結果建一個資料夾，放 PNG 和重現用的 `repro.sh`，細節見下一節。

### 輸出與重現

每張結果各有一個資料夾，預設放在 `outputs/`，`-o DIR` 可以改位置：

```
outputs/
├── 20261013-225301_photo_2511_s3418827001/      # -n 2 的第一張
│   ├── 20261013-225301_photo_2511_s3418827001.png
│   └── repro.sh
└── 20261013-225301_photo_2511_s3418827002/      # 第二張：時間相同，seed 加 1
    ├── 20261013-225301_photo_2511_s3418827002.png
    └── repro.sh
```

資料夾名稱是 `<開始時間>_<第一張圖的檔名>_<模型>_s<seed>`：

- 時間在最前面，`ls` 的順序就是時間順序。同一次 `-n N` 的資料夾共用指令開始的時間，所以會排在一起，只有 seed 不同。
- 圖檔名稱裡的空白和符號換成 `_`，中文保留，超過 40 字的部分截掉。
- PNG 和資料夾同名，複製到別處也認得出來，也找得回它的 `repro.sh`。
- 同一秒內撞名時（例如同一個 seed 連跑兩次），後來的資料夾加上 `-2`。

`repro.sh` 只重做這一張。它是這次的指令再做三項調整：seed 換成這張自己的；影響畫面的設定全部寫明，連沒給、走預設值的 `--steps`、`--cfg`、`--lightning` 也寫進去；路徑一律改成絕對路徑。
所以之後就算改了 script 的預設值，重跑的結果也不會變。跟畫面無關的參數（`-o`、`--server`、`--comfy-args`、`--keep-server` 等）有給才會帶上，`-n` 不帶。
開頭的註解記下出圖時間，以及 ComfyUI 版本與啟動參數、torch 版本、GPU；用了 `--enhance` 的話，也會記下 enhancer 改寫後的 prompt。
它直接呼叫 `~/i2i/.venv/bin/python3`，不必先 activate。

```bash
outputs/20261013-225301_photo_2511_s3418827001/repro.sh            # 再出一次，結果放進新的資料夾
outputs/20261013-225301_photo_2511_s3418827001/repro.sh --dry-run  # 加在後面的參數會傳下去：只印 workflow
outputs/20261013-225301_photo_2511_s3418827001/repro.sh -s 7       # 其他設定不變，改用 seed 7
```

`-o result.png` 這類指定 `.png` 檔名的用法照舊只寫那個檔，不建資料夾，也沒有 `repro.sh`。

### 支援的圖片格式

| 輸入 | 處理方式 |
|---|---|
| 單張的 PNG、JPEG、WebP | 原樣上傳 |
| HEIC/HEIF、AVIF、JPEG XL、BMP、TIFF、PSD 等 Pillow 讀得到的格式 | 先轉成 PNG |
| 相機 RAW，例如 DNG、CR2、CR3、NEF、ARW、RAF、ORF、RW2 | 用相機的白平衡顯影，再轉成 PNG |
| 動畫或多頁檔，例如 GIF、動態 WebP、APNG | 只取第一格，再轉成 PNG |
| SVG 等向量格式 | 讀不了，請先匯出成 PNG |

轉換時會套用 EXIF 方向，帶 ICC profile 的圖片會轉成 sRGB，例如 iPhone 的 Display P3。轉換結果放在暫存資料夾，原檔不會被修改。

### 兩個模型的預設值

| | 2511（預設） | 2.1 |
|---|---|---|
| 模型 | Qwen-Image-Edit-2511 fp8mixed | Qwen-Image-2.1 int8 |
| 輸入圖片 | 最多 3 張 | 最多 10 張 |
| 步數與 CFG | 搭配 Lightning 4-step LoRA：4 步、CFG 1 | 25 步、CFG 1 |
| 輸出尺寸 | 第一張圖縮放到約 1 MP | 第一張圖縮放到約 1024×1024 的面積 |
| 專屬參數 | `--lightning 4\|8\|off` | `--resolution N`、`--enhance` |

### 參數

| 參數 | 說明 |
|---|---|
| `-m 2511\|2.1` | 選模型，預設 2511 |
| `-o PATH` | 結果資料夾要放在哪裡，預設 `./outputs`；給 `.png` 檔名則只寫那個檔 |
| `-n N` | 出 N 張，seed 依序加 1，每張一個資料夾 |
| `-s SEED` | 第一張的 seed；不給就隨機。每張的 seed 都寫在資料夾名稱和 `repro.sh` 裡 |
| `--negative TEXT` | negative prompt，只有 CFG 大於 1 時才有作用 |
| `--steps N`、`--cfg X` | 覆寫上表的預設值 |
| `--lora FILE[:強度]` | 套用 `models/loras` 裡的 LoRA，可重複；強度預設 1.0 |
| `--lightning off` | 2511 改跑完整模型：40 步、CFG 4。品質較好，取樣運算量約為 Lightning 的 20 倍 |
| `--lightning 8` | 2511 改用 8-step LoRA，要先跑 `uv run download_models.py 2511 --lightning 8` |
| `--resolution N` | 2.1 的參考圖縮放到約 N×N 畫素，預設 1024；0 保留原尺寸，大照片會很慢 |
| `--enhance` | 2.1 先用 9B prompt enhancer 改寫 prompt 並印出結果，會多載入 9.5 GB 的模型 |
| `--server URL` | ComfyUI 的位址 |
| `--keep-server` | 這次指令啟動的 ComfyUI 跑完不關，下一次就不必重新載入模型 |
| `--no-start` | 沒有 ComfyUI 時直接報錯，不自動啟動 |
| `--comfy-args "..."` | 自動啟動 ComfyUI 時多帶的參數，例如 `"--reserve-vram 2"` |
| `--dry-run` | 只印 API workflow JSON，不上傳也不執行 |

### 使用建議

- **連續出圖時讓 ComfyUI 一直開著。** 在另一個 zellij pane 用 `run_comfyui.sh` 開好，`img2img.py` 會直接用它，模型只在第一次載入。若每次都讓指令自己啟動，每次都要重新載入約 30 GB 的模型。
- **Ctrl-C 會取消。** 還在排隊的工作會被移除，正在跑的會被中斷；這次指令自動啟動的 ComfyUI 也會一併關掉。
- **輸出的 PNG 內含 workflow。** 把它拖進 ComfyUI 網頁，就會還原成節點圖，可以從這裡開始學 UI。
- **缺模型檔時會告訴你怎麼補。** 例如還沒下載 8-step LoRA，ComfyUI 會在執行前拒絕這個 workflow，script 會印出對應的 `download_models.py` 指令。

### 延伸閱讀

- [Qwen-Image-Edit-2511 ComfyUI 範例](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511)：官方對同一份 template 的說明。
- [Qwen-Image-2.1 ComfyUI 範例](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1)：2.1 的 template 與 prompt enhancer。
- [ComfyUI Server Routes](https://docs.comfy.org/development/comfyui-server/comms_routes)：`/prompt`、`/upload/image`、websocket 事件等 API 的說明。
- `ComfyUI/script_examples/`：官方的 API 使用範例，其中 `websockets_api_example.py` 的做法和 `img2img.py` 相同。
- [Qwen-Image-Edit-2511 model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)：模型能力與 prompt 寫法。
