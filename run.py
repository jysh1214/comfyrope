"""A batch of img2img runs: the combinations of some settings, made in one go and kept in rope, to compare side by
side. A template: copy it, fill in the three numbered parts, and run it from the repo root:

    .venv/bin/python run.py --list    # only print the batch, after checking every request
    .venv/bin/python run.py           # make them all, through one ComfyUI

Every result lands in a rope folder named after its combination, with its recipe. README.md, "From Python",
describes Request, run() and Result."""
import dataclasses
import itertools
import sys

import img2img

# 1. What every request shares: the inputs, the prompt and the fixed settings. Keep the seed fixed, so the images
#    differ only by the settings being compared.
BASE = img2img.Request(
    images=["photo.jpg"],
    prompt="make the subjects skin details more prominent and natural",
    count=1,  # images per combination, one seed each from `seed`
    seed=42,
    into="tests/skin",  # the prefix of the rope folders: into_model_loras, see folder()
    # the rest, with their defaults: negative="", steps=None, cfg=None, resolution=None, enhance=False, tags=[],
    # note=None
)

# 2. The settings to vary: every model in LORAS, with every choice of its LoRAs, each one in or out, the empty
#    choice included (2 ** len(LoRAs) choices), and for 2511 every Lightning setting as well.
LORAS = {  # model -> its LoRAs, name[:strength] as on the command line (.safetensors implied); [] for plain only
    "2511": ["qwen-edit-skin-1.1", "Skin_Fix_rank64:0.8"],
    "2.1": [],
}
LIGHTNING = ["4", "8"]  # 2511 only

# The ComfyUI options, as run() takes them: server, keep_server, no_start, comfy_args, timeout, fresh, library.
OPTIONS = {"fresh": False}


def choices(items: list) -> list[tuple]:
    """Every way of picking some of `items`, from none to all of them."""
    return [c for k in range(len(items) + 1) for c in itertools.combinations(items, k)]


def folder(model: str, loras: tuple[str, ...]) -> str:
    """The rope folder of a combination: into_model_loras, e.g. tests/skin_2511_qwen-edit-skin-1.1+Skin_Fix_rank64:0.8;
    without a LoRA, into_model."""
    bits = [BASE.into, model]
    if loras:
        bits.append("+".join(l.replace(".safetensors", "") for l in loras))
    return "_".join(bits)


def batch():
    """The requests, one per combination, each BASE with the combination's fields changed."""
    # 3. The grid.
    for model, loras_of_model in LORAS.items():
        for lightning, loras in itertools.product(LIGHTNING if model == "2511" else [None], choices(loras_of_model)):
            yield dataclasses.replace(BASE, model=model, lightning=lightning, lora=list(loras), into=folder(model, loras))


def main() -> None:
    requests = [r.settled() for r in batch()]  # a bad field is a ValueError here, before anything runs
    if "--list" in sys.argv[1:]:
        for n, r in enumerate(requests, start=1):
            setting = f"lightning {r.lightning}" if r.model == "2511" else f"resolution {r.resolution}"
            loras = ", ".join(f"{name}:{strength:g}" for name, strength in r.lora) or "none"
            print(f"#{n} {r.into}: model {r.model}, {setting}, {r.steps} steps, cfg {r.cfg:g}, seed {r.seed}, "
                  f"LoRA {loras}")
        folders = dict.fromkeys(r.into for r in requests)
        print(f"{len(requests)} requests, {sum(r.count for r in requests)} images, {len(folders)} folders")
        return
    results = img2img.run(requests, **OPTIONS)
    print()
    for r in results:
        made = f"{r.seconds:.1f}s" if r.created else "cached"
        print(f"{r.short}  seed {r.seed}  {made:>7}  {r.folder}")
    print(f"{len(results)} images in {len(dict.fromkeys(r.folder for r in results))} rope folders")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, img2img.Error) as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        sys.exit(130)
