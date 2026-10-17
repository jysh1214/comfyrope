"""A batch of img2vid runs: the combinations of some settings, made in one go, to compare side by side. A template:
copy it, fill in the three numbered parts, and run it from the repo root:

    .venv/bin/python run_img2vid.py --list    # only print the batch, after checking every request
    .venv/bin/python run_img2vid.py           # queue them all in rope-queue, which has to be running

Every clip lands in a rope folder named after its combination, with its recipe. README.md, "Making a clip from an
image", describes Request, run() and Submitted; run_img2img.py is the same template for images."""
import dataclasses
import itertools
import sys

import img2vid

# 1. What every request shares: the start image, the prompt and the fixed settings. Keep the seed fixed, so the
#    clips differ only by the settings being compared.
BASE = img2vid.Request(
    images=["photo.jpg"],
    prompt="",  # the motion: what moves, and how the camera moves; a LoRA's trigger (TRIGGERS) is added where it is in
    count=1,  # clips per combination, one seed each from `seed`
    seed=42,
    seconds=5,
    size=640,
    into="tests/motion",  # the prefix of the rope folders: into_setting_loras, see folder()
    # the rest, with their defaults: negative=None (Wan's own), steps=None, cfg=None, shift=None,
    # lightning=None ("4"), lora=[], tags=[], note=None
)

# 2. The settings to vary: every Lightning setting with every shift and every fps, and every choice of the LoRAs,
#    each one in or out, the empty choice included (2 ** len(LORAS) choices).
LIGHTNING = ["4"]  # "4", the 4-step LoRAs (about 1.5 minutes a clip), or "off", the full model (about 9 minutes)
SHIFT = [3, 5, 8]  # ModelSamplingSD3's shift; the template's value is 5
FPS = [16]  # frames per second; 16 is Wan's own rate, another value changes the frame count (seconds * fps) and the speed
LORAS: list[str] = []  # name[:strength] as on the command line (.safetensors implied); applied to both models
TRIGGERS: dict[str, str] = {}  # LoRA name (no strength, no .safetensors) -> the prompt its author asks for

# run()'s keyword arguments: fresh (compute again even when ComfyUI has an identical result cached) and library (the
# rope library; default $ROPE_LIBRARY, else ~/.local/share/rope).
OPTIONS = {"fresh": False}


def choices(items: list) -> list[tuple]:
    """Every way of picking some of `items`, from none to all of them."""
    return [c for k in range(len(items) + 1) for c in itertools.combinations(items, k)]


def lora_name(lora: str | tuple[str, float]) -> str:
    """A LoRA's name as TRIGGERS keys it: "motion:0.8", "motion.safetensors" and ("motion", 0.8) are all "motion"."""
    spec = lora if isinstance(lora, str) else lora[0]
    return spec.split(":")[0].removesuffix(".safetensors")


def folder(lightning: str, shift: float, fps: int, loras: tuple[str, ...]) -> str:
    """The rope folder of a combination: into_L<lightning>_shift<shift>_fps<fps>[_loras], e.g.
    tests/motion_L4_shift5_fps16 or tests/motion_Loff_shift8_fps24_motion:0.8. BASE's own LoRAs are in every
    combination, so they are not in the name."""
    if not BASE.into:
        raise ValueError("BASE.into is needed: the folders are named from it")
    bits = [BASE.into, f"L{lightning}_shift{shift:g}_fps{fps}"]
    if loras:
        bits.append("+".join(l.replace(".safetensors", "") for l in loras))
    return "_".join(bits)


def prompt(loras: tuple[str, ...]) -> str:
    """BASE.prompt followed by the triggers of BASE's LoRAs and of `loras`, each trigger once."""
    triggers = dict.fromkeys(TRIGGERS.get(lora_name(l), "") for l in [*BASE.lora, *loras])
    return " ".join(p for p in [BASE.prompt, *triggers] if p)


def batch():
    """The requests, one per combination, each BASE with the combination's fields changed; BASE's LoRAs stay."""
    # 3. The grid.
    for lightning, shift, fps, loras in itertools.product(LIGHTNING, SHIFT, FPS, choices(LORAS)):
        yield dataclasses.replace(BASE, prompt=prompt(loras), lightning=lightning, shift=shift, fps=fps,
                                  lora=[*BASE.lora, *loras], into=folder(lightning, shift, fps, loras))


def main() -> None:
    requests = [r.settled() for r in batch()]  # a bad field is a ValueError here, before anything runs
    if "--list" in sys.argv[1:]:
        for n, r in enumerate(requests, start=1):
            loras = ", ".join(f"{name}:{strength:g}" for name, strength in r.lora) or "none"
            print(f"#{n} {r.into}: Lightning {r.lightning}, {r.steps} steps, cfg {r.cfg:g}, shift {r.shift:g}, "
                  f"{r.seconds:g} s at {r.fps} fps, {r.size} px, seed {r.seed}, LoRA {loras}; prompt {r.prompt!r}")
        folders = dict.fromkeys(r.into for r in requests)
        print(f"{len(requests)} requests, {sum(r.count for r in requests)} clips, {len(folders)} folders")
        return
    submitted = img2vid.run(requests, **OPTIONS)
    queued = [s for s in submitted if s.group is not None]
    print()
    if queued:
        groups = [s.group for s in queued]
        print(f"{len(queued)} requests queued as groups {min(groups)}–{max(groups)}, {sum(s.count for s in queued)} clips")
    if len(queued) < len(submitted):
        sys.exit(f"{len(submitted) - len(queued)} requests skipped; see above")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, img2vid.Error) as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        sys.exit(130)
