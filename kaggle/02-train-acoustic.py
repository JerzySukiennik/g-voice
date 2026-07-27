"""Kaggle cell 2 of 3 — train the acoustic model (phonemes -> mel).

Settings: Accelerator GPU T4 x2, Internet ON, Persistence "Variables and Files".
Inputs: the 'g-voice-data' Dataset from 01-prep.py, plus — on any resume run — the
previous output as 'g-voice-acoustic-ckpt'.

Built to be interrupted: a Kaggle session is capped at 12h and can die sooner,
so everything needed to continue lands in /kaggle/working/run_acoustic every
CKPT_EVERY steps. Add that output as an input to the next session and it resumes
mid-stride (same pattern as G-Micro/Gedit).
"""

import glob
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-voice.git"
WORK = "/kaggle/working"
OUT = f"{WORK}/run_acoustic"

# STEPS is a CEILING, not a target — the checkpoint saves every CKPT_EVERY steps
# and you can stop whenever samples sound good. Duration for a real dataset is
# unknown until the first on-GPU measurement; this is a starting point.
BATCH, ACCUM, STEPS, WARMUP = 16, 1, 30000, 500

if os.path.exists(f"{WORK}/g-voice"):
    subprocess.run(["git", "-C", f"{WORK}/g-voice", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-voice"], check=True)
os.chdir(f"{WORK}/g-voice")

# --- data -------------------------------------------------------------------
hits = glob.glob("/kaggle/input/**/g-voice_meta.json", recursive=True)
if not hits:
    print("g-voice_meta.json not found. /kaggle/input contains:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count("/") - 2
        if depth > 3:
            continue
        print("  " * depth + os.path.basename(root) + "/")
        for f in sorted(files)[:12]:
            print("  " * (depth + 1) + f)
    raise SystemExit("attach the g-voice-data dataset")
data_prefix = hits[0][: -len("_meta.json")]
print(f"data prefix: {data_prefix}")

# --- resume if a checkpoint is attached -------------------------------------
os.makedirs(OUT, exist_ok=True)
ckpts = sorted(glob.glob("/kaggle/input/**/run_acoustic/ckpt.pt", recursive=True)
               or glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
resume = []
if ckpts:
    shutil.copy(ckpts[0], f"{OUT}/ckpt.pt")
    resume = ["--resume"]
    print(f"resuming from {ckpts[0]}")
else:
    print("no checkpoint attached — starting from scratch")

cmd = [sys.executable, "train/train_acoustic.py",
       "--data", data_prefix, "--out", OUT,
       "--batch-size", str(BATCH), "--grad-accum", str(ACCUM),
       "--max-steps", str(STEPS), "--warmup", str(WARMUP),
       "--log-every", "20", "--eval-every", "200", "--ckpt-every", "200"] + resume
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

print("\nsave this notebook's output as a Dataset ('g-voice-acoustic-ckpt') to "
      "continue next session, or download run_acoustic/ckpt.pt when done.")
