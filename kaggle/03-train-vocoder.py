"""Kaggle cell 3 of 3 — train the vocoder (mel -> waveform), HiFi-GAN style.

Settings: Accelerator GPU T4 x2, Internet ON, Persistence "Variables and Files".
Inputs: 'voxg-data' from 01-prep.py, plus — on any resume run — the previous
output as 'voxg-vocoder-ckpt'.

Trains INDEPENDENTLY of the acoustic model (standard TTS practice — see README).
It only needs the audio + mels from voxg-data, not the acoustic checkpoint. The
checkpoint carries both nets and both optimisers so a resume restores the full
GAN state (resuming only the generator would let the discriminators forget and
the loss would jump). Same 12h-resumable scaffolding as the acoustic cell.
"""

import glob
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/voxg.git"
WORK = "/kaggle/working"
OUT = f"{WORK}/run_vocoder"

# GAN vocoders need many steps; HiFi-GAN trains for ~hundreds of k. STEPS is a
# ceiling — stop when synthesised audio sounds clean. Segment length 32 mel
# frames = 8192 audio samples per crop.
BATCH, STEPS, SEGMENT = 16, 200000, 32

if os.path.exists(f"{WORK}/voxg"):
    subprocess.run(["git", "-C", f"{WORK}/voxg", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/voxg"], check=True)
os.chdir(f"{WORK}/voxg")

hits = glob.glob("/kaggle/input/**/voxg_meta.json", recursive=True)
if not hits:
    raise SystemExit("attach the voxg-data dataset (voxg_meta.json not found)")
data_prefix = hits[0][: -len("_meta.json")]
print(f"data prefix: {data_prefix}")

os.makedirs(OUT, exist_ok=True)
ckpts = sorted(glob.glob("/kaggle/input/**/run_vocoder/ckpt.pt", recursive=True)
               or glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
resume = []
if ckpts:
    shutil.copy(ckpts[0], f"{OUT}/ckpt.pt")
    resume = ["--resume"]
    print(f"resuming from {ckpts[0]}")
else:
    print("no checkpoint attached — starting from scratch")

cmd = [sys.executable, "train/train_vocoder.py",
       "--data", data_prefix, "--out", OUT,
       "--batch-size", str(BATCH), "--segment-frames", str(SEGMENT),
       "--max-steps", str(STEPS), "--log-every", "50", "--ckpt-every", "1000"] + resume
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

print("\nsave this notebook's output as a Dataset ('voxg-vocoder-ckpt') to "
      "continue next session, or download run_vocoder/ckpt.pt when done.")
