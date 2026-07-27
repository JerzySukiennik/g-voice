"""Kaggle cell 1 of 3 — build the training binaries from Jurek's recordings.

Run this ONCE. It clones the repo, installs espeak-ng + phonemizer, runs
data/prepare_dataset.py over the recordings you attached as a Kaggle Dataset,
and leaves g-voice_*.bin + g-voice_meta.json under /kaggle/working — which you then
save as a Dataset and feed to the two training notebooks.

Unlike G-Micro/Gedit (which download a public corpus on Kaggle's fast link),
G-Voice's data is YOUR voice — there's no public source to stream. So the raw
recordings do have to reach Kaggle once, as an uploaded Dataset. Everything
heavy after that (G2P, mel extraction, F0) happens here on Kaggle, not at home.

Settings: Accelerator None (CPU is enough), Internet ON.
Inputs (Add Input): your recordings Dataset — see kaggle/README.md for its
expected layout (a wavs/ folder + manifest.json).
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-voice.git"
WORK = "/kaggle/working"
OUT_PREFIX = f"{WORK}/g-voice"

# --- code -------------------------------------------------------------------
if os.path.exists(f"{WORK}/g-voice"):
    subprocess.run(["git", "-C", f"{WORK}/g-voice", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-voice"], check=True)
os.chdir(f"{WORK}/g-voice")

# espeak-ng is a *system* package (apt), phonemizer is pip. On Kaggle apt works.
subprocess.run(["apt-get", "-qq", "install", "-y", "espeak-ng"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "phonemizer"], check=True)

# --- already built? ---------------------------------------------------------
# Kaggle batch runs are all-or-nothing; a re-run should cost seconds, not redo
# the whole prep (same guard as G-Micro's 01-prep).
if os.path.exists(f"{OUT_PREFIX}_meta.json") and os.path.getsize(f"{OUT_PREFIX}_mel.bin") > 0:
    print(f"{OUT_PREFIX}_* already built — nothing to do")
    sys.exit(0)

# --- find the attached recordings -------------------------------------------
# Kaggle's input mount depth isn't fixed — search recursively for the manifest
# (same lesson as G-Micro's recursive glob for pl_train.bin).
manifests = (glob.glob("/kaggle/input/**/manifest.json", recursive=True)
             + glob.glob("/kaggle/input/**/manifest.csv", recursive=True))
if not manifests:
    print("No manifest.json/.csv found under /kaggle/input. Tree:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count("/") - 2
        if depth > 3:
            continue
        print("  " * depth + os.path.basename(root) + "/")
        for f in sorted(files)[:12]:
            print("  " * (depth + 1) + f)
    raise SystemExit("attach your recordings Dataset (wavs/ + manifest.json) as an Input")

manifest = manifests[0]
data_root = os.path.dirname(manifest)
# wavs/ next to the manifest, or the same folder
wav_dir = os.path.join(data_root, "wavs")
if not os.path.isdir(wav_dir):
    wav_dir = data_root
print(f"manifest: {manifest}\nwav_dir:  {wav_dir}")

# --- prepare ----------------------------------------------------------------
# NOTE: no --allow-g2p-fallback here — real training MUST use real espeak-ng
# phonemes. If espeak-ng failed to install above, prepare_dataset will raise a
# clear error rather than silently produce garbage phonemes.
subprocess.run([sys.executable, "data/prepare_dataset.py",
                "--wav-dir", wav_dir, "--manifest", manifest,
                "--out", OUT_PREFIX], check=True)

print("\ndone — save this notebook's output as a Dataset named 'g-voice-data'")
for f in sorted(os.listdir(WORK)):
    p = f"{WORK}/{f}"
    if os.path.isfile(p) and f.startswith("g-voice"):
        print(f"  {f}  {os.path.getsize(p)/1e6:.1f} MB")
