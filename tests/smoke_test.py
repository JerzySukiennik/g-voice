"""End-to-end smoke test for the VoxG pipeline on synthetic data.

This is the cross-module test the individual `if __name__ == "__main__"` blocks
can't cover: it drives the *real* prepare_dataset + train scripts (via
subprocess, exactly as Kaggle runs them) over a handful of synthetic WAVs, and
checks the thing that actually matters for a 12h-capped Kaggle run — that
training checkpoints and then **resumes on a continuous loss curve** instead of
silently restarting from random init.

Same validation philosophy as the siblings: prove the plumbing on fake data
before spending GPU-hours on real data (this caught bugs in both MicroG and
Gedit before Kaggle ever ran).

Run:  .venv/bin/python tests/smoke_test.py
Needs only torch + numpy. No espeak-ng: G2P runs in --allow-g2p-fallback mode.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from model.audio import AudioConfig, save_wav

PY = sys.executable
TEXTS = [
    "Cześć, jestem Jurek.",
    "Dzisiaj budujemy model.",
    "Rakieta leci w kosmos.",
    "Gzowo Space Program startuje.",
    "Minimalizm to filozofia.",
    "Głos brzmi coraz lepiej.",
    "Tesla jest minimalistyczna.",
    "Sztuczna inteligencja uczy się.",
    "Fortepian gra piękną melodię.",
    "Druk trójwymiarowy działa.",
    "Model mówi po polsku.",
    "Testujemy cały potok danych.",
]


def make_synthetic_dataset(root):
    """Write voiced-ish synthetic WAVs (summed harmonics -> a real F0 so the
    pitch estimator has something to find) + a JSON manifest.
    """
    cfg = AudioConfig()
    wav_dir = os.path.join(root, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    manifest = []
    rng = np.random.default_rng(0)
    for i, text in enumerate(TEXTS):
        dur_s = 0.8 + 0.4 * rng.random()          # 0.8-1.2 s
        t = np.arange(int(dur_s * cfg.sample_rate)) / cfg.sample_rate
        f0 = 110 + 40 * rng.random()              # a plausible male F0
        sig = sum(0.4 / h * np.sin(2 * np.pi * f0 * h * t) for h in (1, 2, 3))
        sig = sig * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))   # slow amplitude wobble
        sig += 0.01 * rng.standard_normal(len(t))
        sig = (sig / np.abs(sig).max() * 0.9).astype(np.float32)
        name = f"{i:04d}.wav"
        save_wav(os.path.join(wav_dir, name), sig, cfg.sample_rate)
        manifest.append({"audio": name, "text": text})
    mpath = os.path.join(root, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    return wav_dir, mpath


def run(cmd, **kw):
    print("  $ " + " ".join(str(c) for c in cmd[1:]))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, **kw)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return r.stdout


def parse_losses(stdout, key="loss"):
    """Pull `<key> <float>` pairs from training stdout, in order."""
    out = []
    for m in re.finditer(rf"\b{key}\s+([0-9]*\.?[0-9]+)", stdout):
        out.append(float(m.group(1)))
    return out


def main():
    root = tempfile.mkdtemp(prefix="voxg_smoke_")
    print(f"workdir: {root}")
    ok = True
    try:
        # ---- 1. synthetic data + prepare_dataset --------------------------
        print("\n[1] prepare_dataset on synthetic WAVs")
        wav_dir, manifest = make_synthetic_dataset(root)
        prefix = os.path.join(root, "voxg")
        out = run([PY, "data/prepare_dataset.py", "--wav-dir", wav_dir,
                   "--manifest", manifest, "--out", prefix,
                   "--allow-g2p-fallback", "--val-n", "2"])
        print("   " + out.strip().replace("\n", "\n   "))
        meta = json.load(open(f"{prefix}_meta.json"))
        assert meta["n_utts"] == len(TEXTS), f"expected {len(TEXTS)} utts, got {meta['n_utts']}"
        for suffix in ("audio", "mel", "phon", "dur", "pitch", "energy"):
            p = f"{prefix}_{suffix}.bin"
            assert os.path.getsize(p) > 0, f"{p} is empty"
        print(f"   OK — {meta['n_utts']} utts, {meta['n_phonemes']} phonemes, "
              f"placeholder_dur={meta['placeholder_durations']}")

        # ---- 2. acoustic train -> ckpt -> resume --------------------------
        print("\n[2] acoustic: train 10 steps, resume to 20, check continuity")
        adir = os.path.join(root, "acoustic")
        base = [PY, "train/train_acoustic.py", "--data", prefix, "--out", adir,
                "--batch-size", "2", "--workers", "0", "--warmup", "3",
                "--log-every", "2", "--eval-every", "1000", "--ckpt-every", "10"]
        s1 = run(base + ["--max-steps", "10"])
        l1 = parse_losses(s1)
        s2 = run(base + ["--max-steps", "20", "--resume"])
        assert "resumed from step 10" in s2, "acoustic resume did not load the checkpoint!"
        l2 = parse_losses(s2)
        assert l1 and l2, "no acoustic losses logged"
        assert all(np.isfinite(l1 + l2)), "acoustic loss went NaN/inf"
        last_before, first_after = l1[-1], l2[0]
        # Continuity: resumed loss must be near where we left off, not back at
        # random init (which for L1 mel + variance losses starts far higher).
        continuous = first_after < last_before * 1.5 + 0.5
        print(f"   acoustic loss: start {l1[0]:.3f} -> step10 {last_before:.3f} "
              f"-> resume step12 {first_after:.3f} -> end {l2[-1]:.3f}")
        print(f"   continuity {'OK' if continuous else 'FAIL'} "
              f"(resumed {first_after:.3f} vs left-off {last_before:.3f})")
        ok &= continuous
        assert os.path.exists(f"{adir}/ckpt.pt")

        # ---- 3. vocoder train -> ckpt -> resume ---------------------------
        print("\n[3] vocoder: train 10 steps, resume to 20, check GAN loss sanity")
        vdir = os.path.join(root, "vocoder")
        vbase = [PY, "train/train_vocoder.py", "--data", prefix, "--out", vdir,
                 "--batch-size", "2", "--workers", "0", "--segment-frames", "16",
                 "--log-every", "2", "--ckpt-every", "10"]
        v1 = run(vbase + ["--max-steps", "10"])
        v2 = run(vbase + ["--max-steps", "20", "--resume"])
        assert "resumed from step 10" in v2, "vocoder resume did not load the checkpoint!"
        d1, d2 = parse_losses(v1, "d"), parse_losses(v2, "d")
        mel1 = parse_losses(v1, "mel")
        assert d1 and d2 and mel1, "no vocoder losses logged"
        assert all(np.isfinite(d1 + d2 + mel1)), "vocoder loss went NaN/inf"
        print(f"   vocoder d-loss: {d1[0]:.3f} -> {d1[-1]:.3f} (resume {d2[0]:.3f})  "
              f"mel-recon: {mel1[0]:.3f} -> {mel1[-1]:.3f}")
        assert os.path.exists(f"{vdir}/ckpt.pt")

        print(f"\n{'='*60}")
        print("SMOKE TEST PASSED" if ok else "SMOKE TEST: pipeline runs but a continuity check was soft-FAIL")
        print("="*60)
        return 0 if ok else 1
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
