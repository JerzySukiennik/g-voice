"""Build VoxG's training binaries from a folder of recordings + transcripts.

Eventually the input is Jurek's own voice: the recorder PWA (../recorder, built
by a sibling agent) writes each take to Firebase Storage (webm/opus) with its
prompt text in Firestore. That export tooling doesn't exist yet, so this script
is written against a **simple, documented local contract** that the export will
later produce — exactly how Gedit's data/fetch_dataset.py was written and
smoke-tested against InstructPix2Pix long before any real Kaggle run.

Input contract
--------------
    <wav_dir>/                 folder of PCM WAV files (any sample rate, mono
                               or stereo — resampled/downmixed here)
    <manifest>                 JSON or CSV mapping filename -> transcript

    JSON:  [{"audio": "0001.wav", "text": "Cześć, jestem Jurek."}, ...]
           optionally with "durations": [ints] (real forced-alignment) —
           see the "durations" note below.
    CSV:   one "filename|transcript" per line (pipe-separated; '|' avoids
           colliding with commas in Polish text).

Real recordings arrive as webm/opus. Transcode them to WAV first, e.g.
    ffmpeg -i take.webm -ac 1 -ar 22050 take.wav
(the resampler here will handle a different input rate too, but ffmpeg's is
better — see model/audio.resample_linear).

Output (mirrors MicroG's pl_train.bin / Gedit's *_images.bin: raw mmap-able
binaries + a JSON index, so Kaggle-side loading is a slice, not a re-parse):
    <out>_audio.bin    float32  all waveforms concatenated (for the vocoder)
    <out>_mel.bin      float32  all log-mels concatenated, row = frame (acoustic)
    <out>_phon.bin     int32    phoneme ids, per utt incl. BOS/EOS
    <out>_dur.bin      int32    per-phoneme durations in frames (sum == #frames)
    <out>_pitch.bin    float32  per-phoneme normalised log-F0 (variance adaptor)
    <out>_energy.bin   float32  per-phoneme normalised energy
    <out>_meta.json    per-utterance offsets + global stats/params
    <out>_phonemes.json  the PhonemeVocab

Durations
---------
FastSpeech2 needs per-phoneme durations to train. The gold source is forced
alignment (Montreal Forced Aligner): pass them per utterance via the manifest's
"durations" field and they are used verbatim. When absent, this script falls
back to a **uniform placeholder** (mel frames split evenly across phonemes) so
the pipeline is runnable today — but that yields robotic, evenly-timed speech
and is NOT good enough for a real voice. Getting real alignments is the single
biggest thing standing between this pipeline and good output (see README /
kaggle/README.md). The placeholder is loud about itself in the logs.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.audio import AudioConfig, MelSpectrogram, load_wav, resample_linear
from model.g2p import text_to_phonemes
from model.symbols import PhonemeVocab

import torch


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------

def read_manifest(path: str):
    """-> list of dicts {audio, text, durations?}."""
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        return [{"audio": r["audio"], "text": r["text"],
                 "durations": r.get("durations")} for r in rows]
    # CSV: filename|transcript
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in csv.reader(f, delimiter="|"):
            if len(line) >= 2 and line[0].strip():
                rows.append({"audio": line[0].strip(),
                             "text": "|".join(line[1:]).strip(),
                             "durations": None})
    return rows


# ---------------------------------------------------------------------------
# Per-phoneme variance targets (pitch / energy)
# ---------------------------------------------------------------------------

def frame_energy(mel: np.ndarray) -> np.ndarray:
    """Per-frame energy = mean over mel bins of the (linear) magnitude.

    mel is log-magnitude; exp brings it back to magnitude before averaging.
    Shape (frames,). Cheap, robust, no external deps.
    """
    return np.exp(mel).mean(axis=1)


def estimate_f0(wav: np.ndarray, cfg: AudioConfig,
                fmin=70.0, fmax=400.0, n_frames=None) -> np.ndarray:
    """Per-frame fundamental frequency (Hz) by autocorrelation, 0 where unvoiced.

    Hand-rolled and intentionally basic — autocorrelation F0 is the textbook
    method and dependency-free, but noisier than pyworld/CREPE. Good enough to
    give the pitch head a target; flagged as the thing to upgrade if pitch
    modelling is a quality ceiling. Framing matches the mel (center-padded, same
    hop) so pitch[f] lines up with mel frame f.
    """
    hop, win = cfg.hop_length, cfg.win_length
    pad = win // 2
    padded = np.pad(wav, (pad, pad), mode="reflect")
    if n_frames is None:
        n_frames = 1 + len(wav) // hop
    min_lag = int(cfg.sample_rate / fmax)
    max_lag = int(cfg.sample_rate / fmin)

    f0 = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = padded[start:start + win]
        if len(frame) < win:
            break
        frame = frame - frame.mean()
        e0 = np.dot(frame, frame)
        if e0 < 1e-6:                      # silence -> unvoiced
            continue
        corr = np.correlate(frame, frame, mode="full")[win - 1:]  # lags >= 0
        seg = corr[min_lag:max_lag + 1]
        if len(seg) == 0:
            continue
        lag = min_lag + int(np.argmax(seg))
        # Voicing gate: peak autocorrelation must be a decent fraction of energy.
        if corr[lag] / e0 > 0.3:
            f0[i] = cfg.sample_rate / lag
    return f0


def phoneme_means(frame_values: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """Average a per-frame quantity within each phoneme's frame span.

    durations sum to (about) the number of frames; walk the span for each
    phoneme and take the mean of the (voiced, for F0) frames inside it.
    """
    out = np.zeros(len(durations), dtype=np.float32)
    pos = 0
    for i, d in enumerate(durations):
        d = int(d)
        if d <= 0:
            continue
        span = frame_values[pos:pos + d]
        pos += d
        voiced = span[span > 0]
        out[i] = voiced.mean() if len(voiced) else (span.mean() if len(span) else 0.0)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(args):
    cfg = AudioConfig()
    melspec = MelSpectrogram(cfg)
    rows = read_manifest(args.manifest)
    print(f"{len(rows)} utterances in manifest")

    # ---- pass 1: G2P everything, build the phoneme vocab -------------------
    phoneme_seqs = []
    kept = []
    used_fallback = False
    for r in rows:
        try:
            phones = text_to_phonemes(r["text"], allow_fallback=args.allow_g2p_fallback)
        except Exception as e:
            print(f"  G2P failed on {r['audio']!r}: {type(e).__name__}: {e}")
            raise
        if not phones:
            print(f"  skip {r['audio']} — empty phoneme sequence")
            continue
        phoneme_seqs.append(phones)
        kept.append(r)
    if args.allow_g2p_fallback:
        from model.g2p import espeak_available
        if not espeak_available():
            used_fallback = True
            print("  !! WARNING: using TEST-ONLY char fallback G2P — phonemes are "
                  "not real Polish phonology. Install espeak-ng for real data.")

    vocab = PhonemeVocab.build(phoneme_seqs)
    print(f"phoneme vocab: {len(vocab)} symbols ({len(vocab.symbols)} + 4 reserved)")

    # ---- pass 2: audio -> mel/dur/pitch/energy, pack -----------------------
    audio_f = open(f"{args.out}_audio.bin", "wb")
    mel_f = open(f"{args.out}_mel.bin", "wb")
    phon_f = open(f"{args.out}_phon.bin", "wb")
    dur_f = open(f"{args.out}_dur.bin", "wb")
    pitch_f = open(f"{args.out}_pitch.bin", "wb")
    energy_f = open(f"{args.out}_energy.bin", "wb")

    index = []
    audio_off = mel_off = phon_off = 0
    all_pitch, all_energy = [], []   # raw (unnormalised) for global stats
    placeholder_dur_count = 0

    for r, phones in zip(kept, phoneme_seqs):
        wav, sr = load_wav(os.path.join(args.wav_dir, r["audio"]))
        wav = resample_linear(wav, sr, cfg.sample_rate)
        mel = melspec(torch.from_numpy(wav)).squeeze(0).transpose(0, 1).numpy()  # (frames, n_mels)
        n_frames = mel.shape[0]

        ids = np.array(vocab.encode(phones, add_bos_eos=True), dtype=np.int32)
        n_phon = len(ids)

        # durations
        if r.get("durations") is not None:
            dur = np.array(r["durations"], dtype=np.int32)
            if len(dur) != n_phon:
                raise ValueError(
                    f"{r['audio']}: manifest gives {len(dur)} durations but "
                    f"{n_phon} phonemes (incl. BOS/EOS). They must match.")
            # snap the sum to the true frame count so length regulation lands.
            drift = n_frames - int(dur.sum())
            if drift != 0:
                dur[dur.argmax()] += drift
        else:
            # uniform placeholder — see module docstring
            dur = np.full(n_phon, n_frames // n_phon, dtype=np.int32)
            dur[: n_frames - dur.sum()] += 1     # spread the remainder
            placeholder_dur_count += 1
        dur = np.clip(dur, 0, None).astype(np.int32)

        # variance targets, aligned to durations
        energy = phoneme_means(frame_energy(mel), dur)
        f0 = estimate_f0(wav, cfg, n_frames=n_frames)
        logf0 = np.where(f0 > 0, np.log(f0 + 1e-5), 0.0)
        pitch = phoneme_means(logf0, dur)
        all_pitch.append(pitch)
        all_energy.append(energy)

        # write streams
        wav.astype(np.float32).tofile(audio_f)
        mel.astype(np.float32).tofile(mel_f)
        ids.tofile(phon_f)
        dur.tofile(dur_f)
        pitch.astype(np.float32).tofile(pitch_f)
        energy.astype(np.float32).tofile(energy_f)

        index.append({
            "audio": r["audio"],
            "audio_off": audio_off, "audio_len": len(wav),
            "mel_off": mel_off, "mel_frames": n_frames,
            "phon_off": phon_off, "phon_len": n_phon,
        })
        audio_off += len(wav)
        mel_off += n_frames
        phon_off += n_phon

    for f in (audio_f, mel_f, phon_f, dur_f, pitch_f, energy_f):
        f.close()

    # ---- normalise pitch/energy (store stats, rewrite normalised bins) -----
    pitch_cat = np.concatenate(all_pitch) if all_pitch else np.zeros(1, np.float32)
    energy_cat = np.concatenate(all_energy) if all_energy else np.zeros(1, np.float32)
    # stats over *nonzero* pitch (unvoiced phonemes are 0 and shouldn't skew it)
    voiced = pitch_cat[pitch_cat > 0]
    p_mean = float(voiced.mean()) if len(voiced) else 0.0
    p_std = float(voiced.std()) if len(voiced) else 1.0
    e_mean, e_std = float(energy_cat.mean()), float(energy_cat.std() or 1.0)

    # rewrite normalised versions in place (zeros stay zero = "unvoiced/silent")
    def _normalise(path, mean, std):
        arr = np.fromfile(path, dtype=np.float32)
        norm = np.where(arr != 0, (arr - mean) / (std or 1.0), 0.0).astype(np.float32)
        norm.tofile(path)
    _normalise(f"{args.out}_pitch.bin", p_mean, p_std)
    _normalise(f"{args.out}_energy.bin", e_mean, e_std)

    meta = {
        "n_utts": len(index),
        "sample_rate": cfg.sample_rate, "n_mels": cfg.n_mels,
        "n_fft": cfg.n_fft, "hop_length": cfg.hop_length, "win_length": cfg.win_length,
        "n_phonemes": len(vocab),
        "pitch_mean": p_mean, "pitch_std": p_std,
        "energy_mean": e_mean, "energy_std": e_std,
        "placeholder_durations": placeholder_dur_count,
        "g2p_fallback_used": used_fallback,
        "val_n": min(args.val_n, max(len(index) // 10, 1)),
        "index": index,
    }
    with open(f"{args.out}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    vocab.save(f"{args.out}_phonemes.json")

    total_frames = mel_off
    print(f"\ndone — {len(index)} utts, {total_frames} mel frames "
          f"(~{total_frames * cfg.hop_length / cfg.sample_rate / 60:.1f} min audio)")
    if placeholder_dur_count:
        print(f"  {placeholder_dur_count}/{len(index)} utts used PLACEHOLDER "
              f"uniform durations — supply forced-alignment for real training.")
    print(f"  pitch  mean/std (log-F0): {p_mean:.3f} / {p_std:.3f}")
    print(f"  energy mean/std:          {e_mean:.3f} / {e_std:.3f}")
    print(f"  wrote {args.out}_*.bin + {args.out}_meta.json + {args.out}_phonemes.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wav-dir", required=True, help="folder of PCM WAV files")
    p.add_argument("--manifest", required=True, help="JSON list or pipe-CSV filename|text")
    p.add_argument("--out", default="./voxg", help="output prefix")
    p.add_argument("--val-n", type=int, default=64, help="max held-out utterances")
    p.add_argument("--allow-g2p-fallback", action="store_true",
                   help="TEST ONLY: use char fallback if espeak-ng is missing")
    build(p.parse_args())
