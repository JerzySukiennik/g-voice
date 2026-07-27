"""Dataset loaders over the binaries built by prepare_dataset.py.

Two views on the same packed files, one per training stage:

- AcousticDataset yields whole utterances (phonemes, durations, pitch, energy,
  mel) with a padding collate — the acoustic model consumes a full sequence.
- VocoderDataset yields fixed-length (mel_segment, wav_segment) crops — HiFi-GAN
  trains on short random windows, never whole clips, so batches are uniform and
  a long recording doesn't blow up memory.

Everything is mmap'd (np.memmap), same reason G-Micro/Gedit mmap their bins: the
OS pages in only the slice a batch touches, so start-up is instant and RAM stays
flat no matter how many hours of audio pile up.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class _PackedData:
    """Shared handle on the packed binaries + meta/index."""

    def __init__(self, prefix: str):
        with open(f"{prefix}_meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.index = self.meta["index"]
        self.n_mels = self.meta["n_mels"]
        self.hop = self.meta["hop_length"]

        self.audio = np.memmap(f"{prefix}_audio.bin", dtype=np.float32, mode="r")
        self.mel = np.memmap(f"{prefix}_mel.bin", dtype=np.float32, mode="r")
        self.phon = np.memmap(f"{prefix}_phon.bin", dtype=np.int32, mode="r")
        self.dur = np.memmap(f"{prefix}_dur.bin", dtype=np.int32, mode="r")
        self.pitch = np.memmap(f"{prefix}_pitch.bin", dtype=np.float32, mode="r")
        self.energy = np.memmap(f"{prefix}_energy.bin", dtype=np.float32, mode="r")

    def mel_of(self, e):
        return np.asarray(self.mel[e["mel_off"] * self.n_mels:
                                   (e["mel_off"] + e["mel_frames"]) * self.n_mels]
                          ).reshape(e["mel_frames"], self.n_mels)

    def audio_of(self, e):
        return np.asarray(self.audio[e["audio_off"]: e["audio_off"] + e["audio_len"]])

    def phon_of(self, e):
        s, n = e["phon_off"], e["phon_len"]
        return (np.asarray(self.phon[s:s + n]),
                np.asarray(self.dur[s:s + n]),
                np.asarray(self.pitch[s:s + n]),
                np.asarray(self.energy[s:s + n]))

    def split_indices(self, split):
        n = len(self.index)
        val_n = self.meta.get("val_n", 0)
        return range(0, n - val_n) if split == "train" else range(n - val_n, n)


# ---------------------------------------------------------------------------
# Acoustic
# ---------------------------------------------------------------------------

class AcousticDataset(Dataset):
    def __init__(self, prefix: str, split: str = "train"):
        self.d = _PackedData(prefix)
        self.ids = list(self.d.split_indices(split))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        e = self.d.index[self.ids[i]]
        phon, dur, pitch, energy = self.d.phon_of(e)
        mel = self.d.mel_of(e)
        return {
            "phon": torch.from_numpy(phon.copy()).long(),
            "dur": torch.from_numpy(dur.copy()).long(),
            "pitch": torch.from_numpy(pitch.copy()).float(),
            "energy": torch.from_numpy(energy.copy()).float(),
            "mel": torch.from_numpy(mel.copy()).float(),
        }


def acoustic_collate(batch, pad_id: int = 0):
    """Pad phonemes and mels to the batch max; return lengths for masking."""
    B = len(batch)
    T_phon = max(b["phon"].size(0) for b in batch)
    T_mel = max(b["mel"].size(0) for b in batch)
    n_mels = batch[0]["mel"].size(1)

    phon = torch.full((B, T_phon), pad_id, dtype=torch.long)
    dur = torch.zeros(B, T_phon, dtype=torch.long)
    pitch = torch.zeros(B, T_phon)
    energy = torch.zeros(B, T_phon)
    mel = torch.zeros(B, T_mel, n_mels)
    phon_lengths = torch.zeros(B, dtype=torch.long)
    mel_lengths = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        p, m = b["phon"].size(0), b["mel"].size(0)
        phon[i, :p] = b["phon"]
        dur[i, :p] = b["dur"]
        pitch[i, :p] = b["pitch"]
        energy[i, :p] = b["energy"]
        mel[i, :m] = b["mel"]
        phon_lengths[i] = p
        mel_lengths[i] = m
    return {"phon": phon, "dur": dur, "pitch": pitch, "energy": energy,
            "mel": mel, "phon_lengths": phon_lengths, "mel_lengths": mel_lengths}


# ---------------------------------------------------------------------------
# Vocoder
# ---------------------------------------------------------------------------

class VocoderDataset(Dataset):
    """Random fixed-length crops of (mel, waveform). segment_frames mel frames
    correspond to segment_frames * hop audio samples.
    """

    def __init__(self, prefix: str, split: str = "train", segment_frames: int = 32):
        self.d = _PackedData(prefix)
        self.hop = self.d.hop
        self.seg_frames = segment_frames
        # Only utterances at least one segment long are usable.
        self.ids = [i for i in self.d.split_indices(split)
                    if self.d.index[i]["mel_frames"] >= segment_frames]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        e = self.d.index[self.ids[i]]
        mel = self.d.mel_of(e)                 # (frames, n_mels)
        wav = self.d.audio_of(e)
        n = e["mel_frames"]
        start = np.random.randint(0, n - self.seg_frames + 1)
        mel_seg = mel[start:start + self.seg_frames]                     # (seg, n_mels)
        a0 = start * self.hop
        wav_seg = wav[a0: a0 + self.seg_frames * self.hop]
        # Pad the audio crop if the clip's tail is short (center-padded STFT can
        # leave the last frames without a full hop of samples).
        need = self.seg_frames * self.hop
        if len(wav_seg) < need:
            wav_seg = np.pad(wav_seg, (0, need - len(wav_seg)))
        return (torch.from_numpy(mel_seg.copy()).float().transpose(0, 1),  # (n_mels, seg)
                torch.from_numpy(wav_seg.copy()).float().unsqueeze(0))      # (1, seg*hop)


if __name__ == "__main__":
    # Requires a packed dataset; the end-to-end tests/smoke_test.py builds a
    # tiny synthetic one and drives these. Here we just report if a prefix given.
    import sys
    if len(sys.argv) > 1:
        pre = sys.argv[1]
        ad = AcousticDataset(pre, "train")
        vd = VocoderDataset(pre, "train")
        print(f"acoustic train items: {len(ad)}  vocoder train crops: {len(vd)}")
        if len(ad):
            b = acoustic_collate([ad[0], ad[min(1, len(ad)-1)]])
            print(f"acoustic batch: phon {tuple(b['phon'].shape)} mel {tuple(b['mel'].shape)}")
        if len(vd):
            mseg, wseg = vd[0]
            print(f"vocoder crop: mel {tuple(mseg.shape)} wav {tuple(wseg.shape)}")
    else:
        print("pass a dataset prefix to exercise; see tests/smoke_test.py")
