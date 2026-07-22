"""Audio front-end for VoxG — waveform <-> mel-spectrogram.

Both models in the family precompute their conditioning once and mmap it (see
MicroG's pl_train.bin, Gedit's *_images.bin). Here the mel-spectrogram is that
conditioning: the acoustic model *predicts* mels, the vocoder *consumes* them,
so the exact same transform has to live in one place or the two nets end up
speaking slightly different dialects and never line up.

Everything is hand-rolled on top of `torch.stft` plus a numpy mel filterbank —
no librosa/torchaudio dependency. That keeps the smoke tests runnable with just
torch+numpy (the sibling venvs have nothing else) and keeps the maths legible,
which is the whole point of the family (see README).

STFT / mel parameters (standard 22.05 kHz TTS defaults, fixed here so the whole
pipeline agrees):
    sample_rate = 22050
    n_fft       = 1024
    hop_length  = 256      -> 256/22050 = 11.6 ms per frame, ~86 frames/sec
    win_length  = 1024
    n_mels      = 80
    fmin, fmax  = 0, 8000  Hz
    log floor   = 1e-5  (clamp before log so silence doesn't become -inf)
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    log_floor: float = 1e-5


# ---------------------------------------------------------------------------
# Mel filterbank (HTK formula, built by hand)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz):
    # HTK mel scale — the same one espeak-era TTS stacks assume.
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(cfg: AudioConfig) -> np.ndarray:
    """Triangular mel filters, shape (n_mels, n_fft//2 + 1).

    Each output mel bin is a triangle over the linear-frequency FFT bins,
    peaking at its centre frequency and falling to zero at its neighbours'
    centres. Building this by hand (rather than importing librosa.filters.mel)
    is deliberate: it is ~15 lines and makes the frequency warping explicit.
    """
    n_freqs = cfg.n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, cfg.sample_rate / 2.0, n_freqs)

    # n_mels+2 equally-spaced points on the mel scale -> centre + both edges.
    mel_min, mel_max = _hz_to_mel(cfg.fmin), _hz_to_mel(cfg.fmax)
    mel_points = np.linspace(mel_min, mel_max, cfg.n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    fb = np.zeros((cfg.n_mels, n_freqs), dtype=np.float32)
    for m in range(cfg.n_mels):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        # rising edge left..center, falling edge center..right
        rising = (fft_freqs - left) / max(center - left, 1e-8)
        falling = (right - fft_freqs) / max(right - center, 1e-8)
        fb[m] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


# ---------------------------------------------------------------------------
# Mel spectrogram
# ---------------------------------------------------------------------------

class MelSpectrogram:
    """Callable waveform -> log-mel. Caches the window and filterbank so a
    training loop that calls it every step pays for them once.
    """

    def __init__(self, cfg: AudioConfig = AudioConfig(), device="cpu"):
        self.cfg = cfg
        self.device = device
        self.window = torch.hann_window(cfg.win_length, device=device)
        self.fb = torch.from_numpy(mel_filterbank(cfg)).to(device)  # (n_mels, n_freqs)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) or (T,) float in [-1, 1]. Returns (B, n_mels, frames).

        A mono (T,) tensor is treated as a single-item batch and returned with
        the batch axis kept — callers that want (n_mels, frames) squeeze it.
        """
        cfg = self.cfg
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(self.device)

        spec = torch.stft(
            wav, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            win_length=cfg.win_length, window=self.window,
            center=True, pad_mode="reflect", normalized=False,
            return_complex=True,
        )                                   # (B, n_freqs, frames)
        mag = spec.abs()                    # magnitude, not power
        mel = torch.matmul(self.fb, mag)    # (B, n_mels, frames)
        mel = torch.clamp(mel, min=cfg.log_floor).log()
        return mel


# ---------------------------------------------------------------------------
# WAV I/O without a soundfile dependency
# ---------------------------------------------------------------------------

def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a PCM WAV into a float32 mono array in [-1, 1] + its sample rate.

    Uses the stdlib `wave` module so the pipeline has no soundfile/librosa
    dependency. Real recordings arrive as webm/opus from Firebase — those must
    be transcoded to WAV first (documented in data/prepare_dataset.py); this
    reader deliberately only handles PCM WAV so an unexpected format fails loud
    rather than silently mis-decoding.
    """
    import wave

    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        raw = w.readframes(w.getnframes())

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"{path}: unsupported sample width {sampwidth} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)  # downmix to mono
    return data, sr


def save_wav(path: str, wav: np.ndarray, sr: int):
    """Write a float32 mono array in [-1, 1] as 16-bit PCM WAV (test/debug)."""
    import wave

    clipped = np.clip(wav, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def resample_linear(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample by linear interpolation.

    Deliberately simple: linear interpolation is not the highest-fidelity
    resampler (a windowed-sinc/polyphase filter would alias less), but it has
    zero dependencies and is perfectly adequate for a first pipeline. If aliasing
    ever shows up as a quality ceiling, this is the one function to upgrade
    (scipy.signal.resample_poly) — flagged here so it is easy to find.
    """
    if sr_in == sr_out:
        return wav.astype(np.float32)
    duration = len(wav) / sr_in
    n_out = int(round(duration * sr_out))
    if n_out <= 1:
        return wav.astype(np.float32)
    x_in = np.linspace(0.0, duration, num=len(wav), endpoint=False)
    x_out = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_out, x_in, wav).astype(np.float32)


if __name__ == "__main__":
    # Smoke test: a 1-second 220 Hz tone should produce a mel with energy
    # concentrated in the low bins, and round-trip cleanly through wav I/O.
    cfg = AudioConfig()
    print(f"mel filterbank: {mel_filterbank(cfg).shape} (expect ({cfg.n_mels}, {cfg.n_fft//2+1}))")

    t = np.arange(cfg.sample_rate) / cfg.sample_rate
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    melspec = MelSpectrogram(cfg)
    mel = melspec(torch.from_numpy(tone))
    print(f"mel of 1s tone: {tuple(mel.shape)}  "
          f"(expect ~{cfg.sample_rate // cfg.hop_length + 1} frames)")
    print(f"mel range: [{mel.min():.2f}, {mel.max():.2f}]  "
          f"peak mel bin: {mel.mean(dim=-1).argmax().item()} (low = correct for 220 Hz)")

    # WAV round trip
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "voxg_audio_smoke.wav")
    save_wav(tmp, tone, cfg.sample_rate)
    back, sr = load_wav(tmp)
    err = np.abs(back[:len(tone)] - tone).max()
    print(f"wav round-trip: sr={sr}, max abs err {err:.4f} (expect < 0.01 from 16-bit quantisation)")

    # Resample check
    down = resample_linear(tone, cfg.sample_rate, 16000)
    print(f"resample 22050->16000: {len(tone)} -> {len(down)} samples "
          f"(expect ~{int(len(tone) * 16000 / cfg.sample_rate)})")
    os.remove(tmp)
    print("audio.py smoke test ok")
