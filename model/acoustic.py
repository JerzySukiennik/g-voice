"""G-Voice acoustic model — a from-scratch, non-autoregressive FastSpeech2-style
transformer that maps a phoneme sequence to a mel-spectrogram.

This is the part of G-Voice where "built from scratch" matters most (same role
G-Micro's hand-rolled Llama-mini and Gedit's hand-rolled U-Net play in their
repos), so the code is written to be *read*: every block is a plain transformer
layer, and the one idea that makes TTS non-autoregressive — length regulation —
is spelled out explicitly.

The problem it solves: text is short (a handful of phonemes), audio is long
(hundreds of mel frames). An autoregressive model would emit one frame at a
time and be slow; FastSpeech2 instead *predicts how many frames each phoneme
lasts* (the duration predictor) and then repeats each phoneme's encoding that
many times (the length regulator) so the decoder can produce the whole
spectrogram in one parallel pass.

    phonemes ─► embed ─► [encoder] ─┬─► duration predictor ─► durations
                                    │                            │
                                    └──► length regulator ◄──────┘
                                              │
                                     (optional pitch/energy added here)
                                              ▼
                                        [decoder] ─► linear ─► mel frames

Duration targets: FastSpeech2 needs per-phoneme durations to train. In this
repo those come from the data pipeline (data/prepare_dataset.py) — either real
forced-alignment durations when available, or a documented placeholder. During
training we length-regulate with the *ground-truth* durations (teacher forcing)
and train the predictor as a side head; at inference we use its predictions.
Pitch and energy predictors are included as an optional stretch (FastSpeech2's
"variance adaptor") and can be turned off with a flag.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AcousticConfig:
    n_phonemes: int = 80        # set from the phoneme vocab at build time
    n_mels: int = 80            # must match model/audio.py AudioConfig.n_mels
    d_model: int = 384          # embedding / transformer width (div by n_heads)
    n_heads: int = 6            # head_dim = 64
    enc_layers: int = 6
    dec_layers: int = 6
    ffn_hidden: int = 1536      # ~4 * d_model
    dropout: float = 0.1
    max_len: int = 4096         # positional-encoding table cap (frames or phones)
    # variance adaptor
    var_conv_kernel: int = 3
    var_conv_channels: int = 256
    use_pitch: bool = True      # optional stretch (FastSpeech2 variance adaptor)
    use_energy: bool = True
    pad_id: int = 0             # PhonemeVocab PAD_ID


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------

def sinusoidal_pos_encoding(length: int, dim: int, device, dtype) -> torch.Tensor:
    """Classic transformer positional table, (length, dim).

    Fixed (not learned) so the model generalises to sequences longer than any
    seen in training — a solo-speaker dataset will be short, but a long sentence
    at inference must still get sensible positions.
    """
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)[None, :]
    angle = pos / (10000.0 ** (i / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe.to(dtype)


# ---------------------------------------------------------------------------
# Transformer block (pre-norm, bidirectional — this is an encoder, not causal)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Multi-head self-attention + FFN, pre-normalised, residual.

    Unlike G-Micro this attention is NOT causal: TTS reads the whole phoneme
    sequence (and later the whole frame sequence) at once, so every position may
    attend to every other. The only masking is padding — batched utterances have
    different lengths, and a query must not attend to a neighbour's pad slots.
    """

    def __init__(self, cfg: AcousticConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        assert cfg.d_model % cfg.n_heads == 0, "d_model must divide by n_heads"

        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_hidden, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, key_padding_mask=None):
        """x: (B, T, C). key_padding_mask: (B, T) bool, True where PAD."""
        B, T, C = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # sdpa's bool mask convention: True = "allowed to attend". Our
        # key_padding_mask is True at PAD keys, so invert it and broadcast to
        # (B, 1, 1, T) over heads and query positions.
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask)[:, None, None, :]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.drop.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Variance predictor (duration / pitch / energy) — shared architecture
# ---------------------------------------------------------------------------

class VariancePredictor(nn.Module):
    """Two 1-D convolutions over the phoneme sequence -> one scalar per phoneme.

    FastSpeech2 uses the identical little conv stack for duration, pitch and
    energy. Convolution (not attention) because these are *local* quantities —
    how long a phoneme lasts depends mostly on the phoneme and its immediate
    neighbours, not on a word ten phonemes away.
    """

    def __init__(self, cfg: AcousticConfig):
        super().__init__()
        c, k = cfg.var_conv_channels, cfg.var_conv_kernel
        pad = k // 2
        # Conv1d wants (B, C, T); LayerNorm wants channels last (B, T, C). We
        # transpose around each conv rather than fight it — two small convs, so
        # the shuffling cost is negligible and the code stays readable.
        self.conv1 = nn.Conv1d(cfg.d_model, c, k, padding=pad)
        self.ln1 = nn.LayerNorm(c)
        self.conv2 = nn.Conv1d(c, c, k, padding=pad)
        self.ln2 = nn.LayerNorm(c)
        self.out = nn.Linear(c, 1)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask=None):
        """x: (B, T, C) -> (B, T) scalar per position."""
        h = F.relu(self.conv1(x.transpose(1, 2))).transpose(1, 2)  # (B,T,c)
        h = self.dropout(self.ln1(h))
        h = F.relu(self.conv2(h.transpose(1, 2))).transpose(1, 2)  # (B,T,c)
        h = self.dropout(self.ln2(h))
        out = self.out(h).squeeze(-1)               # (B, T)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


# ---------------------------------------------------------------------------
# Length regulator
# ---------------------------------------------------------------------------

def length_regulate(x, durations, max_out_len=None):
    """Expand each phoneme encoding by its (integer) duration in frames.

    x: (B, T_phon, C). durations: (B, T_phon) non-negative ints.
    Returns (B, T_frames, C) and the per-item output lengths.

    This is the heart of non-autoregressive TTS: phoneme i, encoded as x[:, i],
    is simply repeated durations[i] times along a new time axis. `torch.repeat_
    interleave` does exactly that per item; we pad the batch to the longest.
    """
    B, T, C = x.shape
    out_lens = durations.sum(dim=1)                 # (B,)
    if max_out_len is None:
        max_out_len = int(out_lens.max().item()) if B else 0
    max_out_len = max(max_out_len, 1)
    out = x.new_zeros(B, max_out_len, C)
    for b in range(B):
        d = durations[b].clamp(min=0).long()
        if d.sum() == 0:
            continue
        expanded = torch.repeat_interleave(x[b], d, dim=0)  # (sum(d), C)
        n = min(expanded.size(0), max_out_len)
        out[b, :n] = expanded[:n]
    return out, out_lens


# ---------------------------------------------------------------------------
# The acoustic model
# ---------------------------------------------------------------------------

class G-VoiceAcoustic(nn.Module):

    def __init__(self, cfg: AcousticConfig):
        super().__init__()
        self.cfg = cfg
        self.phon_emb = nn.Embedding(cfg.n_phonemes, cfg.d_model, padding_idx=cfg.pad_id)
        self.encoder = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.enc_layers)])
        self.decoder = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.dec_layers)])

        self.duration_predictor = VariancePredictor(cfg)
        self.pitch_predictor = VariancePredictor(cfg) if cfg.use_pitch else None
        self.energy_predictor = VariancePredictor(cfg) if cfg.use_energy else None
        # Project a scalar pitch/energy back up to d_model so it can be *added*
        # to the frame encodings (FastSpeech2 embeds the quantised value; a
        # linear from the raw scalar is a simpler, equally-differentiable choice).
        if cfg.use_pitch:
            self.pitch_embed = nn.Linear(1, cfg.d_model)
        if cfg.use_energy:
            self.energy_embed = nn.Linear(1, cfg.d_model)

        self.mel_linear = nn.Linear(cfg.d_model, cfg.n_mels)
        self.dropout = nn.Dropout(cfg.dropout)
        self._pe_cache = None

    def _pos(self, T, device, dtype):
        if (self._pe_cache is None or self._pe_cache.size(0) < T
                or self._pe_cache.device != device or self._pe_cache.dtype != dtype):
            self._pe_cache = sinusoidal_pos_encoding(
                max(T, self.cfg.max_len), self.cfg.d_model, device, dtype)
        return self._pe_cache[:T]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, phonemes, phon_lengths=None,
                durations=None, pitch=None, energy=None,
                mel_lengths=None):
        """Training/inference forward.

        phonemes:    (B, T_phon) long
        phon_lengths:(B,) valid phoneme counts (rest is PAD); optional
        durations:   (B, T_phon) GT durations in frames -> teacher forcing.
                     If None, the predicted durations are used (inference).
        pitch/energy:(B, T_phon) GT variance targets -> teacher forcing; else
                     predicted values are used.
        mel_lengths: (B,) GT mel frame counts, used to build the decoder pad
                     mask during training (so the decoder ignores pad frames).

        Returns dict: mel (B, T_frames, n_mels), log_duration_pred (B, T_phon),
        pitch_pred, energy_pred, mel_out_lengths.
        """
        B, T = phonemes.shape
        device = phonemes.device

        # Phoneme padding mask (True where PAD).
        if phon_lengths is not None:
            ar = torch.arange(T, device=device)[None, :]
            phon_mask = ar >= phon_lengths[:, None]
        else:
            phon_mask = phonemes.eq(self.cfg.pad_id)

        # ---- encoder --------------------------------------------------------
        x = self.phon_emb(phonemes) * math.sqrt(self.cfg.d_model)
        x = self.dropout(x + self._pos(T, device, x.dtype)[None])
        for blk in self.encoder:
            x = blk(x, key_padding_mask=phon_mask)

        # ---- duration -------------------------------------------------------
        # Predictor works in log space (durations are positive and vary over
        # orders of magnitude); expm1 back to frames for the regulator.
        log_dur_pred = self.duration_predictor(x, mask=phon_mask)
        if durations is None:
            # exp(x)-1 rather than expm1: numerically identical here (durations
            # are O(1-20), not tiny), and expm1 has no ONNX opset-17 export.
            dur = torch.clamp(torch.round(torch.exp(log_dur_pred) - 1.0), min=0)
            dur = dur.masked_fill(phon_mask, 0).long()
        else:
            dur = durations.long()

        # ---- pitch / energy (variance adaptor, optional) --------------------
        pitch_pred = energy_pred = None
        if self.pitch_predictor is not None:
            pitch_pred = self.pitch_predictor(x, mask=phon_mask)
            use_pitch = pitch if pitch is not None else pitch_pred
            x = x + self.pitch_embed(use_pitch.unsqueeze(-1))
        if self.energy_predictor is not None:
            energy_pred = self.energy_predictor(x, mask=phon_mask)
            use_energy = energy if energy is not None else energy_pred
            x = x + self.energy_embed(use_energy.unsqueeze(-1))

        # ---- length regulation ---------------------------------------------
        frames, out_lens = length_regulate(x, dur)
        Tf = frames.size(1)
        # decoder padding mask
        if mel_lengths is not None and durations is not None:
            ar = torch.arange(Tf, device=device)[None, :]
            frame_mask = ar >= mel_lengths[:, None]
        else:
            ar = torch.arange(Tf, device=device)[None, :]
            frame_mask = ar >= out_lens[:, None]

        # ---- decoder --------------------------------------------------------
        frames = self.dropout(frames + self._pos(Tf, device, frames.dtype)[None])
        for blk in self.decoder:
            frames = blk(frames, key_padding_mask=frame_mask)

        mel = self.mel_linear(frames)               # (B, Tf, n_mels)
        return {
            "mel": mel,
            "log_duration_pred": log_dur_pred,
            "pitch_pred": pitch_pred,
            "energy_pred": energy_pred,
            "mel_out_lengths": out_lens,
            "frame_mask": frame_mask,
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def acoustic_loss(out, mel_target, durations, phon_mask,
                  pitch_target=None, energy_target=None):
    """Combined FastSpeech2 loss.

    - mel: L1 between predicted and target mel over non-pad frames. L1 (not L2)
      is the FastSpeech2 choice — it gives sharper spectrograms than MSE.
    - duration: MSE in log space between predicted log-duration and log(1+GT).
    - pitch/energy: MSE over non-pad phonemes, if those heads are enabled.
    """
    mel_pred = out["mel"]
    frame_mask = out["frame_mask"]                  # (B, Tf) True where pad
    Tf = min(mel_pred.size(1), mel_target.size(1))
    mel_pred = mel_pred[:, :Tf]
    mel_target = mel_target[:, :Tf]
    valid = (~frame_mask[:, :Tf]).unsqueeze(-1).float()
    mel_l1 = (F.l1_loss(mel_pred, mel_target, reduction="none") * valid).sum() / valid.sum().clamp(min=1) / mel_pred.size(-1)

    log_dur_target = torch.log1p(durations.clamp(min=0).float())
    dvalid = (~phon_mask).float()
    dur_mse = (F.mse_loss(out["log_duration_pred"], log_dur_target, reduction="none")
               * dvalid).sum() / dvalid.sum().clamp(min=1)

    total = mel_l1 + dur_mse
    parts = {"mel_l1": mel_l1.item(), "dur_mse": dur_mse.item()}

    if out["pitch_pred"] is not None and pitch_target is not None:
        p_mse = (F.mse_loss(out["pitch_pred"], pitch_target, reduction="none")
                 * dvalid).sum() / dvalid.sum().clamp(min=1)
        total = total + p_mse
        parts["pitch_mse"] = p_mse.item()
    if out["energy_pred"] is not None and energy_target is not None:
        e_mse = (F.mse_loss(out["energy_pred"], energy_target, reduction="none")
                 * dvalid).sum() / dvalid.sum().clamp(min=1)
        total = total + e_mse
        parts["energy_mse"] = e_mse.item()

    return total, parts


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = AcousticConfig()
    model = G-VoiceAcoustic(cfg)
    print(f"G-VoiceAcoustic params: {model.num_params()/1e6:.2f}M "
          f"(target 15-30M)")

    B, T = 2, 12
    phon = torch.randint(4, cfg.n_phonemes, (B, T))
    phon_lengths = torch.tensor([12, 9])
    phon[1, 9:] = cfg.pad_id
    durations = torch.randint(1, 6, (B, T))
    durations[1, 9:] = 0
    mel_lengths = durations.sum(dim=1)
    Tf = int(mel_lengths.max())
    mel_target = torch.randn(B, Tf, cfg.n_mels)
    pitch = torch.randn(B, T)
    energy = torch.randn(B, T)

    # Teacher-forced forward (training path)
    out = model(phon, phon_lengths, durations, pitch, energy, mel_lengths)
    print(f"train forward: mel {tuple(out['mel'].shape)} "
          f"(expect (B, {Tf}, {cfg.n_mels}))")
    phon_mask = torch.arange(T)[None, :] >= phon_lengths[:, None]
    loss, parts = acoustic_loss(out, mel_target, durations, phon_mask, pitch, energy)
    print(f"loss {loss.item():.4f}  parts {parts}")

    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"backward ok — total grad norm {gnorm:.2f} (finite: {math.isfinite(gnorm)})")
    assert math.isfinite(loss.item()) and math.isfinite(gnorm)

    # Inference path (no GT durations -> predicted)
    model.eval()
    with torch.no_grad():
        inf = model(phon[:1, :phon_lengths[0]])
    print(f"inference forward (predicted durations): mel {tuple(inf['mel'].shape)}")
    print("acoustic.py smoke test ok")
