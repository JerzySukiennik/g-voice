"""VoxG vocoder — a from-scratch, HiFi-GAN-style neural vocoder that turns a
mel-spectrogram into a raw waveform.

Why a second net at all? The acoustic model stops at the mel-spectrogram — a
compact, smooth picture of *which frequencies are loud when*. It is not audio:
it has thrown away the phase, and it is at frame rate (~86/sec), not sample rate
(22050/sec). Recovering a natural-sounding waveform from a mel is itself a
learned problem, and training it jointly with the acoustic model is unstable, so
TTS universally trains the two separately (see README). This is VoxG's second
from-scratch-trained network, in the same spirit as the acoustic model.

HiFi-GAN in one paragraph: a fully-convolutional **generator** upsamples the mel
back to sample rate through a stack of transposed convolutions, and after each
upsample a **multi-receptive-field (MRF)** module — several residual blocks with
different kernel sizes and dilations, summed — lets the net model both fine and
coarse temporal structure. Two **discriminators** judge realism: a
multi-period discriminator (MPD) reshapes the 1-D signal into 2-D at several
prime periods to catch periodic artefacts (a voice is highly periodic), and a
multi-scale discriminator (MSD) looks at the raw and downsampled waveform to
catch broad-band ones. Training is a GAN: generator vs. discriminators, plus a
mel-reconstruction loss and a feature-matching loss for stability.

This is a lighter variant than the paper's V1 (see VocoderConfig) to suit a
solo-speaker, limited-hours dataset. Upsampling factors multiply to exactly the
mel hop length (256) so one mel frame becomes exactly hop_length samples.
"""

from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

LRELU = 0.1  # HiFi-GAN's LeakyReLU slope


@dataclass
class VocoderConfig:
    n_mels: int = 80
    # Upsample factors must multiply to model/audio.py's hop_length (256):
    #   8 * 8 * 2 * 2 = 256. Each transposed conv's kernel is 2x its stride.
    upsample_rates: tuple = (8, 8, 2, 2)
    upsample_kernel_sizes: tuple = (16, 16, 4, 4)
    upsample_initial_channel: int = 512
    # MRF: for each upsample stage, these residual-block kernel sizes run in
    # parallel and their outputs are averaged.
    resblock_kernel_sizes: tuple = (3, 7, 11)
    resblock_dilations: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))

    def __post_init__(self):
        assert math.prod(self.upsample_rates) == 256, \
            f"upsample rates {self.upsample_rates} must multiply to hop_length 256"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """One MRF residual block: dilated conv -> LReLU -> conv, added back to the
    input, repeated for each dilation. Dilation widens the receptive field
    without extra parameters — a dilation-5 conv sees 5x further than dilation-1.
    """

    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d,
                                  padding=(kernel_size - 1) * d // 2))
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=1,
                                  padding=(kernel_size - 1) // 2))
            for _ in dilations
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            h = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU)), LRELU))
            x = x + h
        return x


class Generator(nn.Module):
    def __init__(self, cfg: VocoderConfig):
        super().__init__()
        self.cfg = cfg
        self.num_upsamples = len(cfg.upsample_rates)
        self.num_kernels = len(cfg.resblock_kernel_sizes)

        self.conv_pre = weight_norm(nn.Conv1d(cfg.n_mels, cfg.upsample_initial_channel, 7, padding=3))

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        ch = cfg.upsample_initial_channel
        for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
            out_ch = ch // 2
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                ch, out_ch, k, stride=u, padding=(k - u) // 2)))
            for ks, dil in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilations):
                self.resblocks.append(ResBlock(out_ch, ks, dil))
            ch = out_ch

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

    def forward(self, mel):
        """mel: (B, n_mels, frames) -> waveform (B, 1, frames * 256)."""
        x = self.conv_pre(mel)
        for i in range(self.num_upsamples):
            x = self.ups[i](F.leaky_relu(x, LRELU))
            # Average the parallel MRF residual blocks for this stage.
            xs = 0.0
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = torch.tanh(self.conv_post(F.leaky_relu(x, LRELU)))
        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Discriminators
# ---------------------------------------------------------------------------

class PeriodDiscriminator(nn.Module):
    """Reshape the 1-D signal to 2-D with a given period and run 2-D convs.

    A voice is quasi-periodic (the pitch period). Folding the waveform every
    `period` samples puts samples that are one period apart into the same
    column, so a 2-D conv can see periodic structure a 1-D conv would smear.
    """

    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        """x: (B, 1, T). Returns (score, [feature maps]) for feature matching."""
        b, c, t = x.shape
        if t % self.period != 0:
            x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
            t = x.shape[-1]
        x = x.view(b, c, t // self.period, self.period)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU)
            feats.append(x)
        x = self.conv_post(x)
        feats.append(x)
        return torch.flatten(x, 1, -1), feats


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x):
        return [d(x) for d in self.discriminators]  # list of (score, feats)


class ScaleDiscriminator(nn.Module):
    """Plain 1-D conv stack over the (possibly downsampled) raw waveform."""

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU)
            feats.append(x)
        x = self.conv_post(x)
        feats.append(x)
        return torch.flatten(x, 1, -1), feats


class MultiScaleDiscriminator(nn.Module):
    """Three scale discriminators: raw, /2, /4. The first uses spectral norm
    (HiFi-GAN detail — it operates on the un-pooled signal)."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.pools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, x):
        out = []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = self.pools[i](x)
            out.append(d(x))
        return out


# ---------------------------------------------------------------------------
# GAN losses (least-squares GAN, as in HiFi-GAN)
# ---------------------------------------------------------------------------

def discriminator_loss(real_outputs, fake_outputs):
    """LSGAN: push real scores to 1, fake scores to 0."""
    loss = 0.0
    for (r_score, _), (f_score, _) in zip(real_outputs, fake_outputs):
        loss = loss + torch.mean((r_score - 1.0) ** 2) + torch.mean(f_score ** 2)
    return loss


def generator_adv_loss(fake_outputs):
    """LSGAN: push fake scores (as seen by D) toward 1."""
    loss = 0.0
    for (f_score, _) in fake_outputs:
        loss = loss + torch.mean((f_score - 1.0) ** 2)
    return loss


def feature_matching_loss(real_outputs, fake_outputs):
    """L1 between D's intermediate feature maps for real vs. generated audio —
    a strong stabiliser that keeps the generator honest beyond the final score.
    """
    loss = 0.0
    for (_, r_feats), (_, f_feats) in zip(real_outputs, fake_outputs):
        for r, f in zip(r_feats, f_feats):
            loss = loss + F.l1_loss(f, r)
    return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = VocoderConfig()
    gen = Generator(cfg)
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    print(f"Generator params:  {gen.num_params()/1e6:.2f}M (target ~10-15M)")
    print(f"MPD params:        {sum(p.numel() for p in mpd.parameters())/1e6:.2f}M")
    print(f"MSD params:        {sum(p.numel() for p in msd.parameters())/1e6:.2f}M")

    B, frames = 2, 32
    mel = torch.randn(B, cfg.n_mels, frames)
    wav = gen(mel)
    print(f"generator: mel {tuple(mel.shape)} -> wav {tuple(wav.shape)} "
          f"(expect (B, 1, {frames*256}))")
    assert wav.shape == (B, 1, frames * 256), "upsample factor != hop_length!"

    real = torch.randn(B, 1, frames * 256)
    # One discriminator + generator step to prove the whole GAN graph is wired.
    # Discriminator step: judge real vs. detached-generated (no generator grad).
    d_real, d_fake = mpd(real) + msd(real), mpd(wav.detach()) + msd(wav.detach())
    d_loss = discriminator_loss(d_real, d_fake)
    d_loss.backward()
    print(f"discriminator loss {d_loss.item():.4f}  (backward ok)")

    # Generator step: D-features on real audio are the feature-matching *target*
    # (no grad — they are what the generator chases), features on generated
    # audio carry the generator gradient. Recompute both here rather than reuse
    # d_real above, whose graph was already freed by d_loss.backward().
    with torch.no_grad():
        d_real_target = mpd(real) + msd(real)
    g_fake = mpd(wav) + msd(wav)
    g_adv = generator_adv_loss(g_fake)
    g_fm = feature_matching_loss(d_real_target, g_fake)
    g_loss = g_adv + 2.0 * g_fm
    g_loss.backward()
    gnorm = sum(p.grad.norm().item() for p in gen.parameters() if p.grad is not None)
    print(f"generator adv {g_adv.item():.4f}  feat-match {g_fm.item():.4f}  "
          f"grad norm {gnorm:.1f} (finite: {math.isfinite(gnorm)})")
    assert math.isfinite(d_loss.item()) and math.isfinite(g_loss.item())
    print("vocoder.py smoke test ok")
