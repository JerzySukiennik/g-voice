"""Train the G-Voice vocoder (mel -> waveform), HiFi-GAN style.

A GAN, so two optimisers: one for the generator, one for the combined
discriminators (MPD + MSD). Each step:
  1. generate a waveform from the mel crop
  2. update the discriminators to tell real from generated
  3. update the generator to fool them, plus a mel-reconstruction loss (the
     generated audio's mel must match the input mel) and a feature-matching loss
     (the biggest stability lever in HiFi-GAN)

Same Kaggle-shaped scaffolding as train_acoustic.py: gradient accumulation,
DataParallel-ready, checkpoint/resume with the old pickle format. The checkpoint
carries *both* nets and *both* optimisers, so a resume restores the full GAN
state — resuming only the generator would let the discriminators forget and the
loss would jump.

Training the vocoder on the acoustic model's own predicted mels later (rather
than ground-truth mels) is a known refinement that closes the train/inference
gap, but it's optional and not done here — v1 trains on real mels.
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.vocoder import (Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator,
                           VocoderConfig, discriminator_loss, generator_adv_loss,
                           feature_matching_loss)
from model.audio import AudioConfig, MelSpectrogram
from data.dataset import VocoderDataset

FM_WEIGHT = 2.0      # feature-matching weight (HiFi-GAN uses 2)
MEL_WEIGHT = 45.0    # mel-reconstruction weight (HiFi-GAN uses 45)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = VocoderDataset(args.data, "train", segment_frames=args.segment_frames)
    print(f"vocoder train crops: {len(train_ds)}")

    acfg = AudioConfig()
    melspec = MelSpectrogram(acfg, device=device)   # to score reconstruction
    cfg = VocoderConfig(n_mels=acfg.n_mels)

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True)

    gen = Generator(cfg).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    print(f"Generator: {gen.num_params()/1e6:.2f}M  "
          f"MPD: {sum(p.numel() for p in mpd.parameters())/1e6:.2f}M  "
          f"MSD: {sum(p.numel() for p in msd.parameters())/1e6:.2f}M")

    if torch.cuda.device_count() > 1 and not args.single_gpu:
        gen, mpd, msd = (torch.nn.DataParallel(m) for m in (gen, mpd, msd))
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
    raw = lambda m: m.module if hasattr(m, "module") else m

    opt_g = torch.optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(msd.parameters()),
                              lr=args.lr, betas=(0.8, 0.99))

    step = 0
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = f"{args.out}/ckpt.pt"
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        raw(gen).load_state_dict(ck["gen"])
        raw(mpd).load_state_dict(ck["mpd"])
        raw(msd).load_state_dict(ck["msd"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        step = ck["step"]
        print(f"resumed from step {step}")

    def cycle(loader):
        while True:
            for batch in loader:
                yield batch
    data_iter = cycle(loader)
    t0 = time.time()

    while step < args.max_steps:
        mel, wav_real = next(data_iter)
        mel, wav_real = mel.to(device), wav_real.to(device)

        wav_fake = gen(mel)
        # crop to matched length (transposed convs can overshoot by a few samples)
        T = min(wav_fake.size(-1), wav_real.size(-1))
        wav_fake_c, wav_real_c = wav_fake[..., :T], wav_real[..., :T]

        # ---- discriminator ------------------------------------------------
        opt_d.zero_grad()
        d_real = mpd(wav_real_c) + msd(wav_real_c)
        d_fake = mpd(wav_fake_c.detach()) + msd(wav_fake_c.detach())
        d_loss = discriminator_loss(d_real, d_fake)
        d_loss.backward()
        opt_d.step()

        # ---- generator ----------------------------------------------------
        opt_g.zero_grad()
        # feature-matching targets: D features on real audio, no grad
        with torch.no_grad():
            d_real_t = mpd(wav_real_c) + msd(wav_real_c)
        g_fake = mpd(wav_fake_c) + msd(wav_fake_c)
        adv = generator_adv_loss(g_fake)
        fm = feature_matching_loss(d_real_t, g_fake)
        # mel reconstruction: generated audio's mel must match the input mel.
        # center=True STFT yields 1 + samples//hop frames, so re-analysing a
        # frames*hop crop gives one extra frame — crop both to the common length.
        mel_fake = melspec(wav_fake_c.squeeze(1))
        Tm = min(mel_fake.size(-1), mel.size(-1))
        mel_l1 = torch.nn.functional.l1_loss(mel_fake[..., :Tm], mel[..., :Tm])
        g_loss = adv + FM_WEIGHT * fm + MEL_WEIGHT * mel_l1
        g_loss.backward()
        opt_g.step()
        step += 1

        if step % args.log_every == 0:
            print(f"step {step}/{args.max_steps}  d {d_loss.item():.3f}  "
                  f"g_adv {adv.item():.3f}  fm {fm.item():.3f}  mel {mel_l1.item():.3f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

        if step % args.ckpt_every == 0 or step == args.max_steps:
            # Old pickle format — same Kaggle auto-unzip trap as the acoustic
            # trainer (see its comment / kaggle/README.md).
            torch.save({
                "gen": raw(gen).state_dict(), "mpd": raw(mpd).state_dict(),
                "msd": raw(msd).state_dict(), "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(), "step": step, "config": cfg.__dict__,
            }, ckpt_path, _use_new_zipfile_serialization=False)

    print(f"done — {args.max_steps} steps in {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="prefix from prepare_dataset.py --out")
    p.add_argument("--out", default="./run_vocoder")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--segment-frames", type=int, default=32,
                   help="mel frames per training crop (32 -> 8192 audio samples)")
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--single-gpu", action="store_true")
    main(p.parse_args())
