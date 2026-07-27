"""Train the G-Voice acoustic model (phonemes -> mel).

Mirrors G-Micro/Gedit training conventions exactly: gradient accumulation,
DataParallel across Kaggle's T4x2, and checkpoint/resume every --ckpt-every
steps so a 12h-capped (or randomly killed) Kaggle session picks up mid-stride
instead of restarting hours of work. The "skip if already done, resume if a
checkpoint exists" pattern has saved real GPU-hours in both siblings.

Teacher forcing: during training the length regulator uses the ground-truth
durations from the data (so the mel target lines up frame-for-frame), and the
duration predictor is trained as a side head against those same durations. At
inference (runtime) the predicted durations drive the regulator instead.
"""

import argparse
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.acoustic import GVoiceAcoustic, AcousticConfig, acoustic_loss
from data.dataset import AcousticDataset, acoustic_collate


def evaluate(model, loader, device, cfg):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(b["phon"], b["phon_lengths"], b["dur"], b["pitch"],
                        b["energy"], b["mel_lengths"])
            phon_mask = (torch.arange(b["phon"].size(1), device=device)[None, :]
                         >= b["phon_lengths"][:, None])
            loss, _ = acoustic_loss(out, b["mel"], b["dur"], phon_mask,
                                    b["pitch"], b["energy"])
            tot += loss.item(); n += 1
    model.train()
    return tot / max(n, 1)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = AcousticDataset(args.data, "train")
    val_ds = AcousticDataset(args.data, "val")
    print(f"train utts: {len(train_ds)}  val utts: {len(val_ds)}")

    n_phonemes = train_ds.d.meta["n_phonemes"]
    n_mels = train_ds.d.meta["n_mels"]
    cfg = AcousticConfig(n_phonemes=n_phonemes, n_mels=n_mels,
                         use_pitch=not args.no_variance, use_energy=not args.no_variance)

    collate = lambda batch: acoustic_collate(batch, pad_id=cfg.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate) if len(val_ds) else None

    model = GVoiceAcoustic(cfg).to(device)
    print(f"GVoiceAcoustic params: {model.num_params()/1e6:.2f}M")
    if torch.cuda.device_count() > 1 and not args.single_gpu:
        model = torch.nn.DataParallel(model)
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
    raw = lambda: model.module if hasattr(model, "module") else model

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)

    step = 0
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = f"{args.out}/ckpt.pt"
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        raw().load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed from step {step}")

    def cycle(loader):
        while True:
            for batch in loader:
                yield batch
    data_iter = cycle(train_loader)
    t0 = time.time()

    while step < args.max_steps:
        opt.zero_grad()
        accum = 0.0
        for _ in range(args.grad_accum):
            b = next(data_iter)
            b = {k: v.to(device) for k, v in b.items()}
            out = model(b["phon"], b["phon_lengths"], b["dur"], b["pitch"],
                        b["energy"], b["mel_lengths"])
            phon_mask = (torch.arange(b["phon"].size(1), device=device)[None, :]
                         >= b["phon_lengths"][:, None])
            loss, parts = acoustic_loss(out, b["mel"], b["dur"], phon_mask,
                                        b["pitch"], b["energy"])
            (loss / args.grad_accum).backward()
            accum += loss.item() / args.grad_accum

        if step < args.warmup:
            for g in opt.param_groups:
                g["lr"] = args.lr * (step + 1) / args.warmup
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % args.log_every == 0:
            print(f"step {step}/{args.max_steps}  loss {accum:.4f}  "
                  f"mel {parts['mel_l1']:.3f} dur {parts['dur_mse']:.3f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

        if val_loader and step % args.eval_every == 0:
            print(f"  val loss {evaluate(model, val_loader, device, cfg):.4f}")

        if step % args.ckpt_every == 0 or step == args.max_steps:
            # _use_new_zipfile_serialization=False: Kaggle's "Add Input ->
            # Upload" auto-unzips the default zip-format checkpoint and corrupts
            # the exact-filename match every resume depends on. Gedit hit this
            # the hard way — use the old pickle format. (See kaggle/README.md.)
            torch.save({"model": raw().state_dict(), "opt": opt.state_dict(),
                        "step": step, "config": cfg.__dict__},
                       ckpt_path, _use_new_zipfile_serialization=False)

    print(f"done — {args.max_steps} steps in {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="prefix from prepare_dataset.py --out")
    p.add_argument("--out", default="./run_acoustic")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--single-gpu", action="store_true")
    p.add_argument("--no-variance", action="store_true", help="disable pitch/energy heads")
    main(p.parse_args())
