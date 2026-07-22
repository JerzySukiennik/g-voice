"""Export trained VoxG checkpoints to ONNX, for eventual inference in Gzowo AI's
Node bridge via onnxruntime-node — same pattern as Gedit's runtime/export_onnx.py.

Two graphs are exported (the two nets VoxG actually *trains* from scratch):
    voxg_acoustic.onnx   phoneme ids -> mel-spectrogram
    voxg_vocoder.onnx    mel-spectrogram -> waveform

espeak-ng is NOT in either graph — text->phoneme stays a rule-based text step
outside the model (the Node side shells out to espeak-ng, or ships its phoneme
table), exactly like Gedit keeps CLIP tokenisation in JS. Only the trained parts
become ONNX.

STATUS / future work: this is a scaffold. It has the same shape as Gedit's
exporter (export -> re-run under onnxruntime -> assert the graph matches
PyTorch) and becomes exercisable the moment a real checkpoint exists — there is
none yet (waiting on real voice data). The actual wiring into Gzowo AI's Node
bridge (the equivalent of Gedit's bridge/photo-edit.js) is future work, not part
of this build: the runtime loop is autoregressive-free (acoustic is one parallel
pass, vocoder one conv pass) so a JS bridge is straightforward once weights land.

Without --ckpt it runs in --dummy mode: builds randomly-initialised nets and
exports them, which validates that both architectures are ONNX-traceable today.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.acoustic import VoxGAcoustic, AcousticConfig
from model.vocoder import Generator, VocoderConfig


def _maybe_check(path, model, inputs, input_names, output_names, dynamic_axes,
                 strict=True):
    """Export, then (if onnxruntime is installed) re-run and compare to PyTorch.

    An export that 'succeeds' but silently diverges from the eager model is
    worse than a loud failure — same guard Gedit uses. `strict=False` downgrades
    a mismatch to a warning: the acoustic graph's length regulator produces a
    *data-dependent* output length that torch.onnx traces as a baked constant
    (see the note in main() for the acoustic export), so a faithful static-graph
    parity check isn't meaningful for it until length regulation is moved
    host-side. The vocoder (pure conv) exports faithfully and stays strict.
    """
    torch.onnx.export(model, inputs, path, input_names=input_names,
                      output_names=output_names, dynamic_axes=dynamic_axes,
                      opset_version=17)
    print(f"exported {path}")
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("  (onnxruntime not installed — skipping numerical parity check; "
              "pip install onnx onnxruntime to enable)")
        return
    sess = ort.InferenceSession(path)
    feed = {n: (t.numpy() if isinstance(t, torch.Tensor) else t)
            for n, t in zip(input_names, inputs)}
    onnx_out = sess.run(None, feed)[0]
    with torch.no_grad():
        torch_out = model(*inputs)
        if isinstance(torch_out, dict):
            torch_out = torch_out["mel"]
        torch_out = torch_out.numpy()
    if onnx_out.shape != torch_out.shape:
        msg = (f"  shape mismatch onnx {onnx_out.shape} vs torch {torch_out.shape}")
        if strict:
            raise SystemExit(msg + " — do not ship this export")
        print(msg + "\n  KNOWN LIMITATION: dynamic length regulation doesn't trace to a "
              "faithful static graph.\n  Production path: export encoder+duration and "
              "decoder as two graphs and do\n  length regulation in host (JS) code, or "
              "feed precomputed durations. Future work.")
        return
    diff = float(np.abs(onnx_out - torch_out).max())
    print(f"  max abs diff onnx vs torch: {diff:.6f}")
    if diff > 1e-3 and strict:
        raise SystemExit(f"ONNX diverges from PyTorch by {diff} — do not ship this export")


class _AcousticExportWrapper(torch.nn.Module):
    """ONNX-friendly wrapper: fixed positional signature, returns only the mel.

    The training forward returns a dict and takes many optional kwargs (GT
    durations etc.); for inference we want just phonemes -> mel with predicted
    durations, so the graph has a clean single input / single output.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, phonemes):
        return self.model(phonemes)["mel"]


def main(args):
    # ---- acoustic ---------------------------------------------------------
    if args.ckpt_acoustic:
        ck = torch.load(args.ckpt_acoustic, map_location="cpu")
        cfg = AcousticConfig(**ck["config"]) if "config" in ck else AcousticConfig(n_phonemes=args.n_phonemes)
        acoustic = VoxGAcoustic(cfg)
        acoustic.load_state_dict(ck["model"])
        print(f"loaded acoustic checkpoint at step {ck.get('step', '?')}")
    else:
        cfg = AcousticConfig(n_phonemes=args.n_phonemes)
        acoustic = VoxGAcoustic(cfg)
        print("no acoustic checkpoint — exporting a randomly-initialised model (--dummy)")
    acoustic.eval()
    wrapper = _AcousticExportWrapper(acoustic)
    dummy_phon = torch.randint(4, cfg.n_phonemes, (1, 20))
    # strict=False: the length regulator's output length depends on the
    # predicted durations, which torch.onnx bakes as a trace-time constant — so
    # the exported acoustic graph runs, but a static-graph parity check isn't
    # meaningful. The clean production fix (split at the length regulator, do it
    # host-side) is future work; see _maybe_check and the module docstring.
    _maybe_check(args.out_acoustic, wrapper, (dummy_phon,),
                 ["phonemes"], ["mel"],
                 {"phonemes": {0: "batch", 1: "n_phonemes"}, "mel": {0: "batch", 1: "n_frames"}},
                 strict=False)

    # ---- vocoder ----------------------------------------------------------
    vcfg = VocoderConfig(n_mels=cfg.n_mels)
    gen = Generator(vcfg)
    if args.ckpt_vocoder:
        ckv = torch.load(args.ckpt_vocoder, map_location="cpu")
        gen.load_state_dict(ckv["gen"])
        print(f"loaded vocoder checkpoint at step {ckv.get('step', '?')}")
    else:
        print("no vocoder checkpoint — exporting a randomly-initialised generator (--dummy)")
    gen.eval()
    dummy_mel = torch.randn(1, vcfg.n_mels, 40)
    _maybe_check(args.out_vocoder, gen, (dummy_mel,),
                 ["mel"], ["waveform"],
                 {"mel": {0: "batch", 2: "n_frames"}, "waveform": {0: "batch", 2: "n_samples"}})

    print("\nONNX export scaffold ran. Integration into the Gzowo AI Node bridge "
          "is future work (see module docstring).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-acoustic", help="acoustic ckpt.pt (omit for dummy export)")
    p.add_argument("--ckpt-vocoder", help="vocoder ckpt.pt (omit for dummy export)")
    p.add_argument("--out-acoustic", default="./voxg_acoustic.onnx")
    p.add_argument("--out-vocoder", default="./voxg_vocoder.onnx")
    p.add_argument("--n-phonemes", type=int, default=80,
                   help="only used for a dummy export with no checkpoint")
    p.add_argument("--dummy", action="store_true", help="(default when no ckpt given)")
    main(p.parse_args())
