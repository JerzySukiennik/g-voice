"""Phoneme vocabulary for G-Voice.

The acoustic model doesn't see letters or IPA strings — it sees integer ids.
This is the equivalent of G-Micro's BPE tokenizer, but far smaller and much
simpler: Polish, run through espeak-ng, produces a small closed set of IPA
phoneme symbols (a few dozen), so there is no need to *learn* a vocabulary the
way a subword tokenizer does. We just collect every distinct symbol that
appears in the training transcripts and assign it an id.

Four ids are reserved before any real phoneme:
    0  PAD   padding for batching variable-length sequences
    1  UNK   a phoneme espeak emitted that wasn't seen at vocab-build time
    2  BOS   beginning of utterance
    3  EOS   end of utterance

The vocab is built once during data preparation and saved as JSON next to the
packed binaries (mirroring how G-Micro ships data/tokenizer.json with its
dataset), so training and inference always agree on the id<->symbol mapping.
"""

import json

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
RESERVED = [PAD, UNK, BOS, EOS]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


class PhonemeVocab:
    def __init__(self, symbols: list[str] | None = None):
        # symbols: the *non-reserved* phoneme symbols, in a stable order.
        self.symbols = list(symbols) if symbols else []
        self._rebuild()

    def _rebuild(self):
        self.itos = RESERVED + self.symbols
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, phoneme_seqs) -> "PhonemeVocab":
        """Collect the distinct symbols across many phoneme sequences.

        Sorted for determinism — the same corpus always yields the same ids, so
        a checkpoint stays compatible with a vocab rebuilt from the same data.
        """
        seen = set()
        for seq in phoneme_seqs:
            seen.update(seq)
        seen.difference_update(RESERVED)
        return cls(sorted(seen))

    def encode(self, phonemes: list[str], add_bos_eos: bool = True) -> list[int]:
        ids = [self.stoi.get(p, UNK_ID) for p in phonemes]
        if add_bos_eos:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] if 0 <= i < len(self.itos) else UNK for i in ids]

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"symbols": self.symbols}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "PhonemeVocab":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["symbols"])


if __name__ == "__main__":
    seqs = [["a", "b", "a"], ["a", "c"], ["ɕ", "t͡ɕ"]]  # incl. Polish IPA
    v = PhonemeVocab.build(seqs)
    print(f"vocab size: {len(v)} (4 reserved + {len(v.symbols)} phonemes: {v.symbols})")
    ids = v.encode(["a", "ɕ", "zzz"])  # 'zzz' is unseen -> UNK
    print(f"encode ['a','ɕ','zzz'] -> {ids} (BOS ... EOS, zzz -> {UNK_ID})")
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID and UNK_ID in ids
    dec = v.decode(ids)
    print(f"decode back -> {dec}")

    import tempfile, os
    p = os.path.join(tempfile.gettempdir(), "g-voice_vocab_smoke.json")
    v.save(p)
    v2 = PhonemeVocab.load(p)
    assert v2.stoi == v.stoi, "save/load changed the mapping!"
    os.remove(p)
    print("symbols.py smoke test ok")
