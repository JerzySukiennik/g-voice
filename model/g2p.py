"""Grapheme-to-phoneme for Polish.

VoxG converts text to phonemes with **espeak-ng** (via the `phonemizer`
package), used as a deterministic linguistic rule engine — NOT a learned
component. This is a conscious, discussed exception to the family's
"everything trained from scratch" rule, and a smaller one than Gedit's frozen
CLIP: CLIP is at least a neural net (used as-is, never trained); espeak-ng is
hand-written pronunciation rules, not a model at all. The thing VoxG actually
learns — how Jurek's voice turns phonemes into sound — is untouched by it.

Interface is intentionally tiny and testable:

    text_to_phonemes("Cześć, Jurek") -> ["t͡ɕ", "ɛ", "ɕ", "t͡ɕ", ...]

espeak-ng is a *system* binary, not a pip package. If it (or `phonemizer`)
isn't installed we raise a clear, actionable error rather than crashing deep in
a C call — see G2PUnavailable. For running the data-prep smoke test on a machine
without espeak-ng, `text_to_phonemes(..., allow_fallback=True)` substitutes a
crude character-level phonemizer. That fallback is TEST-ONLY: it produces
linguistically wrong phonemes and must never be used for a real training run.
"""

import re
import unicodedata


class G2PUnavailable(RuntimeError):
    """Raised when espeak-ng / phonemizer aren't installed."""


_INSTALL_HINT = (
    "espeak-ng + the phonemizer package are required for real G2P.\n"
    "  macOS:  brew install espeak-ng && pip install phonemizer\n"
    "  Linux:  sudo apt-get install espeak-ng && pip install phonemizer\n"
    "For the offline smoke test only, pass allow_fallback=True (test-only, "
    "linguistically wrong)."
)

# Cache the phonemizer backend — constructing it spins up espeak, which is slow
# to do per call over thousands of transcripts.
_backend = None


def _get_backend():
    global _backend
    if _backend is not None:
        return _backend
    try:
        from phonemizer.backend import EspeakBackend
    except ImportError as e:
        raise G2PUnavailable(f"phonemizer not importable: {e}\n{_INSTALL_HINT}") from e
    try:
        # with_stress=False: VoxG models phoneme durations, not lexical stress
        # markers, in v1 — keeping the symbol set small. preserve_punctuation so
        # commas/full stops survive as prosodic boundaries.
        _backend = EspeakBackend(
            language="pl",
            preserve_punctuation=True,
            with_stress=False,
        )
    except Exception as e:  # espeak-ng binary missing / not found by phonemizer
        raise G2PUnavailable(f"could not start espeak-ng backend: {e}\n{_INSTALL_HINT}") from e
    return _backend


# Punctuation we keep as explicit boundary phonemes (prosody), everything else
# espeak strips. These get their own ids in the vocab like any other symbol.
_KEEP_PUNCT = {",", ".", "?", "!", ";", ":"}


def _split_ipa(phoneme_string: str) -> list[str]:
    """Split an espeak IPA word/utterance string into individual phoneme symbols.

    espeak separates phonemes within a word with no delimiter, so we can't just
    split on spaces. We segment greedily: a base IPA letter plus any following
    combining marks / tie bars (e.g. affricate t͡ɕ, palatalised consonants) stays
    glued to it as one symbol. Whitespace becomes a word-boundary marker; kept
    punctuation becomes its own symbol.
    """
    out: list[str] = []
    cur = ""
    for ch in phoneme_string:
        if ch.isspace():
            if cur:
                out.append(cur)
                cur = ""
            if out and out[-1] != " ":
                out.append(" ")  # word boundary
            continue
        if ch in _KEEP_PUNCT:
            if cur:
                out.append(cur)
                cur = ""
            out.append(ch)
            continue
        cat = unicodedata.category(ch)
        # Combining mark (Mn/Mc) or tie bar U+0361/U+035C -> attach to current.
        if cat in ("Mn", "Mc") or ch in ("͡", "͜") or (cur and cur[-1] in ("͡", "͜")):
            cur += ch
        else:
            if cur:
                out.append(cur)
            cur = ch
    if cur:
        out.append(cur)
    # Trim leading/trailing word-boundary markers.
    while out and out[0] == " ":
        out.pop(0)
    while out and out[-1] == " ":
        out.pop()
    return out


def _fallback_char_phonemes(text: str) -> list[str]:
    """TEST-ONLY crude phonemizer: lowercase, keep Polish letters + kept
    punctuation, one symbol per character, spaces as word boundaries.

    This exists purely so the data pipeline is exercisable end-to-end on a
    machine without espeak-ng. It is NOT Polish phonology.
    """
    text = text.lower().strip()
    out: list[str] = []
    for ch in text:
        if ch.isspace():
            if out and out[-1] != " ":
                out.append(" ")
        elif ch in _KEEP_PUNCT:
            out.append(ch)
        elif ch.isalpha():
            out.append(ch)
        # drop everything else
    while out and out[0] == " ":
        out.pop(0)
    while out and out[-1] == " ":
        out.pop()
    return out


def text_to_phonemes(text: str, allow_fallback: bool = False) -> list[str]:
    """Convert Polish text to a list of phoneme symbols.

    allow_fallback=True: if espeak-ng/phonemizer are unavailable, use the crude
    char-level fallback instead of raising (TEST-ONLY — see module docstring).
    """
    text = text.strip()
    if not text:
        return []
    try:
        backend = _get_backend()
    except G2PUnavailable:
        if allow_fallback:
            return _fallback_char_phonemes(text)
        raise
    # phonemize returns one string per input line; we pass a single utterance.
    phonemized = backend.phonemize([text], strip=True)[0]
    phonemized = re.sub(r"\s+", " ", phonemized).strip()
    return _split_ipa(phonemized)


def espeak_available() -> bool:
    try:
        _get_backend()
        return True
    except G2PUnavailable:
        return False


if __name__ == "__main__":
    sample = "Cześć, jestem Jurek."
    print(f"espeak-ng available: {espeak_available()}")

    if espeak_available():
        phones = text_to_phonemes(sample)
        print(f"real G2P '{sample}':\n  {phones}")
    else:
        print("espeak-ng not installed — exercising the TEST-ONLY fallback path")
        phones = text_to_phonemes(sample, allow_fallback=True)
        print(f"fallback '{sample}':\n  {phones}")
        # And confirm the strict path raises cleanly with a helpful message.
        try:
            text_to_phonemes(sample)
            raise SystemExit("BUG: strict path should have raised G2PUnavailable")
        except G2PUnavailable as e:
            print(f"strict path correctly raised G2PUnavailable "
                  f"(first line: {str(e).splitlines()[0]})")

    # IPA splitter must keep an affricate tie-bar glued as one symbol.
    parts = _split_ipa("t͡ɕa b")  # t͡ɕa b
    print(f"_split_ipa('t͡ɕa b') -> {parts}  (affricate stays one symbol)")
    assert parts[0] == "t͡ɕ", f"tie-bar split wrong: {parts}"
    assert " " in parts, "word boundary lost"
    print("g2p.py smoke test ok")
