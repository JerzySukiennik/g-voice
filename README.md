# VoxG

Czwarty model w rodzinie [Gzowo AI](https://github.com/JerzySukiennik) — a
from-scratch, single-speaker **Polish text-to-speech** model, trained by
[Jurek](https://github.com/JerzySukiennik). Rodzeństwo: [MicroG](https://github.com/JerzySukiennik/microg)
(LLM 117M od zera), [Gedit](https://github.com/JerzySukiennik/gedit) (dyfuzja do
edycji zdjęć). "Vox" (głos) + "G" (Gzowo, ten sam rdzeń co MicroG / Gedit).

Cel docelowy: **klon głosu Jurka** — model mówiący jego głosem po polsku. Głos
zbierany jest osobno przez PWA-nagrywarkę (`recorder/`, robi ją drugi agent),
która wrzuca nagrania do Firebase. Ten potok konsumuje te dane, gdy już powstaną.

**Zakres (ustalony):** TTS (tekst → audio), po **polsku**, **jeden głos**.
Rozmowa głosowa w czasie rzeczywistym jest świadomie **poza zakresem** — to
osobny, znacznie większy problem, za dużo na ten sprzęt.

## Status

**Architektura napisana i przetestowana dymnie (smoke-tested) na danych
syntetycznych. Czeka na realne nagrania głosu.** Dokładnie ta sama kolejność co
w Gedit: cały potok (model, trening, dane, eksport) powstaje i jest zweryfikowany
zanim ruszy pierwszy realny trening na Kaggle. Gdy nagrania będą gotowe →
`kaggle/README.md` prowadzi krok po kroku.

## Jak to działa (dwa modele, trenowane osobno — standard w TTS)

```
tekst ──► [G2P: espeak-ng] ──► fonemy ──► [model akustyczny] ──► mel ──► [wokoder] ──► fala
          (reguły, nie sieć)              (FastSpeech2, od zera)         (HiFi-GAN, od zera)
```

1. **G2P** (`model/g2p.py`) — tekst → fonemy przez **espeak-ng** (pakiet
   `phonemizer`). To deterministyczny silnik reguł, **nie** komponent uczony —
   świadomy wyjątek od reguły "wszystko od zera", mniejszy niż zamrożony CLIP w
   Gedit (CLIP to przynajmniej sieć; espeak to ręcznie pisane reguły wymowy).
   To, czego VoxG faktycznie się uczy — jak fonemy stają się głosem Jurka — jest
   przez to nietknięte. espeak-ng to pakiet **systemowy**: `brew install espeak-ng`
   (macOS) / `apt-get install espeak-ng` (Linux). Bez niego G2P rzuca jasny błąd.

2. **Model akustyczny** (`model/acoustic.py`, **~22.8M param**) — nieautoregresyjny
   transformer w stylu **FastSpeech2**, napisany od zera: embedding fonemów →
   enkoder-transformer → predyktor duracji (ile klatek trwa fonem) → regulator
   długości (rozciąga sekwencję do tempa klatek) → dekoder-transformer → mel
   (80 pasm). Opcjonalny adaptor wariancji (pitch + energia). To tu "od zera"
   waży najwięcej — kod pisany, żeby się go **czytało** (główny cel Jurka: nauka,
   nie tylko działający produkt).

3. **Wokoder** (`model/vocoder.py`, generator **~13.9M param**) — **HiFi-GAN** od
   zera: generator z upsamplingiem transponowanymi splotami + bloki MRF, plus
   dyskryminatory wielookresowy (MPD) i wieloskalowy (MSD). Zamienia mel na
   surową falę. Trenowany niezależnie od modelu akustycznego.

## Layout

- `model/` — `audio.py` (mel STFT + I/O WAV, bez zależności audio), `g2p.py`,
  `symbols.py` (słownik fonemów), `acoustic.py`, `vocoder.py`
- `data/prepare_dataset.py` — nagrania + transkrypcje → spakowane binaria
  (mel, fonemy, duracje, pitch, energia); `data/dataset.py` — loadery mmap
- `train/train_acoustic.py`, `train/train_vocoder.py` — pętle treningowe
  (grad-accum, DataParallel na T4×2, checkpoint/resume — jak MicroG/Gedit)
- `kaggle/` — komórki notebooków + `README.md` do treningu na darmowym Kaggle T4×2
- `runtime/export_onnx.py` — eksport do ONNX (scaffold, integracja z mostkiem
  Node w Gzowo AI to przyszła praca)
- `tests/smoke_test.py` — test całego potoku end-to-end na danych syntetycznych
- `Design/`, `recorder/` — należą do drugiego agenta, nie ruszać

## Uruchomienie

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r runtime/requirements.txt      # Intel Mac: pins mają znaczenie
brew install espeak-ng                        # dla realnego G2P (opcjonalne do testów)

# smoke testy pojedynczych modułów:
python model/acoustic.py
python model/vocoder.py
# pełny potok end-to-end na danych syntetycznych (nie wymaga espeak-ng):
python tests/smoke_test.py
```

Wersje przypięte (Intel Mac, x86_64): `torch==2.2.2` (ostatnie koło torcha na
Intel macOS), `numpy<2` — te same ściany, o które uderzył Gedit. Kaggle ma własny
nowszy torch, więc `requirements.txt` (dla Kaggle) jest nieprzypięty.

## Zanim ruszy realny trening — do decyzji

**Wyrównanie (durations).** FastSpeech2 potrzebuje duracji per-fonem jako celu.
Najlepsze źródło to forced-alignment (Montreal Forced Aligner) — wpisywane do
manifestu. Bez nich `prepare_dataset.py` robi **równomierny placeholder**, który
sprawia, że potok działa, ale daje robotyczne, równo-rytmiczne audio. **To jest
główna rzecz stojąca między tym potokiem a dobrym głosem** — szczegóły i opcje w
`kaggle/README.md`.

Licencja: MIT.
