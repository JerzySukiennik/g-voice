# Trening G-Voice na Kaggle — instrukcja

Ten sam plan co przy [[g-micro|G-Micro]] i [[gedit|Gedit]]: **30 h GPU tygodniowo za darmo** (T4 ×2), sesja ucinana po 12 h, wszystko poniżej zbudowane pod wznawianie.

G-Voice to **dwa** osobne treningi (standard w TTS): najpierw model akustyczny (fonemy → mel-spektrogram), potem osobno wokoder (mel → fala dźwiękowa). Oba trenują niezależnie z tego samego datasetu `g-voice-data`.

Czego potrzebujesz: konta na kaggle.com ze **zweryfikowanym numerem telefonu**.

---

## Uwaga: danych jeszcze nie ma

W przeciwieństwie do G-Micro (publiczny korpus PL) i Gedit (publiczny InstructPix2Pix), dane G-Voice to **Twój własny głos**. Zbiera je osobna PWA (`../recorder`, robi ją drugi agent) do Firebase. Ten potok jest napisany i przetestowany na danych syntetycznych (`tests/smoke_test.py`) — realny trening rusza dopiero, gdy nagrasz materiał. **Nie odpalaj treningu na przykładowych/syntetycznych danych — nic sensownego z tego nie wyjdzie.**

---

## Jak wygląda dataset wejściowy

Eksport z Firebase (jeszcze nie istnieje) ma wyprodukować folder w tym układzie, spakowany i wgrany na Kaggle jako Dataset (np. `g-voice-recordings`):

```
wavs/               folder plików PCM WAV (nagrania, po transkodowaniu z webm/opus)
manifest.json       [{"audio": "0001.wav", "text": "Cześć, jestem Jurek."}, ...]
```

Nagrania z przeglądarki to webm/opus — przed wgraniem przekonwertuj na WAV:
```
ffmpeg -i take.webm -ac 1 -ar 22050 wavs/0001.wav
```
(`data/prepare_dataset.py` i tak zresampluje, ale ffmpeg robi to lepiej niż wbudowany resampler liniowy.)

`manifest.json` może opcjonalnie mieć per-nagranie pole `"durations": [...]` (wyrównanie z forced-alignera). **To jest najważniejsza decyzja jakościowa** — patrz sekcja "Durations / wyrównanie" na dole.

---

## Krok 1 — przygotowanie danych (raz, CPU)

1. **New Notebook**
2. Prawy panel → **Accelerator: None**, **Internet: On**
3. **Add Input** → Twój dataset z nagraniami (`wavs/` + `manifest.json`)
4. Wklej zawartość [`01-prep.py`](01-prep.py) do komórki i uruchom
5. **Save Version → Save & Run All**
6. Na stronie wersji: **Output → New Dataset**, nazwij **`g-voice-data`**

Skrypt instaluje `espeak-ng` (apt) + `phonemizer`, uruchamia G2P (tekst→fonemy IPA, PL), wyciąga mel-spektrogramy, F0 i energię, i pakuje wszystko do `g-voice_*.bin` + `g-voice_meta.json` + `g-voice_phonemes.json`. **Bez `--allow-g2p-fallback`** — realny trening musi mieć prawdziwe fonemy espeak-ng, nie zastępczy tryb testowy.

## Krok 2 — trening modelu akustycznego (czas nieznany do 1. pomiaru na GPU)

1. **New Notebook**
2. **Accelerator: GPU T4 x2**, **Internet: On**, **Persistence: Variables and Files**
3. **Add Input** → `g-voice-data`
4. Wklej [`02-train-acoustic.py`](02-train-acoustic.py) i uruchom
5. **Save Version → Save & Run All (Commit)** — nie zwykłe odpalenie komórki. Batch-commit automatycznie wypełnia zakładkę **Output**; sesja interaktywna ("Edit") tego nie robi (trzeba wtedy ręcznie: Notebook → Output → `/kaggle/working` → `run_acoustic/` → `ckpt.pt`).
6. Po zakończeniu (albo ucięciu na 12h): **Output → New Dataset**, nazwij **`g-voice-acoustic-ckpt`**

Checkpoint co 200 kroków. `STEPS` w skrypcie to **sufit, nie cel** — można pobrać `ckpt.pt` i zatrzymać, gdy mele wyglądają dobrze.

## Krok 3 — trening wokodera (osobno, niezależnie)

Dokładnie jak krok 2, ale wklej [`03-train-vocoder.py`](03-train-vocoder.py), a output nazwij **`g-voice-vocoder-ckpt`**. Wokoder potrzebuje **tylko** `g-voice-data` (audio + mele) — NIE checkpointu akustycznego. To GAN (HiFi-GAN), więc trenuje się dłużej niż model akustyczny; `STEPS` znów jest sufitem, oceniaj po tym, jak brzmi zsyntezowane audio.

## Wznowienie (jeśli 12 h nie starczyło)

Nowy notebook jak w kroku 2/3, ale **Add Input** dodatkowo checkpoint (`g-voice-acoustic-ckpt` albo `g-voice-vocoder-ckpt`). Skrypt sam wykryje `ckpt.pt` i podejmie od ostatniego kroku — razem z momentami optymalizatora (u wokodera: OBU optymalizatorów i OBU sieci), więc bez skoku loss. Po każdej sesji nadpisuj checkpoint nowym outputem (**Output → Update Dataset**).

Jeśli checkpoint osiągnął już `STEPS` z aktualnej wersji skryptu, wznowienie nic nie zrobi (pętla `while step < max_steps` od razu kończy) — najpierw podnieś `STEPS`.

---

## Pułapka: format checkpointu (jak w Gedit)

Checkpointy zapisujemy starym formatem pickle (`_use_new_zipfile_serialization=False`), nie domyślnym zipem torcha. Gedit odkrył na własnej skórze, że kaggle'owy **Add Input → Upload** auto-rozpakowuje zip-owy `ckpt.pt` na wewnętrzne `data.pkl`/`data/N`, psując dokładne dopasowanie nazwy pliku, od którego zależy każde wznowienie. Stary format tego unika. Komentarz jest w obu `train/train_*.py` — nie zmieniaj tego.

## Jak sprawdzić, czy idzie dobrze

**Model akustyczny:** loss to suma L1 na melu + MSE na (log-)duracji (+ opcjonalnie pitch/energy). Startowy loss zależy od skali meli w danych; kluczowe, że **spada i nie jest `nan`**. Val loss (log co `--eval-every`) powinien systematycznie schodzić. `nan` od razu = zbij `--lr`.

**Wokoder (GAN):** patrz na trzy liczby w logu: `d` (dyskryminator, powinien oscylować ~0.5–3, NIE spaść do zera — zero = dyskryminator wygrał, generator przestał się uczyć), `g_adv` (generator), `mel` (rekonstrukcja mela — TA ma systematycznie spadać; to najlepszy sygnał realnej jakości). Jak w Gedit: **loss GAN-a bywa mylący, oceniaj po odsłuchu**, nie po samej krzywej.

## Durations / wyrównanie — najważniejsza decyzja jakościowa

Model akustyczny (FastSpeech2) potrzebuje **duracji per-fonem** (ile klatek trwa każdy fonem) jako celu treningowego. Skąd je wziąć:

- **Najlepiej:** forced-alignment (Montreal Forced Aligner, model PL) → wpisz duracje do `manifest.json` jako pole `"durations"`. Wtedy `prepare_dataset.py` użyje ich wprost.
- **Zastępczo (domyślnie):** jeśli nie ma duracji, `prepare_dataset.py` rozdziela klatki **równomiernie** po fonemach. To sprawia, że potok działa, ale daje **robotyczne, równo-rytmiczne** audio. `g-voice_meta.json` liczy, ile nagrań poszło na placeholderze (`placeholder_durations`), a prep głośno o tym krzyczy w logu.

**To jest główna rzecz stojąca między tym potokiem a dobrym głosem.** Zanim ruszy poważny trening, zdecyduj: albo dorzucamy MFA do `01-prep.py` (osobny krok), albo akceptujemy placeholder na pierwszą iterację i poprawiamy później. (Nowoczesna alternatywa bez zewnętrznego alignera — wewnętrzny aligner w stylu RAD-TTS/one-TTS-alignment uczony razem z modelem — jest możliwa, ale to spora dokładka; świadomie poza zakresem v1.)

## Jak coś nie działa

- **`CUDA out of memory`** → zbij `BATCH` (akustyczny: podnieś `ACCUM` dla tej samej liczby przykładów na krok; wokoder: zmniejsz `SEGMENT`).
- **DataParallel się sypie** → dopisz `"--single-gpu"` do listy `cmd` w skrypcie.
- **Notebook nie widzi danych** → sprawdź, czy dataset jest w **Add Input**, nie tylko utworzony.
- **`espeak not installed`** przy prep → apt-get espeak-ng nie przeszedł; sprawdź, czy Internet: On.
