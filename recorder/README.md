# G-Voice Recorder

Voice-data collection tool for **G-Voice** — the from-scratch, single-speaker Polish
TTS model (4th in the Gzowo AI family, after G-Micro and Gedit). This little PWA is
how Jurek records his own voice, sentence by sentence, over weeks/months. Those
recordings become G-Voice's training corpus.

Plain HTML/CSS/JS, no build step, Firebase loaded from CDN as ES modules — same
philosophy as the rest of the family.

## Entry point (important)

The app is **`nagrywaj.html`** — not `index.html`, not `record.html`.

The repo is public, so there is deliberately no `index.html` and nothing links to
the recorder. The "obvious URL" resolves to nothing; you have to know the filename.
Open it directly:

```
https://<host>/recorder/nagrywaj.html
```

On the iPhone: open that URL in Safari → Share → **Add to Home Screen**. It installs
as a standalone PWA ("G-Voice") with the dark VG icon, no browser chrome.

## Target device

iPhone XR / **Safari iOS** is the real target. Desktop works too (and is used for
testing) but isn't the priority.

## How recording works

1. Fetches `corpus.json` (a JSON array of `{ "id": "p0001", "text": "..." }`),
   picks a sentence not yet recorded on this device.
2. Big sentence, big record button. Tap to record, tap again to stop.
   - Mic DSP is **deliberately disabled** (`echoCancellation`, `noiseSuppression`,
     `autoGainControl` all `false`) — those processors distort voice timbre, which
     is the one thing we must capture faithfully for voice cloning.
3. **Playback before sending.** You hear your own take, then choose **Powtórz**
   (re-record the same sentence) or **Wyślij** (send).
4. On **Wyślij** the take is saved to the local queue first, then the UI advances
   to the next sentence immediately — the upload happens in the background.
5. An always-visible counter shows total recorded across all devices/time
   (`progress/counter` in Firestore) plus how many takes are still queued locally.

Fully resumable — close the tab/app anytime, reopen, keep going. No sessions, no
batches, no forced count. Soft milestone banner at 3000 (a checkpoint, **not** a
"done" marker).

## The reliability core: IndexedDB retry queue

A recording is real effort and must never be lost. So every take is written to
**IndexedDB** (raw audio blob + metadata) *before* any network call. Upload is a
2-step, resumable, idempotent process stored per item:

1. write the `recordings/{id}` Firestore doc (audio embedded as a bytes field),
2. increment `progress/counter` via a transaction.

Each step is persisted, so a retry never re-does a completed step (no duplicate
docs, no double counting). Failed/pending items stay in the queue and retry
automatically on the next page load and on the `online` event. The counter shows
`N w kolejce` while anything is pending.

## Why the audio lives inside Firestore (no Storage)

Firebase Storage now requires the **Blaze (pay-as-you-go)** billing plan, and
Jurek (14) has no independent access to a billing account. So Storage is **out of
the picture entirely** — no bucket, no `storage.rules`, no `firebase-storage.js`.

Instead each recording's audio is stored **directly inside its Firestore document**
as a native `bytes` field. Firestore's free **Spark** plan needs no billing, ever,
and its **1 MiB per-document limit** is generously larger than a short spoken
sentence — even at a healthy **192 kbps** (well above the low-32kbps first cut,
which sounded bad, especially on iOS's AAC encoder), a generous 20-second take is
only ~480 KB. The client caps a take at ~900 KB before review and refuses oversized
takes with a "nagraj krócej" message, keeping every write comfortably under the
950 KB rule cap — there was never actually a reason to compress this hard, the
per-document cap applies per short sentence, not per session.

## Firebase data model

- `progress/counter` — `{ count: number }`. Created as `{count:0}` on first load;
  every confirmed upload increments it by exactly 1 in a transaction.
- `recordings/{id}` — `{ sentenceId, text, audio, createdAt }` where `audio` is a
  Firestore `bytes` value (the raw recording, capped ~950 KB by the security rule).
  **Write-only by design** — the client can't read recordings back (privacy). So
  there's no "list my recordings" UI; that's intentional, not missing.

## Files

| File | What |
|------|------|
| `nagrywaj.html` | the app (entry point) |
| `app.js` | all logic (recording, queue, Firestore writes, counter) |
| `styles.css` | monochrome UI |
| `manifest.json` | PWA manifest |
| `sw.js` | offline app-shell service worker |
| `corpus.json` | sentence list (owned by a separate step; fetched at runtime) |
| `firebase.json`, `.firebaserc`, `firestore.rules`, `firestore.indexes.json` | Firebase config (final, do not edit) |
| `icons/` | G-Voice app icon (final) |

## Notes on the recording format

MediaRecorder support differs by browser. Desktop Chrome records `audio/webm;opus`;
**Safari iOS records `audio/mp4`** (it has no webm/opus support). The app detects the
supported type at runtime and tracks the extension accordingly (`.webm` / `.m4a`),
so it produces valid audio on the actual iPhone target. The raw bytes are embedded
in the Firestore doc regardless of container. G-Voice's training pipeline should expect
a mix of container/codec (webm/opus from desktop tests, mp4/aac from the phone) and
transcode to a common format (e.g. 16 kHz mono WAV) at preprocessing time.
