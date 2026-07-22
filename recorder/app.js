/* VoxG Recorder — voice-data collection tool for the VoxG TTS model.
   Plain ES module, no build step. Firebase loaded lazily via CDN so the app
   still boots (and can record + queue) when offline or when Firebase is down. */

const FB_VERSION = "10.12.2";
const FB = (m) => `https://www.gstatic.com/firebasejs/${FB_VERSION}/firebase-${m}.js`;

const firebaseConfig = {
  projectId: "voxg-recorder",
  appId: "1:361358543792:web:03960191697d9ae9bfa475",
  storageBucket: "voxg-recorder.firebasestorage.app",
  apiKey: "AIzaSyC-AbxkV2NXDyo17hsAm69nF_Zv1rlak0s",
  authDomain: "voxg-recorder.firebaseapp.com",
  messagingSenderId: "361358543792",
};

/* ----------------------------------------------------------------------------
   localStorage keys
---------------------------------------------------------------------------- */
const LS_RECORDED = "voxg_recorded_ids";   // ids recorded on THIS device (repeat-avoidance)
const LS_CONFIRMED = "voxg_confirmed";      // last known server counter (offline display)
const LS_MILESTONE = "voxg_milestone_3000"; // "1" once the 3000 banner was dismissed
const MILESTONE = 3000;

/* ----------------------------------------------------------------------------
   Lazy Firebase loader (memoized). Returns null if it can't load (e.g. offline).
---------------------------------------------------------------------------- */
let _fb = null;
let _fbPromise = null;
async function loadFirebase() {
  if (_fb) return _fb;
  if (_fbPromise) return _fbPromise;
  _fbPromise = (async () => {
    const [appMod, fsMod] = await Promise.all([
      import(FB("app")),
      import(FB("firestore")),
    ]);
    const app = appMod.initializeApp(firebaseConfig);
    const db = fsMod.getFirestore(app);
    // fsMod exposes Bytes (for Uint8Array → Firestore-native bytes) alongside
    // doc/setDoc/serverTimestamp/runTransaction/getDoc — all from the same module.
    _fb = { app, db, fs: fsMod };
    return _fb;
  })();
  try {
    return await _fbPromise;
  } catch (e) {
    _fbPromise = null; // allow a later retry (e.g. after coming back online)
    throw e;
  }
}

/* ----------------------------------------------------------------------------
   IndexedDB pending-upload queue  (the reliability core: never lose a take)
---------------------------------------------------------------------------- */
const DB_NAME = "voxg";
const STORE = "queue";
let _idb = null;

function idb() {
  if (_idb) return _idb;
  _idb = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "id" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return _idb;
}

async function tx(mode, fn) {
  const db = await idb();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const store = t.objectStore(STORE);
    let out;
    Promise.resolve(fn(store)).then((v) => (out = v));
    t.oncomplete = () => resolve(out);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

const queuePut = (item) => tx("readwrite", (s) => s.put(item));
const queueDelete = (id) => tx("readwrite", (s) => s.delete(id));
const queueAll = () =>
  tx("readonly", (s) =>
    new Promise((res) => {
      const r = s.getAll();
      r.onsuccess = () => res(r.result || []);
    })
  );
async function queueCount() {
  return tx("readonly", (s) =>
    new Promise((res) => {
      const r = s.count();
      r.onsuccess = () => res(r.result || 0);
    })
  );
}

/* ----------------------------------------------------------------------------
   DOM refs
---------------------------------------------------------------------------- */
const $ = (id) => document.getElementById(id);
const el = {
  counter: $("counter"),
  stage: $("stage"),
  gate: $("gate"),
  gateTitle: $("gate-title"),
  gateText: $("gate-text"),
  gateRetry: $("gate-retry"),
  sentence: $("sentence"),
  sentenceId: $("sentence-id"),
  idleControls: $("idle-controls"),
  recBtn: $("rec-btn"),
  hint: $("hint"),
  timer: $("timer"),
  review: $("review"),
  player: $("player"),
  redo: $("redo"),
  send: $("send"),
  msg: $("msg"),
  banner: $("banner"),
  bannerClose: $("banner-close"),
};

/* ----------------------------------------------------------------------------
   State
---------------------------------------------------------------------------- */
let corpus = [];
let current = null;            // { id, text }
let recorder = null;
let chunks = [];
let stream = null;
let takeBlob = null;
let takeExt = "webm";
let takeMime = "audio/webm";
let recStart = 0;
let timerRAF = 0;
let confirmedCount = readNum(LS_CONFIRMED, 0);
let pendingCount = 0;
let mode = "loading";          // loading | idle | recording | review

/* ----------------------------------------------------------------------------
   Helpers
---------------------------------------------------------------------------- */
function readNum(key, fallback) {
  const v = parseInt(localStorage.getItem(key) || "", 10);
  return Number.isFinite(v) ? v : fallback;
}
function recordedSet() {
  try {
    return new Set(JSON.parse(localStorage.getItem(LS_RECORDED) || "[]"));
  } catch {
    return new Set();
  }
}
function saveRecordedSet(set) {
  localStorage.setItem(LS_RECORDED, JSON.stringify([...set]));
}
const fmt = (n) => n.toLocaleString("pl-PL");

function pad(n) {
  return String(n).padStart(2, "0");
}

/* ----------------------------------------------------------------------------
   Counter display
---------------------------------------------------------------------------- */
function renderCounter() {
  const parts = [`<b>${fmt(confirmedCount)}</b> wysłanych`];
  if (pendingCount > 0) parts.push(`<span class="pending">${fmt(pendingCount)} w kolejce</span>`);
  el.counter.innerHTML = parts.join(" · ");
}

async function refreshPending() {
  pendingCount = await queueCount();
  renderCounter();
}

function bumpConfirmed() {
  confirmedCount += 1;
  localStorage.setItem(LS_CONFIRMED, String(confirmedCount));
  renderCounter();
  maybeMilestone();
}

function maybeMilestone() {
  if (confirmedCount >= MILESTONE && localStorage.getItem(LS_MILESTONE) !== "1") {
    el.banner.hidden = false;
  }
}

/* ----------------------------------------------------------------------------
   Sentence selection
---------------------------------------------------------------------------- */
function pickSentence() {
  if (!corpus.length) {
    current = null;
    return;
  }
  const done = recordedSet();
  let pool = corpus.filter((s) => !done.has(s.id));
  if (pool.length === 0) {
    // whole corpus exhausted on this device — reshuffle, start fresh
    saveRecordedSet(new Set());
    pool = corpus.slice();
  }
  current = pool[Math.floor(Math.random() * pool.length)];
  el.sentence.textContent = current.text;
  el.sentenceId.textContent = current.id;
}

/* ----------------------------------------------------------------------------
   View switching
---------------------------------------------------------------------------- */
function setMode(next) {
  mode = next;
  el.gate.hidden = next !== "gate";
  el.stage.hidden = next === "gate" || next === "loading";
  if (next === "idle") {
    el.idleControls.hidden = false;
    el.review.hidden = true;
    el.recBtn.classList.remove("recording");
    el.hint.textContent = "Dotknij, aby nagrać";
    el.timer.textContent = "";
  } else if (next === "recording") {
    el.idleControls.hidden = false;
    el.review.hidden = true;
    el.recBtn.classList.add("recording");
    el.hint.textContent = "Nagrywam… dotknij, aby zatrzymać";
  } else if (next === "review") {
    el.idleControls.hidden = true;
    el.review.hidden = false;
    el.recBtn.classList.remove("recording");
  }
}

/* ----------------------------------------------------------------------------
   Recording
---------------------------------------------------------------------------- */
function chooseMime() {
  const candidates = [
    ["audio/webm;codecs=opus", "webm"],
    ["audio/webm", "webm"],
    ["audio/mp4", "m4a"],   // Safari iOS produces this
    ["audio/aac", "aac"],
    ["", "webm"],           // let the browser decide
  ];
  const canCheck = typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported;
  for (const [type, ext] of candidates) {
    if (type === "" || (canCheck && MediaRecorder.isTypeSupported(type))) {
      return { type, ext };
    }
  }
  return { type: "", ext: "webm" };
}

async function startRecording() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        // Intentionally OFF: browser DSP distorts timbre, which is exactly
        // the thing these recordings exist to capture faithfully.
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
    });
  } catch (err) {
    showPermissionGate(err);
    return;
  }

  const { type, ext } = chooseMime();
  takeExt = ext;
  // Explicit low bitrate keeps takes small & predictable so they fit inside a
  // single Firestore document's bytes field (~950KB cap). Some browsers reject
  // the audioBitsPerSecond option combo, so fall back progressively.
  try {
    recorder = type
      ? new MediaRecorder(stream, { mimeType: type, audioBitsPerSecond: 32000 })
      : new MediaRecorder(stream, { audioBitsPerSecond: 32000 });
  } catch {
    try {
      recorder = type ? new MediaRecorder(stream, { mimeType: type }) : new MediaRecorder(stream);
    } catch {
      recorder = new MediaRecorder(stream);
    }
  }
  takeMime = recorder.mimeType || type || "audio/webm";

  chunks = [];
  recorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };
  recorder.onstop = onRecordingStopped;
  recorder.start();
  recStart = performance.now();
  setMode("recording");
  tickTimer();
}

function tickTimer() {
  const secs = Math.floor((performance.now() - recStart) / 1000);
  el.timer.textContent = `${pad(Math.floor(secs / 60))}:${pad(secs % 60)}`;
  timerRAF = requestAnimationFrame(tickTimer);
}

function stopRecording() {
  if (recorder && recorder.state !== "inactive") recorder.stop();
  cancelAnimationFrame(timerRAF);
}

function releaseMic() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
}

// Headroom under the 950000-byte Firestore rule cap: Firestore's binary
// encoding adds overhead and the doc also carries sentenceId/text/createdAt.
// At 32kbps a short spoken sentence stays far below this; hitting it means the
// take ran too long, so we degrade gracefully instead of failing the write later.
const MAX_TAKE_BYTES = 900000;

function onRecordingStopped() {
  cancelAnimationFrame(timerRAF);
  releaseMic();
  takeBlob = new Blob(chunks, { type: takeMime.split(";")[0] || "audio/webm" });
  chunks = [];

  if (takeBlob.size >= MAX_TAKE_BYTES) {
    // Too big to fit one Firestore doc — drop the oversized take and let the
    // user re-record the SAME sentence, shorter.
    takeBlob = null;
    if (el.player.src) {
      URL.revokeObjectURL(el.player.src);
      el.player.removeAttribute("src");
    }
    setMode("idle");
    setMsg("Nagranie za długie — spróbuj ponownie i nagraj krócej.", true);
    return;
  }

  const url = URL.createObjectURL(takeBlob);
  if (el.player.src) URL.revokeObjectURL(el.player.src);
  el.player.src = url;
  setMode("review");
}

/* ----------------------------------------------------------------------------
   Send → enqueue (never lose the take) → optimistic advance → background upload
---------------------------------------------------------------------------- */
async function sendTake() {
  if (!takeBlob || !current) return;
  const id = (crypto.randomUUID && crypto.randomUUID()) || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const item = {
    id,
    sentenceId: current.id,
    text: current.text,
    // Store the raw Blob — IndexedDB handles Blobs natively; the Blob→bytes
    // conversion happens at upload time in the processor, not here.
    blob: takeBlob,
    ext: takeExt,  // kept for debugging only; not required by the Firestore rules
    createdAt: Date.now(),
    step: 0,       // 0 = write recordings/{id} doc (with audio bytes), 1 = counter
    attempts: 0,
  };

  // 1) Persist to IndexedDB FIRST — from here the take cannot be lost.
  await queuePut(item);

  // 2) Mark this sentence recorded on this device + advance immediately.
  const done = recordedSet();
  done.add(current.id);
  saveRecordedSet(done);

  takeBlob = null;
  if (el.player.src) {
    URL.revokeObjectURL(el.player.src);
    el.player.removeAttribute("src");
  }
  pickSentence();
  setMode("idle");
  await refreshPending();
  setMsg("");

  // 3) Fire the upload in the background.
  processQueue();
}

/* ----------------------------------------------------------------------------
   Queue processor — step-based so retries are idempotent (no dup docs/counts)
---------------------------------------------------------------------------- */
let processing = false;

async function processQueue() {
  if (processing) return;
  processing = true;
  try {
    const items = (await queueAll()).sort((a, b) => a.createdAt - b.createdAt);
    if (!items.length) return;

    let fb;
    try {
      fb = await loadFirebase();
    } catch {
      setMsg("Brak połączenia — nagrania czekają w kolejce i wyślą się później.");
      return;
    }

    const { db, fs } = fb;

    for (const item of items) {
      try {
        if (item.step === 0) {
          // Convert the stored Blob to a Firestore-native bytes value and write
          // the recordings/{id} doc in one shot (audio lives inside the doc now).
          const buf = await item.blob.arrayBuffer();
          const bytes = fs.Bytes.fromUint8Array(new Uint8Array(buf));
          await fs.setDoc(fs.doc(db, "recordings", item.id), {
            sentenceId: item.sentenceId,
            text: item.text,
            audio: bytes,
            createdAt: fs.serverTimestamp(),
          });
          item.step = 1;
          await queuePut(item);
        }
        if (item.step === 1) {
          const counterRef = fs.doc(db, "progress", "counter");
          await fs.runTransaction(db, async (t) => {
            const snap = await t.get(counterRef);
            if (!snap.exists()) throw new Error("counter-missing");
            t.update(counterRef, { count: snap.data().count + 1 });
          });
          await queueDelete(item.id);
          bumpConfirmed();
          await refreshPending();
        }
      } catch (err) {
        item.attempts = (item.attempts || 0) + 1;
        await queuePut(item);
        reportUploadError(err);
        // Stop this pass; retry on next online event / page load / manual send.
        break;
      }
    }
  } finally {
    processing = false;
    await refreshPending();
  }
}

function reportUploadError(err) {
  const blob = ((err && (err.message || err.code)) || "").toString();
  // "document too large" shouldn't happen thanks to the client-side size guard,
  // but if it slips through, say so clearly instead of the generic message.
  if (/too.?large|exceeds the maximum|invalid-argument|resource-exhausted/i.test(blob)) {
    setMsg("Nagranie jest za duże do wysłania — nagraj tę wypowiedź krócej.", true);
  } else {
    setMsg("Nie udało się wysłać — nagranie zostało zapisane lokalnie, spróbuj ponownie później.", true);
  }
}

function setMsg(text, warn = false) {
  el.msg.textContent = text;
  el.msg.classList.toggle("warn", !!warn);
}

/* ----------------------------------------------------------------------------
   Firestore counter bootstrap
---------------------------------------------------------------------------- */
async function loadCounter() {
  let fb;
  try {
    fb = await loadFirebase();
  } catch {
    renderCounter(); // show cached value while offline
    return;
  }
  const { db, fs } = fb;
  const ref = fs.doc(db, "progress", "counter");
  try {
    const snap = await fs.getDoc(ref);
    if (!snap.exists()) {
      try {
        await fs.setDoc(ref, { count: 0 });
      } catch {
        /* another device won the race — fine, re-read below */
      }
      const again = await fs.getDoc(ref);
      confirmedCount = again.exists() ? again.data().count : 0;
    } else {
      confirmedCount = snap.data().count;
    }
    localStorage.setItem(LS_CONFIRMED, String(confirmedCount));
  } catch {
    /* keep cached value */
  }
  renderCounter();
  maybeMilestone();
}

/* ----------------------------------------------------------------------------
   Permission / error gate
---------------------------------------------------------------------------- */
function showPermissionGate(err) {
  releaseMic();
  const name = (err && err.name) || "";
  el.gateTitle.textContent = "Potrzebny dostęp do mikrofonu";
  if (name === "NotAllowedError" || name === "SecurityError") {
    el.gateText.innerHTML =
      "Dostęp do mikrofonu został zablokowany. W iOS: <b>Ustawienia → Safari → Mikrofon</b> " +
      "(lub ikona „aA” w pasku adresu → Ustawienia strony) i zezwól tej stronie. Potem dotknij „Spróbuj ponownie”.";
  } else if (name === "NotFoundError" || name === "OverconstrainedError") {
    el.gateText.textContent = "Nie znaleziono mikrofonu. Podłącz mikrofon i spróbuj ponownie.";
  } else {
    el.gateText.textContent = "Nie udało się uruchomić mikrofonu. Spróbuj ponownie.";
  }
  setMode("gate");
}

/* ----------------------------------------------------------------------------
   Init
---------------------------------------------------------------------------- */
async function loadCorpus() {
  try {
    const res = await fetch("./corpus.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(String(res.status));
    const data = await res.json();
    corpus = Array.isArray(data) ? data.filter((s) => s && s.id && typeof s.text === "string") : [];
  } catch {
    corpus = [];
  }
}

function wireEvents() {
  el.recBtn.addEventListener("click", () => {
    if (mode === "recording") stopRecording();
    else startRecording();
  });
  el.redo.addEventListener("click", () => {
    // discard take, re-record the SAME sentence
    takeBlob = null;
    if (el.player.src) {
      URL.revokeObjectURL(el.player.src);
      el.player.removeAttribute("src");
    }
    setMode("idle");
  });
  el.send.addEventListener("click", () => {
    el.send.disabled = true;
    sendTake().finally(() => (el.send.disabled = false));
  });
  el.gateRetry.addEventListener("click", () => {
    setMode("idle");
    startRecording();
  });
  el.bannerClose.addEventListener("click", () => {
    localStorage.setItem(LS_MILESTONE, "1");
    el.banner.hidden = true;
  });
  window.addEventListener("online", () => processQueue());
}

async function init() {
  wireEvents();
  renderCounter();
  await loadCorpus();

  if (!corpus.length) {
    el.gateTitle.textContent = "Brak zdań do nagrania";
    el.gateText.textContent =
      "Nie udało się wczytać corpus.json. Odśwież stronę, gdy będzie dostępny (offline: sprawdź połączenie).";
    el.gateRetry.hidden = true;
    setMode("gate");
  } else {
    pickSentence();
    setMode("idle");
  }

  await refreshPending();
  loadCounter();       // async, non-blocking
  processQueue();      // retry anything left from a previous session

  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  }
}

init();
