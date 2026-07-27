/* G-Voice Recorder — offline app-shell service worker.
   Caches the shell + corpus.json + the Firebase CDN modules so the app boots
   with no signal. Firestore/Storage API traffic is never cached (dynamic).   */

const CACHE = "g-voice-shell-v1";

const SHELL = [
  "./nagrywaj.html",
  "./styles.css",
  "./app.js",
  "./manifest.json",
  "./corpus.json",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./icons/apple-touch-icon.png",
  "./icons/favicon-32.png",
];

const FB_VERSION = "10.12.2";
const FB_MODULES = ["app", "firestore"].map(
  (m) => `https://www.gstatic.com/firebasejs/${FB_VERSION}/firebase-${m}.js`
);

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      // Shell must cache; Firebase modules are best-effort (may be blocked offline).
      await cache.addAll(SHELL);
      await Promise.allSettled(FB_MODULES.map((u) => cache.add(u)));
      self.skipWaiting();
    })()
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)));
      await self.clients.claim();
    })()
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  const sameOrigin = url.origin === self.location.origin;
  const isFirebaseCdn = url.origin === "https://www.gstatic.com" && url.pathname.includes("/firebasejs/");

  // Only handle our own shell + the Firebase CDN modules. Everything else
  // (firestore.googleapis.com, firebasestorage, etc.) goes straight to network.
  if (!sameOrigin && !isFirebaseCdn) return;

  if (isFirebaseCdn) {
    // Cache-first: these are versioned/immutable.
    event.respondWith(
      caches.match(req).then(
        (hit) =>
          hit ||
          fetch(req).then((res) => {
            const copy = res.clone();
            caches.open(CACHE).then((c) => c.put(req, copy));
            return res;
          })
      )
    );
    return;
  }

  // Same-origin shell: stale-while-revalidate so corpus.json / code updates land.
  event.respondWith(
    caches.match(req).then((hit) => {
      const fetchPromise = fetch(req)
        .then((res) => {
          if (res && res.status === 200 && res.type === "basic") {
            const copy = res.clone();
            caches.open(CACHE).then((c) => c.put(req, copy));
          }
          return res;
        })
        .catch(() => hit);
      return hit || fetchPromise;
    })
  );
});
