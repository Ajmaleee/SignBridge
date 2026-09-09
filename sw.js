/* SignSpeak service worker
   Caches the local app shell so the interface loads offline.
   Hand-tracking model files and library code are fetched from
   a CDN at runtime and cached opportunistically (stale-while-
   revalidate) — they require network on first use.

   Cache version bumped to v2: the app shell now uses a
   network-first strategy (falling back to cache only when
   offline), so future edits to index.html/style.css/script.js
   show up on next reload instead of being stuck behind an old
   cached copy. */

const SHELL_CACHE = "signspeak-shell-v2";
const RUNTIME_CACHE = "signspeak-runtime-v2";

const SHELL_FILES = [
  "./",
  "./index.html",
  "./style.css",
  "./script.js",
  "./manifest.json",
  "./icon-192.png",
  "./icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL_FILES)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== SHELL_CACHE && key !== RUNTIME_CACHE)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  const isSameOrigin = url.origin === self.location.origin;

  if (isSameOrigin) {
    // App shell: network-first, falling back to cache when offline.
    // This means edits to these files are picked up on the very next
    // reload while online, instead of being stuck behind a stale cache.
    event.respondWith(
      fetch(request)
        .then((response) => {
          const clone = response.clone();
          caches.open(SHELL_CACHE).then((cache) => cache.put(request, clone));
          return response;
        })
        .catch(() => caches.match(request))
    );
  } else {
    // CDN / model assets: stale-while-revalidate
    event.respondWith(
      caches.open(RUNTIME_CACHE).then((cache) =>
        cache.match(request).then((cached) => {
          const network = fetch(request)
            .then((response) => {
              if (response && response.status === 200) {
                cache.put(request, response.clone());
              }
              return response;
            })
            .catch(() => cached);
          return cached || network;
        })
      )
    );
  }
});

