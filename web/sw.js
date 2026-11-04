// Offline cache for the app shell and the card index (prices always come live).
const CACHE = "scandex-v3";
const SHELL = ["./", "index.html", "css/app.css", "js/app.js", "js/features.js", "js/matcher.js",
  "js/ocr.js", "js/api.js", "js/binder.js", "model/fusion.json", "model/pokemon.json",
  "icon.svg", "manifest.webmanifest"];

self.addEventListener("install", (e) => e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL))));
self.addEventListener("activate", (e) => e.waitUntil(
  caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))));
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return; // APIs, images, CDN: network only
  e.respondWith(fetch(e.request).then((r) => {
    const copy = r.clone();
    caches.open(CACHE).then((c) => c.put(e.request, copy));
    return r;
  }).catch(() => caches.match(e.request)));
});
