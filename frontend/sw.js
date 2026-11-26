/* NFL ML Predictions — service worker.
 *
 * Caches the app shell only. Predictions, live scores and player stats are
 * network-only and are never written to or served from a cache.
 *
 * Bump CACHE_VERSION whenever a shell asset (HTML/CSS/JS/icons) changes so
 * clients pick up the new shell on their next visit. The activate handler
 * deletes every older `nfl-ml-shell-*` cache.
 */

const CACHE_VERSION = 'nfl-ml-v2';
const SHELL_CACHE = `nfl-ml-shell-${CACHE_VERSION}`;

const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/style.css',
  '/app.js',
  '/theme.js',
  '/table-sort.js',
  '/pwa.js',
  '/manifest.webmanifest',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
  '/icons/icon-maskable-512.png',
  '/icons/apple-touch-icon.png',
  '/icons/favicon-32.png',
  '/icons/favicon.svg',
];

const NETWORK_TIMEOUT_MS = 2500;

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key.startsWith('nfl-ml-shell-') && key !== SHELL_CACHE)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

function withTimeout(promise, ms) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('network-timeout')), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      }
    );
  });
}

function isShellAsset(pathname) {
  return PRECACHE_URLS.includes(pathname) || pathname.startsWith('/icons/');
}

async function networkFirst(request, cache) {
  const response = await withTimeout(fetch(request), NETWORK_TIMEOUT_MS);
  if (response && response.ok) {
    // Await the write so it survives the worker being suspended. The response
    // clone is consumed here; the original is still returned to the page.
    await cache.put(request, response.clone()).catch(() => {});
  }
  return response;
}

async function handleNavigate(request) {
  const cache = await caches.open(SHELL_CACHE);
  try {
    return await networkFirst(request, cache);
  } catch (error) {
    const cached = await cache.match(request, { ignoreSearch: true });
    if (cached) return cached;
    const shell = (await cache.match('/index.html')) || (await cache.match('/'));
    if (shell) return shell;
    throw error;
  }
}

async function handleShellAsset(request) {
  const cache = await caches.open(SHELL_CACHE);
  try {
    return await networkFirst(request, cache);
  } catch (error) {
    const cached = await cache.match(request);
    if (cached) return cached;
    throw error;
  }
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  const { pathname } = url;

  // Internal diagnostics page: always network-only, never cached.
  if (pathname === '/admin' || pathname === '/admin.html' || pathname.startsWith('/admin/')) {
    return;
  }

  // Live data and API proxies: network-only, never cached.
  if (pathname.startsWith('/data/') || pathname.startsWith('/api/')) {
    return;
  }

  // Navigations: network-first with a short timeout, then the cached page
  // (team pages) or the app shell as an offline fallback.
  if (request.mode === 'navigate') {
    event.respondWith(handleNavigate(request));
    return;
  }

  // Static shell assets: network-first with cache fallback. Filenames are
  // unversioned, so cache-first would serve stale JS/CSS after a deploy.
  if (isShellAsset(pathname)) {
    event.respondWith(handleShellAsset(request));
  }
});
