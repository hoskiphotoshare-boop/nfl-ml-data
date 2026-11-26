/* NFL ML Predictions — PWA registration + offline banner.
 * Loaded by every public page. No install-prompt UI (phase 2). */
(function () {
  'use strict';

  var root = document.documentElement;

  // Mark standalone launches so CSS can add a status-bar inset.
  var standalone = false;
  try {
    standalone =
      (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches) ||
      navigator.standalone === true;
  } catch (e) {
    standalone = false;
  }
  if (standalone) root.classList.add('standalone');

  // Minimal offline banner. Fixed-positioned by CSS so it never shifts layout
  // or covers the fixed mobile bottom nav.
  var banner = null;

  function makeBanner() {
    var el = document.createElement('div');
    el.className = 'offline-banner';
    el.setAttribute('role', 'status');
    el.setAttribute('aria-live', 'polite');
    var dot = document.createElement('span');
    dot.className = 'offline-dot';
    dot.setAttribute('aria-hidden', 'true');
    var text = document.createElement('span');
    text.textContent = "You’re offline — predictions will refresh when you reconnect";
    el.appendChild(dot);
    el.appendChild(text);
    return el;
  }

  function showOffline() {
    root.classList.add('offline');
    if (!banner && document.body) {
      banner = makeBanner();
      document.body.appendChild(banner);
    }
  }

  function hideOffline() {
    root.classList.remove('offline');
    if (banner && banner.parentNode) {
      banner.parentNode.removeChild(banner);
    }
    banner = null;
  }

  window.addEventListener('offline', showOffline);
  window.addEventListener('online', hideOffline);
  if (navigator.onLine === false) showOffline();

  // Register only on secure origins (HTTPS or localhost).
  var host = location.hostname;
  var secure = location.protocol === 'https:' || host === 'localhost' || host === '127.0.0.1';
  if (!secure || !('serviceWorker' in navigator)) return;

  // Reload once when a new service worker takes control; the guard prevents a
  // loop if controllerchange fires again. Skip the reload on the very first
  // install, where there was no previous controller and nothing is stale.
  var hadController = !!navigator.serviceWorker.controller;
  var refreshing = false;
  navigator.serviceWorker.addEventListener('controllerchange', function () {
    if (!hadController || refreshing) return;
    refreshing = true;
    location.reload();
  });

  navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch(function () {
    /* registration is best-effort; the app still works online */
  });
})();