/* First-party, cookie-free analytics.
 *
 * Batches a page's interactions into one KV write. Respects Do Not Track and
 * Global Privacy Control, and stores no cookies, IPs or user agents. Set
 * localStorage.nfl_no_analytics = "1" to opt out on this device.
 */
(function () {
  "use strict";

  // Always expose a callable so app code never needs to guard every call.
  window.nflTrack = function () {};

  var optedOut = false;
  try {
    optedOut =
      navigator.doNotTrack === "1" ||
      window.globalPrivacyControl === true ||
      localStorage.getItem("nfl_no_analytics") === "1";
  } catch (e) {
    optedOut = navigator.doNotTrack === "1";
  }
  if (optedOut) return;

  var queue = [];
  var timer = null;
  var path = location.pathname || "/";
  var device = window.matchMedia("(max-width: 760px)").matches
    ? "mobile"
    : window.matchMedia("(max-width: 1100px)").matches
    ? "tablet"
    : "desktop";

  function flush() {
    if (!queue.length) return;
    var events = queue.slice(0, 25);
    queue = queue.slice(events.length);
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
    var body = JSON.stringify({
      path: path,
      events: events,
      referrer: document.referrer || "",
      device: device,
    });
    try {
      if (navigator.sendBeacon) {
        var blob = new Blob([body], { type: "application/json" });
        if (navigator.sendBeacon("/api/analytics/collect", blob)) return;
      }
      fetch("/api/analytics/collect", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: body,
        keepalive: true,
        cache: "no-store",
      }).catch(function () {});
    } catch (e) {
      /* analytics must never break the page */
    }
  }

  function track(event, label) {
    var name = String(event || "").slice(0, 32);
    if (!name) return;
    queue.push({ e: name, l: label == null ? "" : String(label).slice(0, 64) });
    if (queue.length >= 10) flush();
    else if (!timer) timer = setTimeout(flush, 2500);
  }
  window.nflTrack = track;

  // Team-page links from the Playoffs table, directory and team pages.
  document.addEventListener(
    "click",
    function (event) {
      var target = event.target;
      var link =
        target && target.closest ? target.closest('a[href^="/teams/"]') : null;
      if (!link) return;
      var parts = (link.getAttribute("href") || "").split("/").filter(Boolean);
      track("team_link", parts[1] || "");
    },
    true
  );

  track("pageview");
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) flush();
  });
  window.addEventListener("pagehide", flush);
  window.addEventListener("beforeunload", flush);
})();
