/**
 * NFL ML external GitHub Actions dispatcher.
 *
 * Replaces GitHub's best-effort schedule queue with an Apps Script 1-minute
 * trigger that dispatches workflows at exact America/New_York wall-clock times.
 *
 * Setup: call nflSetupTriggers() once (reuses GITHUB_TOKEN or NFL_GITHUB_TOKEN).
 */

var NFL_CONFIG = {
  owner: "hoskiphotoshare-boop",
  repo: "nfl-ml-data",
  ref: "main",
  timeZone: "America/New_York",
  catchupWindowMinutes: 15,
  tokenProperty: "NFL_GITHUB_TOKEN",
  fallbackTokenProperty: "GITHUB_TOKEN",
  lastDispatchProperty: "NFL_LAST_DISPATCH",
  firedPrefix: "NFL_FIRED_"
};

var NFL_SLOTS = [
  // Tuesday full retrain after MNF. Workflow waits for the latest week.
  { id: "weekly_retrain_tue_0600", days: [2], hour: 6, minute: 0, workflow: "ingest-and-train.yml", inputs: { fetch_odds: true, create_report: true } },
  // Inference-only refresh Wednesday-Monday.
  { id: "daily_inference_0600", days: [1, 3, 4, 5, 6, 7], hour: 6, minute: 0, workflow: "daily-refresh.yml", inputs: {} },

  // Odds windows. Every odds run refreshes inference inside the workflow.
  { id: "odds_mon_sat_0700", days: [1, 2, 3, 4, 5, 6], hour: 7, minute: 0, workflow: "odds-refresh.yml", inputs: { force: false } },
  { id: "odds_tue_wed_fri_sat_1730", days: [2, 3, 5, 6], hour: 17, minute: 30, workflow: "odds-refresh.yml", inputs: { force: false } },
  { id: "odds_sun_0800", days: [7], hour: 8, minute: 0, workflow: "odds-refresh.yml", inputs: { force: false } },
  { id: "odds_sun_1130", days: [7], hour: 11, minute: 30, workflow: "odds-refresh.yml", inputs: { force: false } },
  { id: "odds_sun_1430", days: [7], hour: 14, minute: 30, workflow: "odds-refresh.yml", inputs: { force: false } },
  { id: "odds_sun_1515", days: [7], hour: 15, minute: 15, workflow: "odds-refresh.yml", inputs: { force: false } },
  { id: "odds_sun_1950", days: [7], hour: 19, minute: 50, workflow: "odds-refresh.yml", inputs: { force: false } },
  { id: "odds_thu_1200", days: [4], hour: 12, minute: 0, workflow: "odds-refresh.yml", inputs: { force: false } },
  { id: "odds_thu_1845", days: [4], hour: 18, minute: 45, workflow: "odds-refresh.yml", inputs: { force: false } },
  { id: "odds_mon_1845", days: [1], hour: 18, minute: 45, workflow: "odds-refresh.yml", inputs: { force: false } }
];

function nflGetToken_(props) {
  var p = props || PropertiesService.getScriptProperties();
  return p.getProperty(NFL_CONFIG.tokenProperty) || p.getProperty(NFL_CONFIG.fallbackTokenProperty) || "";
}

function nflDispatchWorkflow_(workflow, inputs, token) {
  var url = "https://api.github.com/repos/" + NFL_CONFIG.owner + "/" + NFL_CONFIG.repo +
    "/actions/workflows/" + encodeURIComponent(workflow) + "/dispatches";
  var payload = { ref: NFL_CONFIG.ref };
  if (inputs && Object.keys(inputs).length > 0) payload.inputs = inputs;
  var response = UrlFetchApp.fetch(url, {
    method: "post",
    contentType: "application/json",
    muteHttpExceptions: true,
    headers: {
      Authorization: "Bearer " + token,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "nfl-ml-gas-dispatcher"
    },
    payload: JSON.stringify(payload)
  });
  var code = response.getResponseCode();
  if (code === 204 || (code >= 200 && code < 300)) return { ok: true, statusCode: code };
  return { ok: false, statusCode: code, error: "HTTP " + code + ": " + response.getContentText().slice(0, 400) };
}

function nflCronTick() {
  var now = new Date();
  var parts = Utilities.formatDate(now, NFL_CONFIG.timeZone, "yyyy-MM-dd|HH|mm|u").split("|");
  var dateStr = parts[0];
  var hour = parseInt(parts[1], 10);
  var minute = parseInt(parts[2], 10);
  var dayOfWeek = parseInt(parts[3], 10);
  var nowMinutes = hour * 60 + minute;
  var etLabel = Utilities.formatDate(now, NFL_CONFIG.timeZone, "yyyy-MM-dd HH:mm z");

  var due = [];
  for (var i = 0; i < NFL_SLOTS.length; i++) {
    var slot = NFL_SLOTS[i];
    if (slot.days.indexOf(dayOfWeek) === -1) continue;
    var diff = nowMinutes - (slot.hour * 60 + slot.minute);
    if (diff >= 0 && diff <= NFL_CONFIG.catchupWindowMinutes) due.push(slot);
  }
  if (due.length === 0) return { ok: true, fastPathExit: true, etTime: etLabel, dispatched: [] };

  var lock = LockService.getScriptLock();
  if (!lock.tryLock(10000)) return { ok: false, error: "lock busy", etTime: etLabel };
  try {
    var props = PropertiesService.getScriptProperties();
    var token = nflGetToken_(props);
    if (!token) return { ok: false, error: "NFL_GITHUB_TOKEN/GITHUB_TOKEN not configured", etTime: etLabel };
    var actions = [];
    for (var j = 0; j < due.length; j++) {
      var s = due[j];
      var propKey = NFL_CONFIG.firedPrefix + s.id;
      if (props.getProperty(propKey) === dateStr) continue;
      var res = nflDispatchWorkflow_(s.workflow, s.inputs, token);
      if (res.ok) {
        props.setProperty(propKey, dateStr);
        var record = { slotId: s.id, workflow: s.workflow, inputs: s.inputs, status: "dispatched", statusCode: res.statusCode, etTime: etLabel };
        props.setProperty(NFL_CONFIG.lastDispatchProperty, JSON.stringify(record));
        Logger.log("Dispatched " + s.id + " -> " + s.workflow + " (" + res.statusCode + ") at " + etLabel);
        actions.push(record);
      } else {
        Logger.log("Failed " + s.id + " -> " + s.workflow + ": " + res.error);
        actions.push({ slotId: s.id, workflow: s.workflow, status: "dispatch_error", error: res.error });
      }
    }
    return { ok: true, etTime: etLabel, dispatched: actions };
  } finally {
    lock.releaseLock();
  }
}

function nflSetupTriggers(githubToken) {
  var props = PropertiesService.getScriptProperties();
  if (githubToken && String(githubToken).trim()) props.setProperty(NFL_CONFIG.tokenProperty, String(githubToken).trim());
  var existing = ScriptApp.getProjectTriggers();
  var removed = 0;
  for (var i = 0; i < existing.length; i++) {
    var handler = existing[i].getHandlerFunction();
    if (handler === "nflCronTick" || handler === "nflCheckRepoAccess" || handler === "nflTestDispatch") {
      ScriptApp.deleteTrigger(existing[i]);
      removed++;
    }
  }
  var created = ScriptApp.newTrigger("nflCronTick").timeBased().everyMinutes(1).create();
  return { ok: true, removedOldTriggers: removed, triggerId: created.getUniqueId(), hasToken: Boolean(nflGetToken_(props)), slots: NFL_SLOTS.length };
}

function nflCheckRepoAccess() {
  var token = nflGetToken_();
  if (!token) return { ok: false, error: "No GitHub token" };
  var response = UrlFetchApp.fetch("https://api.github.com/repos/" + NFL_CONFIG.owner + "/" + NFL_CONFIG.repo + "/actions/workflows", {
    method: "get", muteHttpExceptions: true,
    headers: { Authorization: "Bearer " + token, Accept: "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "nfl-ml-gas-dispatcher" }
  });
  var code = response.getResponseCode();
  var names = [];
  try {
    var data = JSON.parse(response.getContentText());
    names = (data.workflows || []).map(function (w) { return w.name + ":" + w.path; });
  } catch (e) {}
  return { ok: code >= 200 && code < 300, statusCode: code, workflows: names, error: code >= 200 && code < 300 ? null : response.getContentText().slice(0, 300) };
}

function nflStatus() {
  var now = new Date();
  var props = PropertiesService.getScriptProperties();
  var triggers = ScriptApp.getProjectTriggers();
  var triggerIds = [];
  for (var i = 0; i < triggers.length; i++) if (triggers[i].getHandlerFunction() === "nflCronTick") triggerIds.push(triggers[i].getUniqueId());
  return {
    ok: true,
    repo: NFL_CONFIG.owner + "/" + NFL_CONFIG.repo,
    timeZone: NFL_CONFIG.timeZone,
    etTime: Utilities.formatDate(now, NFL_CONFIG.timeZone, "yyyy-MM-dd HH:mm:ss z"),
    triggerInstalled: triggerIds.length > 0,
    triggerIds: triggerIds,
    hasToken: Boolean(nflGetToken_(props)),
    slots: NFL_SLOTS.length,
    lastDispatch: props.getProperty(NFL_CONFIG.lastDispatchProperty) || null
  };
}
