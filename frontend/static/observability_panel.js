/* ── Observability Panel (embedded in main page) ── */
(function () {
  "use strict";

  // ── DOM refs ──
  var panel        = document.getElementById("obsPanel");
  var toggleBtn    = document.getElementById("obsToggleBtn");
  var closeBtn     = document.getElementById("obsPanelCloseBtn");
  var mainEl       = document.getElementById("obsPanelMain");
  var perfBarEl    = document.getElementById("obsPanelPerfBar");
  var resizeHandle = document.getElementById("obsResizeHandle");

  if (!panel || !toggleBtn) return; // safety: not on the right page

  // ── State ──
  var activeSessionId = null;
  var traceData = null;
  var activeCategory = "all";
  var textFilter = "";
  var expandedRow = null;
  var collapsedGroups = {};
  var refreshTimer = null;
  var _searchDebounceTimer = null;

  // ── Resize constraints ──
  var MIN_PANEL_W = 280;
  var MAX_PANEL_W = 900;

  // ── Performance tracking state ──
  var perfHistory = [];
  var perfHistoryOpen = false;
  var PERF_HISTORY_MAX = 20;

  // ── Helpers (adapted from observability.js) ──
  function fmt(ms) {
    if (ms == null) return "";
    if (ms < 1000) return Math.round(ms) + "ms";
    if (ms < 60000) return (ms / 1000).toFixed(1) + "s";
    return (ms / 60000).toFixed(1) + "m";
  }

  function esc(s) {
    if (!s) return "";
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function truncate(s, n) {
    if (!s) return "";
    return s.length > n ? s.slice(0, n) + "\u2026" : s;
  }

  function nodeDepth(nodeId) {
    if (!nodeId) return 0;
    return nodeId.split(".").length;
  }

  // ── Performance helpers (ported from observability.js) ──

  /** Parse Server-Timing header into { total, file_read, parse } in ms */
  function parseServerTiming(resp) {
    var header = resp.headers.get("Server-Timing");
    if (!header) return null;
    var result = {};
    var parts = header.split(",");
    for (var i = 0; i < parts.length; i++) {
      var seg = parts[i].trim();
      var nameMatch = seg.match(/^([^;]+)/);
      var durMatch = seg.match(/dur=([\d.]+)/);
      if (nameMatch && durMatch) {
        result[nameMatch[1].trim()] = parseFloat(durMatch[1]);
      }
    }
    return result;
  }

  /** Color class for a timing value */
  function perfColor(ms) {
    if (ms == null) return "";
    if (ms < 100) return "obs-perf-green";
    if (ms <= 500) return "obs-perf-yellow";
    return "obs-perf-red";
  }

  /** Format KB value */
  function fmtKB(kb) {
    if (kb == null) return "\u2014";
    return Math.round(kb) + "KB";
  }

  /** Record a perf measurement and update the bar */
  function recordPerf(rec) {
    rec.timestamp = new Date().toLocaleTimeString();
    perfHistory.push(rec);
    if (perfHistory.length > PERF_HISTORY_MAX) {
      perfHistory.shift();
    }
    updatePerfBar();
  }

  /** Helper to create a perfbar metric item */
  function perfItem(label, ms) {
    var val = ms != null ? Math.round(ms) + "ms" : "\u2014";
    return '<span class="obs-perfbar-item"><span class="obs-perf-key">' + label + ':</span> ' +
      '<span class="obs-perf-val ' + perfColor(ms) + '">' + val + '</span></span>';
  }

  /** Render the Performance Bar UI */
  function updatePerfBar() {
    if (!perfBarEl) return;

    if (!perfHistory.length) {
      perfBarEl.innerHTML = '<div class="obs-perfbar-row">' +
        '<span class="obs-perfbar-label">\u26a1 Perf</span>' +
        '<span class="obs-perfbar-sep">\u2502</span>' +
        '<span style="color:var(--muted)">\u2014</span>' +
        '</div>';
      return;
    }

    var latest = perfHistory[perfHistory.length - 1];

    var html = '<div class="obs-perfbar-row">';
    html += '<span class="obs-perfbar-label">\u26a1 Perf</span>';
    html += '<span class="obs-perfbar-sep">\u2502</span>';

    // Action
    html += '<span class="obs-perfbar-item"><span class="obs-perf-key">' +
      esc(latest.action) + '</span></span>';
    html += '<span class="obs-perfbar-sep">\u2502</span>';

    // Total
    html += perfItem("Total", latest.total_ms);
    html += '<span class="obs-perfbar-sep">\u2502</span>';

    // API (only for load_session)
    if (latest.api_ms != null) {
      var serverSub = "";
      if (latest.server_ms && latest.server_ms.total != null) {
        serverSub = ' <span class="obs-perf-sub">(srv:' + Math.round(latest.server_ms.total) + 'ms)</span>';
      }
      html += '<span class="obs-perfbar-item"><span class="obs-perf-key">API:</span> ' +
        '<span class="obs-perf-val ' + perfColor(latest.api_ms) + '">' + Math.round(latest.api_ms) + 'ms</span>' +
        serverSub + '</span>';
      html += '<span class="obs-perfbar-sep">\u2502</span>';
    }

    // Render
    html += perfItem("Render", latest.render_ms);
    html += '<span class="obs-perfbar-sep">\u2502</span>';

    // Rows
    if (latest.entry_count != null) {
      html += '<span class="obs-perfbar-item"><span class="obs-perf-key">Rows:</span> ' +
        '<span class="obs-perf-val">' + latest.entry_count + '</span></span>';
      html += '<span class="obs-perfbar-sep">\u2502</span>';
    }

    // History toggle
    html += '<button class="obs-perfbar-history-btn" id="obsPanelPerfHistoryToggle">' +
      'History ' + (perfHistoryOpen ? '\u25b4' : '\u25be') + '</button>';

    html += '</div>';

    // History table
    html += '<div class="obs-perfbar-history' + (perfHistoryOpen ? ' open' : '') + '" id="obsPanelPerfHistoryPanel">';
    html += '<table><thead><tr>' +
      '<th>Time</th><th>Action</th><th>Total</th><th>API</th><th>Server</th>' +
      '<th>Render</th><th>Rows</th>' +
      '</tr></thead><tbody>';

    for (var i = perfHistory.length - 1; i >= 0; i--) {
      var r = perfHistory[i];
      var serverTotal = (r.server_ms && r.server_ms.total != null) ? Math.round(r.server_ms.total) + "ms" : "\u2014";
      html += '<tr>' +
        '<td>' + esc(r.timestamp) + '</td>' +
        '<td>' + esc(r.action) + '</td>' +
        '<td class="' + perfColor(r.total_ms) + '">' + (r.total_ms != null ? Math.round(r.total_ms) + 'ms' : '\u2014') + '</td>' +
        '<td class="' + perfColor(r.api_ms) + '">' + (r.api_ms != null ? Math.round(r.api_ms) + 'ms' : '\u2014') + '</td>' +
        '<td>' + serverTotal + '</td>' +
        '<td class="' + perfColor(r.render_ms) + '">' + (r.render_ms != null ? Math.round(r.render_ms) + 'ms' : '\u2014') + '</td>' +
        '<td>' + (r.entry_count != null ? r.entry_count : '\u2014') + '</td>' +
        '</tr>';
    }

    html += '</tbody></table></div>';

    perfBarEl.innerHTML = html;

    // Bind history toggle
    var histToggle = document.getElementById("obsPanelPerfHistoryToggle");
    if (histToggle) {
      histToggle.addEventListener("click", function () {
        perfHistoryOpen = !perfHistoryOpen;
        updatePerfBar();
      });
    }
  }

  // ── Panel open/close ──
  function isPanelOpen() {
    return document.body.classList.contains("obs-panel-open");
  }

  /** Read the currently active session ID from the page URL */
  function getCurrentSessionFromUrl() {
    try {
      var url = new URL(window.location.href);
      return (url.searchParams.get("session_id") || url.searchParams.get("run_id") || "").trim() || null;
    } catch (_) { return null; }
  }

  function openPanel() {
    document.body.classList.add("obs-panel-open");
    panel.classList.remove("hidden");
    if (resizeHandle) resizeHandle.classList.remove("hidden");
    localStorage.setItem("sb:obs-panel-open", "1");
    // Restore persisted width
    var savedW = parseInt(localStorage.getItem("sb:obs-panel-width"), 10);
    if (savedW && savedW >= MIN_PANEL_W && savedW <= MAX_PANEL_W) {
      document.documentElement.style.setProperty("--obs-panel-width", savedW + "px");
    }
    updatePerfBar();
    // Auto-load the currently selected session from URL
    var currentSid = getCurrentSessionFromUrl();
    if (currentSid && currentSid !== activeSessionId) {
      selectSession(currentSid);
    }
  }

  function closePanel() {
    document.body.classList.remove("obs-panel-open");
    panel.classList.add("hidden");
    if (resizeHandle) resizeHandle.classList.add("hidden");
    localStorage.setItem("sb:obs-panel-open", "0");
    if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null; }
  }

  function togglePanel() {
    if (isPanelOpen()) {
      closePanel();
    } else {
      openPanel();
    }
  }

  // Bind toggle/close buttons
  toggleBtn.addEventListener("click", togglePanel);
  if (closeBtn) closeBtn.addEventListener("click", closePanel);

  // Restore persisted state
  if (localStorage.getItem("sb:obs-panel-open") === "1") {
    openPanel();
  }

  // ── Resize handle drag logic ──
  if (resizeHandle) {
    var isDragging = false;

    resizeHandle.addEventListener("mousedown", function (e) {
      e.preventDefault();
      isDragging = true;
      document.body.classList.add("obs-resizing");
      resizeHandle.classList.add("is-dragging");

      function onMouseMove(e) {
        if (!isDragging) return;
        // Panel width = distance from mouse to right edge of viewport
        var newW = window.innerWidth - e.clientX;
        if (newW < MIN_PANEL_W) newW = MIN_PANEL_W;
        if (newW > MAX_PANEL_W) newW = MAX_PANEL_W;
        document.documentElement.style.setProperty("--obs-panel-width", newW + "px");
      }

      function onMouseUp() {
        isDragging = false;
        document.body.classList.remove("obs-resizing");
        resizeHandle.classList.remove("is-dragging");
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("mouseup", onMouseUp);
        // Persist the width
        var finalW = parseInt(getComputedStyle(document.documentElement).getPropertyValue("--obs-panel-width"), 10);
        if (finalW) localStorage.setItem("sb:obs-panel-width", String(finalW));
      }

      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
    });
  }

  // ── Select session & load trace (with perf timing) ──
  async function selectSession(sid) {
    var t0 = performance.now();

    activeSessionId = sid;
    expandedRow = null;
    collapsedGroups = {};
    activeCategory = "all";
    textFilter = "";
    mainEl.innerHTML = '<div class="obs-loading">Loading trace\u2026</div>';

    if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null; }

    try {
      var url = "/api/observability/sessions/" + encodeURIComponent(sid) + "/trace";
      var resp = await fetch(url);
      if (!resp.ok) throw new Error("HTTP " + resp.status);

      var serverTiming = parseServerTiming(resp);
      var t2 = performance.now();

      traceData = await resp.json();
      var t3 = performance.now();

      renderTrace();
      var t4 = performance.now();

      var responseKB = null;
      if (traceData.summary && traceData.summary.response_size_bytes != null) {
        responseKB = traceData.summary.response_size_bytes / 1024;
      }

      recordPerf({
        action: "load_session",
        total_ms: t4 - t0,
        api_ms: t2 - t0,
        server_ms: serverTiming,
        parse_ms: t3 - t2,
        render_ms: t4 - t3,
        response_kb: responseKB,
        entry_count: (traceData.entries || []).length,
      });

      // auto-refresh if running
      var status = (traceData.status || "").toLowerCase();
      if (status === "running" || status === "started" || status === "unknown") {
        refreshTimer = setInterval(async function () {
          try {
            traceData = await (await fetch("/api/observability/sessions/" + encodeURIComponent(sid) + "/trace")).json();
            var rt0 = performance.now();
            renderTrace();
            var rt1 = performance.now();
            recordPerf({
              action: "auto_refresh",
              total_ms: rt1 - rt0,
              render_ms: rt1 - rt0,
              entry_count: (traceData.entries || []).length,
            });
          } catch (_) {}
        }, 3000);
      }
    } catch (e) {
      mainEl.innerHTML = '<div class="obs-empty">Failed to load trace</div>';
    }
  }

  // ── Render trace ──
  function renderTrace() {
    if (!traceData) return;
    var t = traceData;
    var sum = t.summary || {};

    // Summary
    var html = '<div class="obs-summary">' +
      '<div class="obs-summary-task">' + esc(t.task || t.session_id) + '</div>' +
      '<div class="obs-summary-row">' +
        statusBadge(t.status) +
        metric("Duration", fmt(t.total_duration_ms)) +
        metric("LLM", sum.llm_calls) +
        metric("Searches", sum.searches) +
        metric("Tokens", sum.total_tokens ? sum.total_tokens.toLocaleString() : "-") +
        metric("Events", sum.events) +
      '</div></div>';

    // Filter bar
    html += '<div class="obs-filters">' +
      filterBtn("all", "All") +
      filterBtn("llm_call", "LLM") +
      filterBtn("search", "Search") +
      filterBtn("agent_event", "Agent") +
      filterBtn("lifecycle", "Lifecycle") +
      filterBtn("error", "Error") +
      '<input type="text" class="obs-filter-search" id="obsPanelTraceSearch" placeholder="Search\u2026" value="' + esc(textFilter) + '" />' +
    '</div>';

    // Timeline table
    html += '<div class="obs-timeline-wrap"><table class="obs-timeline">' +
      '<thead><tr><th>Offset</th><th>Event</th><th>Dur</th><th>Bar</th></tr></thead><tbody>';

    var entries = filterEntries(t.entries || []);
    var maxDur = 0;
    for (var i = 0; i < entries.length; i++) {
      var d = entries[i].duration_ms;
      if (d && d > maxDur) maxDur = d;
    }

    for (var i = 0; i < entries.length; i++) {
      var e = entries[i];
      var origIdx = e._origIdx;
      var depth = nodeDepth(e.node_id);
      if (depth > 4) depth = 4;
      var isExpanded = expandedRow === origIdx;

      // check if hidden by collapsed group
      if (isGroupHidden(e)) continue;

      // is this a group header?
      var isGroupHead = e.node_id && hasChildren(entries, e.node_id, i);
      var groupArrow = "";
      if (isGroupHead) {
        groupArrow = collapsedGroups[e.node_id] ? "\u25b8 " : "\u25be ";
      }

      html += '<tr class="' + (isExpanded ? "expanded" : "") +
              (isGroupHead ? " obs-group-header" : "") +
              '" data-idx="' + origIdx + '" data-nid="' + esc(e.node_id || "") + '">' +
        '<td>' + fmt(e.offset_ms) + '</td>' +
        '<td class="obs-indent-' + depth + '">' +
          groupArrow + esc(e.message) +
          (e.node_id ? ' <span style="color:var(--muted);font-size:9px">[' + esc(e.node_id) + ']</span>' : '') +
        '</td>' +
        '<td>' + fmt(e.duration_ms) + '</td>' +
        '<td class="obs-bar-cell">' + barHTML(e, maxDur) + '</td>' +
      '</tr>';

      // expanded detail row
      if (isExpanded) {
        html += '<tr class="obs-detail-row"><td colspan="4">' +
          '<div class="obs-detail-panel">' +
            '<button class="obs-copy-btn" data-idx="' + origIdx + '">Copy JSON</button>' +
            '<pre>' + esc(JSON.stringify(e.detail || e, null, 2)) + '</pre>' +
          '</div></td></tr>';
      }
    }

    html += '</tbody></table></div>';
    mainEl.innerHTML = html;

    // Bind events
    bindTraceEvents();
  }

  function filterEntries(entries) {
    var out = [];
    for (var i = 0; i < entries.length; i++) {
      var e = entries[i];
      e._origIdx = i;
      var cat = e.category || "lifecycle";
      if (activeCategory !== "all" && cat !== activeCategory) continue;
      if (textFilter && (e.message || "").toLowerCase().indexOf(textFilter) < 0) continue;
      out.push(e);
    }
    return out;
  }

  function isGroupHidden(entry) {
    if (!entry.node_id) return false;
    var parts = entry.node_id.split(".");
    for (var i = 1; i < parts.length; i++) {
      var parent = parts.slice(0, i).join(".");
      if (collapsedGroups[parent]) return true;
    }
    return false;
  }

  function hasChildren(entries, nodeId, startIdx) {
    var prefix = nodeId + ".";
    for (var i = startIdx + 1; i < entries.length; i++) {
      var nid = entries[i].node_id || "";
      if (nid.indexOf(prefix) === 0) return true;
    }
    return false;
  }

  function barHTML(entry, maxDur) {
    if (!entry.duration_ms || !maxDur) {
      return '<span class="obs-dot cat-' + (entry.category || "lifecycle") + '"></span>';
    }
    var pct = Math.max(2, Math.min(100, (entry.duration_ms / maxDur) * 100));
    return '<span class="obs-bar cat-' + (entry.category || "lifecycle") + '" style="width:' + pct + '%"></span>' +
           '<span class="obs-bar-label">' + fmt(entry.duration_ms) + '</span>';
  }

  function statusBadge(status) {
    var cls = (status || "unknown").toLowerCase();
    if (["completed", "running", "failed"].indexOf(cls) < 0) cls = "unknown";
    return '<span class="obs-status-badge ' + cls + '">' + esc(status || "?") + '</span>';
  }

  function metric(label, val) {
    return '<div class="obs-metric"><span>' + esc(label) + '</span> <strong>' + esc(String(val != null ? val : "-")) + '</strong></div>';
  }

  function filterBtn(cat, label) {
    var cls = activeCategory === cat ? " active" : "";
    return '<button class="obs-filter-btn' + cls + '" data-cat="' + cat + '">' + esc(label) + '</button>';
  }

  // ── Bind trace interaction events (with perf timing) ──
  function bindTraceEvents() {
    // row click → expand/collapse
    var rows = mainEl.querySelectorAll(".obs-timeline tr[data-idx]");
    for (var i = 0; i < rows.length; i++) {
      rows[i].addEventListener("click", function () {
        var idx = parseInt(this.getAttribute("data-idx"), 10);
        var nid = this.getAttribute("data-nid");

        // if it's a group header, toggle group
        if (this.classList.contains("obs-group-header") && nid) {
          collapsedGroups[nid] = !collapsedGroups[nid];
          var rt0 = performance.now();
          renderTrace();
          var rt1 = performance.now();
          recordPerf({
            action: "toggle_group",
            total_ms: rt1 - rt0,
            render_ms: rt1 - rt0,
            entry_count: (traceData.entries || []).length,
          });
          return;
        }

        // toggle detail
        if (expandedRow === idx) {
          expandedRow = null;
        } else {
          expandedRow = idx;
        }
        var rt0 = performance.now();
        renderTrace();
        var rt1 = performance.now();
        recordPerf({
          action: "toggle_row",
          total_ms: rt1 - rt0,
          render_ms: rt1 - rt0,
          entry_count: (traceData.entries || []).length,
        });
      });
    }

    // filter buttons
    var btns = mainEl.querySelectorAll(".obs-filter-btn");
    for (var i = 0; i < btns.length; i++) {
      btns[i].addEventListener("click", function (ev) {
        ev.stopPropagation();
        activeCategory = this.getAttribute("data-cat");
        expandedRow = null;
        var rt0 = performance.now();
        renderTrace();
        var rt1 = performance.now();
        recordPerf({
          action: "filter",
          total_ms: rt1 - rt0,
          render_ms: rt1 - rt0,
          entry_count: (traceData.entries || []).length,
        });
      });
    }

    // text search (debounced)
    var searchInput = document.getElementById("obsPanelTraceSearch");
    if (searchInput) {
      searchInput.addEventListener("input", function () {
        var self = this;
        if (_searchDebounceTimer) clearTimeout(_searchDebounceTimer);
        _searchDebounceTimer = setTimeout(function () {
          textFilter = self.value.toLowerCase().trim();
          expandedRow = null;
          var rt0 = performance.now();
          renderTrace();
          var rt1 = performance.now();
          recordPerf({
            action: "filter",
            total_ms: rt1 - rt0,
            render_ms: rt1 - rt0,
            entry_count: (traceData.entries || []).length,
          });
        }, 300);
      });
      // keep focus after re-render
      searchInput.focus();
      searchInput.setSelectionRange(searchInput.value.length, searchInput.value.length);
    }

    // copy buttons
    var copyBtns = mainEl.querySelectorAll(".obs-copy-btn");
    for (var i = 0; i < copyBtns.length; i++) {
      copyBtns[i].addEventListener("click", function (ev) {
        ev.stopPropagation();
        var idx = parseInt(this.getAttribute("data-idx"), 10);
        var entry = (traceData.entries || [])[idx];
        if (entry) {
          navigator.clipboard.writeText(JSON.stringify(entry.detail || entry, null, 2))
            .then(function () { ev.target.textContent = "Copied!"; })
            .catch(function () {});
        }
      });
    }
  }

  // ── Listen for session-selected event from main app ──
  document.addEventListener("sb:session-selected", function (e) {
    var sid = e.detail && e.detail.sessionId;
    if (!sid) return;
    if (isPanelOpen()) {
      selectSession(sid);
    } else {
      // Panel is closed — just remember the sid so we can load when opened
      activeSessionId = null;
    }
  });

  // ── Init ──
  updatePerfBar(); // show initial empty state

  // ── Expose for external use (optional) ──
  window.ObsPanel = {
    open: openPanel,
    close: closePanel,
    toggle: togglePanel,
    isOpen: isPanelOpen,
    selectSession: selectSession,
  };

})();
