/* ── Observability Dashboard – frontend logic ── */
(function () {
  "use strict";

  // ── DOM refs ──
  const sessionListEl = document.getElementById("sessionList");
  const sessionSearchEl = document.getElementById("sessionSearch");
  const mainPanel = document.getElementById("mainPanel");
  const emptyHint = document.getElementById("emptyHint");
  const perfBarEl = document.getElementById("perfBar");

  // ── State ──
  let sessions = [];
  let activeSessionId = null;
  let traceData = null;
  let activeCategory = "all";
  let textFilter = "";
  let expandedRow = null;   // entry index currently expanded
  let collapsedGroups = {}; // node_id -> bool (true = collapsed)
  let refreshTimer = null;

  // ── Performance tracking state ──
  var perfHistory = [];          // last 20 perf records
  var perfHistoryOpen = false;   // history panel expanded?
  var PERF_HISTORY_MAX = 20;
  var _searchDebounceTimer = null;

  // ── Helpers ──
  function fmt(ms) {
    if (ms == null) return "";
    if (ms < 1000) return Math.round(ms) + "ms";
    if (ms < 60000) return (ms / 1000).toFixed(1) + "s";
    return (ms / 60000).toFixed(1) + "m";
  }

  function fmtTime(iso) {
    if (!iso) return "";
    try {
      var d = new Date(iso);
      return d.toLocaleTimeString();
    } catch (_) { return iso; }
  }

  function fmtDate(iso) {
    if (!iso) return "";
    try {
      var d = new Date(iso);
      return d.toLocaleDateString() + " " + d.toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});
    } catch (_) { return iso; }
  }

  function esc(s) {
    if (!s) return "";
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function truncate(s, n) {
    if (!s) return "";
    return s.length > n ? s.slice(0, n) + "…" : s;
  }

  function nodeDepth(nodeId) {
    if (!nodeId) return 0;
    return nodeId.split(".").length;
  }

  // ── API ──
  async function fetchJSON(url) {
    var r = await fetch(url);
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }

  // ── Performance helpers ──

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

  /** Format KB value for display */
  function fmtKB(kb) {
    if (kb == null) return "—";
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

  /** Render the Performance Bar UI */
  function updatePerfBar() {
    if (!perfBarEl) return;

    if (!perfHistory.length) {
      perfBarEl.innerHTML = '<div class="obs-perfbar-row">' +
        '<span class="obs-perfbar-label">⚡ Perf</span>' +
        '<span class="obs-perfbar-sep">│</span>' +
        '<span style="color:var(--obs-muted)">—</span>' +
        '</div>';
      return;
    }

    var latest = perfHistory[perfHistory.length - 1];

    // Build the main row
    var html = '<div class="obs-perfbar-row">';
    html += '<span class="obs-perfbar-label">⚡ Perf</span>';
    html += '<span class="obs-perfbar-sep">│</span>';

    // Action
    html += '<span class="obs-perfbar-item"><span class="obs-perf-key">' +
      esc(latest.action) + '</span></span>';
    html += '<span class="obs-perfbar-sep">│</span>';

    // Total
    html += perfItem("Total", latest.total_ms);
    html += '<span class="obs-perfbar-sep">│</span>';

    // API (only for load_session)
    if (latest.api_ms != null) {
      var serverSub = "";
      if (latest.server_ms && latest.server_ms.total != null) {
        serverSub = ' <span class="obs-perf-sub">(server: ' + Math.round(latest.server_ms.total) + 'ms)</span>';
      }
      html += '<span class="obs-perfbar-item"><span class="obs-perf-key">API:</span> ' +
        '<span class="obs-perf-val ' + perfColor(latest.api_ms) + '">' + Math.round(latest.api_ms) + 'ms</span>' +
        serverSub + '</span>';
      html += '<span class="obs-perfbar-sep">│</span>';

      // Parse
      html += perfItem("Parse", latest.parse_ms);
      html += '<span class="obs-perfbar-sep">│</span>';
    }

    // Render
    html += perfItem("Render", latest.render_ms);
    html += '<span class="obs-perfbar-sep">│</span>';

    // Size (only for load_session)
    if (latest.response_kb != null) {
      html += '<span class="obs-perfbar-item"><span class="obs-perf-key">Size:</span> ' +
        '<span class="obs-perf-val">' + fmtKB(latest.response_kb) + '</span></span>';
      html += '<span class="obs-perfbar-sep">│</span>';
    }

    // Rows
    if (latest.entry_count != null) {
      html += '<span class="obs-perfbar-item"><span class="obs-perf-key">Rows:</span> ' +
        '<span class="obs-perf-val">' + latest.entry_count + '</span></span>';
      html += '<span class="obs-perfbar-sep">│</span>';
    }

    // History toggle
    html += '<button class="obs-perfbar-history-btn" id="perfHistoryToggle">' +
      'History ' + (perfHistoryOpen ? '▴' : '▾') + '</button>';

    html += '</div>';

    // History table
    html += '<div class="obs-perfbar-history' + (perfHistoryOpen ? ' open' : '') + '" id="perfHistoryPanel">';
    html += '<table><thead><tr>' +
      '<th>Time</th><th>Action</th><th>Total</th><th>API</th><th>Server</th>' +
      '<th>Parse</th><th>Render</th><th>Size</th><th>Rows</th>' +
      '</tr></thead><tbody>';

    // Show newest first
    for (var i = perfHistory.length - 1; i >= 0; i--) {
      var r = perfHistory[i];
      var serverTotal = (r.server_ms && r.server_ms.total != null) ? Math.round(r.server_ms.total) + "ms" : "—";
      html += '<tr>' +
        '<td>' + esc(r.timestamp) + '</td>' +
        '<td>' + esc(r.action) + '</td>' +
        '<td class="' + perfColor(r.total_ms) + '">' + (r.total_ms != null ? Math.round(r.total_ms) + 'ms' : '—') + '</td>' +
        '<td class="' + perfColor(r.api_ms) + '">' + (r.api_ms != null ? Math.round(r.api_ms) + 'ms' : '—') + '</td>' +
        '<td>' + serverTotal + '</td>' +
        '<td class="' + perfColor(r.parse_ms) + '">' + (r.parse_ms != null ? Math.round(r.parse_ms) + 'ms' : '—') + '</td>' +
        '<td class="' + perfColor(r.render_ms) + '">' + (r.render_ms != null ? Math.round(r.render_ms) + 'ms' : '—') + '</td>' +
        '<td>' + (r.response_kb != null ? fmtKB(r.response_kb) : '—') + '</td>' +
        '<td>' + (r.entry_count != null ? r.entry_count : '—') + '</td>' +
        '</tr>';
    }

    html += '</tbody></table></div>';

    perfBarEl.innerHTML = html;

    // Bind history toggle
    var toggleBtn = document.getElementById("perfHistoryToggle");
    if (toggleBtn) {
      toggleBtn.addEventListener("click", function () {
        perfHistoryOpen = !perfHistoryOpen;
        updatePerfBar();
      });
    }
  }

  /** Helper to create a perfbar metric item */
  function perfItem(label, ms) {
    var val = ms != null ? Math.round(ms) + "ms" : "—";
    return '<span class="obs-perfbar-item"><span class="obs-perf-key">' + label + ':</span> ' +
      '<span class="obs-perf-val ' + perfColor(ms) + '">' + val + '</span></span>';
  }

  // ── Session list ──
  async function loadSessions() {
    try {
      var data = await fetchJSON("/api/observability/sessions");
      sessions = data.sessions || [];
      renderSessionList();
    } catch (e) {
      sessionListEl.innerHTML = '<div class="obs-empty">Failed to load sessions</div>';
    }
  }

  function renderSessionList() {
    var q = sessionSearchEl.value.toLowerCase().trim();
    var filtered = sessions;
    if (q) {
      filtered = sessions.filter(function (s) {
        return (s.task || "").toLowerCase().indexOf(q) >= 0 ||
               (s.session_id || "").toLowerCase().indexOf(q) >= 0;
      });
    }
    if (!filtered.length) {
      sessionListEl.innerHTML = '<div class="obs-empty">No sessions found</div>';
      return;
    }
    var html = "";
    for (var i = 0; i < filtered.length; i++) {
      var s = filtered[i];
      var cls = s.session_id === activeSessionId ? " active" : "";
      var statusCls = (s.status || "unknown").toLowerCase();
      if (["completed","running","failed"].indexOf(statusCls) < 0) statusCls = "unknown";
      html += '<div class="obs-session-item' + cls + '" data-sid="' + esc(s.session_id) + '">' +
        '<div class="obs-session-task">' + esc(truncate(s.task || s.session_id, 40)) + '</div>' +
        '<div class="obs-session-meta">' +
          '<span class="obs-status-badge ' + statusCls + '">' + esc(s.status || "?") + '</span>' +
          '<span>' + esc(fmtDate(s.modified_at)) + '</span>' +
          '<span>' + (s.entry_count || 0) + ' events</span>' +
        '</div></div>';
    }
    sessionListEl.innerHTML = html;

    // click handlers
    var items = sessionListEl.querySelectorAll(".obs-session-item");
    for (var j = 0; j < items.length; j++) {
      items[j].addEventListener("click", onSessionClick);
    }
  }

  function onSessionClick(e) {
    var sid = this.getAttribute("data-sid");
    if (sid) selectSession(sid);
  }

  sessionSearchEl.addEventListener("input", renderSessionList);

  // ── Select session & load trace (with perf timing) ──
  async function selectSession(sid) {
    var t0 = performance.now();

    activeSessionId = sid;
    expandedRow = null;
    collapsedGroups = {};
    renderSessionList(); // highlight active
    mainPanel.innerHTML = '<div class="obs-loading">Loading trace…</div>';

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
            traceData = await fetchJSON("/api/observability/sessions/" + encodeURIComponent(sid) + "/trace");
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
      mainPanel.innerHTML = '<div class="obs-empty">Failed to load trace</div>';
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
        metric("LLM calls", sum.llm_calls) +
        metric("Searches", sum.searches) +
        metric("Tokens", sum.total_tokens ? sum.total_tokens.toLocaleString() : "-") +
        metric("LLM time", fmt(sum.total_llm_ms)) +
        metric("Search time", fmt(sum.total_search_ms)) +
        metric("Events", sum.events) +
      '</div></div>';

    // Filter bar
    html += '<div class="obs-filters">' +
      filterBtn("all", "All") +
      filterBtn("llm_call", "LLM") +
      filterBtn("search", "Search") +
      filterBtn("agent_event", "Agent") +
      filterBtn("lifecycle", "Lifecycle") +
      filterBtn("frontend", "Frontend") +
      filterBtn("error", "Error") +
      '<div class="obs-legend">' +
        legendDot("var(--obs-running)", "LLM") +
        legendDot("var(--obs-success)", "Search") +
        legendDot("#9a8ec2", "Agent") +
        legendDot("#bbb", "Lifecycle") +
        legendDot("var(--obs-orange)", "Frontend") +
        legendDot("var(--obs-failed)", "Error") +
      '</div>' +
      '<input type="text" class="obs-filter-search" id="traceSearch" placeholder="Search events…" value="' + esc(textFilter) + '" />' +
    '</div>';

    // Timeline table
    html += '<div class="obs-timeline-wrap"><table class="obs-timeline">' +
      '<thead><tr><th>Offset</th><th>Event</th><th>Duration</th><th>Bar</th></tr></thead><tbody>';

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
      var cat = e.category || "lifecycle";
      var isExpanded = expandedRow === origIdx;

      // check if hidden by collapsed group
      if (isGroupHidden(e)) continue;

      // is this a group header?
      var isGroupHead = e.node_id && hasChildren(entries, e.node_id, i);
      var groupArrow = "";
      if (isGroupHead) {
        groupArrow = collapsedGroups[e.node_id] ? "▸ " : "▾ ";
      }

      html += '<tr class="' + (isExpanded ? "expanded" : "") +
              (isGroupHead ? " obs-group-header" : "") +
              '" data-idx="' + origIdx + '" data-nid="' + esc(e.node_id || "") + '">' +
        '<td>' + fmt(e.offset_ms) + '</td>' +
        '<td class="obs-indent-' + depth + '">' +
          groupArrow + esc(e.message) +
          (e.node_id ? ' <span style="color:var(--obs-muted);font-size:10px">[' + esc(e.node_id) + ']</span>' : '') +
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
    mainPanel.innerHTML = html;

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
    // walk up parent chain
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
      // also check if next entries have same node_id (sub-events)
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
    if (["completed","running","failed"].indexOf(cls) < 0) cls = "unknown";
    return '<span class="obs-status-badge ' + cls + '">' + esc(status || "?") + '</span>';
  }

  function metric(label, val) {
    return '<div class="obs-metric"><span>' + esc(label) + '</span> <strong>' + esc(String(val != null ? val : "-")) + '</strong></div>';
  }

  function filterBtn(cat, label) {
    var cls = activeCategory === cat ? " active" : "";
    return '<button class="obs-filter-btn' + cls + '" data-cat="' + cat + '">' + esc(label) + '</button>';
  }

  function legendDot(color, label) {
    return '<span><span class="obs-legend-dot" style="background:' + color + '"></span>' + esc(label) + '</span>';
  }

  // ── Bind trace interaction events (with perf timing) ──
  function bindTraceEvents() {
    // row click → expand/collapse
    var rows = mainPanel.querySelectorAll(".obs-timeline tr[data-idx]");
    for (var i = 0; i < rows.length; i++) {
      rows[i].addEventListener("click", function (ev) {
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
    var btns = mainPanel.querySelectorAll(".obs-filter-btn");
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

    // text search (debounced to avoid flooding perfHistory)
    var searchInput = document.getElementById("traceSearch");
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
    var copyBtns = mainPanel.querySelectorAll(".obs-copy-btn");
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

  // ── Init ──
  loadSessions();
  updatePerfBar(); // show initial empty state

})();
