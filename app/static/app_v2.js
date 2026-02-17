const taskEl = document.getElementById("task");
const maxDepthEl = document.getElementById("maxDepth");
const resultsPerQueryEl = document.getElementById("resultsPerQuery");
const runBtn = document.getElementById("runBtn");
const reportBtn = document.getElementById("reportBtn");
const downloadBtn = document.getElementById("downloadBtn");
const runMeta = document.getElementById("runMeta");
const stopReasonEl = document.getElementById("stopReason");
const errorBanner = document.getElementById("errorBanner");

const runStatusEl = document.getElementById("runStatus");
const researchStatusEl = document.getElementById("researchStatus");
const reportStatusEl = document.getElementById("reportStatus");
const tokenSummaryEl = document.getElementById("tokenSummary");
const canvasHintEl = document.getElementById("canvasHint");
const zoomOutBtn = document.getElementById("zoomOutBtn");
const zoomInBtn = document.getElementById("zoomInBtn");
const zoomFitBtn = document.getElementById("zoomFitBtn");
const zoomPctEl = document.getElementById("zoomPct");
const fitDebugEl = document.getElementById("fitDebug");

const canvasEl = document.getElementById("canvas");
const thoughtStreamEl = document.getElementById("thoughtStream");
const latestThoughtEl = document.getElementById("latestThought");
const coverageNoteEl = document.getElementById("coverageNote");
const reportRawEl = document.getElementById("reportRaw");
const reportRenderedEl = document.getElementById("reportRendered");
const usageEl = document.getElementById("usage");
const rawToggleEl = document.getElementById("rawToggle");
const APP_CONFIG = (window.APP_CONFIG && typeof window.APP_CONFIG === "object")
  ? window.APP_CONFIG
  : {};

let currentRunId = null;
let currentSnapshot = null;
let es = null;
let reportGenerating = false;
let abortRequested = false;
let autoFollowActiveNode = true;
let suppressUserScrollDetection = false;
const thoughts = [];
const openDetailKeys = new Set();
const forceClosedDetailKeys = new Set();
let canvasZoom = 1;
let autoFitCanvas = true;
const MIN_CANVAS_ZOOM = clamp(Number(APP_CONFIG.min_canvas_zoom ?? 0.45), 0.1, 0.95);
const MAX_CANVAS_ZOOM = 1.6;
const BASE_NODE_CARD_WIDTH = 340;
const MIN_NODE_FONT_SCALE = 0.72;
const MAX_NODE_FONT_SCALE = 1.25;
const AUTO_FIT_SAFETY_PX = Math.max(0, Math.floor(Number(APP_CONFIG.auto_fit_safety_px ?? 10)));

function esc(s) {
  return String(s || "").replace(/[&<>\"]/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
  })[c]);
}

function setRunIdInUrl(runId) {
  const url = new URL(window.location.href);
  if (runId) {
    url.searchParams.set("run_id", runId);
  } else {
    url.searchParams.delete("run_id");
  }
  history.replaceState({}, "", url);
}

function readRunIdFromUrl() {
  const url = new URL(window.location.href);
  return (url.searchParams.get("run_id") || "").trim();
}

function showError(msg) {
  if (!msg) {
    errorBanner.classList.add("hidden");
    errorBanner.textContent = "";
    return;
  }
  errorBanner.textContent = msg;
  errorBanner.classList.remove("hidden");
}

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

function getCanvasGraphEls() {
  return Array.from(canvasEl.querySelectorAll(".depth-rows"));
}

function setCanvasScaleVariables(scale) {
  const z = clamp(scale, MIN_CANVAS_ZOOM, MAX_CANVAS_ZOOM);
  const cardWidth = Math.max(96, Math.round(BASE_NODE_CARD_WIDTH * z));
  const fontScale = clamp(z, MIN_NODE_FONT_SCALE, MAX_NODE_FONT_SCALE);
  canvasEl.style.setProperty("--node-card-width", `${cardWidth}px`);
  canvasEl.style.setProperty("--node-font-scale", String(fontScale));
}

function hasHorizontalOverflow() {
  const rowEls = Array.from(canvasEl.querySelectorAll(".depth-row-nodes"));
  const viewportLeft = 0;
  const viewportRight = Math.max(1, window.innerWidth || document.documentElement.clientWidth || 1);
  return rowEls.some((row) => {
    const rect = row.getBoundingClientRect();
    const visibleWidth = Math.max(
      1,
      Math.floor(Math.min(rect.right, viewportRight) - Math.max(rect.left, viewportLeft))
    );
    return Math.ceil(row.scrollWidth) > visibleWidth + 1;
  });
}

function getFitDiagnostics() {
  const rowEls = Array.from(canvasEl.querySelectorAll(".depth-row-nodes"));
  if (rowEls.length === 0) {
    return {
      rowCount: 0,
      worstRow: -1,
      worstOverflowPx: 0,
      worstClientWidth: 0,
      worstScrollWidth: 0,
      hasOverflow: false,
    };
  }
  let worstOverflowPx = -Infinity;
  let worstRow = -1;
  let worstClientWidth = 0;
  let worstScrollWidth = 0;
  let worstVisibleWidth = 0;
  const viewportLeft = 0;
  const viewportRight = Math.max(1, window.innerWidth || document.documentElement.clientWidth || 1);
  for (let i = 0; i < rowEls.length; i += 1) {
    const row = rowEls[i];
    const cw = Math.floor(row.clientWidth);
    const sw = Math.ceil(row.scrollWidth);
    const rect = row.getBoundingClientRect();
    const vw = Math.max(
      1,
      Math.floor(Math.min(rect.right, viewportRight) - Math.max(rect.left, viewportLeft))
    );
    const ov = sw - vw;
    if (ov > worstOverflowPx) {
      worstOverflowPx = ov;
      worstRow = i + 1;
      worstClientWidth = cw;
      worstScrollWidth = sw;
      worstVisibleWidth = vw;
    }
  }
  return {
    rowCount: rowEls.length,
    worstRow,
    worstOverflowPx: Math.max(0, worstOverflowPx),
    worstClientWidth,
    worstScrollWidth,
    worstVisibleWidth,
    hasOverflow: worstOverflowPx > 1,
  };
}

function computeAutoFitZoom() {
  const rowEls = Array.from(canvasEl.querySelectorAll(".depth-row-nodes"));
  if (rowEls.length === 0) return 1;

  // Measurement-only search: update scale vars only, avoid UI label churn.
  const originalZoom = canvasZoom;
  setCanvasScaleVariables(1);
  void canvasEl.offsetHeight;
  if (!hasHorizontalOverflow()) {
    setCanvasScaleVariables(originalZoom);
    return 1;
  }

  setCanvasScaleVariables(MIN_CANVAS_ZOOM);
  void canvasEl.offsetHeight;
  if (hasHorizontalOverflow()) {
    setCanvasScaleVariables(originalZoom);
    return MIN_CANVAS_ZOOM;
  }

  let lo = MIN_CANVAS_ZOOM;
  let hi = 1;
  let best = MIN_CANVAS_ZOOM;
  for (let i = 0; i < 18; i += 1) {
    const mid = (lo + hi) / 2;
    setCanvasScaleVariables(mid);
    void canvasEl.offsetHeight;
    if (hasHorizontalOverflow()) {
      hi = mid;
    } else {
      best = mid;
      lo = mid;
    }
  }

  setCanvasScaleVariables(originalZoom);
  const viewportWidth = Math.max(1, window.innerWidth || document.documentElement.clientWidth || 1);
  const safetyFactor = Math.max(0.9, 1 - (AUTO_FIT_SAFETY_PX / viewportWidth));
  return clamp(best * safetyFactor, MIN_CANVAS_ZOOM, 1);
}

function applyCanvasZoom() {
  canvasZoom = clamp(canvasZoom, MIN_CANVAS_ZOOM, MAX_CANVAS_ZOOM);
  setCanvasScaleVariables(canvasZoom);
  // Keep a stable viewport; content grows via scroll instead of inflating canvas height.
  canvasEl.style.minHeight = "320px";
  zoomPctEl.textContent = `${Math.round(canvasZoom * 100)}%`;
  canvasHintEl.textContent = autoFitCanvas ? "Auto-fit" : "Manual zoom";
  if (fitDebugEl) {
    const d = getFitDiagnostics();
    const canvasCw = Math.floor(canvasEl.clientWidth || 0);
    const canvasSw = Math.ceil(canvasEl.scrollWidth || 0);
    const winW = Math.floor(window.innerWidth || 0);
    fitDebugEl.textContent = `dbg rows=${d.rowCount} worst=r${d.worstRow} ov=${d.worstOverflowPx}px vw=${d.worstVisibleWidth} cw=${d.worstClientWidth} sw=${d.worstScrollWidth} safety=${AUTO_FIT_SAFETY_PX}px canvas=${canvasCw}/${canvasSw} win=${winW} ${d.hasOverflow ? "OVERFLOW" : "ok"}`;
  }
}

function refreshCanvasZoomForCurrentLayout() {
  if (autoFitCanvas) {
    canvasZoom = computeAutoFitZoom();
  }
  applyCanvasZoom();
}

function shortStatus(v) {
  return String(v || "").replaceAll("_", " ") || "-";
}

function shortSourceLabel(url) {
  const u = String(url || "").trim();
  if (!u) return "";
  try {
    const parsed = new URL(u);
    const host = String(parsed.hostname || "").replace(/^www\./i, "");
    return host || u;
  } catch (_err) {
    return u;
  }
}

function toSafeHttpUrl(raw) {
  const u = String(raw || "").trim();
  if (!u) return "";
  try {
    const parsed = new URL(u);
    const proto = String(parsed.protocol || "").toLowerCase();
    if (proto === "http:" || proto === "https:") {
      return parsed.href;
    }
  } catch (_err) {
    // ignore invalid URLs
  }
  return "";
}

function queryStepStatusLabel(v) {
  const s = String(v || "").toLowerCase();
  if (s === "success") return "synthesized";
  return shortStatus(v);
}

function statusClass(v) {
  const s = String(v || "").toLowerCase();
  if (["running", "researching"].includes(s)) return "running";
  if (["success", "completed", "complete", "solved", "solved_via_children"].includes(s)) return "success";
  if (["sufficient"].includes(s)) return "sufficient";
  if (["failed"].includes(s)) return "failed";
  if (["skipped", "cached", "blocked", "queued", "pending"].includes(s)) return "skipped";
  if (["stopped", "aborted", "dead_end", "unresolved"].includes(s)) return "stopped";
  return "";
}

function renderTokenSummary(usage) {
  if (!usage || typeof usage !== "object") {
    tokenSummaryEl.textContent = "-";
    usageEl.textContent = "";
    return;
  }
  const total = usage.total || {};
  const inTok = Number(total.input_tokens || 0);
  const outTok = Number(total.output_tokens || 0);
  const cost = Number(total.estimated_cost_usd || 0);
  tokenSummaryEl.textContent = `${inTok + outTok} tokens | $${cost.toFixed(4)}`;
  usageEl.textContent = JSON.stringify(total, null, 2);
}

function markdownToHtml(md) {
  const lines = String(md || "").split(/\r?\n/);
  const out = [];
  let inList = false;
  for (const line of lines) {
    if (/^\s*[-*]\s+/.test(line)) {
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      const item = esc(line.replace(/^\s*[-*]\s+/, ""));
      out.push(`<li>${item}</li>`);
      continue;
    }
    if (inList) {
      out.push("</ul>");
      inList = false;
    }
    if (/^###\s+/.test(line)) {
      out.push(`<h3>${esc(line.replace(/^###\s+/, ""))}</h3>`);
    } else if (/^##\s+/.test(line)) {
      out.push(`<h2>${esc(line.replace(/^##\s+/, ""))}</h2>`);
    } else if (/^#\s+/.test(line)) {
      out.push(`<h1>${esc(line.replace(/^#\s+/, ""))}</h1>`);
    } else if (!line.trim()) {
      out.push("<p></p>");
    } else {
      out.push(`<p>${esc(line)}</p>`);
    }
  }
  if (inList) {
    out.push("</ul>");
  }
  return out.join("\n");
}

function toggleReportMode() {
  const raw = !!rawToggleEl.checked;
  reportRawEl.classList.toggle("hidden", !raw);
  reportRenderedEl.classList.toggle("hidden", raw);
}

function upsertThought(text, meta) {
  const msg = String(text || "").trim();
  if (!msg) return;
  thoughts.push({ text: msg, meta: meta || "" });
  if (thoughts.length > 120) {
    thoughts.shift();
  }
  thoughtStreamEl.innerHTML = thoughts
    .slice()
    .reverse()
    .map((t) => `<div class="thought"><div class="t-meta">${esc(t.meta)}</div>${esc(t.text)}</div>`)
    .join("");
}

function makeDetailKey(parts) {
  return JSON.stringify(parts.map((v) => String(v || "").trim()));
}

function parseDetailKey(key) {
  try {
    const parsed = JSON.parse(String(key || ""));
    if (Array.isArray(parsed)) {
      return parsed.map((v) => String(v || "").trim());
    }
  } catch (_err) {
    // backward-compat fallback
  }
  return String(key || "").split("||").map((v) => String(v || "").trim());
}

function isNestedUnderClosedDetail(el) {
  let cur = el.parentElement;
  while (cur) {
    if (
      cur.tagName &&
      String(cur.tagName).toLowerCase() === "details" &&
      cur.hasAttribute("data-detail-key") &&
      !cur.open
    ) {
      return true;
    }
    cur = cur.parentElement;
  }
  return false;
}

function captureOpenDetailState() {
  const openEls = canvasEl.querySelectorAll("details[data-detail-key][open]");
  const latest = new Set();
  for (const el of openEls) {
    if (isNestedUnderClosedDetail(el)) {
      continue;
    }
    const key = String(el.getAttribute("data-detail-key") || "");
    if (!key) continue;
    latest.add(key);
  }
  openDetailKeys.clear();
  for (const k of latest) {
    openDetailKeys.add(k);
  }
}

function restoreAndTrackDetailState() {
  const els = canvasEl.querySelectorAll("details[data-detail-key]");
  for (const el of els) {
    const key = String(el.getAttribute("data-detail-key") || "");
    if (!key) continue;
    if (openDetailKeys.has(key)) {
      el.open = true;
    }
    el.addEventListener("toggle", () => {
      const parts = parseDetailKey(key);
      const isQueryWorkSection =
        parts.length >= 3 &&
        parts[0] === "node_section" &&
        parts[2] === "query_work";
      if (el.open) {
        openDetailKeys.add(key);
        if (isQueryWorkSection) {
          forceClosedDetailKeys.delete(key);
        }
      } else {
        openDetailKeys.delete(key);
        // If user collapses "Query Work", also collapse all child query detail states.
        if (isQueryWorkSection) {
          forceClosedDetailKeys.add(key);
          const nested = el.querySelectorAll("details[data-detail-key]");
          for (const childEl of nested) {
            if (childEl.open) {
              childEl.open = false;
            }
            const childKey = String(childEl.getAttribute("data-detail-key") || "");
            if (childKey) {
              openDetailKeys.delete(childKey);
            }
          }
          const nodeName = parts[1];
          for (const existing of Array.from(openDetailKeys)) {
            const eParts = parseDetailKey(existing);
            if (
              eParts.length >= 3 &&
              eParts[0] === "query_detail" &&
              eParts[1] === nodeName
            ) {
              openDetailKeys.delete(existing);
            }
          }
        }
      }
    });
  }
}

function eventNarration(ev) {
  const et = String(ev.event_type || "");
  const p = (ev.payload && typeof ev.payload === "object") ? ev.payload : {};
  if (et === "run_started") return "Research started.";
  if (et === "sub_question_started") return `Exploring task: ${p.sub_question || "(unknown)"}`;
  if (et === "queries_generated") return `Prepared ${Number(p.count || 0)} query variants for this task.`;
  if (et === "query_diagnostic") {
    const cls = p.classification || "";
    const d = p.decision || "";
    return `Query diagnostic: ${cls} -> ${d}.`;
  }
  if (et === "query_started") return `Running query: ${p.query || ""}`;
  if (et === "query_skipped_cached") return `Cache hit: reused prior evidence for query.`;
  if (et === "query_blocked_diminishing_returns") return "Skipped query due to diminishing returns.";
  if (et === "search_completed") return `Search selected ${Number(p.selected_results_count || 0)} results.`;
  if (et === "synthesis_completed") return "Synthesized evidence from selected sources.";
  if (et === "node_sufficiency_started") return "Running node sufficiency check, please wait a moment...";
  if (et === "node_sufficiency_completed") return `Node sufficiency: ${p.is_sufficient ? "pass" : "fail"}.`;
  if (et === "node_decomposition_started") return "Node sufficiency failed, decomposing into child tasks...";
  if (et === "node_decomposed") return `Insufficient evidence, decomposed into ${Array.isArray(p.children) ? p.children.length : 0} child tasks.`;
  if (et === "node_completed") return "Task node resolved.";
  if (et === "sufficiency_started") return "Running pass sufficiency check, please wait a moment...";
  if (et === "sufficiency_completed") return `Pass sufficiency: ${p.is_sufficient ? "pass" : "fail"}.`;
  if (et === "run_abort_requested") return "Stop requested. Finishing current atomic step.";
  if (et === "abort_requested") return "Abort requested.";
  if (et === "run_aborted") return "Run stopped by user request.";
  if (et === "report_generation_started") return "Agent is writing the report, please wait a few moments...";
  if (et === "report_generation_completed") return "Report generation completed.";
  if (et === "report_generation_failed") return "Report generation failed.";
  if (et === "partial_report_generated") return "Generated report from current findings.";
  if (et === "run_heartbeat") {
    const phase = p.phase || "processing";
    if (p.query) return `Still working (${phase}): ${p.query}`;
    if (p.sub_question) return `Still working (${phase}): ${p.sub_question}`;
    return `Still working (${phase})...`;
  }
  if (et === "run_completed") return "Run complete. Final report ready.";
  if (et === "run_failed") return `Run failed: ${p.error || "unknown error"}`;
  return "";
}

function mergeChildren(a, b) {
  const out = [];
  const seen = new Set();
  for (const v of [...a, ...b]) {
    const s = String(v || "").trim();
    if (!s || seen.has(s)) continue;
    seen.add(s);
    out.push(s);
  }
  return out;
}

function mergeNodeStatus(prevStatus, nextStatus) {
  const rank = {
    running: 6,
    decomposing: 6,
    solved_via_children: 5,
    solved: 5,
    success: 5,
    decomposed: 4,
    unresolved: 3,
    blocked: 2,
    skipped: 2,
    cached: 2,
    queued: 1,
    pending: 1,
  };
  const p = String(prevStatus || "queued");
  const n = String(nextStatus || "queued");
  return (rank[n] || 0) >= (rank[p] || 0) ? n : p;
}

function normalizeQuestionKey(v) {
  return String(v || "").trim().toLowerCase().replace(/\s+/g, " ");
}

function buildQuestionsByDepth(questions) {
  const byKey = new Map();
  const byDepth = new Map();
  const list = Array.isArray(questions) ? questions : [];
  for (const q of list) {
    if (!q || typeof q !== "object") continue;
    const label = String(q.sub_question || "").trim();
    const key = normalizeQuestionKey(label);
    if (!key) continue;
    const depth = Number(q.depth || 1);
    const prev = byKey.get(key);
    const merged = prev
      ? {
          ...prev,
          ...q,
          sub_question: label || prev.sub_question || "",
          status: mergeNodeStatus(prev.status, q.status),
          depth: Number(prev.depth || depth),
          query_steps: Array.isArray(q.query_steps) ? q.query_steps : (prev.query_steps || []),
          children: mergeChildren(
            Array.isArray(prev.children) ? prev.children : [],
            Array.isArray(q.children) ? q.children : []
          ),
          child_node_ids: Array.isArray(q.child_node_ids)
            ? q.child_node_ids
            : (Array.isArray(prev.child_node_ids) ? prev.child_node_ids : []),
          parent: q.parent || prev.parent || "",
          is_placeholder: false,
        }
      : {
          ...q,
          sub_question: label,
          depth,
          status: String(q.status || "queued"),
          query_steps: Array.isArray(q.query_steps) ? q.query_steps : [],
          children: Array.isArray(q.children) ? q.children : [],
          child_node_ids: Array.isArray(q.child_node_ids) ? q.child_node_ids : [],
          parent: q.parent || "",
          is_placeholder: false,
        };
    byKey.set(key, merged);
  }

  // Ensure decomposed children are always visible, even before they become active.
  const snapshotNodes = Array.from(byKey.values());
  for (const node of snapshotNodes) {
    const parentDepth = Number(node.depth || 1);
    const children = Array.isArray(node.children) ? node.children : [];
    for (const childNameRaw of children) {
      const childLabel = String(childNameRaw || "").trim();
      const childKey = normalizeQuestionKey(childLabel);
      if (!childKey) continue;
      const childIndex = children.findIndex((v) => String(v || "").trim() === childLabel) + 1;
      const parentNodeId = String(node.node_id || "").trim();
      const childNodeIds = Array.isArray(node.child_node_ids) ? node.child_node_ids : [];
      const mappedChildNodeId = String(childNodeIds[childIndex - 1] || "").trim();
      if (!byKey.has(childKey)) {
        byKey.set(childKey, {
          sub_question: childLabel,
          depth: parentDepth + 1,
          parent: String(node.sub_question || ""),
          node_id:
            mappedChildNodeId
            || (parentNodeId && childIndex > 0 ? `${parentNodeId}.${childIndex}` : ""),
          status: "queued",
          query_steps: [],
          children: [],
          child_node_ids: [],
          is_placeholder: true,
        });
      } else {
        const existing = byKey.get(childKey);
        if (existing && !existing.parent) {
          existing.parent = String(node.sub_question || "");
        }
        if (existing && !existing.node_id) {
          existing.node_id =
            mappedChildNodeId
            || (parentNodeId && childIndex > 0 ? `${parentNodeId}.${childIndex}` : "");
        }
      }
    }
  }

  for (const [key, q] of byKey.entries()) {
    const depth = Number(q.depth || 1);
    if (!byDepth.has(depth)) byDepth.set(depth, new Map());
    byDepth.get(depth).set(key, q);
  }

  for (const depthMap of byDepth.values()) {
    const sorted = Array.from(depthMap.entries()).sort((a, b) => {
      const ida = String(a[1].node_id || "");
      const idb = String(b[1].node_id || "");
      if (ida && idb && ida !== idb) {
        return ida.localeCompare(idb, undefined, { numeric: true, sensitivity: "base" });
      }
      const pa = String(a[1].parent || "");
      const pb = String(b[1].parent || "");
      if (pa !== pb) return pa.localeCompare(pb);
      return String(a[0]).localeCompare(String(b[0]));
    });
    depthMap.clear();
    for (const [k, v] of sorted) {
      depthMap.set(k, v);
    }
  }

  return byDepth;
}

function collectQuestionsFromRounds(rounds) {
  const out = [];
  const arr = Array.isArray(rounds) ? rounds : [];
  for (const r of arr) {
    const qs = Array.isArray(r && r.questions) ? r.questions : [];
    for (const q of qs) {
      if (q && typeof q === "object") out.push(q);
    }
  }
  return out;
}

function buildNodeIdMap(byDepth) {
  const nodes = [];
  const byName = new Map();
  for (const depth of Array.from(byDepth.keys()).sort((a, b) => a - b)) {
    for (const q of byDepth.get(depth).values()) {
      const name = String(q.sub_question || "").trim();
      const key = normalizeQuestionKey(name);
      if (!key) continue;
      nodes.push(q);
      byName.set(key, q);
    }
  }

  const childrenByParent = new Map();
  for (const q of nodes) {
    const parent = normalizeQuestionKey(q.parent || "");
    if (!parent) continue;
    if (!childrenByParent.has(parent)) childrenByParent.set(parent, []);
    childrenByParent.get(parent).push(q);
  }
  for (const arr of childrenByParent.values()) {
    arr.sort((a, b) => String(a.sub_question || "").localeCompare(String(b.sub_question || "")));
  }

  const roots = nodes
    .filter((q) => {
      const parent = normalizeQuestionKey(q.parent || "");
      return !parent || !byName.has(parent);
    })
    .sort((a, b) => String(a.sub_question || "").localeCompare(String(b.sub_question || "")));

  const idByName = new Map();
  const visit = (node, baseId) => {
    const name = normalizeQuestionKey(node.sub_question || "");
    if (!name || idByName.has(name)) return;
    idByName.set(name, baseId);
    const children = childrenByParent.get(name) || [];
    for (let i = 0; i < children.length; i += 1) {
      visit(children[i], `${baseId}.${i + 1}`);
    }
  };

  for (let i = 0; i < roots.length; i += 1) {
    visit(roots[i], String(i + 1));
  }
  return idByName;
}

function buildNodeVisualStateMap(byDepth) {
  const byKey = new Map();
  const parentByKey = new Map();
  const runningKeys = new Set();

  for (const depth of Array.from(byDepth.keys()).sort((a, b) => a - b)) {
    for (const q of byDepth.get(depth).values()) {
      const key = normalizeQuestionKey(q.sub_question || "");
      if (!key) continue;
      byKey.set(key, q);
      const parentKey = normalizeQuestionKey(q.parent || "");
      parentByKey.set(key, parentKey);
      const status = String(q.status || "");
      const steps = Array.isArray(q.query_steps) ? q.query_steps : [];
      const hasRunningStep = steps.some((s) => String((s && s.status) || "") === "running");
      if (status === "running" || status === "decomposing" || hasRunningStep) {
        runningKeys.add(key);
      }
    }
  }

  // "Active" means running or ancestor of a running child.
  const activeKeys = new Set();
  for (const key of runningKeys) {
    let cur = key;
    while (cur) {
      if (activeKeys.has(cur)) break;
      activeKeys.add(cur);
      cur = parentByKey.get(cur) || "";
    }
  }

  const terminal = new Set(["solved", "solved_via_children", "success", "failed"]);

  const visitedKeys = new Set();
  const plannedKeys = new Set();
  for (const [key, q] of byKey.entries()) {
    if (activeKeys.has(key)) continue;
    const status = String(q.status || "queued");
    const hasQueryWork = Array.isArray(q.query_steps) && q.query_steps.length > 0;
    const hasNotes = !!q.node_sufficiency || !!q.unresolved_reason;
    const hasChildren = Array.isArray(q.children) && q.children.length > 0;
    const isVisitedStatus = !["queued", "pending"].includes(status);
    if (isVisitedStatus || hasQueryWork || hasNotes || hasChildren) {
      visitedKeys.add(key);
    } else {
      plannedKeys.add(key);
    }
  }

  return {
    activeKeys,
    visitedKeys,
    plannedKeys,
  };
}

function renderCanvas(tree) {
  captureOpenDetailState();
  if (!tree || typeof tree !== "object") {
    canvasEl.innerHTML = "<em>No run yet.</em>";
    return;
  }
  const renderGraph = (questions, scopeKey, passNo = null) => {
    const byDepth = buildQuestionsByDepth(questions);
    const nodeIdMap = buildNodeIdMap(byDepth);
    const visualStateMap = buildNodeVisualStateMap(byDepth);
    const depths = Array.from(byDepth.keys()).sort((a, b) => a - b);
    if (depths.length === 0) {
      return `<div class="muted">No task nodes yet.</div>`;
    }

    let html = `<div class="depth-rows">`;
    for (const d of depths) {
      html += `<section class="depth-row"><p class="depth-title">Depth ${d}</p>`;
      html += `<div class="depth-row-nodes">`;
      for (const q of byDepth.get(d).values()) {
        const key = normalizeQuestionKey(q.sub_question || "");
        let visualCls = "state-planned";
        let visualLabel = "planned";
        if (visualStateMap.activeKeys.has(key)) {
          visualCls = "state-active";
          visualLabel = "active";
        } else if (visualStateMap.visitedKeys.has(key)) {
          visualCls = "state-visited";
          visualLabel = "visited";
        } else if (visualStateMap.plannedKeys.has(key)) {
          visualCls = "state-planned";
          visualLabel = "planned";
        }
        const st = String(q.status || "queued");
        const cls = statusClass(st);
        const solvedStatuses = new Set(["solved", "solved_via_children", "success", "completed"]);
        const unresolvedStatuses = new Set(["unresolved", "failed", "dead_end", "stopped", "aborted"]);
        const visitedOutcomeCls =
          visualCls === "state-visited"
            ? (solvedStatuses.has(st) ? "visited-solved" : (unresolvedStatuses.has(st) ? "visited-unresolved" : "visited-neutral"))
            : "";
        const dim = visualCls === "state-planned" ? "dimmed" : "";
        const parentCls = q.parent ? "has-parent" : "";
        const stLower = st.toLowerCase();
        const isRunning = ["running", "researching", "decomposing"].includes(stLower);
        const isDecomposed = ["decomposed", "decomposed_child"].includes(stLower);
        const activeStateCls = (visualCls === "state-active" && isRunning)
          ? "active-running"
          : ((visualCls === "state-active" && isDecomposed) ? "active-decomposed" : "");
        const activeEmphasisCls = visualCls === "state-active" ? "active-emphasis" : "";
        const pathId = String(q.node_id || nodeIdMap.get(key) || "");
        const nodeId = (passNo !== null && passNo !== undefined && passNo !== "")
          ? (pathId ? `P${passNo}/${pathId}` : "")
          : pathId;
        html += `<article class="node ${esc(cls)} ${esc(dim)} ${esc(parentCls)} ${esc(activeStateCls)} ${esc(activeEmphasisCls)} ${esc(visualCls)} ${esc(visitedOutcomeCls)}" data-node-depth="${Number(d)}" data-node-status="${esc(stLower)}">`;
        html += `<div class="node-head"><span class="badge ${esc(cls)}">${esc(shortStatus(st))}</span>`;
        html += `<span class="badge">${esc(visualLabel)}</span>`;
        html += `<span class="badge">d${d}</span></div>`;
        if (nodeId) {
          html += `<div class="query-mini"><strong>Node ${esc(nodeId)}</strong></div>`;
        }
        html += `<p class="node-title">${esc(q.sub_question || "")}</p>`;
        if (q.parent) {
          html += `<p class="node-parent"><span class="parent-link">from</span> ${esc(q.parent)}</p>`;
        }
        if (q.is_placeholder) {
          // Keep placeholder node without extra explanatory text.
        }

        const steps = Array.isArray(q.query_steps) ? q.query_steps : [];
        if (steps.length > 0) {
          const queryWorkKey = makeDetailKey(["node_section", q.sub_question, "query_work", scopeKey]);
          const hasOpenChildDetail = steps.some((step) =>
            openDetailKeys.has(
              makeDetailKey(["query_detail", q.sub_question, step.query, scopeKey])
            )
          );
          const forceClosedQueryWork = forceClosedDetailKeys.has(queryWorkKey);
          const queryWorkOpen = (!forceClosedQueryWork && (openDetailKeys.has(queryWorkKey) || hasOpenChildDetail)) ? " open" : "";
          html += `<details class="node-section" data-detail-key="${esc(queryWorkKey)}"${queryWorkOpen}><summary>Query Work (${steps.length})</summary>`;
          html += `<ul class="query-list">`;
          for (const step of steps) {
            const qst = String(step.status || "queued");
            const diag = step.diagnostic || {};
            const freshness = String(diag.classification || "");
            const simDisplay = Math.max(
              Number(diag.similarity || 0),
              Number(diag.intent_map_similarity || 0)
            );
            const queryDetailKey = makeDetailKey(["query_detail", q.sub_question, step.query, scopeKey]);
            const queryDetailOpen = openDetailKeys.has(queryDetailKey) ? " open" : "";
            html += `<li><div><span class="badge ${esc(statusClass(qst))}">${esc(queryStepStatusLabel(qst))}</span> ${esc(step.query || "")}</div>`;
            html += `<div class="query-mini">selected=${Number(step.selected_results_count || 0)} primary=${Number(step.primary_count || 0)} ${freshness ? `| freshness=${esc(freshness)}` : ""}</div>`;
            html += `<details class="query-details" data-detail-key="${esc(queryDetailKey)}"${queryDetailOpen}><summary>details</summary>`;
            html += `<div class="query-meta">decision=${esc(diag.decision || "")} sim=${simDisplay.toFixed(2)} intent_mapped=${diag.intent_mapped ? "yes" : "no"}</div>`;
            if (Array.isArray(diag.new_tokens) && diag.new_tokens.length) {
              html += `<div class="query-meta">new_tokens: ${diag.new_tokens.map(esc).join(", ")}</div>`;
            }
            if (Array.isArray(diag.dropped_tokens) && diag.dropped_tokens.length) {
              html += `<div class="query-meta">dropped_tokens: ${diag.dropped_tokens.map(esc).join(", ")}</div>`;
            }
            if (step.synthesis_summary) {
              html += `<div class="query-meta">synthesis: ${esc(step.synthesis_summary)}</div>`;
            }
            if (step.search_error) {
              html += `<div class="query-meta">error: ${esc(step.search_error)}</div>`;
            }
            const selectedSources = Array.isArray(step.selected_sources) ? step.selected_sources : [];
            if (selectedSources.length > 0) {
              html += `<div class="query-meta">picked sources:</div>`;
              html += `<ul class="source-links">`;
              for (const src of selectedSources) {
                if (!src || typeof src !== "object") continue;
                const srcUrl = toSafeHttpUrl(src.url);
                if (!srcUrl) continue;
                const srcTitle = String(src.title || "").trim();
                const linkLabel = srcTitle || shortSourceLabel(srcUrl);
                html += `<li><a href="${esc(srcUrl)}" target="_blank" rel="noopener noreferrer">${esc(linkLabel)}</a> <span class="source-host">(${esc(shortSourceLabel(srcUrl))})</span></li>`;
              }
              html += `</ul>`;
            }
            html += `</details></li>`;
          }
          html += `</ul></details>`;
        }

        if (q.node_sufficiency || q.unresolved_reason) {
          const nodeNotesKey = makeDetailKey(["node_section", q.sub_question, "node_notes", scopeKey]);
          html += `<details class="node-section" data-detail-key="${esc(nodeNotesKey)}"><summary>Node Notes</summary>`;
          if (q.node_sufficiency) {
            const isSufficient = !!q.node_sufficiency.is_sufficient;
            html += `<div class="query-mini">node sufficiency: ${isSufficient ? "pass" : "fail"}</div>`;
            if (!isSufficient) {
              const reason = String(q.node_sufficiency.reasoning || "").trim();
              const gaps = Array.isArray(q.node_sufficiency.gaps) ? q.node_sufficiency.gaps : [];
              if (reason) {
                html += `<div class="query-meta">reason: ${esc(reason)}</div>`;
              }
              if (gaps.length > 0) {
                html += `<div class="query-meta">gaps: ${gaps.map((g) => esc(g)).join(" | ")}</div>`;
              }
            }
          }
          if (q.unresolved_reason) {
            html += `<div class="query-mini">unresolved: ${esc(q.unresolved_reason)}</div>`;
          }
          html += `</details>`;
        }
        html += `</article>`;
      }
      html += `</div></section>`;
    }
    html += `</div>`;
    return html;
  };

  const rounds = (Array.isArray(tree.rounds) ? tree.rounds : [])
    .filter((r) => r && typeof r === "object")
    .sort((a, b) => Number(a.round || 0) - Number(b.round || 0));

  const mergedQuestions = collectQuestionsFromRounds(rounds);
  let html = "";
  if (mergedQuestions.length === 0) {
    html = `<div class="muted">No task nodes yet. The root task is: ${esc(tree.task || "")}</div>`;
  } else {
    html = renderGraph(mergedQuestions, "merged", null);
  }

  canvasEl.innerHTML = html;
  restoreAndTrackDetailState();
  refreshCanvasZoomForCurrentLayout();
  const active = pickAutoFollowTargetNode();
  if (autoFollowActiveNode && active && typeof active.scrollIntoView === "function") {
    suppressUserScrollDetection = true;
    active.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
    window.setTimeout(() => {
      suppressUserScrollDetection = false;
    }, 250);
  }
}

function pickAutoFollowTargetNode() {
  const nodes = Array.from(canvasEl.querySelectorAll(".node.state-active"));
  if (!nodes.length) return null;
  let best = null;
  let bestScore = -1;
  for (let i = 0; i < nodes.length; i += 1) {
    const node = nodes[i];
    const status = String(node.getAttribute("data-node-status") || "").toLowerCase();
    const depth = Number(node.getAttribute("data-node-depth") || 0);
    const isRunning = node.classList.contains("running")
      || node.classList.contains("active-running")
      || status === "running"
      || status === "decomposing"
      || status === "researching";
    const score = (isRunning ? 1_000_000 : 0) + (depth * 1_000) + i;
    if (score > bestScore) {
      bestScore = score;
      best = node;
    }
  }
  return best;
}

function applySnapshot(snap) {
  currentSnapshot = snap;
  runStatusEl.textContent = shortStatus(snap.status);
  researchStatusEl.textContent = shortStatus(snap.research_status);
  reportStatusEl.textContent = shortStatus(snap.report_status || (snap.tree && snap.tree.report_status) || "pending");
  runMeta.textContent = `Run ID: ${snap.run_id}`;
  stopReasonEl.textContent = snap.stop_reason ? `Stop rationale: ${snap.stop_reason}` : "";
  latestThoughtEl.textContent = snap.latest_thought || "";
  coverageNoteEl.textContent = snap.coverage_note || "";

  showError(snap.error || "");
  renderCanvas(snap.tree || {});
  const hasReport = !!snap.report_file_path || !!(snap.report_text || "").trim();
  if (hasReport) {
    renderTokenSummary(snap.token_usage);
  } else {
    tokenSummaryEl.textContent = "-";
    usageEl.textContent = "";
  }

  const reportText = snap.report_text || "";
  reportRawEl.textContent = reportText;
  reportRenderedEl.innerHTML = markdownToHtml(reportText);
  toggleReportMode();

  const running = snap.status === "running" || snap.status === "queued";
  const canStop = running && !abortRequested;
  runBtn.textContent = abortRequested ? "Stopping..." : (running ? "Stop Research" : "Start Research");
  runBtn.classList.toggle("primary", !running);
  runBtn.disabled = abortRequested;

  const hasDownloadableReport = !!snap.report_file_path;
  downloadBtn.disabled = !hasDownloadableReport;
  const allowManualReport = !!currentRunId && !reportGenerating && !running && !hasDownloadableReport;
  reportBtn.disabled = !allowManualReport;
  reportBtn.textContent = reportGenerating ? "Generating Report..." : "Generate Report From Current Findings";
}

async function fetchSnapshot(runId) {
  const rsp = await fetch(`/api/runs/${runId}`);
  if (!rsp.ok) throw new Error(`Failed to fetch snapshot: ${rsp.status}`);
  const data = await rsp.json();
  applySnapshot(data);
}

function connectEvents(runId) {
  if (es) es.close();
  es = new EventSource(`/api/runs/${runId}/events`);

  es.onmessage = async (evt) => {
    try {
      if (evt.data) {
        const parsed = JSON.parse(evt.data);
        const narration = eventNarration(parsed);
        if (narration) {
          upsertThought(narration, parsed.event_type || "event");
        }
      }
    } catch (_err) {
      // no-op
    }
    await fetchSnapshot(runId);
  };

  const events = [
    "run_started", "plan_created", "round_started", "sub_question_started", "queries_generated",
    "query_started", "query_diagnostic", "query_skipped_cached", "query_rerun_allowed",
    "query_broadened", "query_blocked_diminishing_returns", "search_completed", "synthesis_completed",
    "node_sufficiency_started", "node_sufficiency_completed", "node_decomposition_started", "node_decomposed", "node_completed", "node_unresolved",
    "sufficiency_started", "sufficiency_completed", "run_abort_requested", "abort_requested", "run_aborted",
    "report_generation_started", "report_generation_completed", "report_generation_failed",
    "partial_report_generated", "run_heartbeat", "run_completed", "run_failed"
  ];
  for (const e of events) {
    es.addEventListener(e, async (evt) => {
      try {
        const parsed = JSON.parse(evt.data || "{}");
        const narration = eventNarration(parsed);
        if (narration) {
          upsertThought(narration, parsed.event_type || e);
        }
        if (parsed.event_type === "run_aborted" || parsed.event_type === "run_completed" || parsed.event_type === "run_failed") {
          abortRequested = false;
        }
      } catch (_err) {
        // no-op
      }
      await fetchSnapshot(runId);
    });
  }
}

async function startRun() {
  const task = taskEl.value.trim();
  if (!task) {
    showError("Task is required.");
    return;
  }
  showError("");
  abortRequested = false;
  autoFollowActiveNode = true;
  thoughts.length = 0;
  thoughtStreamEl.innerHTML = "";

  const rsp = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      task,
      max_depth: Number(maxDepthEl.value || 3),
      results_per_query: Number(resultsPerQueryEl.value || 3),
    }),
  });
  if (!rsp.ok) {
    throw new Error(`Run creation failed: ${rsp.status}`);
  }
  const data = await rsp.json();
  currentRunId = data.run_id;
  setRunIdInUrl(currentRunId);
  await fetchSnapshot(currentRunId);
  connectEvents(currentRunId);
}

async function stopRun() {
  if (!currentRunId) return;
  abortRequested = true;
  runBtn.textContent = "Stopping...";
  runBtn.disabled = true;
  const rsp = await fetch(`/api/runs/${currentRunId}/abort`, { method: "POST" });
  if (!rsp.ok) {
    abortRequested = false;
    throw new Error(`Abort failed: ${rsp.status}`);
  }
}

runBtn.addEventListener("click", async () => {
  try {
    const running = currentSnapshot && (currentSnapshot.status === "queued" || currentSnapshot.status === "running");
    if (running) {
      await stopRun();
    } else {
      await startRun();
    }
  } catch (err) {
    showError(err.message || String(err));
  }
});

reportBtn.addEventListener("click", async () => {
  if (!currentRunId || reportGenerating) return;
  reportGenerating = true;
  reportBtn.disabled = true;
  reportBtn.textContent = "Generating Report...";
  showError("");
  upsertThought("Generating report from accumulated findings.", "report");

  try {
    const rsp = await fetch(`/api/runs/${currentRunId}/report`, { method: "POST" });
    if (!rsp.ok) {
      throw new Error(`Report generation failed: ${rsp.status}`);
    }
    await fetchSnapshot(currentRunId);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    reportGenerating = false;
    if (currentRunId) {
      await fetchSnapshot(currentRunId);
    }
  }
});

downloadBtn.addEventListener("click", () => {
  if (!currentRunId) return;
  window.location.href = `/api/runs/${currentRunId}/report/download`;
});

rawToggleEl.addEventListener("change", toggleReportMode);

zoomOutBtn.addEventListener("click", () => {
  autoFitCanvas = false;
  canvasZoom = clamp(canvasZoom - 0.05, MIN_CANVAS_ZOOM, MAX_CANVAS_ZOOM);
  applyCanvasZoom();
});

zoomInBtn.addEventListener("click", () => {
  autoFitCanvas = false;
  canvasZoom = clamp(canvasZoom + 0.05, MIN_CANVAS_ZOOM, MAX_CANVAS_ZOOM);
  applyCanvasZoom();
});

zoomFitBtn.addEventListener("click", () => {
  autoFitCanvas = true;
  refreshCanvasZoomForCurrentLayout();
});

window.addEventListener("resize", () => {
  if (autoFitCanvas) {
    refreshCanvasZoomForCurrentLayout();
  }
});

async function bootstrapFromUrl() {
  const runId = readRunIdFromUrl();
  if (!runId) {
    renderCanvas(null);
    return;
  }
  currentRunId = runId;
  try {
    await fetchSnapshot(runId);
    connectEvents(runId);
    upsertThought("Restored run state from URL.", "bootstrap");
  } catch (err) {
    showError(err.message || String(err));
  }
}

bootstrapFromUrl();

function disableAutoFollowFromUserAction(evt) {
  if (suppressUserScrollDetection) return;
  if (evt) {
    if (evt.type === "keydown") {
      // Ignore typing/navigation keys when user is editing inputs.
      const ae = document.activeElement;
      if (ae) {
        const tag = String(ae.tagName || "").toLowerCase();
        const editable = !!ae.isContentEditable;
        if (editable || tag === "input" || tag === "textarea" || tag === "select") {
          return;
        }
      }
    }
  }
  autoFollowActiveNode = false;
}

canvasEl.addEventListener("wheel", disableAutoFollowFromUserAction, { passive: true });
canvasEl.addEventListener("touchstart", disableAutoFollowFromUserAction, { passive: true });

window.addEventListener("wheel", disableAutoFollowFromUserAction, { passive: true });
window.addEventListener("touchstart", disableAutoFollowFromUserAction, { passive: true });

window.addEventListener("keydown", (evt) => {
  const key = String(evt.key || "");
  if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(key)) {
    disableAutoFollowFromUserAction(evt);
  }
});
