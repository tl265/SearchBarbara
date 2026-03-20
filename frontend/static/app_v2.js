const taskEl = document.getElementById("task");
const maxDepthEl = document.getElementById("maxDepth");
const runBtn = document.getElementById("runBtn");
const reportBtn = document.getElementById("reportBtn");
const downloadBtn = document.getElementById("downloadBtn");
const planningActionBtn = document.getElementById("planningActionBtn");
const pauseBtn = document.getElementById("pauseBtn");
const resumeBtn = document.getElementById("resumeBtn");
const abortBtn = document.getElementById("abortBtn");
const swapBatchCanvasBtn = document.getElementById("swapBatchCanvasBtn");
const planningCommitCanvasBtn = document.getElementById("planningCommitCanvasBtn");
const runMeta = document.getElementById("runMeta");
const stopReasonEl = document.getElementById("stopReason");
const errorBanner = document.getElementById("errorBanner");
const sessionListEl = document.getElementById("sessionList");
const sessionRailToggleBtn = document.getElementById("sessionRailToggleBtn");
const newSessionBtn = document.getElementById("newSessionBtn");
const refreshSessionsBtn = document.getElementById("refreshSessionsBtn");
const contextMetaEl = document.getElementById("contextMeta");
const contextUploadInput = document.getElementById("contextUploadInput");
const contextUploadBtn = document.getElementById("contextUploadBtn");
const contextAddBtn = document.getElementById("contextAddBtn");
const inlineContextFilesEl = document.getElementById("inlineContextFiles");
const contextDiffBannerEl = document.getElementById("contextDiffBanner");
const contextFilesEl = document.getElementById("contextFiles");
const contextAggregateDigestEl = document.getElementById("contextAggregateDigest");
const contextFileDigestEl = document.getElementById("contextFileDigest");
const contextDigestPaneEl = document.querySelector(".context-digest-pane");
const appMainEl = document.querySelector(".app-main");
const liveIntentEl = document.getElementById("liveIntent");

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
const canvasTreePanel = document.getElementById("canvasTreePanel");
const canvasDetailPanel = document.getElementById("canvasDetailPanel");
const progressSection = document.getElementById("progressSection");
const progressTitleEl = document.querySelector(".progress-title");
const thoughtStreamEl = document.getElementById("thoughtStream");
const latestThoughtEl = document.getElementById("latestThought");
const progressLog = document.getElementById("progressLog");
const coverageNoteEl = document.getElementById("coverageNote");
const reportRenderedEl = document.getElementById("reportModalBody");
const viewReportBtn = document.getElementById("viewReportBtn");
const generateReportBtn = document.getElementById("generateReportBtn");
const reportModal = document.getElementById("reportModal");
const reportModalCloseBtn = document.getElementById("reportModalCloseBtn");
const reportModalDownloadBtn = document.getElementById("reportModalDownloadBtn");
const usageEl = document.getElementById("usage");
const reportPrevBtn = document.getElementById("reportPrevBtn");
const reportNextBtn = document.getElementById("reportNextBtn");
const reportVersionLabel = document.getElementById("reportVersionLabel");
const reportTemplateToggleBtn = document.getElementById("reportTemplateToggleBtn");
const reportTemplateSelectEl = document.getElementById("reportTemplateSelect");
const reportTemplateEditorEl = document.getElementById("reportTemplateEditor");
const reportTplNameEl = document.getElementById("reportTplName");
const reportTplBackgroundTypeEl = document.getElementById("reportTplBackgroundType");
const reportTplAudienceEl = document.getElementById("reportTplAudience");
const reportTplPresentationSetupEl = document.getElementById("reportTplPresentationSetup");
const reportTplDosEl = document.getElementById("reportTplDos");
const reportTplDontsEl = document.getElementById("reportTplDonts");
const reportTplToneEl = document.getElementById("reportTplTone");
const reportTplFocusEl = document.getElementById("reportTplFocus");
const reportTplNewBtn = document.getElementById("reportTplNewBtn");
const reportTplSaveBtn = document.getElementById("reportTplSaveBtn");
const reportTplDeleteBtn = document.getElementById("reportTplDeleteBtn");
const reportTplPreviewBtn = document.getElementById("reportTplPreviewBtn");
const reportTplPreviewEl = document.getElementById("reportTplPreview");
const APP_CONFIG = (window.APP_CONFIG && typeof window.APP_CONFIG === "object")
  ? window.APP_CONFIG
  : {};
const SWAP_BATCH_LABEL = "Refresh the lineup";
const SESSION_SYNC_CHANNEL = "searchbarbara:sessions:v1";
const SESSION_SYNC_STORAGE_KEY = "searchbarbara:sessions:signal";
const SESSION_RAIL_COLLAPSED_KEY = "sb_sessions_rail_collapsed";
const SESSION_SYNC_TAB_ID = (() => {
  try {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return `tab_${window.crypto.randomUUID()}`;
    }
  } catch (_err) {
    // no-op
  }
  return `tab_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
})();

let currentRunId = null;
let currentWorkspaceId = "";
let currentSnapshot = null;
let es = null;
let activeSessionSwitchToken = 0;
let snapshotFetchInFlight = false;
let snapshotFetchQueued = false;
let startRunInFlight = false;
const reportGeneratingRunIds = new Set();
let pauseRequested = false;
let abortRequested = false;
let uiClearedByAbort = false;
let selectedReportVersionIndex = null;
let autoFollowActiveNode = true;
let lastAutoFollowNodeKey = "";
let selectedCanvasNodeKey = "";
let canvasNodeDataMap = new Map();
let skipNextCanvasScrollDisable = false;
const thoughts = [];
const openDetailKeys = new Set();
const forceClosedDetailKeys = new Set();
let canvasZoom = 1;
let autoFitCanvas = true;
let sessions = [];
let sessionSyncChannel = null;
let sessionsRefreshTimer = null;
let sessionsRefreshInFlight = false;
const lastEventSeqByRun = new Map();
// DEBUG ONLY: surface session-list load failures prominently while session
// management is being stabilized.
let debugSessionLoadError = "";
const MIN_CANVAS_ZOOM = clamp(Number(APP_CONFIG.min_canvas_zoom ?? 0.45), 0.1, 0.95);
const MAX_CANVAS_ZOOM = 1.6;
const DEFAULT_MAX_DEPTH = Math.max(1, Math.floor(Number(APP_CONFIG.default_max_depth ?? 3)));
const DEFAULT_RESULTS_PER_QUERY = Math.max(
  1,
  Math.floor(Number(APP_CONFIG.default_results_per_query ?? 3))
);
const LIVE_INTENT_ENABLED = !!APP_CONFIG.live_intent_enabled;
const LIVE_INTENT_MIN_CHARS_DEFAULT = Math.max(
  1,
  Math.floor(Number(APP_CONFIG.live_intent_min_chars_default ?? 12))
);
const LIVE_INTENT_MIN_CHARS_CJK = Math.max(
  1,
  Math.floor(Number(APP_CONFIG.live_intent_min_chars_cjk ?? 6))
);
const LIVE_INTENT_DEBOUNCE_MS = Math.max(
  80,
  Math.floor(Number(APP_CONFIG.live_intent_debounce_ms ?? 320))
);
const LIVE_INTENT_CONFIDENCE_THRESHOLD = clamp(
  Number(APP_CONFIG.live_intent_confidence_threshold ?? 0.45),
  0,
  0.95
);
const BASE_NODE_CARD_WIDTH = 340;
const MIN_NODE_FONT_SCALE = 0.72;
const MAX_NODE_FONT_SCALE = 1.25;
const AUTO_FIT_SAFETY_PX = Math.max(0, Math.floor(Number(APP_CONFIG.auto_fit_safety_px ?? 10)));
const UI_DEBUG = !!APP_CONFIG.ui_debug;
let lastDebugReportPhaseKey = "";
let contextEs = null;
let currentContextSet = null;
let selectedContextFileId = "";
let contextMutationInFlight = false;
let pendingUploadFiles = [];
let contextInputLocked = false;
let planningMutationInFlight = false;
let reportTemplateMutationInFlight = false;
let reportTemplates = [];
let selectedReportTemplateId = "executive";
if (reportTemplateSelectEl) {
  reportTemplates = [{
    template_id: "executive",
    name: "Executive / Senior Management",
    background_type: "executive",
    is_builtin: true,
    is_default_manual: true,
    fields: { audience: "", presentation_setup: "", dos: [], donts: [], tone: "", focus: "" },
    rendered_background_prompt: "",
  }];
}
const contextFetchStateByKey = new Map();
const contextRefreshDebounceTimerByKey = new Map();
const contextFetchSeqByKey = new Map();
const stagedContextBySessionId = new Map();
const startupParseStateByRunId = new Map();
let currentContextSource = "session";
let errorBannerTimer = null;
let contextAggregateViewBase = null;
let contextAggregateViewMode = "";
let liveIntentDebounceTimer = null;
let liveIntentRequestSeq = 0;
let liveIntentAppliedServerFingerprint = "";
let liveIntentAppliedText = "";
let liveIntentPendingServerFingerprint = "";
let liveIntentPendingServerCount = 0;
let liveIntentDisplayedPrediction = null;
let liveIntentFetchController = null;
if (document && document.body) {
  document.body.classList.toggle("ui-debug-on", UI_DEBUG);
}
if (contextDigestPaneEl) {
  contextDigestPaneEl.style.display = "";
}

function setReportButtonVisual(generating) {
  if (!reportBtn) return;
  const isGenerating = !!generating;
  reportBtn.classList.toggle("is-loading", isGenerating);
  const label = isGenerating
    ? "Generating report..."
    : "Generate report from current findings";
  reportBtn.setAttribute("title", label);
  reportBtn.setAttribute("aria-label", label);
}

function splitLines(value, maxItems = 20) {
  const out = [];
  const raw = String(value || "").split(/\r?\n/);
  for (let i = 0; i < raw.length; i += 1) {
    const s = String(raw[i] || "").trim();
    if (!s) continue;
    out.push(s);
    if (out.length >= maxItems) break;
  }
  return out;
}

function reportTemplateDraftFromForm() {
  return {
    name: String(reportTplNameEl && reportTplNameEl.value || "").trim(),
    background_type: String(reportTplBackgroundTypeEl && reportTplBackgroundTypeEl.value || "custom").trim() || "custom",
    fields: {
      audience: String(reportTplAudienceEl && reportTplAudienceEl.value || "").trim(),
      presentation_setup: String(reportTplPresentationSetupEl && reportTplPresentationSetupEl.value || "").trim(),
      dos: splitLines(reportTplDosEl && reportTplDosEl.value || "", 20),
      donts: splitLines(reportTplDontsEl && reportTplDontsEl.value || "", 20),
      tone: String(reportTplToneEl && reportTplToneEl.value || "").trim(),
      focus: String(reportTplFocusEl && reportTplFocusEl.value || "").trim(),
    },
  };
}

function findReportTemplateById(templateId) {
  const tid = String(templateId || "").trim();
  if (!tid) return null;
  for (let i = 0; i < reportTemplates.length; i += 1) {
    const tpl = reportTemplates[i];
    if (!tpl || typeof tpl !== "object") continue;
    if (String(tpl.template_id || "").trim() === tid) return tpl;
  }
  return null;
}

function setReportTemplateForm(template) {
  const tpl = template && typeof template === "object" ? template : {};
  const fields = tpl.fields && typeof tpl.fields === "object" ? tpl.fields : {};
  if (reportTplNameEl) reportTplNameEl.value = String(tpl.name || "");
  if (reportTplBackgroundTypeEl) reportTplBackgroundTypeEl.value = String(tpl.background_type || "custom");
  if (reportTplAudienceEl) reportTplAudienceEl.value = String(fields.audience || "");
  if (reportTplPresentationSetupEl) reportTplPresentationSetupEl.value = String(fields.presentation_setup || "");
  if (reportTplDosEl) reportTplDosEl.value = Array.isArray(fields.dos) ? fields.dos.map((v) => String(v || "")).join("\n") : "";
  if (reportTplDontsEl) reportTplDontsEl.value = Array.isArray(fields.donts) ? fields.donts.map((v) => String(v || "")).join("\n") : "";
  if (reportTplToneEl) reportTplToneEl.value = String(fields.tone || "");
  if (reportTplFocusEl) reportTplFocusEl.value = String(fields.focus || "");
}

function refreshReportTemplateActionAvailability() {
  const selected = findReportTemplateById(selectedReportTemplateId);
  const isBuiltin = !!(selected && selected.is_builtin);
  if (reportTplDeleteBtn && !reportTemplateMutationInFlight) {
    reportTplDeleteBtn.disabled = isBuiltin;
  }
}

function clearReportTemplateFormForNew() {
  setReportTemplateForm({
    name: "",
    background_type: "custom",
    fields: { audience: "", presentation_setup: "", dos: [], donts: [], tone: "", focus: "" },
  });
  if (reportTplPreviewEl) reportTplPreviewEl.textContent = "";
}

function renderReportTemplateSelect() {
  if (!reportTemplateSelectEl) return;
  const previous = String(selectedReportTemplateId || "").trim();
  reportTemplateSelectEl.innerHTML = "";
  for (let i = 0; i < reportTemplates.length; i += 1) {
    const tpl = reportTemplates[i];
    if (!tpl || typeof tpl !== "object") continue;
    const option = document.createElement("option");
    const tid = String(tpl.template_id || "").trim();
    option.value = tid;
    option.textContent = String(tpl.name || tid || "Template");
    if (tpl.is_builtin) option.textContent += " (Built-in)";
    reportTemplateSelectEl.appendChild(option);
  }
  let nextSelected = previous;
  if (!findReportTemplateById(nextSelected)) {
    const defaultTpl = reportTemplates.find((tpl) => tpl && tpl.is_default_manual);
    nextSelected = String((defaultTpl && defaultTpl.template_id) || "");
  }
  if (!findReportTemplateById(nextSelected) && reportTemplates.length) {
    nextSelected = String((reportTemplates[0] && reportTemplates[0].template_id) || "");
  }
  selectedReportTemplateId = nextSelected || "executive";
  reportTemplateSelectEl.value = selectedReportTemplateId;
  const selected = findReportTemplateById(selectedReportTemplateId);
  if (selected) {
    setReportTemplateForm(selected);
    if (reportTplPreviewEl) {
      reportTplPreviewEl.textContent = String(selected.rendered_background_prompt || "");
    }
  } else {
    clearReportTemplateFormForNew();
  }
  refreshReportTemplateActionAvailability();
}

function setReportTemplateControlsDisabled(disabled) {
  const off = !!disabled;
  if (reportTemplateToggleBtn) reportTemplateToggleBtn.disabled = off;
  if (reportTemplateSelectEl) reportTemplateSelectEl.disabled = off;
  if (reportTplNameEl) reportTplNameEl.disabled = off;
  if (reportTplBackgroundTypeEl) reportTplBackgroundTypeEl.disabled = off;
  if (reportTplAudienceEl) reportTplAudienceEl.disabled = off;
  if (reportTplPresentationSetupEl) reportTplPresentationSetupEl.disabled = off;
  if (reportTplDosEl) reportTplDosEl.disabled = off;
  if (reportTplDontsEl) reportTplDontsEl.disabled = off;
  if (reportTplToneEl) reportTplToneEl.disabled = off;
  if (reportTplFocusEl) reportTplFocusEl.disabled = off;
  if (reportTplNewBtn) reportTplNewBtn.disabled = off;
  if (reportTplSaveBtn) reportTplSaveBtn.disabled = off;
  if (reportTplDeleteBtn) reportTplDeleteBtn.disabled = off;
  if (reportTplPreviewBtn) reportTplPreviewBtn.disabled = off;
  if (!off) {
    refreshReportTemplateActionAvailability();
  }
}

function syncReportTemplateToggleState() {
  if (!reportTemplateToggleBtn) return;
  const isOpen = !!(reportTemplateEditorEl && reportTemplateEditorEl.open);
  reportTemplateToggleBtn.classList.toggle("is-active", isOpen);
  reportTemplateToggleBtn.setAttribute("aria-pressed", isOpen ? "true" : "false");
  reportTemplateToggleBtn.setAttribute(
    "aria-label",
    isOpen ? "Collapse template editor" : "Expand template editor"
  );
  reportTemplateToggleBtn.setAttribute(
    "title",
    isOpen ? "Collapse template editor" : "Expand template editor"
  );
}

function setReportTemplateEditorOpen(open) {
  if (!reportTemplateEditorEl) return;
  reportTemplateEditorEl.open = !!open;
  syncReportTemplateToggleState();
}

async function fetchReportTemplates() {
  const rsp = await fetch("/api/report/templates");
  if (!rsp.ok) {
    throw new Error(await responseDetail(rsp, `Load templates failed: ${rsp.status}`));
  }
  const data = await rsp.json();
  const list = Array.isArray(data.templates) ? data.templates : [];
  reportTemplates = list.filter((tpl) => tpl && typeof tpl === "object");
  if (!reportTemplates.length) {
    reportTemplates = [{
      template_id: "executive",
      name: "Executive / Senior Management",
      background_type: "executive",
      is_builtin: true,
      is_default_manual: true,
      fields: { audience: "", presentation_setup: "", dos: [], donts: [], tone: "", focus: "" },
      rendered_background_prompt: "",
    }];
  }
  renderReportTemplateSelect();
}

function setSessionsRailCollapsed(collapsed, persist = true) {
  const on = !!collapsed;
  if (document && document.body) {
    document.body.classList.toggle("sessions-collapsed", on);
  }
  if (sessionRailToggleBtn) {
    sessionRailToggleBtn.classList.toggle("is-collapsed", on);
    sessionRailToggleBtn.setAttribute("aria-label", on ? "Expand sessions panel" : "Collapse sessions panel");
    sessionRailToggleBtn.setAttribute("title", on ? "Expand sessions panel" : "Collapse sessions panel");
  }
  if (persist) {
    try {
      localStorage.setItem(SESSION_RAIL_COLLAPSED_KEY, on ? "1" : "0");
    } catch (_err) {
      // no-op
    }
  }
}

function loadSessionsRailCollapsed() {
  try {
    return localStorage.getItem(SESSION_RAIL_COLLAPSED_KEY) === "1";
  } catch (_err) {
    return false;
  }
}

function esc(s) {
  return String(s || "").replace(/[&<>\"]/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
  })[c]);
}

function newIdempotencyKey(prefix) {
  const p = String(prefix || "req").trim() || "req";
  try {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return `${p}_${window.crypto.randomUUID()}`;
    }
  } catch (_err) {
    // fall through to non-crypto fallback
  }
  return `${p}_${Date.now()}_${Math.random().toString(36).slice(2, 12)}`;
}

function setRunIdInUrl(runId) {
  const url = new URL(window.location.href);
  if (runId) {
    url.searchParams.set("session_id", runId);
    url.searchParams.set("run_id", runId);
  } else {
    url.searchParams.delete("session_id");
    url.searchParams.delete("run_id");
  }
  history.replaceState({}, "", url);
}

function resetWorkspaceForNewSession() {
  if (es) {
    es.close();
    es = null;
  }
  if (contextEs) {
    contextEs.close();
    contextEs = null;
  }
  currentRunId = null;
  clearLiveIntent();
  if (appMainEl) appMainEl.classList.add("is-empty");
  // Start a truly fresh workspace context for each new session/reset.
  currentWorkspaceId = newWorkspaceId();
  currentSnapshot = null;
  pauseRequested = false;
  abortRequested = false;
  uiClearedByAbort = false;
  reportGeneratingRunIds.clear();
  activeSessionSwitchToken += 1;
  autoFollowActiveNode = true;
  lastAutoFollowNodeKey = "";
  selectedCanvasNodeKey = "";
  canvasNodeDataMap = new Map();
  thoughts.length = 0;
  lastDebugReportPhaseKey = "";
  openDetailKeys.clear();
  forceClosedDetailKeys.clear();
  if (thoughtStreamEl) thoughtStreamEl.innerHTML = "";

  if (progressLog) progressLog.innerHTML = "";

  runMeta.textContent = "";
  stopReasonEl.textContent = "";
  if (latestThoughtEl) latestThoughtEl.textContent = "";
  if (coverageNoteEl) coverageNoteEl.textContent = "";
  reportRenderedEl.innerHTML = "";
  viewReportBtn.disabled = true;
  reportModal.classList.add("hidden");
  tokenSummaryEl.textContent = "-";
  usageEl.textContent = "";
  maxDepthEl.value = String(DEFAULT_MAX_DEPTH);
  if (depthDropdownValue) depthDropdownValue.textContent = String(DEFAULT_MAX_DEPTH);
  if (taskEl) taskEl.value = "";
  updateTaskCharCount();
  setTaskBoxLocked(false);
  setRunConfigLocked(false);
  setContextEnabled(true);
  clearContextPane("No context files uploaded.");
  setRunIdInUrl("");
  renderCanvas(null);
  showError("");
  runStatusEl.textContent = "idle";
  researchStatusEl.textContent = "idle";
  reportStatusEl.textContent = "pending";
  switchRunBtnToStart();
  if (planningActionBtn) {
    planningActionBtn.textContent = "Start Planning";
    planningActionBtn.disabled = false;
  }
  reportBtn.disabled = true;
  if (generateReportBtn) generateReportBtn.disabled = true;
  setReportButtonVisual(false);
  if (downloadBtn) {
    downloadBtn.setAttribute("title", "Download report");
    downloadBtn.setAttribute("aria-label", "Download report");
    downloadBtn.disabled = true;
  }
  pauseBtn.disabled = true;
  resumeBtn.disabled = true;
  abortBtn.disabled = true;
  if (reportPrevBtn) reportPrevBtn.disabled = true;
  if (reportNextBtn) reportNextBtn.disabled = true;
  if (reportVersionLabel) reportVersionLabel.textContent = "-/-";
  selectedReportVersionIndex = null;
  currentSnapshot = null;
  renderPlanningPanel(null);
}

function isDraftSnapshot(snap) {
  const tree = snap && typeof snap.tree === "object" ? snap.tree : {};
  return !!tree.is_draft;
}

function planningPhase(snap) {
  if (!snap || typeof snap !== "object") return "research";
  const tree = snap.tree && typeof snap.tree === "object" ? snap.tree : {};
  const phase = String(snap.phase || tree.phase || "research").trim().toLowerCase();
  return phase === "planning" ? "planning" : "research";
}

function planningState(snap) {
  if (!snap || typeof snap !== "object") return "idle";
  const tree = snap.tree && typeof snap.tree === "object" ? snap.tree : {};
  const st = String(snap.planning_state || tree.planning_state || "").trim().toLowerCase();
  return st || "idle";
}

function planningCandidates(snap) {
  if (!snap || typeof snap !== "object") return [];
  const tree = snap.tree && typeof snap.tree === "object" ? snap.tree : {};
  const planning = tree.planning && typeof tree.planning === "object" ? tree.planning : {};
  const rows = Array.isArray(planning.root_children_candidates) ? planning.root_children_candidates : [];
  return rows.filter((r) => r && typeof r === "object");
}

function planningDepthBonusLimit(snap) {
  if (!snap || typeof snap !== "object") return null;
  const tree = snap.tree && typeof snap.tree === "object" ? snap.tree : {};
  const planning = tree.planning && typeof tree.planning === "object" ? tree.planning : {};
  const defaultMax = Number(planning.default_max_depth);
  const systemMax = Number(planning.system_max_depth);
  if (!Number.isFinite(defaultMax) || !Number.isFinite(systemMax)) return null;
  return Math.max(0, Math.floor(systemMax) - Math.floor(defaultMax));
}

function planningResearchDepthBonusMap(snap) {
  const out = new Map();
  if (!snap || typeof snap !== "object") return out;
  const tree = snap.tree && typeof snap.tree === "object" ? snap.tree : {};
  const seed = tree.planning_seed && typeof tree.planning_seed === "object" ? tree.planning_seed : {};
  const perBranch = seed.child_branch_max_depths && typeof seed.child_branch_max_depths === "object"
    ? seed.child_branch_max_depths
    : {};
  const planning = tree.planning && typeof tree.planning === "object" ? tree.planning : {};
  const baselineRaw = Number(planning.default_max_depth);
  const fallbackRaw = Number(snap.max_depth);
  const baseline = Number.isFinite(baselineRaw)
    ? Math.max(1, Math.floor(baselineRaw))
    : (Number.isFinite(fallbackRaw) ? Math.max(1, Math.floor(fallbackRaw)) : 1);

  for (const [question, branchRaw] of Object.entries(perBranch)) {
    const key = normalizeQuestionKey(question);
    const branchMax = Number(branchRaw);
    if (!key || !Number.isFinite(branchMax)) continue;
    const branch = Math.max(1, Math.floor(branchMax));
    const bonus = Math.max(0, branch - baseline);
    if (bonus <= 0) continue;
    out.set(key, { bonus, branchMax: branch, baseline });
  }
  return out;
}

function isPlanningSwapInProgress(snap) {
  if (!snap || typeof snap !== "object") return false;
  if (planningPhase(snap) !== "planning") return false;
  const pState = planningState(snap);
  if (!["running", "idle"].includes(pState)) return false;
  const tree = snap.tree && typeof snap.tree === "object" ? snap.tree : {};
  const pendingAction = String(tree.planning_pending_action || "").trim().toLowerCase();
  return pendingAction === "swap_batch" || pendingAction === "swap_batch_retry";
}

function derivePlanningUiState(phase, pState, opts = {}) {
  const inPlanning = phase === "planning";
  const isReview = inPlanning && pState === "review";
  const isMutating = !!planningMutationInFlight;
  const swapInProgress = !!opts.swapInProgress;
  const reportBusy = !!opts.reportBusy;
  const transitionPending = !!opts.transitionPending;
  const startRunBusy = !!opts.startRunBusy;
  const canEdit = isReview && !isMutating;
  const swapBatchText = (isMutating || swapInProgress) ? "Swapping..." : "Swap Batch";
  let planningActionText = "Start Planning";
  let planningActionDisabled = true;

  if (!inPlanning) {
    planningActionDisabled = true;
  } else if (isReview) {
    planningActionText = isMutating ? "Working..." : "Go for a full run";
    planningActionDisabled = !!(reportBusy || transitionPending || isMutating || startRunBusy);
  } else if (pState === "running" || pState === "idle") {
    planningActionText = "Planning...";
    planningActionDisabled = true;
  }

  return {
    inPlanning,
    canEdit,
    swapBatchText,
    planningActionText,
    planningActionDisabled,
  };
}

function isReportGeneratingForRun(runId) {
  const rid = String(runId || "").trim();
  return !!rid && reportGeneratingRunIds.has(rid);
}

function newWorkspaceId() {
  try {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return `ws_${window.crypto.randomUUID()}`;
    }
  } catch (_err) {
    // no-op
  }
  return `ws_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

function ensureWorkspaceId() {
  const ws = String(currentWorkspaceId || "").trim();
  if (ws) return ws;
  currentWorkspaceId = newWorkspaceId();
  return currentWorkspaceId;
}

function contextBasePath(runIdOverride = "", workspaceIdOverride = "") {
  const sid = String(runIdOverride || currentRunId || "").trim();
  if (sid) return `/api/sessions/${encodeURIComponent(sid)}/context`;
  const wid = String(workspaceIdOverride || ensureWorkspaceId()).trim();
  return `/api/workspaces/${encodeURIComponent(wid)}/context`;
}

function currentExpectedVersion() {
  const v = Number(currentSnapshot && currentSnapshot.version);
  if (Number.isFinite(v) && v >= 1) return Math.floor(v);
  return null;
}

function readRunIdFromUrl() {
  const url = new URL(window.location.href);
  return (url.searchParams.get("session_id") || url.searchParams.get("run_id") || "").trim();
}

function setTaskBoxLocked(locked) {
  taskEl.disabled = !!locked;
  taskEl.classList.toggle("locked", !!locked);
  contextInputLocked = !!locked;
  if (contextAddBtn) contextAddBtn.disabled = !!locked;
  if (inlineContextFilesEl) inlineContextFilesEl.classList.toggle("is-locked", !!locked);
  refreshContextUploadButtonState();
}

function setRunConfigLocked(locked) {
  const on = !!locked;
  maxDepthEl.disabled = on;
  if (depthDropdownBtn) depthDropdownBtn.disabled = on;
  /* Don't touch runBtn here – managed by switchRunBtnTo* helpers */
}

function switchRunBtnToAbort() {
  runBtn.textContent = "Abort";
  runBtn.classList.remove("primary");
  runBtn.disabled = false;
  runBtn.dataset.mode = "abort";
}

function switchRunBtnToStart() {
  runBtn.textContent = "Start";
  runBtn.classList.add("primary");
  delete runBtn.dataset.mode;
  runBtn.disabled = false;
}

function fmtTime(iso) {
  const s = String(iso || "").trim();
  if (!s) return "-";
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return s;
  const now = new Date();
  const diffMs = now - d;
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 7) return `${diffDay}d ago`;
  return d.toLocaleDateString();
}

function showError(msg) {
  if (errorBannerTimer) {
    clearTimeout(errorBannerTimer);
    errorBannerTimer = null;
  }
  if (!msg) {
    errorBanner.classList.add("hidden");
    errorBanner.textContent = "";
    return;
  }
  errorBanner.textContent = msg;
  errorBanner.classList.remove("hidden");
}

function showTransientError(msg, ms = 2000) {
  showError(msg);
  errorBannerTimer = setTimeout(() => {
    if (errorBanner.textContent === String(msg || "")) {
      showError("");
    }
  }, Math.max(200, Number(ms) || 2000));
}

async function responseDetail(rsp, fallback) {
  let detail = fallback;
  try {
    const body = await rsp.json();
    if (typeof body === "string" && body.trim()) {
      detail = body.trim();
    } else if (body && typeof body === "object") {
      const d = body.detail;
      if (typeof d === "string" && d.trim()) {
        detail = d.trim();
      } else if (d && typeof d === "object") {
        const msg = String(d.message || d.error || "").trim();
        if (msg) detail = msg;
      }
    }
  } catch (_err) {
    // no-op
  }
  return detail;
}

const LIVE_INTENT_FIELD_ORDER = [
  "task_type",
  "sophistication",
  "audience",
  "stake_level",
  "time_horizon",
];
const LIVE_INTENT_LABELS = {
  task_type: "Task",
  sophistication: "Depth",
  audience: "Audience",
  stake_level: "Stakes",
  time_horizon: "Horizon",
};
const LIVE_INTENT_VALUES = {
  task_type: {
    explain: "Explain",
    analyze: "Analyze",
    compare: "Compare",
    create: "Create",
    persuade: "Persuade",
    troubleshoot: "Troubleshoot",
    plan: "Plan",
  },
  sophistication: {
    intro: "Intro",
    intermediate: "Intermediate",
    deep: "Deep",
  },
  audience: {
    general_public: "General public",
    practitioner: "Practitioner",
    mid_management: "Mid-management",
    senior_management: "Senior management",
    academic: "Academic",
  },
  stake_level: {
    low: "Low",
    medium: "Medium",
    high: "High",
  },
  time_horizon: {
    immediate: "Immediate",
    near_term: "Near term",
    strategic: "Strategic",
  },
};
const LIVE_INTENT_HINTS = {
  task_type: {
    compare: ["compare", "versus", " vs ", "pros and cons", "tradeoff", "对比", "比较", "区别", "优缺点", "差异"],
    persuade: ["persuade", "convince", "memo", "pitch", "business case", "说服", "打动", "争取", "汇报材料", "立项"],
    troubleshoot: ["error", "failing", "bug", "debug", "fix", "broken", "issue", "incident", "502", "排查", "报错", "故障", "异常", "修复", "问题", "出错"],
    plan: ["plan", "roadmap", "strategy", "strategic", "next steps", "rollout", "migration plan", "规划", "计划", "路线图", "方案", "实施", "战略"],
    create: ["draft", "write", "create", "design", "build", "generate", "写", "起草", "设计", "生成", "做一份", "整理一份"],
    explain: ["explain", "what is", "why", "how does", "overview", "introduction", "解释", "为什么", "怎么", "介绍", "原理", "是什么"],
    analyze: ["analyze", "assess", "evaluate", "implications", "recommend", "recommendation", "decision", "分析", "评估", "判断", "建议", "决策"],
  },
  sophistication: {
    intro: ["simple", "simply", "beginner", "basic", "introduction", "eli5", "入门", "基础", "简单", "小白", "通俗"],
    deep: ["deep", "detailed", "rigorous", "exhaustive", "comprehensive", "technical", "architecture", "benchmark", "academic", "深入", "详细", "严谨", "全面", "系统", "技术细节"],
  },
  audience: {
    senior_management: ["ceo", "cto", "leadership", "executive", "senior management", "board", "exec team", "领导", "管理层", "高层", "老板"],
    mid_management: ["manager", "managers", "director", "directors", "team lead", "stakeholder", "经理", "主管", "负责人", "项目负责人"],
    academic: ["academic", "literature", "journal", "paper", "citation", "citations", "学术", "论文", "文献", "研究"],
    practitioner: ["engineer", "developer", "operator", "implementation", "architecture", "production", "deploy", "debugging", "工程师", "开发", "运维", "研发", "程序员", "架构师"],
    general_public: ["customer", "non-technical", "public", "general audience", "everyone", "普通人", "大众", "非技术", "小白用户"],
  },
  stake_level: {
    high: ["urgent", "asap", "critical", "board", "compliance", "risk", "production", "incident", "outage", "紧急", "高风险", "事故", "线上", "生产", "宕机", "故障"],
    medium: ["decide", "decision", "recommend", "migration", "launch", "plan", "persuade", "tradeoff", "决定", "决策", "建议", "迁移", "上线", "规划", "说服"],
  },
  time_horizon: {
    immediate: ["now", "today", "asap", "immediate", "right away", "fix", "incident", "urgent", "现在", "今天", "马上", "尽快", "立刻", "当前"],
    near_term: ["this week", "this month", "next sprint", "rollout", "migration", "near term", "launch", "本周", "这周", "近期", "这个月", "下周", "下个月"],
    strategic: ["next quarter", "quarter", "next year", "yearly", "long term", "strategic", "roadmap", "multi-year", "下季度", "明年", "长期", "战略", "年度", "路线图"],
  },
};
const LIVE_INTENT_CJK_RE = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]/g;
const LIVE_INTENT_LATIN_RE = /[A-Za-z]/g;

function liveIntentNormalizedText(raw) {
  return String(raw || "").replace(/\s+/g, " ").trim();
}

function liveIntentSignalChars(raw) {
  const text = String(raw || "");
  let total = 0;
  for (let i = 0; i < text.length; i += 1) {
    if (!/\s/.test(text[i])) total += 1;
  }
  return total;
}

function liveIntentCjkCharCount(raw) {
  const matches = String(raw || "").match(LIVE_INTENT_CJK_RE);
  return matches ? matches.length : 0;
}

function liveIntentLatinCharCount(raw) {
  const matches = String(raw || "").match(LIVE_INTENT_LATIN_RE);
  return matches ? matches.length : 0;
}

function liveIntentHasMeaningfulCjk(raw) {
  return liveIntentCjkCharCount(raw) >= 2;
}

function liveIntentMinimumChars(raw) {
  return liveIntentHasMeaningfulCjk(raw)
    ? LIVE_INTENT_MIN_CHARS_CJK
    : LIVE_INTENT_MIN_CHARS_DEFAULT;
}

function liveIntentTokenCount(raw) {
  const text = liveIntentNormalizedText(raw);
  return text ? text.split(" ").length : 0;
}

function liveIntentPhraseScore(text, phrases) {
  let score = 0;
  for (let i = 0; i < phrases.length; i += 1) {
    const phrase = String(phrases[i] || "").trim().toLowerCase();
    if (!phrase) continue;
    if (text.includes(phrase)) {
      score += 1 + (0.15 * Math.min(2, Math.max(0, phrase.split(" ").length - 1)));
    }
  }
  return score;
}

function liveIntentPick(scores, minimum) {
  let bestLabel = "";
  let bestScore = 0;
  for (const [label, scoreRaw] of Object.entries(scores || {})) {
    const score = Number(scoreRaw || 0);
    if (!Number.isFinite(score) || score <= bestScore) continue;
    bestLabel = label;
    bestScore = score;
  }
  if (!bestLabel || bestScore < Number(minimum || 0)) return null;
  return bestLabel;
}

function liveIntentIsMaterialShift(prevText, nextText) {
  const prev = liveIntentNormalizedText(prevText);
  const next = liveIntentNormalizedText(nextText);
  if (!prev || !next) return true;
  if (prev === next) return false;
  if (liveIntentHasMeaningfulCjk(prev) || liveIntentHasMeaningfulCjk(next)) {
    const sharedLimit = Math.min(prev.length, next.length);
    let sharedPrefix = 0;
    while (sharedPrefix < sharedLimit && prev[sharedPrefix] === next[sharedPrefix]) {
      sharedPrefix += 1;
    }
    if (sharedPrefix >= Math.max(3, Math.floor(sharedLimit * 0.6)) && Math.abs(next.length - prev.length) < 4) {
      return false;
    }
    return sharedPrefix < Math.max(2, Math.floor(sharedLimit * 0.35));
  }
  if ((next.startsWith(prev) || prev.startsWith(next)) && Math.abs(next.length - prev.length) < 20) {
    return false;
  }
  const prevTokens = prev.split(" ");
  const nextTokens = next.split(" ");
  const prevSet = new Set(prevTokens);
  let overlap = 0;
  for (let i = 0; i < nextTokens.length; i += 1) {
    if (prevSet.has(nextTokens[i])) overlap += 1;
  }
  const base = Math.max(prevTokens.length, nextTokens.length, 1);
  return (overlap / base) < 0.45;
}

function normalizeLiveIntentPrediction(raw, fallbackSource = "heuristic") {
  if (!raw || typeof raw !== "object") return null;
  const out = { source: fallbackSource, confidence: null };
  for (let i = 0; i < LIVE_INTENT_FIELD_ORDER.length; i += 1) {
    const key = LIVE_INTENT_FIELD_ORDER[i];
    const allowed = LIVE_INTENT_VALUES[key] || {};
    const value = String(raw[key] || "").trim();
    out[key] = Object.prototype.hasOwnProperty.call(allowed, value) ? value : "";
  }
  const confidence = Number(raw.confidence);
  if (Number.isFinite(confidence)) {
    out.confidence = clamp(confidence, 0, 1);
  }
  const source = String(raw.source || fallbackSource).trim().toLowerCase();
  out.source = source === "model" ? "model" : "heuristic";
  let visible = 0;
  for (let i = 0; i < LIVE_INTENT_FIELD_ORDER.length; i += 1) {
    if (out[LIVE_INTENT_FIELD_ORDER[i]]) visible += 1;
  }
  if (!visible) return null;
  return out;
}

function liveIntentFingerprint(prediction) {
  const normalized = normalizeLiveIntentPrediction(prediction);
  if (!normalized) return "";
  return LIVE_INTENT_FIELD_ORDER.map((key) => normalized[key] || "-").join("|");
}

function computeLocalLiveIntentPrediction(rawText) {
  const normalized = liveIntentNormalizedText(rawText);
  if (liveIntentSignalChars(normalized) < liveIntentMinimumChars(normalized)) return null;
  const lowered = ` ${normalized.toLowerCase()} `;
  const tokenCount = liveIntentTokenCount(lowered);
  const cjkCount = liveIntentCjkCharCount(normalized);
  const cjkMode = liveIntentHasMeaningfulCjk(normalized);
  const lengthScore = Math.max(tokenCount, cjkCount);
  const taskScores = {
    compare: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.task_type.compare),
    persuade: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.task_type.persuade),
    troubleshoot: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.task_type.troubleshoot),
    plan: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.task_type.plan),
    create: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.task_type.create),
    explain: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.task_type.explain),
    analyze: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.task_type.analyze),
  };
  if (normalized.endsWith("?") || normalized.endsWith("？")) {
    taskScores.explain += 0.35;
    taskScores.analyze += 0.2;
  }
  if (tokenCount >= 10 || cjkCount >= 10) taskScores.analyze += 0.15;
  const taskType = liveIntentPick(taskScores, 0.6) || "analyze";

  const sophisticationScores = {
    intro: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.sophistication.intro),
    intermediate: 0.3 + (lengthScore >= 12 ? 0.15 : 0),
    deep: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.sophistication.deep),
  };
  if (["compare", "troubleshoot", "plan", "persuade"].includes(taskType)) {
    sophisticationScores.intermediate += 0.25;
  }
  const sophistication = liveIntentPick(sophisticationScores, 0.4) || "intermediate";

  const audienceScores = {
    senior_management: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.audience.senior_management),
    mid_management: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.audience.mid_management),
    academic: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.audience.academic),
    practitioner: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.audience.practitioner),
    general_public: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.audience.general_public),
  };
  if (taskType === "troubleshoot") audienceScores.practitioner += 0.8;
  if (taskType === "explain" && sophistication === "intro") audienceScores.general_public += 0.55;
  const audience = liveIntentPick(audienceScores, 0.5);

  const stakeScores = {
    high: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.stake_level.high),
    medium: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.stake_level.medium),
    low: 0.3,
  };
  if (["persuade", "compare", "plan"].includes(taskType)) stakeScores.medium += 0.35;
  if (taskType === "troubleshoot") stakeScores.high += 0.4;
  const stakeLevel = liveIntentPick(stakeScores, 0.35) || "low";

  const horizonScores = {
    immediate: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.time_horizon.immediate),
    near_term: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.time_horizon.near_term),
    strategic: liveIntentPhraseScore(lowered, LIVE_INTENT_HINTS.time_horizon.strategic),
  };
  if (taskType === "plan") {
    horizonScores.near_term += 0.35;
    horizonScores.strategic += 0.25;
  }
  const timeHorizon = liveIntentPick(horizonScores, 0.35);

  const scores = [];
  [taskScores[taskType], sophisticationScores[sophistication], audience ? audienceScores[audience] : 0, stakeScores[stakeLevel], timeHorizon ? horizonScores[timeHorizon] : 0]
    .forEach((score) => {
      const n = Number(score || 0);
      if (Number.isFinite(n) && n > 0) scores.push(n);
    });
  const confidence = Math.min(
    0.88,
    0.38 + (scores.reduce((sum, value) => sum + value, 0) / Math.max(1, scores.length * 3.8))
  );

  const prediction = normalizeLiveIntentPrediction({
    task_type: (taskScores[taskType] >= 0.75 || cjkMode) ? taskType : "",
    sophistication,
    audience: audience && audienceScores[audience] >= 0.85 ? audience : "",
    stake_level: stakeLevel,
    time_horizon: timeHorizon && horizonScores[timeHorizon] >= 0.75 ? timeHorizon : "",
    confidence,
    source: "heuristic",
  }, "heuristic");
  if (!prediction) return null;
  if ((prediction.confidence || 0) < LIVE_INTENT_CONFIDENCE_THRESHOLD) {
    prediction.audience = "";
    prediction.time_horizon = "";
  }
  return normalizeLiveIntentPrediction(prediction, "heuristic");
}

function renderLiveIntent(prediction) {
  if (!liveIntentEl) return;
  const normalized = normalizeLiveIntentPrediction(prediction);
  if (!normalized) {
    liveIntentEl.innerHTML = "";
    liveIntentEl.classList.add("hidden");
    liveIntentDisplayedPrediction = null;
    return;
  }
  liveIntentDisplayedPrediction = normalized;
  const sourceClass = normalized.source === "model" ? "is-model" : "is-local";
  const parts = [];
  for (let i = 0; i < LIVE_INTENT_FIELD_ORDER.length; i += 1) {
    const key = LIVE_INTENT_FIELD_ORDER[i];
    const value = normalized[key];
    if (!value) continue;
    const label = LIVE_INTENT_LABELS[key] || key;
    const display = (LIVE_INTENT_VALUES[key] && LIVE_INTENT_VALUES[key][value]) || value;
    parts.push(
      `<span class="live-intent-chip ${sourceClass}"><span class="live-intent-chip-label">${esc(label)}</span><span class="live-intent-chip-value">${esc(display)}</span></span>`
    );
  }
  if (!parts.length) {
    liveIntentEl.innerHTML = "";
    liveIntentEl.classList.add("hidden");
    liveIntentDisplayedPrediction = null;
    return;
  }
  liveIntentEl.innerHTML = parts.join("");
  liveIntentEl.classList.remove("hidden");
}

function clearLiveIntent() {
  if (liveIntentDebounceTimer) {
    clearTimeout(liveIntentDebounceTimer);
    liveIntentDebounceTimer = null;
  }
  if (liveIntentFetchController) {
    try {
      liveIntentFetchController.abort();
    } catch (_err) {
      // no-op
    }
    liveIntentFetchController = null;
  }
  liveIntentRequestSeq = 0;
  liveIntentAppliedServerFingerprint = "";
  liveIntentPendingServerFingerprint = "";
  liveIntentPendingServerCount = 0;
  liveIntentAppliedText = "";
  renderLiveIntent(null);
}

function mergeLiveIntentPrediction(nextPrediction, source, requestText = "") {
  const next = normalizeLiveIntentPrediction(nextPrediction, source);
  if (!next) return;
  const current = normalizeLiveIntentPrediction(liveIntentDisplayedPrediction || null);
  const preserveExisting = current && !liveIntentIsMaterialShift(
    liveIntentAppliedText || requestText,
    requestText || liveIntentAppliedText || ""
  );
  const merged = {
    source: source === "model" ? "model" : next.source,
    confidence: next.confidence,
  };
  for (let i = 0; i < LIVE_INTENT_FIELD_ORDER.length; i += 1) {
    const key = LIVE_INTENT_FIELD_ORDER[i];
    merged[key] = next[key] || ((preserveExisting && current) ? current[key] : "");
  }
  const normalized = normalizeLiveIntentPrediction(merged, merged.source);
  if (!normalized) return;
  liveIntentAppliedText = requestText || liveIntentAppliedText;
  renderLiveIntent(normalized);
}

async function fetchLiveIntentPrediction(requestText) {
  if (!LIVE_INTENT_ENABLED) return;
  const normalizedText = liveIntentNormalizedText(requestText);
  if (liveIntentSignalChars(normalizedText) < liveIntentMinimumChars(normalizedText)) return;
  const requestSeq = liveIntentRequestSeq + 1;
  liveIntentRequestSeq = requestSeq;
  if (liveIntentFetchController) {
    try {
      liveIntentFetchController.abort();
    } catch (_err) {
      // no-op
    }
  }
  const controller = (typeof AbortController !== "undefined") ? new AbortController() : null;
  liveIntentFetchController = controller;
  try {
    const rsp = await fetch("/api/intent/live", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: normalizedText }),
      signal: controller ? controller.signal : undefined,
    });
    if (!rsp.ok) return;
    const payload = await rsp.json();
    if (requestSeq !== liveIntentRequestSeq) return;
    if (liveIntentNormalizedText(taskEl && taskEl.value) !== normalizedText) return;
    const normalized = normalizeLiveIntentPrediction(payload, "heuristic");
    if (!normalized) return;
    const fingerprint = liveIntentFingerprint(normalized);
    if (fingerprint && fingerprint === liveIntentPendingServerFingerprint) {
      liveIntentPendingServerCount += 1;
    } else {
      liveIntentPendingServerFingerprint = fingerprint;
      liveIntentPendingServerCount = 1;
    }
    const confidence = Number(normalized.confidence || 0);
    const shouldApply = (
      !liveIntentDisplayedPrediction
      || !liveIntentAppliedServerFingerprint
      || fingerprint === liveIntentAppliedServerFingerprint
      || liveIntentPendingServerCount >= 2
      || confidence >= Math.max(0.68, LIVE_INTENT_CONFIDENCE_THRESHOLD + 0.15)
      || liveIntentIsMaterialShift(liveIntentAppliedText, normalizedText)
    );
    if (!shouldApply) return;
    liveIntentAppliedServerFingerprint = fingerprint;
    mergeLiveIntentPrediction(normalized, normalized.source || "model", normalizedText);
  } catch (_err) {
    // fail-soft for live typing assistance
  } finally {
    if (liveIntentFetchController === controller) {
      liveIntentFetchController = null;
    }
  }
}

function scheduleLiveIntentServerPrediction(rawText) {
  if (liveIntentDebounceTimer) {
    clearTimeout(liveIntentDebounceTimer);
    liveIntentDebounceTimer = null;
  }
  const normalizedText = liveIntentNormalizedText(rawText);
  if (!LIVE_INTENT_ENABLED || liveIntentSignalChars(normalizedText) < liveIntentMinimumChars(normalizedText)) return;
  liveIntentDebounceTimer = setTimeout(() => {
    liveIntentDebounceTimer = null;
    void fetchLiveIntentPrediction(normalizedText);
  }, LIVE_INTENT_DEBOUNCE_MS);
}

function updateLiveIntentFromInput() {
  if (!LIVE_INTENT_ENABLED || !taskEl || taskEl.disabled) {
    clearLiveIntent();
    return;
  }
  const normalizedText = liveIntentNormalizedText(taskEl.value);
  if (liveIntentSignalChars(normalizedText) < liveIntentMinimumChars(normalizedText)) {
    clearLiveIntent();
    return;
  }
  const localPrediction = computeLocalLiveIntentPrediction(normalizedText);
  if (localPrediction) {
    mergeLiveIntentPrediction(localPrediction, "heuristic", normalizedText);
  }
  scheduleLiveIntentServerPrediction(normalizedText);
}

function contextStateClass(v) {
  const s = String(v || "").toLowerCase();
  if (s === "ready" || s === "completed" || s === "partial_ready" || s === "parsed") return "ready";
  if (s === "running" || s === "parsing" || s === "generating") return "parsing";
  if (s === "error" || s === "failed") return "error";
  if (s === "uploaded") return "stale";
  return "stale";
}

function contextStatusLabel(v) {
  const s = String(v || "").trim().toLowerCase();
  if (!s) return "uploaded";
  if (s === "stale") return "uploaded";
  if (s === "parsing") return "reading, please wait...";
  if (s === "ready" || s === "completed" || s === "partial_ready") return "parsed";
  return s.replaceAll("_", " ");
}

function fileSizeLabel(n) {
  const bytes = Number(n || 0);
  if (!Number.isFinite(bytes) || bytes < 0) return "-";
  if (bytes < 1024) return `${Math.floor(bytes)} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setContextEnabled(enabled) {
  // TODO: implement context pane enable/disable (graying out upload controls when no session is active)
  void enabled;
}

function clearContextPane(msg = "No context files uploaded.") {
  currentContextSet = null;
  currentContextSource = "session";
  selectedContextFileId = "";
  contextAggregateViewBase = null;
  contextAggregateViewMode = "";
  contextMetaEl.textContent = msg;
  contextDiffBannerEl.textContent = "";
  contextFilesEl.innerHTML = `<div class="muted">No files.</div>`;
  contextAggregateDigestEl.textContent = "";
  if (inlineContextFilesEl) inlineContextFilesEl.innerHTML = "";
  if (contextFileDigestEl) {
    contextFileDigestEl.textContent = "";
  }
}

function getStartupParseState(runId) {
  const rid = String(runId || "").trim();
  if (!rid) return null;
  const st = startupParseStateByRunId.get(rid);
  return st && typeof st === "object" ? st : null;
}

function hasStartupParseProgress(runId) {
  const st = getStartupParseState(runId);
  if (!st) return false;
  const perIndex = st.perIndex && typeof st.perIndex === "object" ? st.perIndex : {};
  return Object.keys(perIndex).length > 0;
}

function clearStartupParseState(runId) {
  const rid = String(runId || "").trim();
  if (!rid) return;
  startupParseStateByRunId.delete(rid);
}

function initStartupParseState(runId, filesTotal = 0) {
  const rid = String(runId || "").trim();
  if (!rid) return;
  const prev = getStartupParseState(rid);
  const next = {
    started: true,
    total: Math.max(0, Number(filesTotal || (prev && prev.total) || 0)),
    perIndex: (prev && prev.perIndex && typeof prev.perIndex === "object") ? { ...prev.perIndex } : {},
    updatedAt: Date.now(),
  };
  startupParseStateByRunId.set(rid, next);
}

function updateStartupParseState(runId, payload) {
  const rid = String(runId || "").trim();
  if (!rid) return;
  const prev = getStartupParseState(rid) || { started: true, total: 0, perIndex: {} };
  const perIndex = (prev.perIndex && typeof prev.perIndex === "object") ? { ...prev.perIndex } : {};
  const idx = Number(payload && payload.index);
  if (!Number.isFinite(idx) || idx < 1) return;
  const status = String((payload && payload.status) || "").trim().toLowerCase();
  const error = String((payload && payload.error) || "").trim();
  perIndex[Math.floor(idx)] = {
    status: (status === "ready" || status === "parsed")
      ? "ready"
      : (status === "error" || status === "failed" ? "error" : "parsing"),
    error,
  };
  startupParseStateByRunId.set(rid, {
    started: true,
    total: Math.max(0, Number((payload && payload.total) || prev.total || 0)),
    perIndex,
    updatedAt: Date.now(),
  });
}

function applyStartupParseOverlay(set, runId) {
  const rid = String(runId || "").trim();
  if (!rid || !set || typeof set !== "object") return set;
  const st = getStartupParseState(rid);
  if (!st || !st.started) return set;
  const baseFiles = Array.isArray(set.files) ? set.files : [];
  if (!baseFiles.length) return set;
  const out = cloneJson(set);
  const files = Array.isArray(out.files) ? out.files : [];
  const perIndex = st.perIndex && typeof st.perIndex === "object" ? st.perIndex : {};
  for (let i = 0; i < files.length; i += 1) {
    const f = files[i];
    if (!f || typeof f !== "object") continue;
    const ov = perIndex[i + 1];
    if (!ov || typeof ov !== "object") continue;
    const current = String(f.digest_status || "").trim().toLowerCase();
    const ovStatus = String(ov.status || "").trim().toLowerCase();
    if (ovStatus === "ready") {
      f.digest_status = "ready";
      f.error = "";
      continue;
    }
    if (ovStatus === "error") {
      f.digest_status = "error";
      if (String(ov.error || "").trim()) f.error = String(ov.error || "").trim();
      continue;
    }
    if (ovStatus === "parsing" && current !== "ready" && current !== "error") {
      f.digest_status = "parsing";
    }
  }
  const agg = String(out.aggregate_digest_status || "").trim().toLowerCase();
  // Keep backend terminal states visible during startup overlays.
  if (!agg || agg === "stale" || agg === "uploaded" || agg === "parsing" || agg === "generating") {
    out.aggregate_digest_status = "parsing";
  }
  return out;
}

function setContextBusy(busy) {
  const on = !!busy;
  contextMutationInFlight = on;
  refreshContextUploadButtonState();
}

function refreshContextUploadButtonState() {
  if (!contextUploadBtn) return;
  const disabled = !!contextInputLocked || !!contextMutationInFlight;
  contextUploadBtn.disabled = disabled;
  contextUploadBtn.textContent = contextMutationInFlight ? "Uploading..." : "Upload Files";
  if (contextUploadInput) {
    contextUploadInput.disabled = disabled;
  }
}

function renderContextPane() {
  const set = currentContextSet;
  if (!set || typeof set !== "object") {
    clearContextPane();
    return;
  }
  const rev = Number(set.revision || 1);
  const files = Array.isArray(set.files) ? set.files : [];
  const dedupFiles = [];
  const seenFileKeys = new Set();
  for (const f of files) {
    if (!f || typeof f !== "object") continue;
    const key = `${String(f.filename || "").trim().toLowerCase()}::${String(f.content_hash || "").trim()}`;
    if (seenFileKeys.has(key)) continue;
    seenFileKeys.add(key);
    dedupFiles.push(f);
  }
  const startupPhase = currentStartupPhase();
  const startupParsing = isStartupBindingPhase(startupPhase) && startupPhase === "parsing_context";
  const aggRaw = String(set.aggregate_digest_status || "stale");
  const aggRawEffective = (startupParsing && (aggRaw === "stale" || aggRaw === "uploaded" || !aggRaw))
    ? "parsing"
    : aggRaw;
  const aggStatus = contextStatusLabel(aggRawEffective || "stale");
  const sourceSuffix = currentContextSource === "workspace" ? " · Source staged workspace" : "";
  contextMetaEl.textContent = `Revision ${rev} · Aggregate ${aggStatus}${sourceSuffix}`;
  contextDiffBannerEl.textContent = "";

  const pendingRows = Array.isArray(pendingUploadFiles) ? pendingUploadFiles : [];
  if (!dedupFiles.length && !pendingRows.length) {
    contextFilesEl.innerHTML = `<div class="muted">No files uploaded.</div>`;
  } else {
    const existingHtml = dedupFiles.map((f) => {
      const fid = String(f.file_id || "");
      const rawStatus = String(f.digest_status || "stale").trim().toLowerCase();
      const effectiveRawStatus = (startupParsing && (rawStatus === "stale" || rawStatus === "uploaded" || !rawStatus))
        ? "parsing"
        : rawStatus;
      const status = contextStatusLabel(effectiveRawStatus);
      const chipClass = contextStateClass(effectiveRawStatus);
      return `<div class="context-file-row" data-file-id="${esc(fid)}">
        <div class="context-file-main">
          <div class="context-file-name">${esc(f.filename || fid)}</div>
          <span class="context-chip ${esc(chipClass)}">${esc(status)}</span>
        </div>
        <div class="context-file-meta">${esc(fileSizeLabel(f.size_bytes))} · ${esc(fmtTime(f.uploaded_at))}</div>
        ${String(f.error || "").trim() ? `<div class="context-file-meta" style="color:#b91c1c;">parse error: ${esc(String(f.error || ""))}</div>` : ""}
        <div class="context-file-actions">
          <button type="button" data-action="delete" data-file-id="${esc(fid)}">Delete</button>
        </div>
      </div>`;
    }).join("");
    const pendingHtml = pendingRows.map((name, i) => {
      const safe = esc(String(name || `upload_${i + 1}`));
      return `<div class="context-file-row">
        <div class="context-file-main">
          <div class="context-file-name">${safe}</div>
          <span class="context-chip parsing">uploading</span>
        </div>
        <div class="context-file-meta">pending parse...</div>
      </div>`;
    }).join("");
    contextFilesEl.innerHTML = existingHtml + pendingHtml;
  }
  renderInlineContextFiles();
}

function renderInlineContextFiles() {
  if (!inlineContextFilesEl) return;
  const set = currentContextSet;
  const files = (set && Array.isArray(set.files)) ? set.files : [];
  const pendingRows = Array.isArray(pendingUploadFiles) ? pendingUploadFiles : [];
  if (!files.length && !pendingRows.length) {
    inlineContextFilesEl.innerHTML = "";
    return;
  }
  const seenKeys = new Set();
  let html = "";
  for (const f of files) {
    if (!f || typeof f !== "object") continue;
    const key = `${String(f.filename || "").trim().toLowerCase()}::${String(f.content_hash || "").trim()}`;
    if (seenKeys.has(key)) continue;
    seenKeys.add(key);
    const fid = String(f.file_id || "");
    const name = String(f.filename || fid);
    html += `<span class="ctx-file-box" title="${esc(name)}" data-file-id="${esc(fid)}">`;
    html += `<span class="ctx-file-name">${esc(name)}</span>`;
    html += `<button class="ctx-file-del" type="button" data-delete-fid="${esc(fid)}" title="Remove">&times;</button>`;
    html += `</span>`;
  }
  for (const name of pendingRows) {
    const safe = esc(String(name || "file"));
    html += `<span class="ctx-file-box is-uploading" title="${safe}">${safe}</span>`;
  }
  inlineContextFilesEl.innerHTML = html;
}

function cloneJson(v) {
  try {
    return JSON.parse(JSON.stringify(v));
  } catch (_err) {
    return v;
  }
}

function buildContextAggregateView(basePayload, mode = "digest") {
  const payload = basePayload && typeof basePayload === "object" ? basePayload : {};
  const viewMode = String(mode || "digest");
  return viewMode === "context_slice"
    ? { context_slice: payload }
    : { digest: payload };
}

function renderContextAggregateView(basePayload, mode = "digest") {
  if (!contextAggregateDigestEl) return;
  contextAggregateViewBase = cloneJson(basePayload && typeof basePayload === "object" ? basePayload : {});
  contextAggregateViewMode = String(mode || "digest");
  contextAggregateDigestEl.textContent = JSON.stringify(
    buildContextAggregateView(contextAggregateViewBase, contextAggregateViewMode),
    null,
    2
  );
}

function refreshContextAggregateViewFromSnapshot() {
  if (!contextAggregateDigestEl) return;
  if (!contextAggregateViewBase || typeof contextAggregateViewBase !== "object") return;
  contextAggregateDigestEl.textContent = JSON.stringify(
    buildContextAggregateView(contextAggregateViewBase, contextAggregateViewMode || "digest"),
    null,
    2
  );
}

function renderPlanningPanel(snap) {
  const phase = planningPhase(snap);
  const pState = planningState(snap);
  const swapInProgress = isPlanningSwapInProgress(snap);
  const sid = String((snap && (snap.session_id || snap.run_id)) || "").trim();
  const executionState = String((snap && snap.execution_state) || "").toLowerCase();
  const researchState = String((snap && snap.research_state) || "").toLowerCase();
  const paused = researchState ? researchState === "paused" : executionState === "paused";
  const terminal = researchState
    ? researchState === "terminal"
    : ["completed", "aborted", "failed"].includes(executionState);
  const pausePending = pauseRequested && !paused && !terminal;
  const abortPending = abortRequested || isAbortPendingSnapshot(snap);
  const transitionPending = pausePending || abortPending;
  const reportGeneratingForThisRun = isReportGeneratingForRun(sid);
  const reportState = String((snap && snap.report_state) || "").toLowerCase();
  const reportPhase = String(
    (snap && (snap.report_status || (snap.tree && snap.tree.report_status))) || "pending"
  ).toLowerCase();
  const reportBusy = reportGeneratingForThisRun || reportPhase === "running" || reportState === "generating";
  const planningUi = derivePlanningUiState(phase, pState, {
    swapInProgress,
    reportBusy,
    transitionPending,
    startRunBusy: startRunInFlight,
  });
  if (swapBatchCanvasBtn) {
    swapBatchCanvasBtn.classList.toggle("hidden", !planningUi.inPlanning);
    swapBatchCanvasBtn.disabled = !planningUi.canEdit;
    swapBatchCanvasBtn.setAttribute("title", SWAP_BATCH_LABEL);
    swapBatchCanvasBtn.setAttribute("aria-label", SWAP_BATCH_LABEL);
  }
  if (planningCommitCanvasBtn) {
    planningCommitCanvasBtn.classList.toggle("hidden", !planningUi.inPlanning);
    planningCommitCanvasBtn.disabled = planningUi.planningActionDisabled;
    planningCommitCanvasBtn.setAttribute("title", "Go for a full run");
    planningCommitCanvasBtn.setAttribute("aria-label", "Go for a full run");
  }
}

function refreshPlanningUi(snap) {
  renderPlanningPanel(snap);
  if (!uiClearedByAbort && !abortRequested && snap && typeof snap === "object" && snap.tree && typeof snap.tree === "object") {
    renderCanvas(snap.tree);
  }
  refreshPrimaryAndPlanningActionButtonsFromSnapshot(snap);
}

async function planningSwapBatch() {
  if (!currentRunId || planningMutationInFlight) return;
  planningMutationInFlight = true;
  refreshPlanningUi(currentSnapshot);
  try {
    const ev = currentExpectedVersion();
    const qs = ev ? `?expected_version=${encodeURIComponent(String(ev))}` : "";
    const rsp = await fetch(`/api/runs/${encodeURIComponent(String(currentRunId))}/planning/swap_batch${qs}`, {
      method: "POST",
      headers: { "Idempotency-Key": newIdempotencyKey("planning_swap") },
    });
    if (!rsp.ok) {
      throw new Error(await responseDetail(rsp, `Swap batch failed: ${rsp.status}`));
    }
    await fetchSnapshot(currentRunId);
  } finally {
    planningMutationInFlight = false;
    refreshPlanningUi(currentSnapshot);
  }
}

async function planningCommit() {
  if (!currentRunId || planningMutationInFlight) return;
  planningMutationInFlight = true;
  refreshPlanningUi(currentSnapshot);
  try {
    const ev = currentExpectedVersion();
    const qs = ev ? `?expected_version=${encodeURIComponent(String(ev))}` : "";
    const rsp = await fetch(`/api/runs/${encodeURIComponent(String(currentRunId))}/planning/commit${qs}`, {
      method: "POST",
      headers: { "Idempotency-Key": newIdempotencyKey("planning_commit") },
    });
    if (!rsp.ok) {
      throw new Error(await responseDetail(rsp, `Start research failed: ${rsp.status}`));
    }
    await fetchSnapshot(currentRunId);
  } finally {
    planningMutationInFlight = false;
    refreshPlanningUi(currentSnapshot);
  }
}

async function planningSetPin(nodeId, pin) {
  if (!currentRunId || planningMutationInFlight) return;
  const nid = String(nodeId || "").trim();
  if (!nid) return;
  planningMutationInFlight = true;
  refreshPlanningUi(currentSnapshot);
  try {
    const ev = currentExpectedVersion();
    const qs = ev ? `?expected_version=${encodeURIComponent(String(ev))}` : "";
    const method = pin ? "POST" : "DELETE";
    const rsp = await fetch(`/api/runs/${encodeURIComponent(String(currentRunId))}/nodes/${encodeURIComponent(nid)}/pin${qs}`, {
      method,
      headers: { "Idempotency-Key": newIdempotencyKey(pin ? "planning_pin" : "planning_unpin") },
    });
    if (!rsp.ok) {
      throw new Error(await responseDetail(rsp, `${pin ? "Pin" : "Unpin"} failed: ${rsp.status}`));
    }
    await fetchSnapshot(currentRunId);
  } finally {
    planningMutationInFlight = false;
    refreshPlanningUi(currentSnapshot);
  }
}

async function planningDepthPlus(nodeId) {
  if (!currentRunId || planningMutationInFlight) return;
  const nid = String(nodeId || "").trim();
  if (!nid) return;
  planningMutationInFlight = true;
  refreshPlanningUi(currentSnapshot);
  try {
    const ev = currentExpectedVersion();
    const qs = ev ? `?expected_version=${encodeURIComponent(String(ev))}` : "";
    const rsp = await fetch(`/api/runs/${encodeURIComponent(String(currentRunId))}/nodes/${encodeURIComponent(nid)}/depth_bonus${qs}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": newIdempotencyKey("planning_depth"),
      },
      body: JSON.stringify({ increment: 1 }),
    });
    if (!rsp.ok) {
      throw new Error(await responseDetail(rsp, `Depth update failed: ${rsp.status}`));
    }
    await fetchSnapshot(currentRunId);
  } finally {
    planningMutationInFlight = false;
    refreshPlanningUi(currentSnapshot);
  }
}

function currentStartupPhase() {
  const tree = currentSnapshot && typeof currentSnapshot.tree === "object" ? currentSnapshot.tree : {};
  return String(tree.startup_phase || "").trim().toLowerCase();
}

function currentPendingWorkspaceId() {
  const tree = currentSnapshot && typeof currentSnapshot.tree === "object" ? currentSnapshot.tree : {};
  return String(tree.pending_workspace_id || "").trim();
}

function isStartupBindingPhase(phase) {
  const p = String(phase || "").trim().toLowerCase();
  return p === "queued_for_binding" || p === "binding_context" || p === "parsing_context";
}

function hasUsableContextData(set) {
  if (!set || typeof set !== "object") return false;
  const files = Array.isArray(set.files) ? set.files : [];
  if (files.length > 0) return true;
  const agg = String(set.aggregate_digest_status || "").trim().toLowerCase();
  return agg === "ready" || agg === "error" || agg === "parsing";
}

async function fetchContextSetRawByPath(path) {
  const rsp = await fetch(path);
  if (!rsp.ok) return null;
  const data = await rsp.json();
  const set = data && data.context_set ? data.context_set : null;
  return set && typeof set === "object" ? set : null;
}

function markContextFilesParsingStarted(runId = "", filesTotal = 0) {
  initStartupParseState(runId || currentRunId, filesTotal);
  if (!currentContextSet || typeof currentContextSet !== "object") return;
  const files = Array.isArray(currentContextSet.files) ? currentContextSet.files : [];
  if (!files.length) return;
  for (const f of files) {
    if (!f || typeof f !== "object") continue;
    const st = String(f.digest_status || "").trim().toLowerCase();
    if (!st || st === "stale" || st === "uploaded" || st === "error") {
      f.digest_status = "parsing";
      if (st !== "error") {
        f.error = "";
      }
    }
  }
  currentContextSet.aggregate_digest_status = "parsing";
  renderContextPane();
}

function markContextFileParsingProgress(payload, runId = "") {
  updateStartupParseState(runId || currentRunId, payload || {});
  if (!currentContextSet || typeof currentContextSet !== "object") return;
  const files = Array.isArray(currentContextSet.files) ? currentContextSet.files : [];
  if (!files.length) return;
  const idx = Number(payload && payload.index);
  const status = String((payload && payload.status) || "").trim().toLowerCase();
  const err = String((payload && payload.error) || "").trim();
  if (!Number.isFinite(idx) || idx < 1 || idx > files.length) return;
  const f = files[Math.floor(idx) - 1];
  if (!f || typeof f !== "object") return;
  if (status === "ready" || status === "parsed") {
    f.digest_status = "ready";
    f.error = "";
  } else if (status === "error" || status === "failed") {
    f.digest_status = "error";
    f.error = err || f.error || "Parse failed.";
  } else {
    f.digest_status = "parsing";
  }
  renderContextPane();
}

async function fetchContextAggregateDigest(runIdOverride = "", workspaceIdOverride = "") {
  const set = currentContextSet && typeof currentContextSet === "object" ? currentContextSet : {};
  const files = Array.isArray(set.files) ? set.files.filter((f) => f && typeof f === "object") : [];
  if (files.length === 0) {
    contextAggregateViewBase = null;
    contextAggregateViewMode = "";
    contextAggregateDigestEl.textContent = "";
    return;
  }
  const base = contextBasePath(runIdOverride, workspaceIdOverride);
  const singleFileId = files.length === 1 ? String(files[0].file_id || "").trim() : "";

  const renderError = (statusCode, label) => {
    const status = String(set.aggregate_digest_status || "").trim();
    const err = String(set.aggregate_error || "").trim();
    const statusLower = status.toLowerCase();
    // During upload/parsing, digest may legitimately be unavailable; keep panel blank.
    if (
      Number(statusCode) === 404 &&
      statusLower !== "error" &&
      statusLower !== "failed" &&
      !err
    ) {
      contextAggregateViewBase = null;
      contextAggregateViewMode = "";
      contextAggregateDigestEl.textContent = "";
      return;
    }
    renderContextAggregateView(
      {
        status: status || "missing",
        error: err || `${label} not ready (${statusCode}).`,
      },
      "digest"
    );
  };

  // Single-file UX: show the per-file digest directly.
  if (singleFileId) {
    let rsp = await fetch(`${base}/files/${encodeURIComponent(singleFileId)}/digest`);
    // Fallback once to aggregate digest on transient or availability mismatch.
    if (!rsp.ok) {
      rsp = await fetch(`${base}/digest`);
    }
    if (!rsp.ok) {
      renderError(rsp.status, "Digest");
      return;
    }
    const data = await rsp.json();
    const digest = data && typeof data === "object" ? data.digest : null;
    renderContextAggregateView(digest || {}, "digest");
    return;
  }

  const rsp = await fetch(`${base}/digest`);
  if (!rsp.ok) {
    renderError(rsp.status, "Aggregate digest");
    return;
  }
  const data = await rsp.json();
  const digest = data && typeof data === "object" ? data.digest : null;
  renderContextAggregateView(digest || {}, "digest");
}

async function fetchContextFileDigest(fileId, runIdOverride = "", workspaceIdOverride = "") {
  const fid = String(fileId || "").trim();
  if (!fid) return;
  const rsp = await fetch(`${contextBasePath(runIdOverride, workspaceIdOverride)}/files/${encodeURIComponent(fid)}/digest`);
  if (!contextFileDigestEl) {
    return;
  }
  if (!rsp.ok) {
    contextFileDigestEl.textContent = "";
    return;
  }
  const data = await rsp.json();
  const digest = data && typeof data === "object" ? data.digest : null;
  contextFileDigestEl.textContent = JSON.stringify(digest || {}, null, 2);
}

function resolveContextTarget(runIdOverride = "", workspaceIdOverride = "") {
  const rid = String(runIdOverride || currentRunId || "").trim();
  const wid = rid ? "" : String(workspaceIdOverride || ensureWorkspaceId()).trim();
  const key = rid ? `s:${rid}` : `w:${wid}`;
  return { rid, wid, key };
}

async function fetchContextSetOnce(runIdOverride = "", workspaceIdOverride = "") {
  const target = resolveContextTarget(runIdOverride, workspaceIdOverride);
  const reqSeq = Number(contextFetchSeqByKey.get(target.key) || 0) + 1;
  contextFetchSeqByKey.set(target.key, reqSeq);
  try {
    const primarySet = await fetchContextSetRawByPath(contextBasePath(target.rid, target.wid));
    if (!primarySet) {
      throw new Error("Context load failed.");
    }
    if (target.rid) {
      if (String(currentRunId || "") !== target.rid) return;
    } else if (String(currentRunId || "").trim()) {
      return;
    }
    if (Number(contextFetchSeqByKey.get(target.key) || 0) !== reqSeq) return;
    let chosenSet = primarySet;
    let chosenSource = target.rid ? "session" : "workspace";
    if (target.rid) {
      const phase = currentStartupPhase();
      const pendingWid = currentPendingWorkspaceId();
      const preferWorkspace =
        isStartupBindingPhase(phase) &&
        !!pendingWid &&
        !hasUsableContextData(primarySet);
      if (preferWorkspace) {
        const cached = stagedContextBySessionId.get(target.rid);
        if (cached && cached.contextSet && typeof cached.contextSet === "object") {
          chosenSet = cloneJson(cached.contextSet);
          chosenSource = "workspace";
        } else if (pendingWid) {
          const wsSet = await fetchContextSetRawByPath(
            `/api/workspaces/${encodeURIComponent(pendingWid)}/context`
          );
          if (wsSet && typeof wsSet === "object") {
            stagedContextBySessionId.set(target.rid, {
              workspaceId: pendingWid,
              contextSet: cloneJson(wsSet),
            });
            chosenSet = wsSet;
            chosenSource = "workspace";
          }
        }
      } else if (hasUsableContextData(primarySet)) {
        stagedContextBySessionId.delete(target.rid);
      }
    }
    if (target.rid) {
      chosenSet = applyStartupParseOverlay(chosenSet, target.rid);
    }
    currentContextSet = chosenSet;
    currentContextSource = chosenSource;
    renderContextPane();
    if (chosenSource === "workspace" && target.rid) {
      const wsid = currentPendingWorkspaceId();
      await fetchContextAggregateDigest("", wsid);
    } else {
      await fetchContextAggregateDigest(target.rid, target.wid);
    }
    if (selectedContextFileId && contextFileDigestEl) {
      if (chosenSource === "workspace" && target.rid) {
        const wsid = currentPendingWorkspaceId();
        await fetchContextFileDigest(selectedContextFileId, "", wsid);
      } else {
        await fetchContextFileDigest(selectedContextFileId, target.rid, target.wid);
      }
    } else if (contextFileDigestEl) {
      contextFileDigestEl.textContent = "";
    }
  } catch (_err) {
    const currentRid = String(currentRunId || "").trim();
    const activeMatches = target.rid ? currentRid === target.rid : !currentRid;
    if (activeMatches && Number(contextFetchSeqByKey.get(target.key) || 0) === reqSeq) {
      clearContextPane("Context unavailable.");
    }
  }
}

async function scheduleContextRefresh(runIdOverride = "", workspaceIdOverride = "") {
  const target = resolveContextTarget(runIdOverride, workspaceIdOverride);
  const st = contextFetchStateByKey.get(target.key) || { inFlight: false, queued: false };
  st.queued = true;
  if (st.inFlight) {
    contextFetchStateByKey.set(target.key, st);
    return;
  }
  st.inFlight = true;
  contextFetchStateByKey.set(target.key, st);
  try {
    do {
      st.queued = false;
      await fetchContextSetOnce(target.rid, target.wid);
    } while (st.queued);
  } finally {
    contextFetchStateByKey.delete(target.key);
  }
}

function scheduleContextRefreshDebounced(runIdOverride = "", workspaceIdOverride = "", delayMs = 180) {
  const target = resolveContextTarget(runIdOverride, workspaceIdOverride);
  if (contextRefreshDebounceTimerByKey.has(target.key)) {
    clearTimeout(contextRefreshDebounceTimerByKey.get(target.key));
  }
  const t = setTimeout(() => {
    contextRefreshDebounceTimerByKey.delete(target.key);
    void scheduleContextRefresh(target.rid, target.wid);
  }, delayMs);
  contextRefreshDebounceTimerByKey.set(target.key, t);
}

function connectContextEvents(runId) {
  if (contextEs) {
    contextEs.close();
    contextEs = null;
  }
  const rid = String(runId || "").trim();
  if (!rid) return;
  contextEs = new EventSource(`/api/sessions/${encodeURIComponent(rid)}/context/stream`);
  const onContextEvt = async () => {
    if (String(currentRunId || "") !== rid) return;
    await scheduleContextRefresh(rid);
  };
  contextEs.onmessage = onContextEvt;
  contextEs.addEventListener("context_updated", onContextEvt);
}

function scheduleSessionsRefresh(_reason) {
  if (sessionsRefreshTimer) {
    clearTimeout(sessionsRefreshTimer);
  }
  sessionsRefreshTimer = setTimeout(async () => {
    sessionsRefreshTimer = null;
    if (sessionsRefreshInFlight) return;
    sessionsRefreshInFlight = true;
    try {
      await fetchSessions();
    } catch (_err) {
      // no-op
    } finally {
      sessionsRefreshInFlight = false;
    }
  }, 250);
}

function emitSessionMutation(reason, sessionId = "") {
  const payload = {
    kind: "sessions_mutated",
    reason: String(reason || "unknown"),
    session_id: String(sessionId || ""),
    ts: Date.now(),
    origin_tab_id: SESSION_SYNC_TAB_ID,
  };
  try {
    if (sessionSyncChannel) {
      sessionSyncChannel.postMessage(payload);
    }
  } catch (_err) {
    // no-op
  }
  try {
    localStorage.setItem(SESSION_SYNC_STORAGE_KEY, JSON.stringify(payload));
  } catch (_err) {
    // no-op
  }
}

function initSessionSyncListeners() {
  try {
    if ("BroadcastChannel" in window) {
      sessionSyncChannel = new BroadcastChannel(SESSION_SYNC_CHANNEL);
      sessionSyncChannel.onmessage = (evt) => {
        const msg = evt && evt.data && typeof evt.data === "object" ? evt.data : {};
        if (String(msg.origin_tab_id || "") === SESSION_SYNC_TAB_ID) return;
        scheduleSessionsRefresh("broadcast");
      };
    }
  } catch (_err) {
    sessionSyncChannel = null;
  }
  window.addEventListener("storage", (evt) => {
    if (!evt || evt.key !== SESSION_SYNC_STORAGE_KEY || !evt.newValue) return;
    try {
      const msg = JSON.parse(String(evt.newValue || "{}"));
      if (String(msg.origin_tab_id || "") === SESSION_SYNC_TAB_ID) return;
    } catch (_err) {
      // ignore malformed values
    }
    scheduleSessionsRefresh("storage");
  });
  window.addEventListener("focus", () => {
    scheduleSessionsRefresh("focus");
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      scheduleSessionsRefresh("visible");
    }
  });
}

function renderSessions() {
  if (!sessionListEl) return;
  const lockLegend = UI_DEBUG
    ? `<div class="session-meta muted">Debug lock legend: mgr=global RunManager mutex (shows owner section + held ms) · session=per-session mutex · index=sessions_index.json write mutex</div>`
    : "";
  if (debugSessionLoadError) {
    if (UI_DEBUG) {
      sessionListEl.innerHTML = `${lockLegend}<div class="error">DEBUG: ${esc(debugSessionLoadError)}</div>`;
    } else {
      sessionListEl.innerHTML = `<div class="error">Session list unavailable. Please refresh.</div>`;
    }
    return;
  }
  if (!Array.isArray(sessions) || sessions.length === 0) {
    sessionListEl.innerHTML = `${lockLegend}<div class="muted">No sessions yet.</div>`;
    return;
  }
  sessionListEl.innerHTML = lockLegend + sessions
    .map((s) => {
      const sid = String(s.session_id || "");
      const active = sid && sid === currentRunId ? " active" : "";
      const title = String(s.title || sid || "Untitled");
      const research = sessionResearchState(s);
      const updated = fmtTime(s.updated_at);
      const statusCls = research === "completed" ? "success"
        : research === "running" ? "running"
        : research === "failed" ? "failed"
        : "queued";
      const lockDbg = (UI_DEBUG && s && s.lock_debug && typeof s.lock_debug === "object")
        ? `Locks: mgr=${esc(String(s.lock_debug.manager_lock || "-"))}${s.lock_debug.manager_lock_owner_section ? `(${esc(String(s.lock_debug.manager_lock_owner_section || ""))})` : ""}${Number(s.lock_debug.manager_lock_held_ms || 0) > 0 ? ` ${Math.round(Number(s.lock_debug.manager_lock_held_ms || 0))}ms` : ""} · session=${esc(String(s.lock_debug.session_lock || "-"))} · index=${esc(String(s.lock_debug.index_write_lock || "-"))}`
        : "";
      return `<div class="session-row${active}" data-session-id="${esc(sid)}">
        <div class="session-main">
          <div class="session-title" title="${esc(title)}"><span class="session-status-dot ${statusCls}"></span>${esc(title)}</div>
          <div class="session-meta">${esc(updated)}</div>
          ${lockDbg ? `<div class="session-meta muted">${lockDbg}</div>` : ""}
        </div>
        <div class="session-actions">
          <button
            type="button"
            class="session-delete-btn"
            data-action="delete"
            data-session-id="${esc(sid)}"
            aria-label="Delete session"
            title="Delete session"
          >
            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M4 7h16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
              <path d="M10 11v6M14 11v6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
              <path d="M6 7l1 12h10l1-12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M9 7V5h6v2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
        </div>
      </div>`;
    })
    .join("");
}

async function fetchSessions() {
  try {
    const rsp = await fetch("/api/sessions");
    if (!rsp.ok) throw new Error(`Failed to list sessions: ${rsp.status}`);
    const data = await rsp.json();
    sessions = Array.isArray(data.sessions) ? data.sessions : [];
    if (currentRunId) {
      const activeExists = sessions.some((s) => String((s && s.session_id) || "") === String(currentRunId || ""));
      if (!activeExists) {
        resetWorkspaceForNewSession();
      }
    }
    debugSessionLoadError = "";
    renderSessions();
  } catch (err) {
    const detail = err && err.message ? err.message : String(err);
    debugSessionLoadError = `Session list load failed. ${detail}`;
    if (UI_DEBUG) {
      // DEBUG ONLY: explicit banner to distinguish backend/API load issues from
      // empty data situations.
      showError(`DEBUG: ${debugSessionLoadError}`);
    } else {
      showError("Session list load failed.");
    }
    renderSessions();
    throw err;
  }
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
  const canvasRect = canvasEl.getBoundingClientRect();
  const viewportLeft = Math.floor(canvasRect.left);
  const viewportRight = Math.ceil(canvasRect.right);
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
  const canvasRect = canvasEl.getBoundingClientRect();
  const viewportLeft = Math.floor(canvasRect.left);
  const viewportRight = Math.ceil(canvasRect.right);
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
  // Split-pane layout handles scrolling per-panel; zoom not needed.
  return 1;
}

function applyCanvasZoom() {
  canvasZoom = clamp(canvasZoom, MIN_CANVAS_ZOOM, MAX_CANVAS_ZOOM);
  setCanvasScaleVariables(canvasZoom);
  // Keep a stable viewport; content grows via scroll instead of inflating canvas height.
  canvasEl.style.minHeight = "320px";
  zoomPctEl.textContent = `${Math.round(canvasZoom * 100)}%`;
  canvasHintEl.textContent = autoFitCanvas ? "Auto-fit" : "Manual zoom";
  if (fitDebugEl && UI_DEBUG) {
    const d = getFitDiagnostics();
    const canvasCw = Math.floor(canvasEl.clientWidth || 0);
    const canvasSw = Math.ceil(canvasEl.scrollWidth || 0);
    const winW = Math.floor(window.innerWidth || 0);
    fitDebugEl.textContent = `dbg rows=${d.rowCount} worst=r${d.worstRow} ov=${d.worstOverflowPx}px vw=${d.worstVisibleWidth} cw=${d.worstClientWidth} sw=${d.worstScrollWidth} safety=${AUTO_FIT_SAFETY_PX}px canvas=${canvasCw}/${canvasSw} win=${winW} ${d.hasOverflow ? "OVERFLOW" : "ok"}`;
  } else if (fitDebugEl) {
    fitDebugEl.textContent = "";
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

function sessionResearchState(s) {
  const explicit = String((s && s.research_state) || "").trim();
  if (explicit) return explicit;
  return String((s && (s.execution_state || s.status)) || "unknown");
}

function sessionReportState(s) {
  const explicit = String((s && s.report_state) || "").trim();
  if (explicit) return explicit;
  return "idle";
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

function normalizeReportVersions(snap) {
  const versions = Array.isArray(snap && snap.report_versions) ? snap.report_versions : [];
  return versions.filter((v) => v && typeof v === "object");
}

function activeReportVersionIndex(snap) {
  const versions = normalizeReportVersions(snap);
  if (!versions.length) return null;
  const idx = Number(snap && snap.current_report_version_index);
  if (Number.isFinite(idx) && idx >= 1 && idx <= versions.length) {
    return Math.floor(idx);
  }
  return versions.length;
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
  const totalCalls = Number(total.calls || 0);
  const totalTokens = Number(total.total_tokens || (inTok + outTok));
  const parts = [
    `Calls: ${totalCalls}`,
    `In: ${inTok.toLocaleString()}`,
    `Out: ${outTok.toLocaleString()}`,
    `Total: ${totalTokens.toLocaleString()}`,
    `Cost: $${cost.toFixed(4)}`,
  ];
  usageEl.textContent = parts.join("  ·  ");
}

function renderInlineMarkdown(text) {
  const raw = String(text || "");
  const linkRe = /\[([^\]\r\n]+)\]\(([^)\r\n]+)\)/g;
  let html = "";
  let lastIdx = 0;
  let m = null;
  while ((m = linkRe.exec(raw)) !== null) {
    const full = String(m[0] || "");
    const label = String(m[1] || "").trim();
    const url = String(m[2] || "").trim();
    html += esc(raw.slice(lastIdx, m.index));
    const safeUrl = toSafeHttpUrl(url);
    if (safeUrl) {
      const safeLabel = esc(label || shortSourceLabel(safeUrl));
      html += `<a href="${esc(safeUrl)}" target="_blank" rel="noopener noreferrer">${safeLabel}</a>`;
    } else {
      html += esc(label || full);
    }
    lastIdx = m.index + full.length;
  }
  html += esc(raw.slice(lastIdx));
  return html;
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
      const item = renderInlineMarkdown(line.replace(/^\s*[-*]\s+/, ""));
      out.push(`<li>${item}</li>`);
      continue;
    }
    if (inList) {
      out.push("</ul>");
      inList = false;
    }
    if (/^###\s+/.test(line)) {
      out.push(`<h3>${renderInlineMarkdown(line.replace(/^###\s+/, ""))}</h3>`);
    } else if (/^##\s+/.test(line)) {
      out.push(`<h2>${renderInlineMarkdown(line.replace(/^##\s+/, ""))}</h2>`);
    } else if (/^#\s+/.test(line)) {
      out.push(`<h1>${renderInlineMarkdown(line.replace(/^#\s+/, ""))}</h1>`);
    } else if (!line.trim()) {
      out.push("<p></p>");
    } else {
      out.push(`<p>${renderInlineMarkdown(line)}</p>`);
    }
  }
  if (inList) {
    out.push("</ul>");
  }
  return out.join("\n");
}

function upsertThought(text, meta) {
  if (abortRequested || uiClearedByAbort) return;
  const msg = String(text || "").trim();
  if (!msg) return;
  thoughts.push({ text: msg, meta: meta || "" });
  if (thoughts.length > 120) {
    thoughts.shift();
  }

  /* Append to scrolling log */
  if (progressLog) {
    const now = new Date();
    const ts = [now.getHours(), now.getMinutes(), now.getSeconds()]
      .map((n) => String(n).padStart(2, "0"))
      .join(":");
    const entry = document.createElement("div");
    entry.className = "progress-log-entry";
    entry.innerHTML =
      '<span class="log-time">' + esc(ts) + "</span><span>" + esc(msg) + "</span>";
    progressLog.appendChild(entry);
    while (progressLog.children.length > 50) {
      progressLog.removeChild(progressLog.firstChild);
    }
    progressLog.scrollTop = progressLog.scrollHeight;
  }
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
    const parts = parseDetailKey(key);
    const isDefaultOpenSection =
      parts.length >= 3 &&
      parts[0] === "node_section" &&
      (parts[2] === "query_work" || parts[2] === "node_notes");

    if (openDetailKeys.has(key)) {
      el.open = true;
    } else if (isDefaultOpenSection && !forceClosedDetailKeys.has(key)) {
      el.open = true;
    }
    el.addEventListener("toggle", () => {
      const parts = parseDetailKey(key);
      const isQueryWorkSection =
        parts.length >= 3 &&
        parts[0] === "node_section" &&
        parts[2] === "query_work";
      const isNodeNotesSection =
        parts.length >= 3 &&
        parts[0] === "node_section" &&
        parts[2] === "node_notes";
      if (el.open) {
        openDetailKeys.add(key);
        if (isQueryWorkSection || isNodeNotesSection) {
          forceClosedDetailKeys.delete(key);
        }
      } else {
        openDetailKeys.delete(key);
        if (isNodeNotesSection) {
          forceClosedDetailKeys.add(key);
        }
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
  if (et === "planning_started") {
    const action = String(p.action || "").trim().toLowerCase();
    if (action === "swap_batch" || action === "swap_batch_retry") {
      return "Swapping planning batch. Current candidates remain visible until replacement is ready.";
    }
    return "Planning started for root decomposition.";
  }
  if (et === "planning_review_ready") return `Planning ready: ${Array.isArray(p.root_children_candidates) ? p.root_children_candidates.length : 0} root child candidates.`;
  if (et === "planning_pin_updated") return "Updated pinned planning branches.";
  if (et === "planning_depth_bonus_updated") return "Updated planning branch depth bonus.";
  if (et === "planning_committed") return "Planning committed. Starting formal research.";
  if (et === "planning_failed") return `Planning failed: ${p.error || "unknown error"}`;
  if (et === "planning_aborted") return "Planning stopped by user request.";
  if (et === "run_started") return "Research started.";
  if (et === "context_binding_parsing_started") return `Parsing uploaded context files (${Number(p.files_total || 0)} total)...`;
  if (et === "context_binding_parsing_file_completed") return `Parsed context file ${Number(p.index || 0)}/${Number(p.total || 0)}: ${p.filename || ""}`;
  if (et === "context_binding_parsing_completed") return "Context parsing completed.";
  if (et === "context_for_node_ready") return `Prepared context for node ${p.node_id || ""} (${Number(p.selected_items_count || 0)} items).`;
  if (et === "context_binding_started") return "Binding uploaded context files...";
  if (et === "context_binding_completed") return "Context binding completed. Starting research...";
  if (et === "context_binding_failed") return `Context binding failed: ${p.error || "unknown error"}`;
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
  if (et === "run_paused") return "Research paused.";
  if (et === "run_resumed") return "Research resumed.";
  if (et === "report_generation_started") return "Agent is writing the report, please wait a few moments...";
  if (et === "report_context_attached") return "Attached full user context for report writing.";
  if (et === "report_heartbeat") return "Still writing report...";
  if (et === "report_generation_completed") return "Report generation completed.";
  if (et === "report_generation_failed") return "Report generation failed.";
  if (et === "partial_report_generated") return "Generated report from current findings.";
  if (et === "report_version_created") return `Saved report version #${Number(p.version_index || 0)}.`;
  if (et === "report_version_selected") return `Switched to report version #${Number(p.version_index || 0)}.`;
  if (et === "run_heartbeat") {
    const phaseLabels = {
      working: "Processing",
      initializing: "Initializing research",
      planning_node: "Planning sub-questions",
      idle: "Waiting for review",
      searching: "Searching the web",
      query_decision: "Evaluating queries",
      synthesizing: "Synthesizing results",
      node_sufficiency: "Checking evidence sufficiency",
      decomposing: "Decomposing into sub-tasks",
      run_sufficiency: "Checking overall sufficiency",
      writing_report: "Writing report",
      binding_context: "Loading context files",
      paused: "Paused",
    };
    const rawPhase = p.phase || "working";
    const label = phaseLabels[rawPhase] || rawPhase;
    if (p.query) return `${label}: ${p.query}`;
    if (p.sub_question) return `${label}: ${p.sub_question}`;
    return `${label}...`;
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
          display_title: String(q.display_title || prev.display_title || label || "").trim(),
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
          display_title: String(q.display_title || label || "").trim(),
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
        const childDisplayTitleRaw = (
          node && node.children_display_titles && typeof node.children_display_titles === "object"
            ? node.children_display_titles[childLabel]
            : ""
        );
        const childDisplayTitle = String(childDisplayTitleRaw || childLabel).trim();
        byKey.set(childKey, {
          sub_question: childLabel,
          display_title: childDisplayTitle,
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
        if (existing && !String(existing.display_title || "").trim()) {
          const fromParent = (
            node && node.children_display_titles && typeof node.children_display_titles === "object"
              ? node.children_display_titles[childLabel]
              : ""
          );
          existing.display_title = String(fromParent || childLabel).trim();
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
  const _renderTracker = (typeof SBTelemetry !== "undefined") ? SBTelemetry.trackRender("canvas_tree") : null;
  captureOpenDetailState();
  if (!tree || typeof tree !== "object") {
    canvasTreePanel.innerHTML = "";
    canvasDetailPanel.innerHTML = "";
    canvasNodeDataMap = new Map();
    return;
  }
  const phase = planningPhase(currentSnapshot || { tree });
  const pState = planningState(currentSnapshot || { tree });
  const isPlanningView = phase === "planning";
  const swapInProgress = isPlanningSwapInProgress(currentSnapshot || { tree });
  const planningSwapDimAll = isPlanningView && swapInProgress;
  const bonusCap = planningDepthBonusLimit(currentSnapshot || { tree });
  const researchDepthBonusByQuestion = planningResearchDepthBonusMap(currentSnapshot || { tree });
  const planningUi = derivePlanningUiState(phase, pState);
  const planningRows = planningCandidates(currentSnapshot || { tree });
  const planningRowByNodeId = new Map();
  for (const row of planningRows) {
    const nid = String((row && row.node_id) || "").trim();
    if (nid) planningRowByNodeId.set(nid, row);
  }

  const rounds = (Array.isArray(tree.rounds) ? tree.rounds : [])
    .filter((r) => r && typeof r === "object")
    .sort((a, b) => Number(a.round || 0) - Number(b.round || 0));
  const mergedQuestions = collectQuestionsFromRounds(rounds);

  if (mergedQuestions.length === 0) {
    canvasTreePanel.innerHTML = "";
    canvasDetailPanel.innerHTML = `<div class="canvas-detail-empty">${esc(tree.task || "")}</div>`;
    canvasNodeDataMap = new Map();
    return;
  }

  const byDepth = buildQuestionsByDepth(mergedQuestions);
  const nodeIdMap = buildNodeIdMap(byDepth);
  const visualStateMap = buildNodeVisualStateMap(byDepth);
  const depths = Array.from(byDepth.keys()).sort((a, b) => a - b);
  const scopeKey = "merged";

  // Build flat ordered node list and store data for detail rendering
  // We need DFS tree order: 1 → 1.1 → 1.1.1 → 1.1.2 → 1.2 → 2 → ...
  canvasNodeDataMap = new Map();
  let autoSelectKey = "";

  // Collect all nodes with their keys into a flat map first
  const nodeByKey = new Map();
  for (const d of depths) {
    for (const q of byDepth.get(d).values()) {
      const key = normalizeQuestionKey(q.sub_question || "");
      const pathId = String(q.node_id || nodeIdMap.get(key) || "");
      const st = String(q.status || "queued").toLowerCase();
      const cls = statusClass(st);
      let visualCls = "state-planned";
      if (visualStateMap.activeKeys.has(key)) visualCls = "state-active";
      else if (visualStateMap.visitedKeys.has(key)) visualCls = "state-visited";
      nodeByKey.set(key, { key, depth: Number(d), nodeId: pathId, status: st, cls, visualCls, q });
    }
  }

  // Build parent→children map using the parent field
  const childrenOf = new Map(); // parentKey → [childEntry, ...]
  const roots = [];
  for (const entry of nodeByKey.values()) {
    const parentQuestion = String(entry.q.parent || "").trim();
    const parentKey = parentQuestion ? normalizeQuestionKey(parentQuestion) : "";
    if (parentKey && nodeByKey.has(parentKey)) {
      if (!childrenOf.has(parentKey)) childrenOf.set(parentKey, []);
      childrenOf.get(parentKey).push(entry);
    } else {
      roots.push(entry);
    }
  }
  // Sort children by node_id for stable ordering
  const numericIdSort = (a, b) => {
    const ida = String(a.nodeId || "");
    const idb = String(b.nodeId || "");
    if (ida && idb) return ida.localeCompare(idb, undefined, { numeric: true, sensitivity: "base" });
    return String(a.key).localeCompare(String(b.key));
  };
  roots.sort(numericIdSort);
  for (const children of childrenOf.values()) children.sort(numericIdSort);

  // DFS walk to produce tree-order flat list
  const orderedNodes = [];
  const dfsWalk = (entries) => {
    for (const entry of entries) {
      orderedNodes.push(entry);
      const kids = childrenOf.get(entry.key);
      if (kids && kids.length) dfsWalk(kids);
    }
  };
  dfsWalk(roots);

  // Also append any orphans that weren't reachable (safety net)
  const visited = new Set(orderedNodes.map((n) => n.key));
  for (const entry of nodeByKey.values()) {
    if (!visited.has(entry.key)) orderedNodes.push(entry);
  }

  // Populate data map and detect auto-select
  for (const n of orderedNodes) {
    const isRunning = ["running", "researching", "decomposing"].includes(n.status);
    if (isRunning && !autoSelectKey) autoSelectKey = n.key;
    canvasNodeDataMap.set(n.key, {
      q: n.q, depth: n.depth, nodeId: n.nodeId, status: n.status,
      cls: n.cls, visualCls: n.visualCls, scopeKey,
      isPlanningView, planningUi, planningRowByNodeId, bonusCap,
      researchDepthBonusByQuestion, planningSwapDimAll,
    });
  }

  // If no running node, keep current selection; if that's gone, select first
  if (!autoSelectKey) {
    if (selectedCanvasNodeKey && canvasNodeDataMap.has(selectedCanvasNodeKey)) {
      autoSelectKey = selectedCanvasNodeKey;
    } else if (orderedNodes.length) {
      autoSelectKey = orderedNodes[0].key;
    }
  }
  selectedCanvasNodeKey = autoSelectKey;

  // ── Left panel: tree directory ──
  let treeHtml = "";
  for (const n of orderedNodes) {
    const indent = Math.max(0, n.depth - 1) * 16;
    const sel = n.key === selectedCanvasNodeKey ? "is-selected" : "";
    const displayTitle = String(n.q.display_title || n.q.sub_question || "").trim();
    treeHtml += `<div class="canvas-tree-item ${sel}" data-tree-key="${esc(n.key)}" style="padding-left:${10 + indent}px">`;
    treeHtml += `<span class="tree-status-dot ${esc(n.cls)}"></span>`;
    if (n.nodeId) treeHtml += `<span class="tree-node-id">${esc(n.nodeId)}</span>`;
    treeHtml += `<span class="tree-node-title" title="${esc(displayTitle)}">${esc(displayTitle)}</span>`;
    treeHtml += `</div>`;
  }
  canvasTreePanel.innerHTML = treeHtml;

  // ── Right panel: detail for selected node ──
  renderCanvasDetailForKey(selectedCanvasNodeKey);

  // ── Tree click handler ──
  canvasTreePanel.onclick = (e) => {
    const item = e.target.closest(".canvas-tree-item");
    if (!item) return;
    const key = item.getAttribute("data-tree-key");
    if (!key || key === selectedCanvasNodeKey) return;
    selectedCanvasNodeKey = key;
    // Update selection highlight
    for (const el of canvasTreePanel.querySelectorAll(".canvas-tree-item")) {
      el.classList.toggle("is-selected", el.getAttribute("data-tree-key") === key);
    }
    renderCanvasDetailForKey(key);
  };

  // Auto-scroll to selected tree item
  const selectedTreeItem = canvasTreePanel.querySelector(".canvas-tree-item.is-selected");
  if (selectedTreeItem) {
    selectedTreeItem.scrollIntoView({ behavior: "auto", block: "nearest" });
  }
  if (_renderTracker) _renderTracker.end();
}

function renderCanvasDetailForKey(key) {
  const data = canvasNodeDataMap.get(key);
  if (!data) {
    canvasDetailPanel.innerHTML = `<div class="canvas-detail-empty">Select a node to view details</div>`;
    return;
  }
  const { q, depth, nodeId, status, cls, visualCls, scopeKey,
    isPlanningView, planningUi, planningRowByNodeId, bonusCap,
    researchDepthBonusByQuestion, planningSwapDimAll } = data;
  const st = status;
  const solvedStatuses = new Set(["solved", "solved_via_children", "success", "completed"]);
  const unresolvedStatuses = new Set(["unresolved", "failed", "dead_end", "stopped", "aborted"]);
  const visitedOutcomeCls =
    visualCls === "state-visited"
      ? (solvedStatuses.has(st) ? "visited-solved" : (unresolvedStatuses.has(st) ? "visited-unresolved" : "visited-neutral"))
      : "";
  const dim = planningSwapDimAll
    ? "dimmed"
    : ((visualCls === "state-planned" && !isPlanningView) ? "dimmed" : "");
  const parentCls = q.parent ? "has-parent" : "";
  const isRunning = ["running", "researching", "decomposing"].includes(st);
  const isDecomposed = ["decomposed", "decomposed_child"].includes(st);
  const activeStateCls = (visualCls === "state-active" && isRunning)
    ? "active-running"
    : ((visualCls === "state-active" && isDecomposed) ? "active-decomposed" : "");
  const activeEmphasisCls = visualCls === "state-active" ? "active-emphasis" : "";
  let visualLabel = "planned";
  if (visualCls === "state-active") visualLabel = "active";
  else if (visualCls === "state-visited") visualLabel = "visited";

  let html = `<article class="node ${esc(cls)} ${esc(dim)} ${esc(parentCls)} ${esc(activeStateCls)} ${esc(activeEmphasisCls)} ${esc(visualCls)} ${esc(visitedOutcomeCls)}" data-node-depth="${Number(depth)}" data-node-status="${esc(st)}" data-node-id="${esc(nodeId || "")}" data-node-key="${esc(key)}">`;
  html += `<div class="node-head"><span class="badge ${esc(cls)}">${esc(shortStatus(st))}</span>`;
  html += `<span class="badge">${esc(visualLabel)}</span>`;
  html += `<span class="badge">d${depth}</span></div>`;
  if (nodeId) {
    html += `<div class="query-mini"><strong>Node ${esc(nodeId)}</strong></div>`;
  }
  const fullQuestion = String(q.sub_question || "").trim();
  const displayTitle = String(q.display_title || fullQuestion || "").trim();
  html += `<p class="node-title" title="${esc(fullQuestion)}">${esc(displayTitle)}</p>`;

  const pathId = nodeId;
  const researchDepthBonus = !isPlanningView
    ? researchDepthBonusByQuestion.get(normalizeQuestionKey(q.sub_question || ""))
    : null;
  const planningRow = planningUi.inPlanning ? planningRowByNodeId.get(pathId) : null;
  if (planningRow) {
    const pinned = !!planningRow.is_pinned;
    const bonus = Math.max(0, Number(planningRow.pin_depth_bonus || 0));
    const canEdit = !!planningUi.canEdit;
    html += `<div class="planning-node-actions">`;
    html += `<button type="button" class="planning-node-icon-btn ${pinned ? "is-active" : ""}" data-planning-action="${pinned ? "unpin" : "pin"}" data-node-id="${esc(pathId)}" ${canEdit ? "" : "disabled"} title="${pinned ? "Unpin" : "Pin"}" aria-label="${pinned ? "Unpin" : "Pin"}">`;
    html += `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 3h8l1 4h-2l-1 5 2 2v1H8v-1l2-2-1-5H7l1-4z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M12 15v6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`;
    html += `</button>`;
    html += `<button type="button" class="planning-node-icon-btn" data-planning-action="depth" data-node-id="${esc(pathId)}" ${(canEdit && pinned) ? "" : "disabled"} title="Depth +1" aria-label="Depth +1">`;
    html += `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`;
    html += `</button>`;
    const capTail = Number.isFinite(bonusCap) ? ` (<=${Math.max(0, Number(bonusCap))})` : "";
    html += `<span class="planning-node-depth">depth +${bonus}${esc(capTail)}</span>`;
    html += `</div>`;
  } else if (researchDepthBonus && Number.isFinite(researchDepthBonus.bonus) && researchDepthBonus.bonus > 0) {
    html += `<div class="planning-node-actions">`;
    html += `<span class="planning-node-depth planning-node-depth-readonly">depth +${Math.max(0, Number(researchDepthBonus.bonus || 0))}</span>`;
    html += `</div>`;
  }
  if (q.parent) {
    html += `<p class="node-parent"><span class="parent-link">from</span> ${esc(q.parent)}</p>`;
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
    const queryWorkOpen = !forceClosedQueryWork ? " open" : "";
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
    const nodeNotesOpen = !forceClosedDetailKeys.has(nodeNotesKey) ? " open" : "";
    html += `<details class="node-section" data-detail-key="${esc(nodeNotesKey)}"${nodeNotesOpen}><summary>Node Notes</summary>`;
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

  canvasDetailPanel.innerHTML = html;
  restoreAndTrackDetailState();
}

function isNodeMostlyVisibleInCanvas(node) {
  if (!node || !canvasEl) return false;
  const nr = node.getBoundingClientRect();
  const cr = canvasEl.getBoundingClientRect();
  const interLeft = Math.max(nr.left, cr.left);
  const interRight = Math.min(nr.right, cr.right);
  const interTop = Math.max(nr.top, cr.top);
  const interBottom = Math.min(nr.bottom, cr.bottom);
  const interW = Math.max(0, interRight - interLeft);
  const interH = Math.max(0, interBottom - interTop);
  const nodeW = Math.max(1, nr.width);
  const nodeH = Math.max(1, nr.height);
  const visibleRatio = (interW * interH) / (nodeW * nodeH);
  return visibleRatio >= 0.75;
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

function isAbortPendingSnapshot(snap) {
  if (!snap || typeof snap !== "object") return false;
  const tree = snap.tree && typeof snap.tree === "object" ? snap.tree : {};
  return !!tree.abort_pending;
}

function setPrimaryAndPlanningActionButtons({
  isNewSessionWorkspace,
  reportBusy,
  transitionPending,
  phase,
  pState,
}) {
  /* runBtn state is managed by switchRunBtnTo* in applySnapshot */
  if (!planningActionBtn) return;
  if (!currentRunId) {
    planningActionBtn.textContent = "Start Planning";
    planningActionBtn.disabled = !!(reportBusy || startRunInFlight);
    return;
  }
  const planningUi = derivePlanningUiState(phase, pState, {
    reportBusy,
    transitionPending,
    startRunBusy: startRunInFlight,
  });
  if (planningUi.inPlanning) {
    planningActionBtn.textContent = planningUi.planningActionText;
    planningActionBtn.disabled = planningUi.planningActionDisabled;
    return;
  }
  planningActionBtn.textContent = "Start Planning";
  planningActionBtn.disabled = true;
}

function refreshPrimaryAndPlanningActionButtonsFromSnapshot(snap) {
  if (!snap || typeof snap !== "object") return;
  const sid = String(snap.session_id || snap.run_id || "").trim();
  const isDraft = isDraftSnapshot(snap);
  const executionState = String(snap.execution_state || "").toLowerCase();
  const researchState = String(snap.research_state || "").toLowerCase();
  const paused = researchState ? researchState === "paused" : executionState === "paused";
  const terminal = researchState
    ? researchState === "terminal"
    : ["completed", "aborted", "failed"].includes(executionState);
  const pausePending = pauseRequested && !paused && !terminal;
  const abortPending = abortRequested || isAbortPendingSnapshot(snap);
  const transitionPending = pausePending || abortPending;
  const reportGeneratingForThisRun = isReportGeneratingForRun(sid);
  const reportState = String(snap.report_state || "").toLowerCase();
  const reportPhase = String(
    snap.report_status || (snap.tree && snap.tree.report_status) || "pending"
  ).toLowerCase();
  const reportBusy = reportGeneratingForThisRun || reportPhase === "running" || reportState === "generating";
  const isNewSessionWorkspace = !currentRunId || isDraft;
  const phase = planningPhase(snap);
  const pState = planningState(snap);
  setPrimaryAndPlanningActionButtons({
    isNewSessionWorkspace,
    reportBusy,
    transitionPending,
    phase,
    pState,
  });
}

function applySnapshot(snap) {
  currentSnapshot = snap;
  refreshContextAggregateViewFromSnapshot();
  const sid = String(snap.session_id || snap.run_id || "").trim();
  setContextEnabled(!!sid);
  if (contextDigestPaneEl) {
    contextDigestPaneEl.style.display = "";
  }
  if (sid && Array.isArray(sessions)) {
    const idx = sessions.findIndex((s) => String((s && s.session_id) || "") === sid);
    if (idx >= 0) {
      sessions[idx] = {
        ...sessions[idx],
        title: snap.title || snap.task || sessions[idx].title,
        status: snap.status,
        version: snap.version || sessions[idx].version,
        execution_state: snap.execution_state || sessions[idx].execution_state,
        research_state: snap.research_state || sessions[idx].research_state,
        report_state: snap.report_state || sessions[idx].report_state,
        report_status: snap.report_status || sessions[idx].report_status,
        updated_at: snap.updated_at || sessions[idx].updated_at,
      };
    }
  }
  runStatusEl.textContent = shortStatus(snap.status);
  researchStatusEl.textContent = shortStatus(snap.research_status);
  reportStatusEl.textContent = shortStatus(snap.report_status || (snap.tree && snap.tree.report_status) || "pending");
  const title = String(snap.title || snap.task || "").trim();
  runMeta.textContent = `${title ? `${title} · ` : ""}Run ID: ${snap.run_id}`;
  const isDraft = isDraftSnapshot(snap);
  const runActive = !!sid && !isDraft;
  taskEl.value = String(snap.task || "").slice(0, TASK_MAX_CHARS);
  updateTaskCharCount();
  alignTaskTextBottom();
  setTaskBoxLocked(runActive);
  setRunConfigLocked(runActive);
  if (Number.isFinite(Number(snap.max_depth)) && Number(snap.max_depth) >= 1) {
    const depthVal = String(Math.floor(Number(snap.max_depth)));
    maxDepthEl.value = depthVal;
    if (depthDropdownValue) depthDropdownValue.textContent = depthVal;
  }
  stopReasonEl.textContent = snap.stop_reason ? `Stop rationale: ${snap.stop_reason}` : "";
  if (latestThoughtEl) latestThoughtEl.textContent = snap.latest_thought || "";
  if (coverageNoteEl) coverageNoteEl.textContent = snap.coverage_note || "";
  if (UI_DEBUG) {
    const tree = (snap && typeof snap.tree === "object" && snap.tree) ? snap.tree : {};
    const dbgStatus = String(snap.report_status || tree.report_status || "pending").trim() || "pending";
    const dbgPhase = String(tree.report_phase || "").trim() || "-";
    const dbgMode = String(tree.report_mode || "").trim() || "-";
    const dbgUpdated = String(tree.report_phase_updated_at || "").trim();
    const dbgError = String(tree.report_error || "").trim();
    const dbgKey = [sid, dbgStatus, dbgPhase, dbgMode, dbgUpdated, dbgError].join("|");
    if (dbgKey !== lastDebugReportPhaseKey) {
      const errTail = dbgError ? ` error=${dbgError.slice(0, 120)}` : "";
      upsertThought(
        `[DEBUG] report phase=${dbgPhase} status=${dbgStatus} mode=${dbgMode}${errTail}`,
        "debug"
      );
      lastDebugReportPhaseKey = dbgKey;
    }
  }

  showError(snap.error || "");

  /* ── Detect execution state early (needed for abort-clear logic) ── */
  const executionState = String(snap.execution_state || "").toLowerCase();
  const researchState = String(snap.research_state || "").toLowerCase();
  const running = researchState ? researchState === "running" : executionState === "running";
  const paused = researchState ? researchState === "paused" : executionState === "paused";
  const terminal = researchState ? researchState === "terminal" : ["completed", "aborted", "failed"].includes(executionState);

  if (progressTitleEl) progressTitleEl.style.display = terminal ? "none" : "";

  /* If user clicked Abort, keep page content intact */
  if (!uiClearedByAbort) {
    renderCanvas(snap.tree || {});
  }

  const versions = normalizeReportVersions(snap);
  const activeIdx = activeReportVersionIndex(snap);
  selectedReportVersionIndex = activeIdx;
  let reportText = "";
  let reportFilePath = "";
  if (versions.length && activeIdx) {
    const selected = versions[activeIdx - 1] || {};
    reportText = String(selected.report_text || "");
    reportFilePath = String(selected.report_file_path || "");
  } else {
    reportText = String(snap.report_text || "");
    reportFilePath = String(snap.report_file_path || "");
  }

  const hasReport = !!reportFilePath || !!reportText.trim();
  if (hasReport) {
    renderTokenSummary(snap.token_usage);
  } else {
    tokenSummaryEl.textContent = "-";
    usageEl.textContent = "";
  }

  reportRenderedEl.innerHTML = markdownToHtml(reportText);
  viewReportBtn.disabled = !reportText.trim();

  if (paused || terminal) {
    pauseRequested = false;
  }
  if (terminal) {
    abortRequested = false;
  }
  const pausePending = pauseRequested && !paused && !terminal;
  const abortPending = abortRequested || isAbortPendingSnapshot(snap);
  const transitionPending = pausePending || abortPending;
  const reportGeneratingForThisRun = isReportGeneratingForRun(sid);
  const reportState = String(snap.report_state || "").toLowerCase();
  const reportPhase = String(snap.report_status || (snap.tree && snap.tree.report_status) || "pending").toLowerCase();
  const reportBusy = reportGeneratingForThisRun || reportPhase === "running" || reportState === "generating";
  const isNewSessionWorkspace = !currentRunId || isDraft;
  const phase = planningPhase(snap);
  const pState = planningState(snap);
  const planningAbortable = phase === "planning" && ["running", "review"].includes(pState);
  setPrimaryAndPlanningActionButtons({
    isNewSessionWorkspace,
    reportBusy,
    transitionPending,
    phase,
    pState,
  });
  if (transitionPending) {
    pauseBtn.disabled = true;
    resumeBtn.disabled = true;
    abortBtn.disabled = true;
  } else {
    pauseBtn.disabled = !(running && !reportBusy);
    resumeBtn.disabled = !(paused && !reportBusy);
    abortBtn.disabled = !((running || paused || planningAbortable) && !reportBusy);
  }
  abortBtn.textContent = abortPending ? "Stopping..." : "Abort";

  /* ── Sync inline run/abort button ── */
  if (uiClearedByAbort) {
    switchRunBtnToStart();
    runBtn.disabled = true;
    setTaskBoxLocked(true);
  } else if (running || paused) {
    switchRunBtnToAbort();
    if (abortPending) {
      runBtn.textContent = "Stopping...";
      runBtn.disabled = true;
    }
    setTaskBoxLocked(true);
  } else if (terminal) {
    switchRunBtnToStart();
    runBtn.disabled = true;
    setTaskBoxLocked(true);
  } else if (!currentRunId) {
    switchRunBtnToStart();
    setTaskBoxLocked(false);
  }

  const hasDownloadableReport = !!reportFilePath;
  if (downloadBtn) downloadBtn.disabled = !hasDownloadableReport;
  const allowManualReport = !!sid && !reportBusy && (paused || terminal);
  reportBtn.disabled = !allowManualReport;
  if (generateReportBtn) generateReportBtn.disabled = !allowManualReport;
  setReportTemplateControlsDisabled(reportBusy || reportTemplateMutationInFlight);
  setReportButtonVisual(reportGeneratingForThisRun);
  if (versions.length && activeIdx) {
    if (reportVersionLabel) reportVersionLabel.textContent = `${activeIdx}/${versions.length}`;
  } else {
    if (reportVersionLabel) reportVersionLabel.textContent = "-/-";
  }
  if (reportPrevBtn) reportPrevBtn.disabled = !(versions.length > 1 && activeIdx && activeIdx > 1);
  if (reportNextBtn) reportNextBtn.disabled = !(versions.length > 1 && activeIdx && activeIdx < versions.length);
  renderPlanningPanel(snap);
  renderSessions();
}

async function fetchSnapshot(runId) {
  const rsp = await fetch(`/api/sessions/${runId}`);
  if (!rsp.ok) throw new Error(`Failed to fetch snapshot: ${rsp.status}`);
  const data = await rsp.json();
  if (String(currentRunId || "") !== String(runId || "")) return;
  applySnapshot(data);
  renderSessions();
}

async function scheduleSnapshotRefresh(runId) {
  const rid = String(runId || "").trim();
  if (!rid) return;
  if (snapshotFetchInFlight) {
    snapshotFetchQueued = true;
    return;
  }
  snapshotFetchInFlight = true;
  try {
    do {
      snapshotFetchQueued = false;
      if (String(currentRunId || "") !== rid) {
        break;
      }
      await fetchSnapshot(rid);
    } while (snapshotFetchQueued);
  } finally {
    snapshotFetchInFlight = false;
  }
}

async function openSession(runId, opts = {}) {
  const sid = String(runId || "").trim();
  if (!sid) return;
  closeMobileDrawer();
  const updateUrl = opts.updateUrl !== false;
  const switchToken = ++activeSessionSwitchToken;
  if (es) {
    es.close();
    es = null;
  }
  clearLiveIntent();
  currentRunId = sid;
  if (typeof SBTelemetry !== "undefined") SBTelemetry.setSession(sid);
  if (appMainEl) appMainEl.classList.remove("is-empty");
  pauseRequested = false;
  abortRequested = false;
  uiClearedByAbort = false;
  if (progressLog) progressLog.innerHTML = "";
  await fetchSnapshot(sid);
  setContextEnabled(true);
  clearContextPane("Loading context...");
  void scheduleContextRefresh(sid);
  if (switchToken !== activeSessionSwitchToken || currentRunId !== sid) {
    return;
  }
  if (updateUrl) {
    setRunIdInUrl(sid);
  }
  // Keep SSE attached for any opened session so manual report generation
  // and version-selection events are visible even when research is not running.
  connectEvents(sid);
  connectContextEvents(sid);
}

function connectEvents(runId) {
  if (es) es.close();
  lastEventSeqByRun.set(String(runId), 0);
  es = new EventSource(`/api/runs/${runId}/events`);
  if (typeof SBTelemetry !== "undefined") SBTelemetry.trackSSE(es, "run_events");

  es.onmessage = async (evt) => {
    if (String(runId) !== String(currentRunId || "")) return;
    try {
      if (evt.data) {
        const parsed = JSON.parse(evt.data);
        const seq = Number(parsed && parsed.event_seq);
        if (Number.isFinite(seq)) {
          const k = String(runId);
          const prev = Number(lastEventSeqByRun.get(k) || 0);
          if (seq <= prev) {
            return;
          }
          lastEventSeqByRun.set(k, seq);
        }
        const narration = eventNarration(parsed);
        if (narration) {
          upsertThought(narration, parsed.event_type || "event");
        }
      }
    } catch (_err) {
      // no-op
    }
    await scheduleSnapshotRefresh(runId);
  };

  const events = [
    "planning_started", "planning_review_ready", "planning_pin_updated", "planning_depth_bonus_updated",
    "planning_committed", "planning_failed", "planning_aborted",
    "run_started", "context_binding_started", "context_binding_completed", "context_binding_failed",
    "context_binding_parsing_started", "context_binding_parsing_file_completed", "context_binding_parsing_completed", "context_for_node_ready",
    "plan_created", "round_started", "sub_question_started", "queries_generated",
    "query_started", "query_diagnostic", "query_skipped_cached", "query_rerun_allowed",
    "query_broadened", "query_blocked_diminishing_returns", "search_completed", "synthesis_completed",
    "node_sufficiency_started", "node_sufficiency_completed", "node_decomposition_started", "node_decomposed", "node_completed", "node_unresolved",
    "sufficiency_started", "sufficiency_completed", "run_paused", "run_resumed", "run_abort_requested", "abort_requested", "run_aborted",
    "report_generation_started", "report_heartbeat", "report_generation_completed", "report_generation_failed",
    "report_context_attached",
    "partial_report_generated", "report_version_created", "report_version_selected", "run_heartbeat", "run_completed", "run_failed"
  ];
  for (const e of events) {
    es.addEventListener(e, async (evt) => {
      if (String(runId) !== String(currentRunId || "")) return;
      try {
        const parsed = JSON.parse(evt.data || "{}");
        const seq = Number(parsed && parsed.event_seq);
        if (Number.isFinite(seq)) {
          const k = String(runId);
          const prev = Number(lastEventSeqByRun.get(k) || 0);
          if (seq <= prev) {
            return;
          }
          lastEventSeqByRun.set(k, seq);
        }
        const narration = eventNarration(parsed);
        if (narration) {
          upsertThought(narration, parsed.event_type || e);
        }
        if (parsed.event_type === "context_binding_parsing_started") {
          const total = Number(parsed.payload && parsed.payload.files_total);
          markContextFilesParsingStarted(runId, Number.isFinite(total) ? total : 0);
          scheduleContextRefreshDebounced(runId);
        } else if (parsed.event_type === "context_binding_parsing_file_completed") {
          markContextFileParsingProgress(parsed.payload || {}, runId);
          scheduleContextRefreshDebounced(runId);
        } else if (
          parsed.event_type === "context_binding_parsing_completed"
          || parsed.event_type === "context_binding_completed"
          || parsed.event_type === "context_binding_failed"
        ) {
          clearStartupParseState(runId);
          scheduleContextRefreshDebounced(runId);
        }
        if (parsed.event_type === "context_for_node_ready" && contextAggregateDigestEl) {
          const ctx = parsed.payload && typeof parsed.payload === "object"
            ? parsed.payload.context_slice
            : null;
          if (ctx && typeof ctx === "object") {
            renderContextAggregateView(ctx, "context_slice");
          }
        }
        if (parsed.event_type === "report_generation_completed" || parsed.event_type === "report_generation_failed") {
          reportGeneratingRunIds.delete(String(runId));
        }
        if (parsed.event_type === "run_paused" || parsed.event_type === "run_resumed") {
          pauseRequested = false;
        }
        if (parsed.event_type === "run_abort_requested" || parsed.event_type === "abort_requested") {
          abortRequested = true;
        }
        if (
          parsed.event_type === "run_aborted"
          || parsed.event_type === "run_completed"
          || parsed.event_type === "run_failed"
          || parsed.event_type === "planning_aborted"
          || parsed.event_type === "planning_failed"
        ) {
          clearStartupParseState(runId);
          pauseRequested = false;
          /* abortRequested is reset inside applySnapshot after clearing the UI */
        }
      } catch (_err) {
        // no-op
      }
      await scheduleSnapshotRefresh(runId);
    });
  }
}

async function startRun(idempotencyKey, startMode = "research") {
  const task = taskEl.value.trim();
  if (!task) {
    showError("Task is required.");
    return;
  }
  clearLiveIntent();
  setRunConfigLocked(true);
  showError("");
  pauseRequested = false;
  abortRequested = false;
  uiClearedByAbort = false;
  autoFollowActiveNode = true;
  lastAutoFollowNodeKey = "";
  selectedCanvasNodeKey = "";
  canvasNodeDataMap = new Map();
  thoughts.length = 0;

  try {
    const endpoint = "/api/runs/start_from_workspace";
    const rsp = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(idempotencyKey ? { "Idempotency-Key": idempotencyKey } : {}),
      },
      body: JSON.stringify({
        workspace_id: ensureWorkspaceId(),
        task,
        max_depth: Number(maxDepthEl.value || DEFAULT_MAX_DEPTH),
        results_per_query: DEFAULT_RESULTS_PER_QUERY,
        start_mode: String(startMode || "research"),
      }),
    });
    if (!rsp.ok) {
      throw new Error(`Run creation failed: ${rsp.status}`);
    }
    const data = await rsp.json();
    const newRunId = String(data.run_id || "").trim();
    if (currentRunId && currentRunId !== newRunId) {
      clearStartupParseState(currentRunId);
    }
    if (newRunId) {
      markContextFilesParsingStarted(newRunId, 0);
    }
    const stagedWorkspaceId = String(ensureWorkspaceId() || "").trim();
    if (newRunId && stagedWorkspaceId && currentContextSet && typeof currentContextSet === "object") {
      stagedContextBySessionId.set(newRunId, {
        workspaceId: stagedWorkspaceId,
        contextSet: cloneJson(currentContextSet),
      });
    }
    currentRunId = String(data.run_id || currentRunId || "");
    if (typeof SBTelemetry !== "undefined") SBTelemetry.setSession(currentRunId);
    if (appMainEl) appMainEl.classList.remove("is-empty");
    setRunIdInUrl(currentRunId);
    // Subscribe immediately so startup parsing events are not missed.
    connectEvents(currentRunId);
    connectContextEvents(currentRunId);
    await fetchSnapshot(currentRunId);
    setContextEnabled(true);
    await scheduleContextRefresh(currentRunId);
    await fetchSessions();
    emitSessionMutation("create", currentRunId);
  } catch (err) {
    if (!currentRunId) {
      setRunConfigLocked(false);
    }
    throw err;
  }
}

async function abortRun() {
  if (!currentRunId) return;
  const ok = window.confirm("Abort this research session? This cannot be resumed.");
  if (!ok) return;
  abortRequested = true;
  uiClearedByAbort = true;
  pauseBtn.disabled = true;
  resumeBtn.disabled = true;
  abortBtn.disabled = true;
  abortBtn.textContent = "Stopping...";
  const runId = String(currentRunId || "").trim();
  try {
    const rsp = await fetch(`/api/runs/${runId}/abort`, {
      method: "POST",
      headers: { "Idempotency-Key": newIdempotencyKey("abort") },
    });
    if (!rsp.ok) {
      abortRequested = false;
      throw new Error(await responseDetail(rsp, `Abort failed: ${rsp.status}`));
    }
    /* Keep page content intact — just lock UI controls */
    switchRunBtnToStart();
    runBtn.disabled = true;
    setTaskBoxLocked(true);
    setRunConfigLocked(true);
  } catch (err) {
    abortRequested = false;
    if (currentSnapshot && typeof currentSnapshot === "object") {
      applySnapshot(currentSnapshot);
    }
    throw err;
  }
}

async function pauseRun() {
  if (!currentRunId) return;
  pauseRequested = true;
  pauseBtn.disabled = true;
  resumeBtn.disabled = true;
  abortBtn.disabled = true;
  try {
    const runId = String(currentRunId || "").trim();
    const ev = currentExpectedVersion();
    const qs = ev ? `?expected_version=${encodeURIComponent(String(ev))}` : "";
    let rsp = await fetch(`/api/runs/${runId}/pause${qs}`, { method: "POST" });
    if (rsp.status === 409) {
      await fetchSnapshot(runId);
      const retryEv = currentExpectedVersion();
      const retryQs = retryEv ? `?expected_version=${encodeURIComponent(String(retryEv))}` : "";
      rsp = await fetch(`/api/runs/${runId}/pause${retryQs}`, { method: "POST" });
    }
    if (!rsp.ok) {
      pauseRequested = false;
      throw new Error(await responseDetail(rsp, `Pause failed: ${rsp.status}`));
    }
  } catch (err) {
    pauseRequested = false;
    if (currentSnapshot && typeof currentSnapshot === "object") {
      applySnapshot(currentSnapshot);
    }
    throw err;
  }
}

async function resumeRun() {
  if (!currentRunId) return;
  const runId = String(currentRunId || "").trim();
  const ev = currentExpectedVersion();
  const qs = ev ? `?expected_version=${encodeURIComponent(String(ev))}` : "";
  let rsp = await fetch(`/api/runs/${runId}/resume${qs}`, { method: "POST" });
  if (rsp.status === 409) {
    await fetchSnapshot(runId);
    const retryEv = currentExpectedVersion();
    const retryQs = retryEv ? `?expected_version=${encodeURIComponent(String(retryEv))}` : "";
    rsp = await fetch(`/api/runs/${runId}/resume${retryQs}`, { method: "POST" });
  }
  if (!rsp.ok) {
    throw new Error(await responseDetail(rsp, `Resume failed: ${rsp.status}`));
  }
}

async function selectReportVersion(versionIndex) {
  if (!currentRunId) return;
  const vi = Number(versionIndex);
  if (!Number.isFinite(vi) || vi < 1) return;
  const ev = currentExpectedVersion();
  const qs = ev
    ? `version_index=${Math.floor(vi)}&expected_version=${encodeURIComponent(String(ev))}`
    : `version_index=${Math.floor(vi)}`;
  const rsp = await fetch(`/api/runs/${currentRunId}/report/select?${qs}`, {
    method: "POST",
  });
  if (!rsp.ok) {
    throw new Error(`Select report version failed: ${rsp.status}`);
  }
  await fetchSnapshot(currentRunId);
}

async function uploadContextFiles(files) {
  if (contextMutationInFlight) {
    throw new Error("Context update already in progress. Please wait.");
  }
  const list = Array.from(files || []).filter((f) => f instanceof File);
  if (!list.length) {
    throw new Error("Please select one or more files to upload.");
  }
  const uploadedNames = list.map((f) => String(f.name || "context.txt"));
  pendingUploadFiles = uploadedNames;
  renderContextPane();
  if (contextAggregateDigestEl) {
    contextAggregateViewBase = null;
    contextAggregateViewMode = "";
    contextAggregateDigestEl.textContent = `Parsing uploaded files, please wait...\n- ${uploadedNames.join("\n- ")}`;
  }
  setContextBusy(true);
  const fd = new FormData();
  for (const f of list) {
    fd.append("files", f, f.name || "context.txt");
  }
  try {
    const send = async () => {
      const headers = { "Idempotency-Key": newIdempotencyKey("ctx_upload") };
      const rev = Number(currentContextSet && currentContextSet.revision);
      if (Number.isFinite(rev) && rev >= 1) {
        headers["If-Match"] = String(Math.floor(rev));
      }
      return fetch(`${contextBasePath()}/files`, {
        method: "POST",
        headers,
        body: fd,
      });
    };
    let rsp = await send();
    if (rsp.status === 409) {
      await scheduleContextRefresh();
      rsp = await send();
    }
    if (!rsp.ok) {
      throw new Error(await responseDetail(rsp, `Context upload failed: ${rsp.status}`));
    }
    const data = await rsp.json();
    currentContextSet = data && data.context_set ? data.context_set : currentContextSet;
    pendingUploadFiles = [];
    renderContextPane();
    await fetchContextAggregateDigest();
  } finally {
    pendingUploadFiles = [];
    setContextBusy(false);
  }
}

async function replaceContextFile(fileId, fileObj) {
  if (contextMutationInFlight) {
    throw new Error("Context update already in progress. Please wait.");
  }
  const fid = String(fileId || "").trim();
  if (!fid || !(fileObj instanceof File)) return;
  setContextBusy(true);
  const fd = new FormData();
  fd.append("file", fileObj, fileObj.name || "context.txt");
  try {
    const send = async () => {
      const headers = { "Idempotency-Key": newIdempotencyKey("ctx_replace") };
      const rev = Number(currentContextSet && currentContextSet.revision);
      if (Number.isFinite(rev) && rev >= 1) {
        headers["If-Match"] = String(Math.floor(rev));
      }
      return fetch(`${contextBasePath()}/files/${encodeURIComponent(fid)}`, {
        method: "PUT",
        headers,
        body: fd,
      });
    };
    let rsp = await send();
    if (rsp.status === 409) {
      await scheduleContextRefresh();
      rsp = await send();
    }
    if (!rsp.ok) {
      throw new Error(await responseDetail(rsp, `Context replace failed: ${rsp.status}`));
    }
    const data = await rsp.json();
    currentContextSet = data && data.context_set ? data.context_set : currentContextSet;
    renderContextPane();
    await fetchContextAggregateDigest();
    if (selectedContextFileId === fid) {
      await fetchContextFileDigest(fid);
    }
  } finally {
    setContextBusy(false);
  }
}

async function deleteContextFile(fileId) {
  if (contextMutationInFlight) {
    throw new Error("Context update already in progress. Please wait.");
  }
  const fid = String(fileId || "").trim();
  if (!fid) return;
  setContextBusy(true);
  try {
    const send = async () => {
      const headers = { "Idempotency-Key": newIdempotencyKey("ctx_delete") };
      const rev = Number(currentContextSet && currentContextSet.revision);
      if (Number.isFinite(rev) && rev >= 1) {
        headers["If-Match"] = String(Math.floor(rev));
      }
      return fetch(`${contextBasePath()}/files/${encodeURIComponent(fid)}`, {
        method: "DELETE",
        headers,
      });
    };
    let rsp = await send();
    if (rsp.status === 409) {
      await scheduleContextRefresh();
      rsp = await send();
    }
    if (!rsp.ok) {
      throw new Error(await responseDetail(rsp, `Context delete failed: ${rsp.status}`));
    }
    const data = await rsp.json();
    currentContextSet = data && data.context_set ? data.context_set : currentContextSet;
    if (selectedContextFileId === fid) {
      selectedContextFileId = "";
      if (contextFileDigestEl) {
        contextFileDigestEl.textContent = "";
      }
    }
    renderContextPane();
    await fetchContextAggregateDigest();
  } finally {
    setContextBusy(false);
  }
}

/* ── Bottom-align textarea content ──
 * Dynamically adjust padding-top so text sits at the bottom of the 3-row
 * textarea and grows upward as the user types more lines.
 * Uses a hidden mirror div to measure true content height.
 */
const _taskMirror = document.createElement("div");
_taskMirror.setAttribute("aria-hidden", "true");
_taskMirror.style.cssText =
  "position:absolute;top:0;left:0;visibility:hidden;pointer-events:none;" +
  "white-space:pre-wrap;word-wrap:break-word;overflow-wrap:break-word;";
document.body.appendChild(_taskMirror);

function alignTaskTextBottom() {
  const el = taskEl;
  if (!el) return;
  const cs = getComputedStyle(el);
  // Sync mirror sizing with textarea
  _taskMirror.style.font = cs.font;
  _taskMirror.style.letterSpacing = cs.letterSpacing;
  _taskMirror.style.width = el.clientWidth + "px";
  _taskMirror.style.paddingLeft = cs.paddingLeft;
  _taskMirror.style.paddingRight = cs.paddingRight;
  _taskMirror.textContent = el.value || "\u200b";
  if (el.value.endsWith("\n")) _taskMirror.textContent += "\u200b";
  const contentH = _taskMirror.offsetHeight;
  const lineHeight = parseFloat(cs.lineHeight) || (parseFloat(cs.fontSize) * 1.5);
  const rows = parseInt(el.getAttribute("rows") || "3", 10);
  const minContentArea = Math.round(lineHeight * rows);
  const targetH = Math.max(minContentArea, contentH) + 4; /* 4px bottom pad */
  el.style.height = targetH + "px";
  const pad = Math.max(0, targetH - 4 - contentH);
  el.style.paddingTop = pad + "px";
}
const TASK_MAX_CHARS = 140;
const taskCharCountEl = document.getElementById("taskCharCount");

taskEl.setAttribute("maxlength", String(TASK_MAX_CHARS));

function updateTaskCharCount() {
  const len = taskEl.value.length;
  if (taskCharCountEl) {
    taskCharCountEl.textContent = len + "/" + TASK_MAX_CHARS;
    taskCharCountEl.classList.toggle("over-limit", len >= TASK_MAX_CHARS);
  }
}

taskEl.addEventListener("input", () => {
  if (taskEl.value.length > TASK_MAX_CHARS) {
    taskEl.value = taskEl.value.slice(0, TASK_MAX_CHARS);
  }
  updateTaskCharCount();
  alignTaskTextBottom();
  updateLiveIntentFromInput();
  if (currentRunId || startRunInFlight) return;
  runBtn.disabled = !taskEl.value.trim();
});
// Also align on initial load and when value is set programmatically
requestAnimationFrame(() => {
  alignTaskTextBottom();
  updateTaskCharCount();
  updateLiveIntentFromInput();
});

taskEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    if (!e.shiftKey && !runBtn.disabled) runBtn.click();
  }
});

/* ── Prompt example chips ── */
document.querySelectorAll(".prompt-example-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    if (currentRunId || startRunInFlight) return;
    taskEl.value = chip.textContent.trim().slice(0, TASK_MAX_CHARS);
    taskEl.focus();
    updateTaskCharCount();
    alignTaskTextBottom();
    updateLiveIntentFromInput();
    runBtn.disabled = false;
  });
});

runBtn.addEventListener("click", async () => {
  /* ── Abort mode: button says "Abort" while a run is active ── */
  if (currentRunId && !startRunInFlight && runBtn.dataset.mode === "abort") {
    try {
      await abortRun();
    } catch (err) {
      showError(err.message || String(err));
    }
    return;
  }

  if (currentRunId || startRunInFlight) {
    return;
  }
  startRunInFlight = true;
  runBtn.disabled = true;
  if (planningActionBtn) {
    planningActionBtn.disabled = true;
  }
  try {
    await startRun(newIdempotencyKey("start_research"), "research");
    /* After run starts: lock textarea (keep content visible), switch to Abort */
    setTaskBoxLocked(true);
    switchRunBtnToAbort();
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    startRunInFlight = false;
    if (!currentRunId) {
      switchRunBtnToStart();
      setTaskBoxLocked(false);
    }
  }
});

if (planningActionBtn) {
  planningActionBtn.addEventListener("click", async () => {
    const phase = planningPhase(currentSnapshot);
    const pState = planningState(currentSnapshot);
    if (currentRunId && phase === "planning" && pState === "review") {
      try {
        await planningCommit();
      } catch (err) {
        showError(err.message || String(err));
      }
      return;
    }
    if (currentRunId || startRunInFlight) {
      return;
    }
    startRunInFlight = true;
    runBtn.disabled = true;
    planningActionBtn.disabled = true;
    try {
      await startRun(newIdempotencyKey("start_planning"), "planning");
    } catch (err) {
      showError(err.message || String(err));
    } finally {
      startRunInFlight = false;
      if (!currentRunId) {
        runBtn.disabled = false;
        planningActionBtn.disabled = false;
      }
    }
  });
}

reportBtn.addEventListener("click", async () => {
  const reportRunId = String(currentRunId || "").trim();
  if (!reportRunId || isReportGeneratingForRun(reportRunId)) return;
  reportGeneratingRunIds.add(reportRunId);
  runBtn.disabled = true;
  if (planningActionBtn) {
    planningActionBtn.disabled = true;
  }
  reportBtn.disabled = true;
  if (generateReportBtn) generateReportBtn.disabled = true;
  setReportButtonVisual(true);
  showError("");
  upsertThought("Generating report from accumulated findings.", "report");

  try {
    const idempotencyKey = newIdempotencyKey("report");
    const ev = currentExpectedVersion();
    const qs = ev ? `?expected_version=${encodeURIComponent(String(ev))}` : "";
    const templateId = String(selectedReportTemplateId || "").trim() || "executive";
    const rsp = await fetch(`/api/runs/${reportRunId}/report${qs}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": idempotencyKey,
      },
      body: JSON.stringify({ template_id: templateId }),
    });
    if (!rsp.ok) {
      throw new Error(`Report generation failed: ${rsp.status}`);
    }
    if (currentRunId === reportRunId) {
      await fetchSnapshot(reportRunId);
    }
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    reportGeneratingRunIds.delete(reportRunId);
    if (currentRunId === reportRunId) {
      await fetchSnapshot(reportRunId);
    }
  }
});

if (generateReportBtn) {
  generateReportBtn.addEventListener("click", () => reportBtn.click());
}

if (reportTemplateSelectEl) {
  reportTemplateSelectEl.addEventListener("change", () => {
    selectedReportTemplateId = String(reportTemplateSelectEl.value || "").trim() || "executive";
    const selected = findReportTemplateById(selectedReportTemplateId);
    if (selected) {
      setReportTemplateForm(selected);
      if (reportTplPreviewEl) {
        reportTplPreviewEl.textContent = String(selected.rendered_background_prompt || "");
      }
    }
    refreshReportTemplateActionAvailability();
  });
}

if (reportTemplateToggleBtn) {
  reportTemplateToggleBtn.addEventListener("click", () => {
    const isOpen = !!(reportTemplateEditorEl && reportTemplateEditorEl.open);
    setReportTemplateEditorOpen(!isOpen);
  });
}

if (reportTemplateEditorEl) {
  reportTemplateEditorEl.addEventListener("toggle", () => {
    syncReportTemplateToggleState();
  });
}
syncReportTemplateToggleState();

if (reportTplNewBtn) {
  reportTplNewBtn.addEventListener("click", () => {
    selectedReportTemplateId = "";
    if (reportTemplateSelectEl) {
      reportTemplateSelectEl.value = "";
    }
    clearReportTemplateFormForNew();
    setReportTemplateEditorOpen(true);
    refreshReportTemplateActionAvailability();
  });
}

if (reportTplSaveBtn) {
  reportTplSaveBtn.addEventListener("click", async () => {
    const draft = reportTemplateDraftFromForm();
    if (!draft.name) {
      showError("Template name is required.");
      return;
    }
    const selected = findReportTemplateById(selectedReportTemplateId);
    if (selected && selected.is_builtin) {
      selectedReportTemplateId = "";
    }
    reportTemplateMutationInFlight = true;
    setReportTemplateControlsDisabled(true);
    try {
      const payload = JSON.stringify(draft);
      let rsp;
      const existing = findReportTemplateById(selectedReportTemplateId);
      if (existing && !existing.is_builtin) {
        rsp = await fetch(`/api/report/templates/${encodeURIComponent(String(existing.template_id || ""))}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: payload,
        });
      } else {
        rsp = await fetch("/api/report/templates", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload,
        });
      }
      if (!rsp.ok) {
        throw new Error(await responseDetail(rsp, `Save template failed: ${rsp.status}`));
      }
      const saved = await rsp.json();
      await fetchReportTemplates();
      selectedReportTemplateId = String(saved.template_id || selectedReportTemplateId || "").trim() || "executive";
      if (reportTemplateSelectEl) reportTemplateSelectEl.value = selectedReportTemplateId;
      const chosen = findReportTemplateById(selectedReportTemplateId);
      if (chosen) setReportTemplateForm(chosen);
    } catch (err) {
      showError(err.message || String(err));
    } finally {
      reportTemplateMutationInFlight = false;
      const sid = String((currentSnapshot && (currentSnapshot.session_id || currentSnapshot.run_id)) || "").trim();
      const reportGeneratingForThisRun = isReportGeneratingForRun(sid);
      const reportState = String((currentSnapshot && currentSnapshot.report_state) || "").toLowerCase();
      const tree = currentSnapshot && currentSnapshot.tree && typeof currentSnapshot.tree === "object" ? currentSnapshot.tree : {};
      const reportPhase = String((currentSnapshot && currentSnapshot.report_status) || tree.report_status || "pending").toLowerCase();
      setReportTemplateControlsDisabled(reportGeneratingForThisRun || reportPhase === "running" || reportState === "generating");
    }
  });
}

if (reportTplDeleteBtn) {
  reportTplDeleteBtn.addEventListener("click", async () => {
    const selected = findReportTemplateById(selectedReportTemplateId);
    if (!selected || selected.is_builtin) {
      showError("Select a custom template to delete.");
      return;
    }
    const ok = window.confirm(`Delete template "${selected.name || selected.template_id}"?`);
    if (!ok) return;
    reportTemplateMutationInFlight = true;
    setReportTemplateControlsDisabled(true);
    try {
      const rsp = await fetch(`/api/report/templates/${encodeURIComponent(String(selected.template_id || ""))}`, {
        method: "DELETE",
      });
      if (!rsp.ok) {
        throw new Error(await responseDetail(rsp, `Delete template failed: ${rsp.status}`));
      }
      selectedReportTemplateId = "executive";
      await fetchReportTemplates();
    } catch (err) {
      showError(err.message || String(err));
    } finally {
      reportTemplateMutationInFlight = false;
      const sid = String((currentSnapshot && (currentSnapshot.session_id || currentSnapshot.run_id)) || "").trim();
      const reportGeneratingForThisRun = isReportGeneratingForRun(sid);
      const reportState = String((currentSnapshot && currentSnapshot.report_state) || "").toLowerCase();
      const tree = currentSnapshot && currentSnapshot.tree && typeof currentSnapshot.tree === "object" ? currentSnapshot.tree : {};
      const reportPhase = String((currentSnapshot && currentSnapshot.report_status) || tree.report_status || "pending").toLowerCase();
      setReportTemplateControlsDisabled(reportGeneratingForThisRun || reportPhase === "running" || reportState === "generating");
    }
  });
}

if (reportTplPreviewBtn) {
  reportTplPreviewBtn.addEventListener("click", async () => {
    reportTemplateMutationInFlight = true;
    setReportTemplateControlsDisabled(true);
    try {
      const draft = reportTemplateDraftFromForm();
      const rsp = await fetch("/api/report/templates/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ draft }),
      });
      if (!rsp.ok) {
        throw new Error(await responseDetail(rsp, `Preview failed: ${rsp.status}`));
      }
      const data = await rsp.json();
      if (reportTplPreviewEl) {
        reportTplPreviewEl.textContent = String(data.rendered_background_prompt || "");
      }
      setReportTemplateEditorOpen(true);
    } catch (err) {
      showError(err.message || String(err));
    } finally {
      reportTemplateMutationInFlight = false;
      const sid = String((currentSnapshot && (currentSnapshot.session_id || currentSnapshot.run_id)) || "").trim();
      const reportGeneratingForThisRun = isReportGeneratingForRun(sid);
      const reportState = String((currentSnapshot && currentSnapshot.report_state) || "").toLowerCase();
      const tree = currentSnapshot && currentSnapshot.tree && typeof currentSnapshot.tree === "object" ? currentSnapshot.tree : {};
      const reportPhase = String((currentSnapshot && currentSnapshot.report_status) || tree.report_status || "pending").toLowerCase();
      setReportTemplateControlsDisabled(reportGeneratingForThisRun || reportPhase === "running" || reportState === "generating");
    }
  });
}

if (downloadBtn) {
  downloadBtn.addEventListener("click", () => {
  if (!currentRunId) return;
  const idx = Number(selectedReportVersionIndex || 0);
  if (Number.isFinite(idx) && idx > 0) {
    window.location.href = `/api/runs/${currentRunId}/report/download?version_index=${Math.floor(idx)}`;
    return;
  }
  window.location.href = `/api/runs/${currentRunId}/report/download`;
  });
}

/* ── Report modal open / close / download ── */
viewReportBtn.addEventListener("click", () => {
  if (viewReportBtn.disabled) return;
  reportModal.classList.remove("hidden");
});
reportModalCloseBtn.addEventListener("click", () => {
  reportModal.classList.add("hidden");
});
reportModal.addEventListener("click", (e) => {
  if (e.target === reportModal) reportModal.classList.add("hidden");
});
reportModalDownloadBtn.addEventListener("click", () => {
  if (!currentRunId) return;
  const idx = Number(selectedReportVersionIndex || 0);
  if (Number.isFinite(idx) && idx > 0) {
    window.location.href = `/api/runs/${currentRunId}/report/download?version_index=${Math.floor(idx)}`;
    return;
  }
  window.location.href = `/api/runs/${currentRunId}/report/download`;
});

pauseBtn.addEventListener("click", async () => {
  try {
    await pauseRun();
  } catch (err) {
    showError(err.message || String(err));
  }
});

resumeBtn.addEventListener("click", async () => {
  try {
    await resumeRun();
  } catch (err) {
    showError(err.message || String(err));
  }
});

abortBtn.addEventListener("click", async () => {
  try {
    await abortRun();
  } catch (err) {
    showError(err.message || String(err));
  }
});

if (swapBatchCanvasBtn) {
  swapBatchCanvasBtn.addEventListener("click", async () => {
    try {
      await planningSwapBatch();
    } catch (err) {
      showError(err.message || String(err));
    }
  });
}

if (planningCommitCanvasBtn) {
  planningCommitCanvasBtn.addEventListener("click", async () => {
    const phase = planningPhase(currentSnapshot);
    const pState = planningState(currentSnapshot);
    if (!currentRunId || phase !== "planning" || pState !== "review") return;
    try {
      await planningCommit();
    } catch (err) {
      showError(err.message || String(err));
    }
  });
}

canvasEl.addEventListener("click", async (evt) => {
  const target = evt && evt.target ? evt.target : null;
  if (!target || typeof target.closest !== "function") return;
  const btn = target.closest("button[data-planning-action][data-node-id]");
  if (!btn) return;
  const action = String(btn.getAttribute("data-planning-action") || "").trim();
  const nodeId = String(btn.getAttribute("data-node-id") || "").trim();
  if (!action || !nodeId) return;
  try {
    if (action === "pin") {
      await planningSetPin(nodeId, true);
    } else if (action === "unpin") {
      await planningSetPin(nodeId, false);
    } else if (action === "depth") {
      await planningDepthPlus(nodeId);
    }
  } catch (err) {
    showError(err.message || String(err));
  }
});

if (reportPrevBtn) {
  reportPrevBtn.addEventListener("click", async () => {
    try {
      const idx = Number(selectedReportVersionIndex || 0);
      if (idx > 1) {
        await selectReportVersion(idx - 1);
      }
    } catch (err) {
      showError(err.message || String(err));
    }
  });
}

if (reportNextBtn) {
  reportNextBtn.addEventListener("click", async () => {
    try {
      const idx = Number(selectedReportVersionIndex || 0);
      const versions = normalizeReportVersions(currentSnapshot || {});
      if (idx >= 1 && idx < versions.length) {
        await selectReportVersion(idx + 1);
      }
    } catch (err) {
      showError(err.message || String(err));
    }
  });
}

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

/* ── Depth dropdown ── */
const depthDropdownBtn = document.getElementById("depthDropdownBtn");
const depthDropdownValue = document.getElementById("depthDropdownValue");
const depthDropdownMenu = document.getElementById("depthDropdownMenu");

function syncDepthDropdownActive() {
  const cur = String(maxDepthEl.value);
  for (const opt of depthDropdownMenu.querySelectorAll(".depth-dropdown-option")) {
    opt.classList.toggle("is-active", opt.getAttribute("data-depth") === cur);
  }
}

depthDropdownBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  const isOpen = !depthDropdownMenu.classList.contains("hidden");
  depthDropdownMenu.classList.toggle("hidden", isOpen);
  if (!isOpen) syncDepthDropdownActive();
});

depthDropdownMenu.addEventListener("click", (e) => {
  const opt = e.target.closest(".depth-dropdown-option");
  if (!opt) return;
  const val = opt.getAttribute("data-depth");
  maxDepthEl.value = val;
  depthDropdownValue.textContent = val;
  depthDropdownMenu.classList.add("hidden");
  syncDepthDropdownActive();
});

document.addEventListener("click", () => {
  depthDropdownMenu.classList.add("hidden");
});

async function bootstrapFromUrl() {
  const runId = readRunIdFromUrl();
  try {
    try {
      await fetchReportTemplates();
    } catch (err) {
      showError(err.message || String(err));
    }
    await fetchSessions();
    const selected = runId ? String(runId) : "";
    if (!selected) {
      resetWorkspaceForNewSession();
      await scheduleContextRefresh();
      return;
    }
    await openSession(selected, { updateUrl: true });
  } catch (err) {
    showError(err.message || String(err));
  }
}

setContextEnabled(false);
clearContextPane("No context files uploaded.");
if (reportTemplateSelectEl) {
  renderReportTemplateSelect();
}
bootstrapFromUrl();
initSessionSyncListeners();

function disableAutoFollowFromUserAction(evt) {
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
canvasEl.addEventListener(
  "scroll",
  () => {
    if (skipNextCanvasScrollDisable) {
      skipNextCanvasScrollDisable = false;
      return;
    }
    autoFollowActiveNode = false;
  },
  { passive: true, capture: true }
);

window.addEventListener("wheel", disableAutoFollowFromUserAction, { passive: true });
window.addEventListener("touchstart", disableAutoFollowFromUserAction, { passive: true });

window.addEventListener("keydown", (evt) => {
  const key = String(evt.key || "");
  if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(key)) {
    disableAutoFollowFromUserAction(evt);
  }
});

if (sessionListEl) {
  sessionListEl.addEventListener("click", async (evt) => {
    const target = evt.target;
    if (!(target instanceof Element)) return;
    const deleteBtn = target.closest("button[data-action='delete']");
    const row = target.closest(".session-row");
    const sid = String(
      (deleteBtn && deleteBtn.getAttribute("data-session-id"))
      || (row && row.getAttribute("data-session-id"))
      || ""
    );
    if (!sid) return;
    try {
      if (deleteBtn) {
        const ok = window.confirm("Delete this session? This removes its saved state.");
        if (!ok) return;
        const rsp = await fetch(`/api/sessions/${sid}`, { method: "DELETE" });
        if (rsp.status === 409) {
          const body = await rsp.json().catch(() => ({}));
          throw new Error(String(body.detail || "Cannot delete running session."));
        }
        if (!rsp.ok) throw new Error(`Delete failed: ${rsp.status}`);
        stagedContextBySessionId.delete(sid);
        if (currentRunId === sid) {
          resetWorkspaceForNewSession();
        }
        await fetchSessions();
        emitSessionMutation("delete", sid);
        return;
      }
      if (row) {
        await openSession(sid, { updateUrl: true });
      }
    } catch (err) {
      const detail = err && err.message ? err.message : String(err);
      showError(`Session action failed: ${detail}`);
    }
  });
}

setSessionsRailCollapsed(loadSessionsRailCollapsed(), false);
if (sessionRailToggleBtn) {
  sessionRailToggleBtn.addEventListener("click", () => {
    const collapsed = !!(document && document.body && document.body.classList.contains("sessions-collapsed"));
    setSessionsRailCollapsed(!collapsed, true);
  });
}

/* ── Mobile session toggle (collapse / expand) ── */
const mobileSessionToggle = document.getElementById("mobileSessionToggle");
const sessionRailEl = document.querySelector(".session-rail");
function closeMobileDrawer() {
  if (sessionRailEl) sessionRailEl.classList.remove("mobile-expanded");
}
if (mobileSessionToggle) {
  mobileSessionToggle.addEventListener("click", () => {
    if (sessionRailEl) sessionRailEl.classList.toggle("mobile-expanded");
  });
}

if (refreshSessionsBtn) {
  refreshSessionsBtn.addEventListener("click", async () => {
    try {
      await fetchSessions();
    } catch (err) {
      showError(err.message || String(err));
    }
  });
}

if (newSessionBtn) {
  newSessionBtn.addEventListener("click", async () => {
    if (currentRunId) {
      clearStartupParseState(currentRunId);
    }
    resetWorkspaceForNewSession();
    if (taskEl && window.matchMedia && window.matchMedia("(max-width: 768px)").matches) {
      taskEl.scrollIntoView({ behavior: "smooth", block: "start" });
      taskEl.focus({ preventScroll: true });
    }
    try {
      await scheduleContextRefresh();
      await fetchSessions();
    } catch (err) {
      showError(err.message || String(err));
      renderSessions();
    }
  });
}

if (contextUploadBtn) {
  contextUploadBtn.addEventListener("click", async () => {
    if (contextMutationInFlight || contextInputLocked) return;
    if (contextUploadInput) {
      contextUploadInput.click();
    }
  });
}

if (contextAddBtn) {
  contextAddBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (contextMutationInFlight || contextInputLocked) return;
    if (contextUploadInput) {
      contextUploadInput.click();
    }
  });
}

if (inlineContextFilesEl) {
  inlineContextFilesEl.addEventListener("click", async (evt) => {
    const delBtn = evt.target.closest(".ctx-file-del[data-delete-fid]");
    if (!delBtn) return;
    evt.stopPropagation();
    const fid = String(delBtn.getAttribute("data-delete-fid") || "");
    if (!fid) return;
    try {
      await deleteContextFile(fid);
    } catch (err) {
      showError(err && err.message ? err.message : String(err));
    }
  });
}

if (contextUploadInput) {
  contextUploadInput.addEventListener("change", async () => {
    refreshContextUploadButtonState();
    const files = contextUploadInput && contextUploadInput.files ? contextUploadInput.files : null;
    if (!files || files.length === 0) return;
    try {
      await uploadContextFiles(files);
      contextUploadInput.value = "";
      refreshContextUploadButtonState();
    } catch (err) {
      const msg = err && err.message ? err.message : String(err);
      if (msg === "Context update already in progress. Please wait.") {
        showTransientError(msg, 2000);
      } else {
        showError(msg);
      }
    }
  });
}

if (contextFilesEl) {
  contextFilesEl.addEventListener("click", async (evt) => {
    const target = evt.target;
    if (!(target instanceof HTMLElement)) return;
    const action = String(target.getAttribute("data-action") || "");
    const fid = String(target.getAttribute("data-file-id") || "");
    if (!action || !fid) return;
    try {
      if (action === "download") {
        window.location.href = `${contextBasePath()}/files/${encodeURIComponent(fid)}/download`;
        return;
      }
      if (action === "delete") {
        const ok = window.confirm("Delete this context file?");
        if (!ok) return;
        await deleteContextFile(fid);
        return;
      }
      if (action === "replace") {
        const picker = document.createElement("input");
        picker.type = "file";
        picker.multiple = false;
        picker.onchange = async () => {
          const f = picker.files && picker.files[0] ? picker.files[0] : null;
          if (!f) return;
          try {
            await replaceContextFile(fid, f);
          } catch (err) {
            const msg = err && err.message ? err.message : String(err);
            if (msg === "Context update already in progress. Please wait.") {
              showTransientError(msg, 2000);
            } else {
              showError(msg);
            }
          }
        };
        picker.click();
      }
    } catch (err) {
      const msg = err && err.message ? err.message : String(err);
      if (msg === "Context update already in progress. Please wait.") {
        showTransientError(msg, 2000);
      } else {
        showError(msg);
      }
    }
  });
}
