const taskEl = document.getElementById("task");
const maxDepthEl = document.getElementById("maxDepth");
const resultsPerQueryEl = document.getElementById("resultsPerQuery");
const startBtn = document.getElementById("startBtn");
const abortBtn = document.getElementById("abortBtn");
const statusPill = document.getElementById("statusPill");
const runMeta = document.getElementById("runMeta");
const errorBanner = document.getElementById("errorBanner");
const treeEl = document.getElementById("tree");
const reportEl = document.getElementById("report");
const partialReportBtn = document.getElementById("partialReportBtn");
const downloadBtn = document.getElementById("downloadBtn");
const usageEl = document.getElementById("usage");

let currentRunId = null;
let es = null;
let currentSnapshot = null;
let partialReportGenerating = false;

function setStatus(status) {
  statusPill.textContent = status || "unknown";
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

function esc(s) {
  return String(s || "").replace(/[&<>"]/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;"
  })[c]);
}

function renderTree(tree) {
  if (!tree) {
    treeEl.innerHTML = "<em>No run yet.</em>";
    return;
  }
  const rounds = tree.rounds || [];
  let html = `<ul>`;
  html += `<li><strong>Task:</strong> ${esc(tree.task || "")}</li>`;
  for (const r of rounds) {
    html += `<li><strong>Pass ${r.round}</strong><ul>`;
    if (Array.isArray(r.frontier_questions) && r.frontier_questions.length > 0) {
      html += `<li><small>Entry tasks: ${r.frontier_questions.map(esc).join(" | ")}</small></li>`;
    }
    for (const q of (r.questions || [])) {
      html += `<li>${esc(q.sub_question)} [depth=${Number(q.depth || 1)} status=${esc(q.status || "pending")}]<ul>`;
      for (const step of (q.query_steps || [])) {
        html += `<li>[${esc(step.status || "queued")}] ${esc(step.query)} | selected=${step.selected_results_count || 0} primary=${step.primary_count || 0}`;
        if (step.search_error) {
          html += ` | error=${esc(step.search_error)}`;
        }
        if (step.diagnostic) {
          const d = step.diagnostic;
          html += `<br><small>diagnostic: ${esc(d.classification || "")} decision=${esc(d.decision || "")}`;
          if (d.prior_query) {
            html += ` prior="${esc(d.prior_query)}"`;
          }
          if (typeof d.similarity === "number" && d.similarity > 0) {
            html += ` sim=${d.similarity.toFixed(2)}`;
          }
          if (d.intent_mapped) {
            html += ` | mapped_intent sim=${Number(d.intent_map_similarity || 0).toFixed(2)}`;
          }
          const newToks = Array.isArray(d.new_tokens) ? d.new_tokens : [];
          const droppedToks = Array.isArray(d.dropped_tokens) ? d.dropped_tokens : [];
          if (newToks.length > 0) {
            html += ` new=[${newToks.map(esc).join(", ")}]`;
          }
          if (droppedToks.length > 0) {
            html += ` dropped=[${droppedToks.map(esc).join(", ")}]`;
          }
          if (d.is_broadened) {
            html += ` broadened ${Number(d.base_k || 0)}->${Number(d.effective_k || 0)}`;
          } else if (d.effective_k) {
            html += ` k=${Number(d.effective_k)}`;
          }
          html += `</small>`;
        }
        if (step.synthesis_summary) {
          html += `<br><small>${esc(step.synthesis_summary)}</small>`;
        }
        html += `</li>`;
      }
      if (q.node_sufficiency) {
        html += `<li><strong>Node sufficiency:</strong> ${q.node_sufficiency.is_sufficient ? "pass" : "fail"}`;
        if (q.node_sufficiency.reasoning) {
          html += `<br><small>reason: ${esc(q.node_sufficiency.reasoning)}</small>`;
        }
        html += `</li>`;
      }
      if (Array.isArray(q.children) && q.children.length > 0) {
        html += `<li><small>children: ${q.children.map(esc).join(" | ")}</small></li>`;
      }
      if (q.unresolved_reason) {
        html += `<li><small>unresolved reason: ${esc(q.unresolved_reason)}</small></li>`;
      }
      html += `</ul></li>`;
    }
    if (r.sufficiency) {
      const suff = r.sufficiency;
      html += `<li><strong>Pass sufficiency:</strong> ${suff.is_sufficient ? "pass" : "fail"}`;
      if (suff.reasoning) {
        html += `<br><small>reason: ${esc(suff.reasoning)}</small>`;
      }
      const gaps = Array.isArray(suff.gaps) ? suff.gaps : [];
      if (gaps.length > 0) {
        html += `<br><small>gaps: ${gaps.map(esc).join(" | ")}</small>`;
      }
      html += `</li>`;
    }
    html += `</ul></li>`;
  }
  html += `<li><strong>Report status:</strong> ${esc(tree.report_status || "pending")}</li>`;
  html += `</ul>`;
  treeEl.innerHTML = html;
}

function renderUsage(usage) {
  if (!usage) {
    usageEl.textContent = "";
    return;
  }
  usageEl.textContent = JSON.stringify(usage.total || {}, null, 2);
}

function applySnapshot(snap) {
  currentSnapshot = snap;
  setStatus(snap.status);
  runMeta.textContent = `Run ID: ${snap.run_id}`;
  renderTree(snap.tree);
  renderUsage(snap.token_usage);
  reportEl.textContent = snap.report_text || "";
  showError(snap.error || "");
  downloadBtn.disabled = !snap.report_file_path;
  abortBtn.disabled = !(snap.status === "queued" || snap.status === "running");
  partialReportBtn.disabled = !(
    !partialReportGenerating &&
    currentRunId &&
    !snap.report_file_path
  );
  partialReportBtn.textContent = partialReportGenerating
    ? "Generating Partial Report..."
    : "Generate Partial Report";
}

async function fetchSnapshot(runId) {
  const rsp = await fetch(`/api/runs/${runId}`);
  if (!rsp.ok) {
    throw new Error(`Failed to fetch snapshot: ${rsp.status}`);
  }
  const data = await rsp.json();
  applySnapshot(data);
}

function connectEvents(runId) {
  if (es) {
    es.close();
  }
  es = new EventSource(`/api/runs/${runId}/events`);
  es.onmessage = async () => {
    await fetchSnapshot(runId);
  };
  const names = [
    "run_started",
    "plan_created",
    "round_started",
    "sub_question_started",
    "queries_generated",
    "query_started",
    "query_diagnostic",
    "query_skipped_cached",
    "query_rerun_allowed",
    "query_broadened",
    "query_blocked_diminishing_returns",
    "search_completed",
    "synthesis_completed",
    "node_sufficiency_completed",
    "node_decomposed",
    "node_completed",
    "node_unresolved",
    "sufficiency_completed",
    "run_completed",
    "run_abort_requested",
    "run_aborted",
    "partial_report_generated",
    "run_failed",
  ];
  for (const n of names) {
    es.addEventListener(n, async () => {
      await fetchSnapshot(runId);
    });
  }
}

startBtn.addEventListener("click", async () => {
  const task = taskEl.value.trim();
  if (!task) {
    showError("Task is required.");
    return;
  }
  showError("");
  setStatus("queued");
  startBtn.disabled = true;
  try {
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
    await fetchSnapshot(currentRunId);
    connectEvents(currentRunId);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    startBtn.disabled = false;
  }
});

abortBtn.addEventListener("click", async () => {
  if (!currentRunId) return;
  try {
    const rsp = await fetch(`/api/runs/${currentRunId}/abort`, { method: "POST" });
    if (!rsp.ok) {
      throw new Error(`Abort failed: ${rsp.status}`);
    }
  } catch (err) {
    showError(err.message || String(err));
  }
});

partialReportBtn.addEventListener("click", async () => {
  if (!currentRunId || partialReportGenerating) return;
  partialReportGenerating = true;
  partialReportBtn.disabled = true;
  partialReportBtn.textContent = "Generating Partial Report...";
  showError("");
  try {
    const rsp = await fetch(`/api/runs/${currentRunId}/report/partial`, {
      method: "POST",
    });
    if (!rsp.ok) {
      throw new Error(`Partial report failed: ${rsp.status}`);
    }
    await fetchSnapshot(currentRunId);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    partialReportGenerating = false;
    if (!currentRunId) {
      partialReportBtn.disabled = true;
      partialReportBtn.textContent = "Generate Partial Report";
      return;
    }
    try {
      await fetchSnapshot(currentRunId);
    } catch (err) {
      // If snapshot refresh fails, still restore button from local state.
      const snap = currentSnapshot;
      const allow = !!(
        snap &&
        currentRunId &&
        !snap.report_file_path
      );
      partialReportBtn.disabled = !allow;
      partialReportBtn.textContent = "Generate Partial Report";
    }
  }
});

downloadBtn.addEventListener("click", () => {
  if (!currentRunId) return;
  window.location.href = `/api/runs/${currentRunId}/report/download`;
});

renderTree(null);
