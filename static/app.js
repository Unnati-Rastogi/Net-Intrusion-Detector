/* NIDS dashboard logic (analyst workflow).
 *
 * Everything here talks to the existing FastAPI endpoints:
 *   GET  /health, /features, /sample, /demo/test-rows, /alerts
 *   POST /analyze, /predict/batch
 * No build step, no framework. All API-derived values are rendered through
 * safe DOM APIs (textContent / createElement); API data never passes
 * through markup string concatenation, so injection is impossible.
 *
 * Terminology honesty: "anomaly score" is the model's P(anomaly) output;
 * "risk" is a model-derived prioritization score, not attack severity;
 * "P1..P4" order analyst review, they do not grade attack danger.
 */

const API = {
  async request(path, options) {
    const res = await fetch(path, options);
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        detail = body.feature_errors || body.detail || detail;
      } catch (_) { /* non-JSON error body */ }
      throw new Error(
        Array.isArray(detail) ? detail.join("\n") : String(detail)
      );
    }
    return res.json();
  },

  get(path) {
    return this.request(path);
  },

  post(path, payload) {
    return this.request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },
};

const els = {
  health: document.getElementById("health"),
  form: document.getElementById("flow-form"),
  formError: document.getElementById("form-error"),
  btnSample: document.getElementById("btn-sample"),
  btnAnalyze: document.getElementById("btn-analyze"),
  btnBatch: document.getElementById("btn-batch"),
  btnAlerts: document.getElementById("btn-alerts"),
  batchSummary: document.getElementById("batch-summary"),
  resultCard: document.getElementById("result-card"),
  whyCard: document.getElementById("why-card"),
  sourceCard: document.getElementById("source-card"),
  detVerdict: document.getElementById("det-verdict"),
  detScore: document.getElementById("det-score"),
  detProbs: document.getElementById("det-probs"),
  riskLevel: document.getElementById("risk-level"),
  riskScore: document.getElementById("risk-score"),
  priority: document.getElementById("priority"),
  priorityNote: document.getElementById("priority-note"),
  evidenceBody: document.querySelector("#evidence-table tbody"),
  evidenceNote: document.getElementById("evidence-note"),
  sourceLine: document.getElementById("source-line"),
  alertsError: document.getElementById("alerts-error"),
  alertsBody: document.querySelector("#alerts-table tbody"),
  alertsNote: document.getElementById("alerts-note"),
  alertFilterRow: document.getElementById("alert-filter-row"),
  alertFilter: document.getElementById("alert-filter"),
  alertCounts: document.getElementById("alert-counts"),
};

const EVIDENCE_SHOWN = 10;
const ALERTS_SHOWN = 100;
const PRIORITY_NOTES = {
  P1: "P1 - highest analyst review priority",
  P2: "P2 - high review priority",
  P3: "P3 - moderate review priority",
  P4: "P4 - review backlog (model did not flag this record)",
};

let allAlerts = [];

/* --- health banner ----------------------------------------------------- */

async function refreshHealth() {
  els.health.className = "health health-loading";
  els.health.textContent = "checking API health…";
  try {
    const h = await API.get("/health");
    if (!h.model_loaded) throw new Error("model not loaded");
    els.health.className = "health health-ok";
    els.health.textContent =
      `API healthy - model loaded (${h.feature_count} features, ` +
      `classes ${h.classes.join("/")})`;
  } catch (err) {
    els.health.className = "health health-bad";
    els.health.textContent = "API unhealthy: " + err.message;
    throw err;
  }
}

/* --- feature form ------------------------------------------------------ */

function defaultFor(column, meta) {
  if (meta.categorical.includes(column)) {
    const values = meta.categoryValues[column];
    return values && values.length ? values[0] : "";
  }
  if (/(_rate)$/.test(column)) return 0.0;      // rate features live in [0,1]
  if (["land", "urgent", "root_shell", "logged_in",
       "is_guest_login", "num_shells"].includes(column)) return 0;
  if (column === "same_srv_rate" || column === "dst_host_same_srv_rate") return 1.0;
  if (column === "dst_host_srv_count") return 255;
  if (column === "dst_host_count") return 30;
  if (column === "count" || column === "srv_count") return 5;
  return 0;
}

function buildForm(meta) {
  els.form.textContent = "";
  for (const column of meta.featureColumns) {
    const wrap = document.createElement("div");
    wrap.className = "field";
    const label = document.createElement("label");
    label.textContent = column;
    label.title = column;
    wrap.appendChild(label);

    let input;
    if (meta.categorical.includes(column)) {
      input = document.createElement("select");
      const values = meta.categoryValues[column] || [];
      for (const v of values) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        input.appendChild(opt);
      }
      if (!values.length) {
        // training CSV unavailable -> free-text fallback (model tolerates it)
        input = document.createElement("input");
        input.type = "text";
      }
    } else {
      input = document.createElement("input");
      input.type = "text";
      input.inputMode = "decimal";
    }
    input.name = column;
    input.value = defaultFor(column, meta);
    wrap.appendChild(input);
    els.form.appendChild(wrap);
  }
}

function collectForm() {
  const features = {};
  for (const el of els.form.elements) {
    if (!el.name) continue;
    if (el.tagName === "SELECT") {
      features[el.name] = el.value;                 // categorical: string
    } else if (/_rate$/.test(el.name)) {
      features[el.name] = Number(el.value) || 0;    // rate features: numeric
    } else {
      const n = Number(el.value);
      features[el.name] =
        el.value.trim() === "" ? 0
        : Number.isFinite(n) ? n
        : el.value;                                 // bad input -> API 422
    }
  }
  return features;
}

function loadSample(sample) {
  for (const [column, value] of Object.entries(sample)) {
    const el = els.form.elements[column];
    if (el) el.value = value;
  }
}

function setFormError(message) {
  if (message) {
    els.formError.textContent = message;
    els.formError.classList.remove("hidden");
  } else {
    els.formError.textContent = "";
    els.formError.classList.add("hidden");
  }
}

/* --- analyst result rendering ------------------------------------------ */

function renderWorkflow(result) {
  // Detection
  const isAnomaly = result.prediction === "anomaly";
  els.detVerdict.textContent = isAnomaly ? "ANOMALY" : "NORMAL";
  els.detVerdict.className =
    "block-value " + (isAnomaly ? "verdict-anomaly" : "verdict-normal");
  els.detScore.textContent =
    `anomaly score (model P(anomaly)): ${result.anomaly_score.toFixed(4)}`;
  els.detProbs.textContent =
    `P(normal) ${result.probability_normal.toFixed(4)} - ` +
    `P(anomaly) ${result.probability_anomaly.toFixed(4)}`;
  els.resultCard.classList.remove("hidden");

  // Risk (model-derived prioritization score)
  els.riskLevel.textContent = result.risk_level;
  els.riskLevel.className =
    "block-value level-" + result.risk_level.toLowerCase();
  els.riskScore.textContent = `risk score: ${result.risk_score.toFixed(1)} / 100`;
  els.resultCard.classList.remove("hidden");

  // Review priority
  els.priority.textContent = result.priority;
  els.priority.className =
    "block-value priority-" + result.priority.toLowerCase();
  els.priorityNote.textContent =
    PRIORITY_NOTES[result.priority] || "analyst review priority";

  // Why was this flagged? (SHAP attributions of the model output)
  els.evidenceBody.textContent = "";
  const rows = result.explanation.slice(0, EVIDENCE_SHOWN);
  for (const item of rows) {
    const tr = document.createElement("tr");
    const cells = [
      item.feature,
      String(item.value),
      (item.shap_value >= 0 ? "+" : "") + item.shap_value.toFixed(4),
      item.direction === "increases_anomaly_score"
        ? "increases anomaly score"
        : "decreases anomaly score",
    ];
    cells.forEach((text, i) => {
      const td = document.createElement("td");
      td.textContent = text;
      if (i === 2) {
        td.className = item.shap_value >= 0 ? "shap-pos" : "shap-neg";
      }
      tr.appendChild(td);
    });
    els.evidenceBody.appendChild(tr);
  }
  els.evidenceNote.textContent =
    `Showing the ${rows.length} strongest attributions of ` +
    `${result.explanation.length} features.`;
  els.whyCard.classList.remove("hidden");

  // Source lineage (only present when the record matches a stored sample)
  els.sourceLine.textContent = "";
  if (result.sample_id !== null && result.sample_id !== undefined) {
    const strong = document.createElement("strong");
    strong.textContent = "known report sample";
    els.sourceLine.appendChild(strong);
    els.sourceLine.appendChild(document.createTextNode(
      ` - Sample ID: ${result.sample_id}` +
      ` - Source row: ${result.source_row_id}` +
      " (zero-based row in Train_data.csv)"
    ));
  } else {
    els.sourceLine.appendChild(document.createTextNode(
      "ad-hoc record - no stored-report match; " +
      "sample ID / source row stay empty (never fabricated)"
    ));
  }
  els.sourceCard.classList.remove("hidden");
  els.resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* --- analyze (features or stored sample) -------------------------------- */

async function analyzeCurrent() {
  els.btnAnalyze.disabled = true;
  setFormError("");
  try {
    const result = await API.post("/analyze", { features: collectForm() });
    renderWorkflow(result);
  } catch (err) {
    setFormError("Analysis failed:\n" + err.message);
  } finally {
    els.btnAnalyze.disabled = false;
  }
}

async function analyzeStoredSample(sampleId) {
  setFormError("");
  try {
    const result = await API.post("/analyze", { sample_id: sampleId });
    renderWorkflow(result);
  } catch (err) {
    setFormError("Analysis failed:\n" + err.message);
  }
}

/* --- prioritized alert queue -------------------------------------------- */

function alertRow(alert) {
  const tr = document.createElement("tr");
  tr.className = "alert-row priority-" + alert.priority.toLowerCase();
  tr.title = "click to analyze this record";

  const cells = [
    alert.priority,
    alert.risk_level,
    alert.risk_score.toFixed(1),
    alert.anomaly_score.toFixed(4),
    alert.prediction,
    String(alert.sample_id),
    alert.source_row_id === null || alert.source_row_id === undefined
      ? "-"
      : String(alert.source_row_id),
  ];
  cells.forEach((text, i) => {
    const td = document.createElement("td");
    td.textContent = text;
    if (i === 0) td.className = "tag tag-" + alert.priority.toLowerCase();
    if (i === 4) td.className = "tag tag-" + alert.prediction;
    tr.appendChild(td);
  });
  tr.addEventListener("click", () => analyzeStoredSample(alert.sample_id));
  return tr;
}

function renderAlerts() {
  const filter = els.alertFilter.value;                 // 'ALL' | 'P1'..'P4'
  const filtered =
    filter === "ALL" ? allAlerts : allAlerts.filter((a) => a.priority === filter);
  const shown = filtered.slice(0, ALERTS_SHOWN);

  els.alertsBody.textContent = "";
  if (!shown.length) {
    const tr = document.createElement("tr");
    tr.className = "empty-row";
    const td = document.createElement("td");
    td.colSpan = 7;
    td.textContent = filter === "ALL"
      ? "no alerts loaded"
      : `no ${filter} alerts in the report`;
    tr.appendChild(td);
    els.alertsBody.appendChild(tr);
  }
  for (const alert of shown) els.alertsBody.appendChild(alertRow(alert));

  if (allAlerts.length) {
    const counts = allAlerts.reduce((acc, a) => {
      acc[a.priority] = (acc[a.priority] || 0) + 1;
      return acc;
    }, {});
    els.alertCounts.textContent = Object.keys(counts)
      .sort()
      .map((p) => `${p}=${counts[p]}`)
      .join("  ");
    els.alertsNote.textContent =
      `Showing ${shown.length} of ${filtered.length} ${filter === "ALL" ? "" : filter + " "}alerts` +
      (filtered.length > ALERTS_SHOWN ? ` (first ${ALERTS_SHOWN})` : "") +
      ". Click a row to analyze that record with full SHAP evidence.";
    els.alertsNote.hidden = false;
  }
}

async function loadAlerts() {
  els.btnAlerts.disabled = true;
  els.alertsError.classList.add("hidden");
  try {
    const report = await API.get("/alerts");
    allAlerts = report.alerts;
    els.alertFilterRow.hidden = false;
    renderAlerts();
  } catch (err) {
    els.alertsError.textContent = "Could not load alerts:\n" + err.message;
    els.alertsError.classList.remove("hidden");
  } finally {
    els.btnAlerts.disabled = false;
  }
}

/* --- fast batch demo (no SHAP) ------------------------------------------ */

function clearNode(node) {
  node.textContent = "";
  return node;
}

function batchLine(parent, text) {
  const p = document.createElement("p");
  p.className = "batch-line";
  p.textContent = text;
  parent.appendChild(p);
  return p;
}

async function runBatchDemo() {
  els.btnBatch.disabled = true;
  clearNode(els.batchSummary).textContent = "fetching test rows…";
  try {
    const demo = await API.get("/demo/test-rows?limit=10");
    const batch = await API.post("/predict/batch", { features: demo.rows });
    const anomalies = batch.predictions.filter(
      (p) => p.prediction === "anomaly"
    ).length;
    const avgConf =
      batch.predictions.reduce((sum, p) => sum + p.confidence, 0) /
      batch.predictions.length;
    clearNode(els.batchSummary);
    batchLine(els.batchSummary,
      `Sent ${batch.count} real rows from Test_data.csv to POST /predict/batch.`);
    batchLine(els.batchSummary,
      `${batch.count - anomalies} NORMAL - ${anomalies} ANOMALY`);
    batchLine(els.batchSummary,
      `Average confidence: ${(avgConf * 100).toFixed(1)}%`);
  } catch (err) {
    clearNode(els.batchSummary);
    const p = batchLine(els.batchSummary, "Batch demo failed: " + err.message);
    p.style.color = "var(--bad)";
  } finally {
    els.btnBatch.disabled = false;
  }
}

/* --- init --------------------------------------------------------------- */

async function init() {
  try {
    await refreshHealth();
  } catch (_) {
    setFormError(
      "API is unreachable or the model is not loaded. Start it with: python run_api.py"
    );
    els.formError.classList.remove("hidden");
    return;
  }
  const meta = await API.get("/features");
  buildForm({
    featureColumns: meta.feature_columns,
    categorical: meta.categorical_columns,
    categoryValues: meta.category_values,
  });
  const sample = await API.get("/sample");
  loadSample(sample.features);

  els.btnSample.addEventListener("click", () => {
    loadSample(sample.features);
    setFormError("");          // a restored sample invalidates old errors
  });
  els.btnAnalyze.addEventListener("click", analyzeCurrent);
  els.btnBatch.addEventListener("click", runBatchDemo);
  els.btnAlerts.addEventListener("click", loadAlerts);
  els.alertFilter.addEventListener("change", renderAlerts);
  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    analyzeCurrent();
  });
}

init();
