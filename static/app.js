/* NetGuard browser application. API data is always inserted with DOM nodes. */
const $ = (id) => document.getElementById(id);
const els = {
  file: $("dataset-file"), dropZone: $("drop-zone"), scanButton: $("scan-button"),
  uploadPrompt: $("upload-prompt"), uploadName: $("upload-file-name"), scanStatus: $("scan-status"),
  detectorStatus: $("detector-status"), result: $("scan-result"), verdict: $("verdict-card"),
  verdictLabel: $("verdict-label"), verdictSummary: $("verdict-summary"), scanned: $("stat-scanned"),
  flagged: $("stat-flagged"), flaggedRate: $("stat-flagged-rate"), normal: $("stat-normal"), model: $("stat-model"),
  labeled: $("labeled-evaluation"), accuracy: $("metric-accuracy"), precision: $("metric-precision"), recall: $("metric-recall"),
  alertsBody: $("alerts-body"), download: $("download-results"), explanation: $("explanation-panel"),
  explanationStatus: $("explanation-status"), explanationContent: $("explanation-content"), explanationSubtitle: $("explanation-subtitle"),
  explanationDecision: $("explanation-decision"), explanationScore: $("explanation-score"), evidenceBody: $("evidence-body"), plainExplanation: $("plain-explanation"),
  validationMetrics: $("validation-metrics"), selectedBadge: $("selected-model-badge"), validationMethod: $("validation-method"),
  comparisonBody: $("comparison-body"), baselineMetrics: $("baseline-metrics"),
};

const api = async (path, options = {}) => {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try { message = (await response.json()).detail || message; } catch (_) { /* ignore */ }
    throw new Error(Array.isArray(message) ? message.join(" ") : String(message));
  }
  return response.json();
};
const percent = (value) => `${(Number(value) * 100).toFixed(1)}%`;
const number = (value) => Number(value).toLocaleString();
const labelForModel = (value) => ({ random_forest: "Random Forest", logistic_regression: "Logistic Regression", xgboost: "XGBoost" }[value] || value || "—");
const plainSignal = (feature) => {
  const labels = {
    "Flow Duration": "how long the connection lasted",
    "Flow Bytes/s": "the rate at which data moved",
    "Flow Packets/s": "the rate at which packets arrived",
    "Src Port": "the source port used",
    "Dst Port": "the destination port used",
    "Protocol": "the network protocol used",
    "Total Fwd Packet": "the number of packets sent",
    "Total Bwd packets": "the number of reply packets",
  };
  if (labels[feature]) return labels[feature];
  if (feature.includes("IAT")) return "the timing gaps between packets";
  if (feature.includes("Packet Length")) return "the packet-size pattern";
  if (feature.includes("Flags")) return "the connection-control flags";
  if (feature.includes("Header Length")) return "the packet-header size";
  return feature.replaceAll("_", " ").toLowerCase();
};

function setStatus(target, message, error = false) {
  target.textContent = message;
  target.classList.toggle("error", error);
}

function cell(row, text, className = "") {
  const item = document.createElement("td");
  item.textContent = text == null || text === "" ? "—" : String(text);
  if (className) item.className = className;
  row.append(item);
  return item;
}

function setView(view) {
  const validation = view === "validation";
  $("scan-view").classList.toggle("hidden", validation);
  $("validation-view").classList.toggle("hidden", !validation);
  document.querySelectorAll("[data-view-link]").forEach((link) => link.classList.toggle("active", link.dataset.viewLink === view));
  if (validation) loadValidation();
}

function updateFile(file) {
  els.scanButton.disabled = !file;
  els.uploadPrompt.textContent = file ? "Dataset ready to scan" : "Choose a CSV file";
  els.uploadName.textContent = file ? `${file.name} · ${Math.max(1, Math.ceil(file.size / 1024)).toLocaleString()} KB` : "or drag it here";
  setStatus(els.scanStatus, "");
}

async function refreshDetector() {
  try {
    const state = await api("/detections/cic/status");
    if (!state.ready) throw new Error(state.detail);
    els.detectorStatus.className = "service-status ready";
    els.detectorStatus.replaceChildren();
    const dot = document.createElement("span"); dot.className = "status-dot"; dot.setAttribute("aria-hidden", "true");
    els.detectorStatus.append(dot, document.createTextNode(`${labelForModel(state.model)} detector ready`));
  } catch (error) {
    els.detectorStatus.className = "service-status error";
    els.detectorStatus.replaceChildren();
    const dot = document.createElement("span"); dot.className = "status-dot"; dot.setAttribute("aria-hidden", "true");
    els.detectorStatus.append(dot, document.createTextNode("Detector unavailable"));
    els.scanButton.disabled = true;
    setStatus(els.scanStatus, error.message, true);
  }
}

function renderAlerts(alerts) {
  els.alertsBody.replaceChildren();
  if (!alerts.length) {
    const row = document.createElement("tr");
    const item = cell(row, "No flow data was returned."); item.colSpan = 5; els.alertsBody.append(row); return;
  }
  for (const alert of alerts) {
    const row = document.createElement("tr");
    cell(row, `#${alert.record}`);
    const decision = document.createElement("span"); decision.className = `decision ${alert.prediction}`; decision.textContent = alert.prediction === "attack" ? "Suspicious" : "Normal";
    const decisionCell = document.createElement("td"); decisionCell.append(decision); row.append(decisionCell);
    cell(row, percent(alert.anomaly_score), "score");
    const connection = [alert["Src IP"], alert["Src Port"], alert["Dst IP"], alert["Dst Port"]].filter(Boolean).join("  →  ");
    cell(row, connection || "Flow metadata unavailable");
    const actionCell = document.createElement("td"); const button = document.createElement("button");
    button.type = "button"; button.className = "row-button"; button.textContent = "Explain"; button.addEventListener("click", () => explainAlert(alert));
    actionCell.append(button); row.append(actionCell); els.alertsBody.append(row);
  }
}

function renderResult(result) {
  const suspicious = result.dataset_assessment === "suspicious_activity_detected";
  const rate = result.records_scanned ? result.attacks_flagged / result.records_scanned : 0;
  els.result.classList.remove("hidden");
  els.verdict.className = `verdict-card ${suspicious ? "warning" : "safe"}`;
  els.verdictLabel.textContent = suspicious ? "Suspicious activity detected" : "No suspicious activity detected";
  els.verdictSummary.textContent = suspicious
    ? `${number(result.attacks_flagged)} of ${number(result.records_scanned)} flows were flagged for review. Inspect the highest-scoring connections below to understand the evidence.`
    : `No flows crossed the detector’s review threshold in this dataset of ${number(result.records_scanned)} connections.`;
  els.scanned.textContent = number(result.records_scanned); els.flagged.textContent = number(result.attacks_flagged);
  els.flaggedRate.textContent = `${percent(rate)} of uploaded flows`; els.normal.textContent = number(result.normal_records); els.model.textContent = labelForModel(result.model);
  els.download.href = result.download_url; els.download.download = "intrusion-scan-results.csv";
  if (result.labeled_evaluation) {
    const metrics = result.labeled_evaluation;
    els.accuracy.textContent = percent(metrics.accuracy);
    els.precision.textContent = metrics.attack_metrics_available === false ? "Not applicable" : percent(metrics.precision_attack);
    els.recall.textContent = metrics.attack_metrics_available === false ? "Not applicable" : percent(metrics.recall_attack);
    els.labeled.classList.remove("hidden");
  } else { els.labeled.classList.add("hidden"); }
  renderAlerts(result.top_alerts || []);
  els.explanation.classList.add("hidden");
  els.result.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function scanDataset() {
  const file = els.file.files[0]; if (!file) return;
  els.scanButton.disabled = true; els.scanButton.textContent = "Scanning…";
  setStatus(els.scanStatus, "Uploading dataset and evaluating network flows…");
  try {
    const result = await api("/detections/cic/scan", { method: "POST", headers: { "Content-Type": "text/csv" }, body: file });
    setStatus(els.scanStatus, "Scan complete."); renderResult(result);
  } catch (error) {
    setStatus(els.scanStatus, error.message, true);
  } finally { els.scanButton.disabled = false; els.scanButton.textContent = "Scan dataset"; }
}

async function explainAlert(alert) {
  els.explanation.classList.remove("hidden"); els.explanationContent.classList.add("hidden");
  els.explanationSubtitle.textContent = `Record #${alert.record} · generating feature evidence`;
  setStatus(els.explanationStatus, "Preparing explanation…");
  els.explanation.scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const result = await api("/detections/cic/explain", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(alert.features) });
    setStatus(els.explanationStatus, ""); els.explanationDecision.textContent = result.prediction === "attack" ? "Suspicious flow" : "Normal flow";
    els.explanationScore.textContent = `Attack score ${percent(result.anomaly_score)} · Review threshold ${percent(result.threshold)}`;
    const strongest = result.evidence.filter((item) => item.shap_value > 0).slice(0, 3).map((item) => plainSignal(item.feature));
    els.plainExplanation.textContent = strongest.length
      ? `In simple terms: this connection stood out because of ${strongest.join(", ")}. Together, these patterns raised the model’s suspicion score.`
      : "In simple terms: the model did not find strong traffic patterns that raised its suspicion score for this connection.";
    els.evidenceBody.replaceChildren();
    for (const evidence of result.evidence) {
      const row = document.createElement("tr"); cell(row, evidence.feature); cell(row, evidence.value);
      cell(row, evidence.direction, evidence.shap_value >= 0 ? "effect-up" : "effect-down"); els.evidenceBody.append(row);
    }
    els.explanationContent.classList.remove("hidden");
  } catch (error) { setStatus(els.explanationStatus, error.message, true); }
}

function metricCard(title, value, note = "") {
  const card = document.createElement("article"); card.className = "stat-card";
  const label = document.createElement("span"); label.textContent = title; const strong = document.createElement("strong"); strong.textContent = value;
  card.append(label, strong); if (note) { const small = document.createElement("small"); small.textContent = note; card.append(small); } return card;
}

function renderCicValidation(report) {
  const selected = report.selected_model_by_validation; const selectedResult = report.models[selected];
  els.selectedBadge.textContent = `${labelForModel(selected)} selected`;
  els.validationMetrics.replaceChildren(
    metricCard("Accuracy", percent(selectedResult.test.accuracy), "Held-out Friday capture"),
    metricCard("Attack precision", percent(selectedResult.test.precision_attack), "Flagged flows that were attacks"),
    metricCard("Attack recall", percent(selectedResult.test.recall_attack), "Known attacks detected"),
    metricCard("Attack F1", percent(selectedResult.test.f1_attack), "Precision and recall combined"),
  );
  els.validationMethod.textContent = `${number(report.evaluation.training_records)} training flows · ${number(report.evaluation.validation_records)} validation flows · ${number(report.evaluation.test_records)} held-out test flows. Model selection used Thursday only; Friday was reserved for final testing.`;
  els.comparisonBody.replaceChildren();
  for (const key of ["logistic_regression", "random_forest", "xgboost"]) {
    const model = report.models[key]; if (!model) continue; const row = document.createElement("tr");
    cell(row, labelForModel(key)); cell(row, percent(model.test.accuracy)); cell(row, percent(model.test.precision_attack)); cell(row, percent(model.test.recall_attack)); cell(row, percent(model.test.f1_attack)); cell(row, key === selected ? "Selected" : "Compared", key === selected ? "score" : ""); els.comparisonBody.append(row);
  }
}

function renderBaseline(report) {
  const metrics = report.metrics;
  if (!metrics) return;
  els.baselineMetrics.replaceChildren();
  [["Accuracy", metrics.accuracy], ["Attack precision", metrics.precision_anomaly], ["Attack recall", metrics.recall_anomaly], ["Attack F1", metrics.f1_anomaly]].forEach(([title, value]) => {
    const item = document.createElement("span"); item.textContent = `${title} `; const strong = document.createElement("b"); strong.textContent = percent(value); item.append(strong); els.baselineMetrics.append(item);
  });
}

let validationLoaded = false;
async function loadValidation() {
  if (validationLoaded) return;
  try { renderCicValidation(await api("/evaluation/cic-ids2017")); } catch (error) { els.validationMethod.textContent = error.message; }
  try { renderBaseline(await api("/evaluation/holdout")); } catch (_) { els.baselineMetrics.textContent = "Saved NSL-KDD baseline results are unavailable."; }
  validationLoaded = true;
}

els.file.addEventListener("change", () => updateFile(els.file.files[0]));
els.dropZone.addEventListener("dragover", (event) => { event.preventDefault(); els.dropZone.classList.add("dragging"); });
els.dropZone.addEventListener("dragleave", () => els.dropZone.classList.remove("dragging"));
els.dropZone.addEventListener("drop", (event) => { event.preventDefault(); els.dropZone.classList.remove("dragging"); const [file] = event.dataTransfer.files; if (file) { const data = new DataTransfer(); data.items.add(file); els.file.files = data.files; updateFile(file); } });
els.scanButton.addEventListener("click", scanDataset);
window.addEventListener("hashchange", () => setView(location.hash === "#validation" ? "validation" : "scan"));
setView(location.hash === "#validation" ? "validation" : "scan"); refreshDetector();
