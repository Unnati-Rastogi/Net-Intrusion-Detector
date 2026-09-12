/* NIDS dashboard logic.
 *
 * Everything here talks to the existing FastAPI endpoints:
 *   GET  /health, /features, /sample, /demo/test-rows
 *   POST /predict, /predict/batch
 * No build step, no framework.
 */

const API = {
  async get(path) {
    const res = await fetch(path);
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

  async post(path, payload) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
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
};

const els = {
  health: document.getElementById("health"),
  form: document.getElementById("flow-form"),
  formError: document.getElementById("form-error"),
  btnSample: document.getElementById("btn-sample"),
  btnPredict: document.getElementById("btn-predict"),
  btnBatch: document.getElementById("btn-batch"),
  batchSummary: document.getElementById("batch-summary"),
  recentBody: document.querySelector("#recent-table tbody"),
};

let recentCount = 0;
const RECENT_LIMIT = 20;

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
  els.form.innerHTML = "";
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

/* --- recent predictions table ------------------------------------------ */

function addRecent(result) {
  recentCount += 1;
  const tr = document.createElement("tr");
  const verdict = result.prediction;
  tr.innerHTML = `
    <td>${recentCount}</td>
    <td class="tag tag-${verdict}">${verdict.toUpperCase()}</td>
    <td>${(result.confidence * 100).toFixed(1)}%</td>
    <td>${result.probability_normal.toFixed(4)}</td>
    <td>${result.probability_anomaly.toFixed(4)}</td>`;
  els.recentBody.prepend(tr);
  while (els.recentBody.rows.length > RECENT_LIMIT) {
    els.recentBody.deleteRow(-1);
  }
}

/* --- predict ------------------------------------------------------------ */

function setFormError(message) {
  if (message) {
    els.formError.textContent = message;
    els.formError.classList.remove("hidden");
  } else {
    els.formError.textContent = "";
    els.formError.classList.add("hidden");
  }
}

async function predictCurrent() {
  els.btnPredict.disabled = true;
  setFormError("");
  try {
    const result = await API.post("/predict", { features: collectForm() });
    showResult(result);
    addRecent(result);
  } catch (err) {
    setFormError("Prediction failed:\n" + err.message);
  } finally {
    els.btnPredict.disabled = false;
  }
}

function showResult(result) {
  let banner = document.getElementById("result-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "result-banner";
    els.form.after(banner);
  }
  const isAnomaly = result.prediction === "anomaly";
  banner.className = "result " + (isAnomaly ? "result-anomaly" : "result-normal");
  banner.innerHTML = `
    <span class="verdict verdict-${result.prediction}">
      ${isAnomaly ? "⚠ ANOMALY" : "✔ NORMAL"}
    </span>
    <span class="probs">
      confidence ${(result.confidence * 100).toFixed(1)}% -
      P(normal) ${result.probability_normal.toFixed(4)} -
      P(anomaly) ${result.probability_anomaly.toFixed(4)}
    </span>`;
}

/* --- batch demo --------------------------------------------------------- */

async function runBatchDemo() {
  els.btnBatch.disabled = true;
  els.batchSummary.textContent = "fetching test rows…";
  try {
    const demo = await API.get("/demo/test-rows?limit=10");
    const batch = await API.post("/predict/batch", { features: demo.rows });
    const anomalies = batch.predictions.filter(
      (p) => p.prediction === "anomaly"
    ).length;
    const avgConf =
      batch.predictions.reduce((sum, p) => sum + p.confidence, 0) /
      batch.predictions.length;
    els.batchSummary.innerHTML = `
      <p class="batch-line">Sent <b>${batch.count}</b> real rows from
        Test_data.csv to POST /predict/batch.</p>
      <p class="batch-verdicts batch-line">
        ${batch.count - anomalies} NORMAL - ${anomalies} ANOMALY
      </p>
      <p class="batch-line">Average confidence: ${(avgConf * 100).toFixed(1)}%</p>`;
    for (const prediction of batch.predictions) addRecent(prediction);
  } catch (err) {
    els.batchSummary.innerHTML =
      `<p class="batch-line" style="color:var(--bad)">Batch demo failed: ${err.message}</p>`;
  } finally {
    els.btnBatch.disabled = false;
  }
}

/* --- init --------------------------------------------------------------- */

async function init() {
  try {
    await refreshHealth();
  } catch (_) {
    els.formError.textContent =
      "API is unreachable or the model is not loaded. Start it with: python run_api.py";
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
  els.btnPredict.addEventListener("click", predictCurrent);
  els.btnBatch.addEventListener("click", runBatchDemo);
  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    predictCurrent();
  });
}

init();
