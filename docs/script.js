(function () {
  "use strict";

  const API = (window.CHURN_API_BASE || "").replace(/\/$/, "");
  const SAMPLE_URL = "sample/Telco-Customer-Churn.csv";
  const $ = (id) => document.getElementById(id);
  const state = { file: null, runId: null };

  // ---------- small helpers ----------
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v; // only used for trusted, server-built <img>
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  const num = (n, d = 0) => Number(n).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (n, d = 1) => (Number(n) * 100).toFixed(d) + "%";
  const cell = (v) => (v == null ? "" : typeof v === "number" ? (Number.isInteger(v) ? num(v) : num(v, 2)) : v);

  function table(columns, rows, opts = {}) {
    const head = h("tr", {}, columns.map((c, i) => h("th", { class: opts.numeric && opts.numeric[i] ? "num" : "" }, c)));
    const body = rows.map((r, ri) =>
      h("tr", { class: opts.rowClass ? opts.rowClass(ri, r) : "" },
        r.map((v, i) => h("td", { class: opts.numeric && opts.numeric[i] ? "num" : "" }, opts.format ? opts.format(v, i, r) : cell(v)))));
    return h("div", { class: "tablewrap" }, h("table", {}, h("thead", {}, head), h("tbody", {}, body)));
  }
  const stat = (value, label, cls) => h("div", { class: "stat " + (cls || "") }, h("b", {}, value), h("span", {}, label));
  const card = (title, ...kids) => h("div", { class: "card" }, title ? h("h3", {}, title) : null, ...kids);
  const bullets = (items) => h("ul", { class: "bullets" }, items.map((t) => h("li", {}, t)));
  function chart(c) {
    const fig = h("figure", { class: "chart" });
    const img = h("img", { alt: c.title + ". " + c.caption, src: c.image });
    fig.append(img, h("figcaption", {}, c.caption));
    return fig;
  }

  // ---------- backend status ----------
  async function checkApi() {
    const el = $("api-status");
    try {
      const r = await fetch(API + "/api/health", { cache: "no-store" });
      if (!r.ok) throw new Error();
      el.textContent = "Backend connected";
      el.className = "api ok";
    } catch {
      el.textContent = "Backend not reachable";
      el.className = "api bad";
    }
  }

  // ---------- error / view helpers ----------
  function showError(msg) {
    const el = $("error");
    el.textContent = msg || "";
    el.hidden = !msg;
  }
  function show(view) {
    $("upload").hidden = view !== "upload";
    $("loading").hidden = view !== "loading";
    $("results").hidden = view !== "results";
  }

  // ---------- file selection ----------
  function setFile(file) {
    state.file = file;
    showError("");
    const drop = $("drop");
    drop.classList.toggle("has-file", !!file);
    $("drop-title").textContent = file ? file.name : "Choose a CSV file";
    $("drop-hint").textContent = file ? `${(file.size / 1024).toFixed(0)} KB. Click to choose a different file.` : "or drag it here. Up to 25 MB and 50,000 rows.";
    $("run").disabled = !file;
    $("target-row").hidden = true;
    if (file) previewFile();
  }

  async function previewFile() {
    const fd = new FormData();
    fd.append("file", state.file);
    try {
      const r = await fetch(API + "/api/preview", { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || "The file could not be read.");
      const sel = $("target");
      sel.replaceChildren(...data.binary_columns.map((c) => h("option", { value: c, selected: c === data.suggested_target }, c)));
      if (!data.binary_columns.length) throw new Error("No column with exactly two values was found, so there is no churn label to predict.");
      $("target-note").textContent = `${num(data.rows)} rows and ${data.columns.length} columns found. Only columns with exactly two values are listed.`;
      $("target-row").hidden = false;
    } catch (e) {
      showError(e instanceof TypeError ? "Cannot reach the Python backend. Start it with 'python app.py' or check config.js." : e.message);
      $("run").disabled = true;
    }
  }

  // ---------- run ----------
  const STEPS = ["Cleaning the data…", "Exploring churn patterns…", "Training four models…", "Scoring every customer…"];
  async function analyze() {
    if (!state.file) return;
    show("loading");
    let i = 0;
    $("loading-step").textContent = STEPS[0];
    const timer = setInterval(() => { i = Math.min(i + 1, STEPS.length - 1); $("loading-step").textContent = STEPS[i]; }, 1800);
    const fd = new FormData();
    fd.append("file", state.file);
    fd.append("target", $("target").value);
    try {
      const r = await fetch(API + "/api/analyze", { method: "POST", body: fd });
      const data = await r.json().catch(() => ({ error: "The server sent an unreadable response." }));
      if (!r.ok) throw new Error(data.error || "Analysis failed.");
      render(data);
      show("results");
      window.scrollTo({ top: 0 });
    } catch (e) {
      show("upload");
      showError(e instanceof TypeError ? "Cannot reach the Python backend. Start it with 'python app.py' or check config.js." : e.message);
    } finally {
      clearInterval(timer);
    }
  }

  // ---------- tabs ----------
  function selectTab(name) {
    document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    document.querySelectorAll(".tabpane").forEach((p) => (p.hidden = p.id !== "tab-" + name));
  }

  // ---------- renderers ----------
  function render(d) {
    state.runId = d.run_id;
    $("result-title").textContent = d.filename;
    $("result-sub").textContent = `${num(d.cleaning.rows_after)} customers analysed. Best model: ${d.evaluation.model}.`;
    $("dl-cleaned").href = `${API}/api/download/${d.run_id}/cleaned`;
    $("dl-scored").href = `${API}/api/download/${d.run_id}/scored`;
    $("tab-cleaning").replaceChildren(...cleaningTab(d.cleaning));
    $("tab-eda").replaceChildren(...edaTab(d.eda));
    $("tab-building").replaceChildren(...buildingTab(d.model_building));
    $("tab-scoring").replaceChildren(...scoringTab(d.scoring, d.run_id));
    $("tab-charts").replaceChildren(h("div", { class: "charts" }, d.eda.charts.map(chart)));
    $("tab-comparison").replaceChildren(...comparisonTab(d.comparison, d.run_id));
    $("tab-evaluation").replaceChildren(...evaluationTab(d.evaluation));
    selectTab("cleaning");
  }

  function cleaningTab(c) {
    return [
      h("div", { class: "stats" },
        stat(num(c.rows_before), "rows uploaded"),
        stat(num(c.rows_after), "rows after cleaning"),
        stat(num(c.columns_before) + " → " + num(c.columns_after), "columns"),
        stat(num(c.empty_cells_found), "empty cells found"),
        stat(num(c.missing_after), "missing values left")),
      card("What the cleaning did",
        h("ul", { class: "steps" }, c.steps.map((s) =>
          h("li", {}, h("div", { class: "tick " + (s.changed ? "changed" : "same"), "aria-hidden": "true" }, s.changed ? "✎" : "✓"),
            h("div", {}, h("b", {}, s.title), h("span", {}, s.detail)))))),
      c.dropped_columns.length || c.missing_filled.length ? h("div", { class: "two" },
        c.dropped_columns.length ? card("Columns dropped", table(["Column", "Reason"], c.dropped_columns.map((x) => [x.column, x.reason]))) : null,
        c.missing_filled.length ? card("Missing values filled", table(["Column", "Missing", "Filled with"], c.missing_filled.map((x) => [x.column, x.missing, x.method]), { numeric: [false, true, false] })) : null) : null,
      card("Cleaned dataset preview (first 10 rows)", h("div", { class: "tablewrap" }, previewTable(c.preview)), h("p", { class: "muted", style: "margin-top:10px" }, "Use the Download cleaned dataset button above for the full file.")),
    ];
  }
  function previewTable(p) {
    return h("table", {}, h("thead", {}, h("tr", {}, p.columns.map((c) => h("th", {}, c)))),
      h("tbody", {}, p.rows.map((r) => h("tr", {}, r.map((v) => h("td", {}, cell(v)))))));
  }

  function edaTab(e) {
    const o = e.overview;
    const out = [
      h("div", { class: "stats" },
        stat(num(o.rows), "customers"),
        stat(num(o.features), `features (${o.numeric} numeric, ${o.categorical} text)`),
        stat(num(o.churned), `churned (${o.churn_rate}%)`),
        stat(num(o.retained), "retained")),
      card("What stands out", bullets(e.insights)),
    ];
    if (e.numeric_summary.length) {
      out.push(card("Numeric columns",
        table(["Column", "Mean", "Std", "Min", "Median", "Max", "Mean (retained)", "Mean (churned)"],
          e.numeric_summary.map((r, i) => [r.column, r.mean, r.std, r.min, r.median, r.max, e.numeric_by_class[i].retained, e.numeric_by_class[i].churned]),
          { numeric: [false, true, true, true, true, true, true, true], format: (v, i) => (typeof v === "number" ? num(v, 2) : v) })));
    }
    if (e.category_rates.length) {
      out.push(h("div", { class: "two" }, e.category_rates.slice(0, 4).map((c) =>
        card("Churn rate by " + c.feature,
          table(["Value", "Customers", "Churn rate"], c.rows.map((r) => [r.value, r.customers, r.churn_rate]),
            { numeric: [false, true, true], format: (v, i) => (i === 2 ? v.toFixed(1) + "%" : cell(v)) })))));
    }
    return out;
  }

  function buildingTab(m) {
    return [
      h("div", { class: "stats" },
        stat(num(m.split.train), "training customers"),
        stat(num(m.split.test), "test customers"),
        stat(num(m.features.total), "features used"),
        stat(m.best_model, "selected model")),
      card("How the models were built", h("ol", { class: "bullets" }, m.steps.map((s) => h("li", {}, s)))),
      card("The four models", table(["Model", "What it is", "Settings"], m.models.map((x) => [x.name, x.description, x.settings]),
        { format: (v) => v })),
      h("p", { class: "muted" }, `Predicting "${m.target}" where ${m.positive_label} means the customer left. The test customers are never seen during training.`),
    ];
  }

  function scoringTab(s, runId) {
    return [
      h("div", { class: "risk" }, s.levels.map((l) =>
        h("div", { class: "stat " + l.level.split(" ")[0] }, h("b", {}, num(l.customers)), h("span", {}, `${l.level} · ${l.share}% · ${l.rule}`)))),
      h("div", { class: "stats" },
        stat(num(s.customers), "customers scored"),
        stat(s.average_probability + "%", "average churn probability"),
        stat(num(s.predicted_churners), "predicted to churn (50% or higher)")),
      chart(s.chart),
      card("15 customers most likely to churn",
        table(s.top_risk.columns, s.top_risk.rows, {
          numeric: s.top_risk.columns.map((c) => c === "Churn_Probability"),
          format: (v, i, row) => {
            const col = s.top_risk.columns[i];
            if (col === "Risk_Category") return h("span", { class: "badge " + String(v).split(" ")[0] }, v);
            if (col === "Churn_Probability") return (v * 100).toFixed(1) + "%";
            return cell(v);
          },
        }),
        h("p", { class: "muted", style: "margin-top:10px" }, "Download scored customers (above) for every customer with probability and risk level.")),
    ];
  }

  function comparisonTab(c, runId) {
    const cols = ["accuracy", "precision", "recall", "f1", "roc_auc"];
    const best = {};
    cols.forEach((k) => (best[k] = Math.max(...c.rows.map((r) => r[k]))));
    const t = table(["Rank", "Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"],
      c.rows.map((r) => [r.rank, r.model, ...cols.map((k) => r[k])]),
      {
        numeric: [true, false, true, true, true, true, true],
        rowClass: (i) => (i === 0 ? "best" : ""),
        format: (v, i, r) => (i < 2 ? v : (v * 100).toFixed(i === 6 ? 1 : 1) + (i === 6 ? "" : "")),
      });
    // ROC AUC is better shown as 0.844
    t.querySelectorAll("tbody tr").forEach((tr, ri) => {
      const tds = tr.querySelectorAll("td");
      tds[6].textContent = c.rows[ri].roc_auc.toFixed(3);
      cols.forEach((k, ci) => { if (c.rows[ri][k] === best[k]) tds[ci + 2].classList.add("best-cell"); });
    });
    return [
      card("Model comparison on the test set",
        h("p", {}, `${c.best_model} ranks first by ROC AUC. A blue value is the best in its column.`), t,
        h("p", { style: "margin-top:12px" }, h("a", { class: "btn", href: `${API}/api/download/${runId}/comparison` }, "Download comparison table"))),
      chart(c.roc_chart),
    ];
  }

  function evaluationTab(e) {
    const m = e.metrics, c = e.confusion;
    const maxImp = Math.max(...e.top_drivers.map((d) => d.importance));
    return [
      h("div", { class: "stats" },
        stat(pct(m.accuracy), "accuracy"), stat(pct(m.precision), "precision"),
        stat(pct(m.recall), "recall"), stat(pct(m.f1), "F1 score"), stat(m.roc_auc.toFixed(3), "ROC AUC")),
      h("div", { class: "two" },
        card("Confusion matrix: " + e.model,
          h("div", { class: "confusion" },
            h("span", {}), h("span", { class: "h" }, "Predicted retained"), h("span", { class: "h" }, "Predicted churned"),
            h("span", { class: "h" }, "Actually retained"),
            h("div", { class: "cell good" }, h("b", {}, num(c.tn)), h("span", {}, "correct")),
            h("div", { class: "cell bad" }, h("b", {}, num(c.fp)), h("span", {}, "false alarm")),
            h("span", { class: "h" }, "Actually churned"),
            h("div", { class: "cell bad" }, h("b", {}, num(c.fn)), h("span", {}, "missed")),
            h("div", { class: "cell good" }, h("b", {}, num(c.tp)), h("span", {}, "caught"))),
          h("div", { style: "margin-top:14px" }, bullets(e.reading))),
        card("Top 5 drivers of churn",
          h("ul", { class: "drivers" }, e.top_drivers.map((d) =>
            h("li", {}, h("span", {}, d.feature), h("div", { class: "bar" }, h("i", { style: `width:${(d.importance / maxImp) * 100}%` })), h("em", {}, (d.importance * 100).toFixed(1) + "%")))))),
      h("div", { class: "charts" }, chart(e.confusion_chart), chart(e.importance_chart)),
    ];
  }

  // ---------- wire up ----------
  const drop = $("drop"), input = $("file");
  input.addEventListener("change", () => input.files[0] && setFile(input.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) setFile(f); });
  $("form").addEventListener("submit", (e) => { e.preventDefault(); analyze(); });
  $("sample").addEventListener("click", async () => {
    try {
      const r = await fetch(SAMPLE_URL);
      if (!r.ok) throw new Error();
      const blob = await r.blob();
      setFile(new File([blob], "Telco-Customer-Churn.csv", { type: "text/csv" }));
    } catch {
      showError("The sample file could not be loaded. Upload your own CSV instead.");
    }
  });
  $("again").addEventListener("click", () => { input.value = ""; setFile(null); show("upload"); });
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));
  document.querySelector(".tabs").addEventListener("keydown", (e) => {
    const tabs = [...document.querySelectorAll(".tabs button")];
    const i = tabs.findIndex((t) => t.getAttribute("aria-selected") === "true");
    const next = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : null;
    if (next != null && tabs[next]) { tabs[next].focus(); selectTab(tabs[next].dataset.tab); }
  });

  checkApi();
})();
