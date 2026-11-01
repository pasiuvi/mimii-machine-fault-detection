// AcoustiGuard frontend – plain JavaScript, no build step.
const $ = (id) => document.getElementById(id);
const state = { type: "pump", file: null, busy: false, explain: null, specSrc: "", overlaySrc: "" };
const ICONS = {
  no_alarm: '<svg viewBox="0 0 24 24"><path d="M5 12l5 5 9-10"/></svg>',
  alarm: '<svg viewBox="0 0 24 24"><path d="M12 3l10 18H2zM12 10v5M12 18v.5"/></svg>',
  unreliable: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 114 2c-1 .6-1.5 1-1.5 2.2M12 17v.5"/></svg>',
};
const ACTIONS = {
  alarm: "Raise an inspection work order and check this machine now.",
  no_alarm: "No action needed now. Keep the planned maintenance schedule.",
  unreliable: "Do not act on this score. Check the recording and the input checks below.",
};
const pct = (v) => (v * 100).toFixed(v < 0.01 && v > 0 ? 2 : 1) + " %";
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------------- start-up ----------------
async function init() {
  try {
    const h = await (await fetch("/api/health")).json();
    $("status-pill").classList.add(h.model_loaded ? "ok" : "bad");
    $("status-pill").lastChild.textContent = h.model_loaded ? `Model ready · ${h.device.toUpperCase()}` : "Model not loaded";
    const card = await (await fetch("/api/model-card")).json();
    $("model-pill").textContent = `FaultCNN · run ${card.deployed_run} · ${card.seed_runs.length} seeds`;
    const ev = card.evaluation || {};
    const dep = Object.keys(ev).find((k) => k.includes("deployed"));
    const loud = Object.keys(ev).find((k) => k.startsWith("Trivial"));
    if (dep) $("lim-test").textContent = pct(ev[dep].test_macro);
    if (loud) $("lim-loud").textContent = pct(ev[loud].test_macro);
  } catch (e) {
    $("status-pill").classList.add("bad");
    $("status-pill").lastChild.textContent = "Server unreachable";
  }
  loadExamples();
}

let EXAMPLES = [];
async function loadExamples() {
  try {
    EXAMPLES = await (await fetch("/api/examples")).json();
    renderExamples();
  } catch (e) { /* examples are optional */ }
}

// Only the demo clips of the selected machine type are listed, so the list always matches the toggle.
function renderExamples() {
  const ex = EXAMPLES.filter((e) => e.machine_type === state.type);
  $("examples").innerHTML = ex.length ? ex.map((e) => `
      <li class="ex">
        <span class="tag ${e.true_label}">${e.true_label}</span>
        <div class="ex-info"><b>${esc(e.machine)}</b><span>${esc(e.file)} · Q3 result: ${esc(e.kind)}</span></div>
        <button class="ghost" data-ex="${e.id}" data-type="${e.machine_type}" data-file="${esc(e.file)}">Run</button>
      </li>`).join("") : `<li class="muted">No ${state.type} demo clips found on this computer.</li>`;
  document.querySelectorAll("[data-ex]").forEach((b) => b.addEventListener("click", () => runExample(b)));
}

// ---------------- input handling ----------------
document.querySelectorAll(".seg").forEach((b) => b.addEventListener("click", () => setType(b.dataset.type)));
function setType(t) {
  state.type = t;
  document.querySelectorAll(".seg").forEach((b) => {
    const on = b.dataset.type === t;
    b.classList.toggle("active", on);
    b.setAttribute("aria-checked", on);
  });
  if (EXAMPLES.length) renderExamples();
}

const dz = $("dropzone");
dz.addEventListener("click", () => $("file-input").click());
dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") $("file-input").click(); });
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));
$("file-input").addEventListener("change", (e) => e.target.files[0] && setFile(e.target.files[0]));
$("file-clear").addEventListener("click", () => setFile(null));

function setFile(f) {
  state.file = f;
  hideError();
  $("file-chip").classList.toggle("hidden", !f);
  $("player").classList.toggle("hidden", !f);
  dz.classList.toggle("hidden", !!f);
  $("analyse").disabled = !f;
  if (f) {
    $("file-name").textContent = f.name;
    $("file-size").textContent = `${(f.size / 1024 / 1024).toFixed(2)} MB`;
    $("player").src = URL.createObjectURL(f);
  } else {
    $("file-input").value = "";
    $("player").removeAttribute("src");
  }
}

$("analyse").addEventListener("click", async () => {
  if (!state.file || state.busy) return;
  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("machine_type", state.type);
  const file = state.file, type = state.type;
  await request("/api/predict", { method: "POST", body: fd }, state.file.name, () => {
    const f2 = new FormData(); f2.append("file", file); f2.append("machine_type", type);
    return fetch("/api/explain", { method: "POST", body: f2 });
  });
});

async function runExample(btn) {
  setFile(null);
  $("player").src = `/api/examples/${btn.dataset.ex}/audio`;
  $("player").classList.remove("hidden");
  btn.disabled = true;
  const id = btn.dataset.ex;
  await request(`/api/examples/${id}/predict`, { method: "POST" }, btn.dataset.file,
    () => fetch(`/api/examples/${id}/explain`, { method: "POST" }));
  btn.disabled = false;
}

async function request(url, opts, name, explainFn) {
  state.busy = true;
  hideError();
  $("analyse").classList.add("loading");
  try {
    const res = await fetch(url, opts);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = body.message || body.detail || `Request failed (${res.status})`;
      showError(msg, body.error);
      addHistory({ file: name, machine_type: state.type, verdict: "rejected", reason: body.error || res.status });
      return;
    }
    state.explain = explainFn || null;
    render(body);
    addHistory(body);
  } catch (e) {
    showError("Could not reach the server. Is it still running?");
  } finally {
    state.busy = false;
    $("analyse").classList.remove("loading");
  }
}

function showError(msg, code) {
  $("error").innerHTML = `<b>Input rejected${code ? ` (${esc(code)})` : ""}.</b> ${esc(msg)}`;
  $("error").classList.remove("hidden");
}
function hideError() { $("error").classList.add("hidden"); }

// ---------------- rendering ----------------
function render(r) {
  $("empty").classList.add("hidden");
  $("result").classList.remove("hidden");
  const v = $("verdict");
  v.className = `verdict ${r.verdict}`;
  void v.offsetWidth; // restart animation
  $("verdict-icon").innerHTML = ICONS[r.verdict];
  $("verdict-title").textContent = r.headline;
  $("verdict-action").textContent = ACTIONS[r.verdict] +
    (r.ood_status === "unusual" ? " The input is unusual compared with the training data, so treat this result with caution." : "");

  const truth = $("truth");
  if (r.true_label) {
    const predicted = r.alarm ? "abnormal" : "normal";
    const right = predicted === r.true_label;
    truth.innerHTML = `Demo clip – true label<b>${r.true_label}</b><span class="${right ? "right" : "wrong"}">${right ? "model correct" : "model wrong"}</span>`;
    truth.classList.remove("hidden");
  } else truth.classList.add("hidden");

  // gauge: arc length ~ 251.3 for r = 80
  const L = Math.PI * 80;
  const fill = $("g-fill");
  fill.style.strokeDasharray = `${L}`;
  fill.style.strokeDashoffset = `${L * (1 - r.score)}`;
  fill.style.stroke = r.verdict === "alarm" ? "var(--red)" : r.verdict === "unreliable" ? "var(--amber)" : "var(--green)";
  const a = Math.PI * (1 - Math.min(Math.max(r.threshold, 0), 1));
  const thr = $("g-thr");
  thr.setAttribute("x1", 100 + 64 * Math.cos(a)); thr.setAttribute("y1", 100 - 64 * Math.sin(a));
  thr.setAttribute("x2", 100 + 96 * Math.cos(a)); thr.setAttribute("y2", 100 - 96 * Math.sin(a));
  $("g-value").textContent = r.score.toFixed(3);

  $("m-thr").textContent = r.threshold > 1 ? "n/a" : r.threshold.toFixed(3);
  $("m-margin").textContent = `score is ${r.margin >= 0 ? "above" : "below"} the threshold by ${Math.abs(r.margin).toFixed(3)}`;
  $("m-far").textContent = pct(r.validation_false_alarm_rate);
  $("m-votes").textContent = `${r.seed_votes} of ${r.seed_count} alarm`;
  $("m-spread").textContent = `scores ${Object.values(r.seed_scores).map((s) => s.toFixed(3)).join(" / ")} (spread ${r.seed_spread.toFixed(3)})`;
  const oodText = { in_distribution: "Typical", unusual: "Unusual", out_of_distribution: "Outside range" };
  $("m-ood").textContent = oodText[r.ood_status];
  $("m-ood").style.color = r.ood_status === "in_distribution" ? "#86efac" : r.ood_status === "unusual" ? "#fcd34d" : "#fca5a5";
  $("m-ood-sub").textContent = `level ${r.ood_values.level_db} dB · shape distance ${r.ood_values.shape_distance}`;

  const sym = { pass: "✓", info: "i", warn: "!", fail: "✕" };
  $("checks").innerHTML = r.checks.map((c) => `
    <li class="${c.status}"><span class="dot">${sym[c.status]}</span><div><b>${esc(c.name)}</b><span>${esc(c.detail)}</span></div></li>`).join("");
  $("spec").src = r.spectrogram_png;
  state.specSrc = r.spectrogram_png;
  resetExplanation(r);
  $("meta").innerHTML = [
    `File: <b>${esc(r.file)}</b>`, `${r.audio.seconds} s · ${r.audio.original_sample_rate} Hz · ${r.audio.channels} ch`,
    `Latency: ${r.timing_ms.total} ms (features ${r.timing_ms.preprocess} ms, model ${r.timing_ms.model} ms)`,
    `Model: ${r.model.deployed_run} on ${r.model.device}`, `Request ID: ${r.request_id}`,
  ].map((s) => `<span>${s}</span>`).join("");
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------- Question 5: explanation ----------------
function resetExplanation(r) {
  $("xai-body").classList.add("hidden");
  $("xai-btn").disabled = !state.explain;
  $("xai-btn").textContent = "Explain this result";
  state.lastVerdict = r.verdict;
}

$("xai-btn").addEventListener("click", async () => {
  if (!state.explain || state.busy) return;
  const b = $("xai-btn");
  b.disabled = true; b.textContent = "Explaining…";
  try {
    const res = await state.explain();
    const body = await res.json().catch(() => ({}));
    if (!res.ok) { showError(body.message || `Explanation failed (${res.status})`, body.error); b.textContent = "Explain this result"; b.disabled = false; return; }
    renderExplanation(body);
    b.textContent = "Explain again";
  } catch (e) {
    showError("Could not reach the server. Is it still running?");
    b.textContent = "Explain this result";
  }
  b.disabled = false;
});

document.querySelectorAll(".mini").forEach((m) => m.addEventListener("click", () => {
  document.querySelectorAll(".mini").forEach((x) => x.classList.toggle("active", x === m));
  $("xai-img").src = m.dataset.view === "spec" ? state.specSrc : state.overlaySrc;
}));

function renderExplanation(r) {
  const e = r.explanation;
  state.overlaySrc = e.overlay_png;
  document.querySelectorAll(".mini").forEach((x) => x.classList.toggle("active", x.dataset.view === "overlay"));
  $("xai-img").src = e.overlay_png;
  const warn = $("xai-warn");
  const msgs = [];
  if (r.verdict === "unreliable") msgs.push("The input is outside what the model knows. The explanation of an unreliable result is not meaningful.");
  else if (r.ood_status === "unusual") msgs.push("The input is unusual compared with the training data – read the explanation with extra caution.");
  warn.innerHTML = msgs.map(esc).join("<br>");
  warn.classList.toggle("hidden", !msgs.length);

  const st = $("xai-status");
  if (r.verdict === "unreliable") {
    st.className = "xai-status bad"; st.textContent = "Not meaningful – input outside what the model knows";
  } else if (e.check && e.check.passed) {
    st.className = "xai-status ok"; st.textContent = "Validated explanation – the model depends on the highlighted area (not a fault diagnosis)";
  } else {
    st.className = "xai-status bad"; st.textContent = "Not validated – the map may look plausible, but hiding it did not lower the fault evidence. Do not rely on it.";
  }
  $("xai-lines").innerHTML = e.summary.lines.map((l) => `<li>${esc(l)}</li>`).join("");
  const q1 = { pump: [[431, 910]], fan: [[0, 50], [479, 2400]] }[r.machine_type] || [];
  const inQ1 = (b) => q1.some(([lo, hi]) => b.lo < hi && b.hi > lo);
  const maxShare = Math.max(...e.bands.map((b) => b.share), 0.01);
  $("xai-bands").innerHTML = e.bands.map((b) => `
    <li class="${inQ1(b) ? "q1" : ""}"><span>${esc(b.band)}</span>
      <span class="bar"><i style="width:${(100 * b.share / maxShare).toFixed(0)}%"></i></span>
      <span>${(100 * b.share).toFixed(0)} %</span></li>`).join("") +
    `<li><span class="note">● = band where ${r.machine_type} faults showed up in the training data (Question 1)</span></li>`;

  const c = e.check;
  if (!c) {
    $("xai-check").innerHTML = `<div class="xc-verdict fail">Nothing was highlighted, so there is nothing to check. The network found no fault-like region in this recording.</div>`;
  } else {
    const row = (label, ev, sc) => `<div class="xc-row"><span>${label}</span><b>${ev.toFixed(2)}<small> evidence · score ${sc.toFixed(3)}</small></b></div>`;
    $("xai-check").innerHTML =
      row("As analysed", c.logit, c.score) +
      row(`Highlighted ${Math.round(100 * c.hidden_share)} % hidden`, c.logit_top_hidden, c.score_top_hidden) +
      row(`Random ${Math.round(100 * c.hidden_share)} % hidden (average of 10)`, c.logit_random_hidden, c.score_random_hidden) +
      `<p class="small muted xc-note">"Evidence" is the network's raw output before it is squeezed into the 0–1 score. Near 0 or 1 the score hardly moves, so the check compares the evidence.</p>` +
      `<div class="xc-verdict ${c.passed ? "pass" : "fail"}">${c.passed
        ? `<b>Passed.</b> Hiding the highlighted area lowered the fault evidence by ${c.logit_drop_top.toFixed(2)}, more than hiding random areas (${c.logit_drop_random.toFixed(2)}). The model does depend on this area.`
        : `<b>Not passed.</b> Hiding the highlighted area lowered the fault evidence by ${c.logit_drop_top.toFixed(2)}, not more than random areas (${c.logit_drop_random.toFixed(2)}). Do not rely on this map for this recording.`}</div>`;
  }
  $("xai-meta").innerHTML = [`Method: ${esc(e.method)} on layer ${esc(e.layer)} (${e.grid.join(" × ")} grid)`,
    `Explanation time: ${e.timing_ms.gradcam} ms + check ${e.timing_ms.check} ms`, `Request ID: ${esc(r.request_id)}`]
    .map((x) => `<span>${x}</span>`).join("");
  $("xai-body").classList.remove("hidden");
  $("xai").scrollIntoView({ behavior: "smooth", block: "start" });
}

function addHistory(r) {
  const tb = $("history");
  if (tb.querySelector(".muted")) tb.innerHTML = "";
  const t = new Date().toLocaleTimeString();
  const label = { alarm: "Alarm", no_alarm: "No alarm", unreliable: "Not reliable", rejected: "Rejected" }[r.verdict];
  const row = document.createElement("tr");
  row.innerHTML = `<td>${t}</td><td>${esc(r.file || "")}</td><td>${esc(r.machine_type || "")}</td>
    <td>${r.score !== undefined ? r.score.toFixed(3) : "–"}</td><td>${r.threshold !== undefined ? r.threshold.toFixed(3) : "–"}</td>
    <td><span class="badge ${r.verdict}">${label}</span></td>
    <td>${r.ood_status ? r.ood_status.replace(/_/g, " ") : esc(r.reason || "")}</td>
    <td>${r.timing_ms ? r.timing_ms.total + " ms" : "–"}</td>`;
  tb.prepend(row);
}

init();
