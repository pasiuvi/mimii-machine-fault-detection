// AcoustiGuard frontend – plain JavaScript, no build step.
const $ = (id) => document.getElementById(id);
const state = { type: "pump", file: null, busy: false };
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
  await request("/api/predict", { method: "POST", body: fd }, state.file.name);
});

async function runExample(btn) {
  setFile(null);
  $("player").src = `/api/examples/${btn.dataset.ex}/audio`;
  $("player").classList.remove("hidden");
  btn.disabled = true;
  await request(`/api/examples/${btn.dataset.ex}/predict`, { method: "POST" }, btn.dataset.file);
  btn.disabled = false;
}

async function request(url, opts, name) {
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
  $("meta").innerHTML = [
    `File: <b>${esc(r.file)}</b>`, `${r.audio.seconds} s · ${r.audio.original_sample_rate} Hz · ${r.audio.channels} ch`,
    `Latency: ${r.timing_ms.total} ms (features ${r.timing_ms.preprocess} ms, model ${r.timing_ms.model} ms)`,
    `Model: ${r.model.deployed_run} on ${r.model.device}`, `Request ID: ${r.request_id}`,
  ].map((s) => `<span>${s}</span>`).join("");
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
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