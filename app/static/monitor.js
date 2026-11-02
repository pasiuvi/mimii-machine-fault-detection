// AcoustiGuard v2 – machine monitoring page
const $ = (id) => document.getElementById(id);
let TYPE = "pump", SELECTED = null, SETTINGS = null;

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.message || j.detail || `Request failed (${r.status})`);
  return j;
}
function busy(btn, on) { btn.disabled = on; btn.classList.toggle("loading", on); }

document.querySelectorAll("#m-type .seg").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#m-type .seg").forEach((x) => { x.classList.remove("active"); x.setAttribute("aria-checked", "false"); });
  b.classList.add("active"); b.setAttribute("aria-checked", "true"); TYPE = b.dataset.type;
}));

function dots(recent) {
  const cells = [];
  for (let i = 0; i < 3; i++) {
    const h = recent[recent.length - 3 + i];
    cells.push(`<i class="${h ? (h.unusual ? "bad" : "ok") : ""}" title="${h ? esc(h.file) + " z=" + h.z : "no check yet"}"></i>`);
  }
  return `<span class="dots">${cells.join("")}</span>`;
}

async function loadMachines() {
  const j = await api("/api/monitor/machines");
  SETTINGS = j.settings;
  $("cfg-pill").textContent = `k=${SETTINGS.k} · threshold z>${SETTINGS.c} · alarm 2 of 3`;
  const m = j.machines;
  if (!m.length) { $("machines").innerHTML = "None yet."; return; }
  $("machines").innerHTML = m.map((x) => `
    <div class="mon-machine ${SELECTED === x.id ? "sel" : ""}" data-id="${esc(x.id)}" data-name="${esc(x.name)}">
      <h4>${esc(x.name)} <span class="tag">${esc(x.machine_type)}</span></h4>
      <div class="small muted">${x.n_enrol} enrolment clips · ${x.checks} checks · last 3: ${dots(x.recent)}</div>
      <div class="mon-row" style="margin-top:8px">
        <button class="ghost mini sel-btn">Select</button>
        <button class="ghost mini del-btn">Remove</button>
      </div>
    </div>`).join("");
  document.querySelectorAll(".mon-machine").forEach((el) => {
    el.querySelector(".sel-btn").onclick = () => select(el.dataset.id, el.dataset.name);
    el.querySelector(".del-btn").onclick = async () => {
      await api(`/api/monitor/${encodeURIComponent(el.dataset.id)}`, { method: "DELETE" });
      if (SELECTED === el.dataset.id) select(null, null);
      loadMachines();
    };
  });
}

function select(id, name) {
  SELECTED = id;
  $("sel-name").textContent = name || "– select on the left –";
  $("check-btn").disabled = !id; $("reset-btn").disabled = !id;
  $("result").innerHTML = "";
  loadMachines(); renderDemoClips();
}

$("enrol-btn").onclick = async () => {
  const files = $("m-files").files, name = $("m-name").value.trim();
  if (!name || !files.length) { $("enrol-msg").innerHTML = `<span class="error">Enter a name and choose the normal clips.</span>`; return; }
  const fd = new FormData(); fd.append("name", name); fd.append("machine_type", TYPE);
  for (const f of files) fd.append("files", f);
  busy($("enrol-btn"), true); $("enrol-msg").textContent = `Enrolling ${files.length} clips… (the first time also loads the CLAP model)`;
  try { const r = await api("/api/monitor/enrol", { method: "POST", body: fd }); enrolMsg(r); select(r.id, r.name); }
  catch (e) { $("enrol-msg").innerHTML = `<span class="error">${esc(e.message)}</span>`; }
  finally { busy($("enrol-btn"), false); }
};

function enrolMsg(r) {
  $("enrol-msg").innerHTML = `✅ <b>${esc(r.name)}</b> enrolled with ${r.n_enrol} clips in ${(r.timing_ms / 1000).toFixed(1)} s.` +
    (r.rejected.length ? `<br>${r.rejected.length} clip(s) rejected: ${r.rejected.map((x) => esc(x.file) + " – " + esc(x.reason)).join("; ")}` : "") +
    r.warnings.map((w) => `<br><span class="warn-ico">⚠</span> ${esc(w)}`).join("");
}

$("check-btn").onclick = async () => {
  const f = $("c-file").files[0];
  if (!f) { $("result").innerHTML = `<p class="error">Choose a recording first.</p>`; return; }
  const fd = new FormData(); fd.append("file", f);
  busy($("check-btn"), true);
  try { showResult(await api(`/api/monitor/${encodeURIComponent(SELECTED)}/check`, { method: "POST", body: fd })); }
  catch (e) { $("result").innerHTML = `<p class="error">${esc(e.message)}</p>`; }
  finally { busy($("check-btn"), false); loadMachines(); }
};

$("reset-btn").onclick = async () => {
  await api(`/api/monitor/${encodeURIComponent(SELECTED)}/reset`, { method: "POST" });
  $("result").innerHTML = `<p class="muted small">History cleared.</p>`; loadMachines();
};

function showResult(r) {
  const title = { normal: "Normal for this machine", watch: "Watch – unusual clip", alarm: "ALARM – inspect this machine" }[r.status];
  const max = Math.max(r.threshold_z * 2.5, r.z + 1, 1);
  const pct = (v) => Math.max(0, Math.min(100, (100 * v) / max));
  $("result").innerHTML = `
    <div class="mon-status ${r.status}">
      <h3>${title}</h3>
      <div>${esc(r.action)}</div>
      ${r.true_label ? `<div class="small muted" style="margin-top:6px">Demo clip – true label: <b>${esc(r.true_label)}</b></div>` : ""}
    </div>
    <div class="zbar"><div class="fill" style="width:${pct(r.z)}%"></div><div class="thr" style="left:${pct(r.threshold_z)}%"></div></div>
    <div class="small muted">Difference from this machine's normal sound: <b>z = ${r.z}</b> (unusual above ${r.threshold_z}) · ${esc(r.file)} · ${r.timing_ms} ms</div>
    <h3>Last 3 checks of ${esc(r.machine)} ${dots(r.window)}</h3>
    <table class="mon-table"><tr><th>Time</th><th>File</th><th>z</th><th>Unusual?</th></tr>
      ${r.window.map((h) => `<tr><td>${esc(h.time.slice(11))}</td><td>${esc(h.file)}</td><td>${h.z}</td><td>${h.unusual ? "yes" : "no"}</td></tr>`).join("")}
    </table>
    ${r.notes.map((n) => `<p class="small"><span class="warn-ico">⚠</span> ${esc(n)}</p>`).join("")}
    <p class="small muted">Input checks: ${r.checks.map((c) => esc(c.name) + " – " + esc(c.status)).join(" · ")}</p>`;
}

// ---------- demo machines ----------
let DEMO = [];
async function loadDemo() {
  try { DEMO = await api("/api/monitor/demo"); } catch { DEMO = []; }
  if (!DEMO.length) { $("demo-card").classList.add("hidden"); return; }
  renderDemoClips();
}
function renderDemoClips() {
  if (!DEMO.length) return;
  $("demo-list").innerHTML = DEMO.map((d) => {
    const id = d.name.replace(" (demo)", "-demo").toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
    const active = SELECTED === id;
    return `<div class="mon-machine ${active ? "sel" : ""}">
      <h4>${esc(d.name)}</h4>
      <button class="ghost mini" data-enrol="${esc(d.key)}">Enrol with ${d.n_enrol} normal clips</button>
      <div class="clip-list" style="margin-top:8px">${active ? d.clips.map((c) =>
        `<button class="mini" data-key="${esc(d.key)}" data-i="${c.i}">${esc(c.file)} (${c.true_label})</button>`).join("") :
        `<span class="small muted">Enrol, then the test clips appear here.</span>`}</div></div>`;
  }).join("");
  document.querySelectorAll("[data-enrol]").forEach((b) => b.onclick = async () => {
    busy(b, true); b.textContent = "Enrolling… (first time loads CLAP)";
    try { const r = await api(`/api/monitor/demo/${b.dataset.enrol}/enrol`, { method: "POST" }); enrolMsg(r); select(r.id, r.name); }
    catch (e) { $("enrol-msg").innerHTML = `<span class="error">${esc(e.message)}</span>`; busy(b, false); }
  });
  document.querySelectorAll("[data-key]").forEach((b) => b.onclick = async () => {
    busy(b, true);
    try { showResult(await api(`/api/monitor/demo/${b.dataset.key}/check/${b.dataset.i}`, { method: "POST" })); }
    catch (e) { $("result").innerHTML = `<p class="error">${esc(e.message)}</p>`; }
    finally { busy(b, false); loadMachines(); }
  });
}

loadMachines(); loadDemo();
