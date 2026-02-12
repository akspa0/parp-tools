// Minimal Tools panel logic: presets, defaults, CLI preview, run + SSE logs

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}

function $(id) { return document.getElementById(id); }

function buildPayloadFromForm() {
  return {
    useDefaults: false,
    uniqueId: {
      max: parseInt($("uniqueIdMax").value || "0", 10) || 125000,
      buryDepth: parseInt($("buryDepth").value || "-5000", 10) || -5000
    },
    terrain: {
      fixHoles: $("fixHoles").checked,
      disableMcsh: $("disableMcsh").checked
    },
    mapping: {
      strictAreaId: $("strictAreaId").checked,
      chainVia060: $("chainVia060").checked
    },
    paths: {
      wdt: $("wdtInput").value.trim(),
      crosswalkDir: $("crosswalkInput").value.trim() || null,
      lkDbcDir: $("lkDbcInput").value.trim() || null,
      outDir: $("outDirInput").value.trim() || null,
      lkOutDir: $("lkOutDirInput").value.trim() || null
    }
  };
}

function guessMapName(wdtPath) {
  if (!wdtPath) return null;
  const m = wdtPath.match(/([^\\\/]+)\.wdt$/i);
  return m ? m[1] : null;
}

function ensureLkOut(payload) {
  const p = payload;
  if (!p.paths.lkOutDir && p.paths.outDir && p.paths.wdt) {
    const map = guessMapName(p.paths.wdt) || "Map";
    p.paths.lkOutDir = `${p.paths.outDir}/lk_adts/World/Maps/${map}`;
  }
  return p;
}

function buildCliPreview(p) {
  const args = ["WoWRollback", "alpha-to-lk"];
  const add = (k, v) => { if (v) { args.push(k, v.includes(" ") ? `\"${v}\"` : v); } };
  args.push("--input", p.paths.wdt || "<wdt>");
  args.push("--max-uniqueid", String(p.uniqueId.max || 125000));
  add("--out", p.paths.outDir);
  args.push("--export-lk-adts");
  add("--lk-out", p.paths.lkOutDir);
  add("--crosswalk-dir", p.paths.crosswalkDir);
  add("--lk-dbc-dir", p.paths.lkDbcDir);
  if (p.terrain.fixHoles) args.push("--fix-holes");
  if (p.terrain.disableMcsh) args.push("--disable-mcsh");
  if (p.mapping.strictAreaId) args.push("--strict-areaid");
  if (p.mapping.chainVia060) args.push("--chain-via-060");
  return args.join(" ");
}

async function loadDefaults() {
  try {
    const d = await fetchJSON(`/api/defaults`);
    if (d.wdt) $("wdtInput").value = d.wdt;
    if (d.crosswalkDir) $("crosswalkInput").value = d.crosswalkDir;
    if (d.lkDbcDir) $("lkDbcInput").value = d.lkDbcDir;
    updatePreview();
  } catch (e) {
    console.error("Failed to load defaults", e);
  }
}

async function loadPresets() {
  try {
    const presets = await fetchJSON(`/api/presets`);
    const sel = $("presetSelect");
    sel.innerHTML = "";
    presets.forEach((p, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = p.name;
      opt.dataset.payload = JSON.stringify(p.request);
      sel.appendChild(opt);
    });
  } catch (e) {
    console.warn("No presets available", e);
  }
}

function applySelectedPreset() {
  const sel = $("presetSelect");
  const opt = sel.options[sel.selectedIndex];
  if (!opt || !opt.dataset.payload) return;
  try {
    const req = JSON.parse(opt.dataset.payload);
    $("uniqueIdMax").value = req.uniqueId?.max ?? 125000;
    $("buryDepth").value = req.uniqueId?.buryDepth ?? -5000;
    $("fixHoles").checked = !!req.terrain?.fixHoles;
    $("disableMcsh").checked = !!req.terrain?.disableMcsh;
    $("strictAreaId").checked = req.mapping?.strictAreaId !== false;
    $("chainVia060").checked = !!req.mapping?.chainVia060;
    $("wdtInput").value = req.paths?.wdt ?? "";
    $("crosswalkInput").value = req.paths?.crosswalkDir ?? "";
    $("lkDbcInput").value = req.paths?.lkDbcDir ?? "";
    $("outDirInput").value = req.paths?.outDir ?? "";
    $("lkOutDirInput").value = req.paths?.lkOutDir ?? "";
    updatePreview();
  } catch {}
}

function updatePreview() {
  let payload = buildPayloadFromForm();
  payload = ensureLkOut(payload);
  $("cliPreview").textContent = buildCliPreview(payload);
}

async function runJob() {
  const logs = $("jobLogs");
  logs.textContent = "";
  let payload = buildPayloadFromForm();
  payload = ensureLkOut(payload);

  try {
    const res = await fetch(`/api/build/alpha-to-lk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(`${res.status}`);
    const { jobId } = await res.json();

    const es = new EventSource(`/api/jobs/${jobId}/events`);
    es.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.type === "log") {
          logs.textContent += data.message + "\n";
          logs.scrollTop = logs.scrollHeight;
        } else if (data.type === "status") {
          // could update a small status pill
        }
      } catch {}
    };
    es.onerror = () => { /* ignore */ };
  } catch (e) {
    logs.textContent = `Failed to start job: ${e.message}`;
  }
}

export async function initToolsPanel() {
  // Wire events
  $("loadDefaultsBtn").addEventListener("click", loadDefaults);
  $("presetSelect").addEventListener("change", applySelectedPreset);
  ["uniqueIdMax","buryDepth","wdtInput","crosswalkInput","lkDbcInput","outDirInput","lkOutDirInput","fixHoles","disableMcsh","strictAreaId","chainVia060"].forEach(id => {
    const el = $(id);
    if (!el) return;
    el.addEventListener("input", updatePreview);
    el.addEventListener("change", updatePreview);
  });
  $("runBtn").addEventListener("click", runJob);

  await loadPresets();
  await loadDefaults();
  updatePreview();
}
