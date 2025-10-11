export function initInspector(state) {
  const el = document.getElementById('ins-content');
  if (!el) return;
  el.textContent = '(Nothing selected)';
}

export function updateInspector(state) {
  const el = document.getElementById('ins-content');
  if (!el) return;

  const sel = Array.isArray(state.selection) ? state.selection : [];
  if (sel.length === 0) {
    el.style.opacity = '.7';
    el.style.fontSize = '12px';
    el.textContent = '(Nothing selected)';
    return;
  }

  // Clear
  el.innerHTML = '';
  el.style.opacity = '1';
  el.style.fontSize = '13px';

  // Summary header
  const hdr = document.createElement('div');
  hdr.style.margin = '0 0 8px 0';
  hdr.style.fontWeight = '600';
  const counts = sel.reduce((acc, s) => { const t = (s.feature?.type || 'feature').toLowerCase(); acc[t] = (acc[t]||0)+1; return acc; }, {});
  const parts = Object.entries(counts).map(([k,v]) => `${k}:${v}`);
  hdr.textContent = `Selected: ${sel.length} (${parts.join(', ')})`;
  el.appendChild(hdr);

  for (const item of sel) {
    const box = document.createElement('div');
    box.className = 'ins-item';

    const f = item.feature || {};
    const type = (f.type || item.pluginId || 'feature').toUpperCase();
    const UniqueID = f.UniqueID ?? f.uid ?? null;
    const FileDataID = f.FileDataID ?? f.fileId ?? null; // normalize legacy
    const FileName = f.FileName ?? f.file ?? null;
    const Flags = f.flags ?? f.Flags ?? null;
    const pos = f.position || f.Position || null;
    const rot = f.rotation || f.Rotation || null;
    const scale = f.scale ?? f.Scale ?? null;
    const textures = f.TextureFileDataIDs || f.textures || null;

    const addKV = (k, v) => {
      const row = document.createElement('div');
      row.className = 'ins-kv';
      const lk = document.createElement('div'); lk.textContent = k; lk.style.opacity = '.8';
      const rv = document.createElement('div'); rv.textContent = v;
      row.appendChild(lk); row.appendChild(rv);
      box.appendChild(row);
    };

    const title = document.createElement('div');
    title.style.margin = '0 0 6px 0';
    title.style.fontWeight = '600';
    title.textContent = type;
    box.appendChild(title);

    if (UniqueID !== null) addKV('UniqueID', String(UniqueID));
    if (FileDataID !== null) addKV('FileDataID', String(FileDataID));
    if (FileName) addKV('FileName', FileName);
    if (Flags !== null) addKV('Flags', `0x${Number(Flags).toString(16)}`);

    if (pos) addKV('Position', vec3(pos));
    if (rot) addKV('Rotation', vec3(rot));
    if (scale !== null) addKV('Scale', String(scale));

    if (Array.isArray(textures) && textures.length > 0) {
      addKV('TextureFileDataIDs', textures.join(', '));
    }

    // Actions
    const actions = document.createElement('div');
    actions.style.marginTop = '8px';
    const btnCopy = document.createElement('button');
    btnCopy.textContent = 'Copy JSON';
    btnCopy.onclick = () => {
      try { navigator.clipboard.writeText(JSON.stringify(f, null, 2)); } catch {}
    };
    actions.appendChild(btnCopy);
    box.appendChild(actions);

    el.appendChild(box);
  }
}

function vec3(v) {
  const x = v.x ?? v.X ?? 0;
  const y = v.y ?? v.Y ?? 0;
  const z = v.z ?? v.Z ?? 0;
  return `${fix(x)}, ${fix(y)}, ${fix(z)}`;
}
function fix(n) { const num = Number(n); return Number.isFinite(num) ? num.toFixed(3) : String(n); }
