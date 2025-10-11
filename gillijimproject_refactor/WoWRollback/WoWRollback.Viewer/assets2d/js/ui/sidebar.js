export function initSidebar(overlayMgr, state) {
  const panel = document.getElementById('layers');
  if (!panel) return; // no sidebar in DOM, do nothing

  // Helper: persistence per-map
  const storageKey = (map) => `wrb_layers_${map || 'default'}`;
  const loadPrefs = () => {
    try { return JSON.parse(localStorage.getItem(storageKey(state.map)) || '{}'); }
    catch { return {}; }
  };
  const savePrefs = (prefs) => {
    try { localStorage.setItem(storageKey(state.map), JSON.stringify(prefs)); } catch {}
  };

  // Clear existing UI
  while (panel.firstChild) panel.removeChild(panel.firstChild);

  const prefs = loadPrefs();

  // Always include Grid toggle at top (built-in overlay)
  const gridItem = document.createElement('div');
  gridItem.className = 'layer-item';
  const gridChk = document.createElement('input');
  gridChk.type = 'checkbox';
  gridChk.checked = !!state.showGrid;
  gridChk.addEventListener('change', () => {
    state.showGrid = gridChk.checked;
    prefs['grid'] = gridChk.checked;
    savePrefs(prefs);
  });
  const gridLbl = document.createElement('label');
  gridLbl.textContent = 'Grid';
  gridItem.appendChild(gridChk);
  gridItem.appendChild(gridLbl);
  panel.appendChild(gridItem);

  // Load manifest-driven layers (optional)
  const manifest = state.overlays?.manifest;
  if (!manifest || !Array.isArray(manifest.layers) || manifest.layers.length === 0) {
    const empty = document.createElement('div');
    empty.style.opacity = '0.6';
    empty.style.fontSize = '12px';
    empty.textContent = '(No overlay layers)';
    panel.appendChild(empty);
    return;
  }

  for (const layer of manifest.layers) {
    const id = layer.id;
    const title = layer.title || id;

    const item = document.createElement('div');
    item.className = 'layer-item';

    const chk = document.createElement('input');
    chk.type = 'checkbox';
    const defaultOn = (layer.enabledByDefault !== false);
    chk.checked = (prefs[id] !== undefined) ? !!prefs[id] : defaultOn;

    // Apply initial toggle to overlay manager
    overlayMgr.toggle(id, chk.checked);

    chk.addEventListener('change', () => {
      overlayMgr.toggle(id, chk.checked);
      prefs[id] = chk.checked;
      savePrefs(prefs);
    });

    const lbl = document.createElement('label');
    lbl.textContent = title;

    item.appendChild(chk);
    item.appendChild(lbl);
    panel.appendChild(item);
  }
}
