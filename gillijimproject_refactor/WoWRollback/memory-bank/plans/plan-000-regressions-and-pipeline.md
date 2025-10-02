# Plan 000 — Regressions & Pipeline Fixes

## Summary
Stabilize the current viewer before adding new layers. Fix marker correctness, ensure tile images load reliably, and make the rebuild process fully reproducible. Align image format configuration with actual outputs.

## Symptoms
- WMOs show as orange but X/Y appear flipped.
- M2/MDX markers no longer visible; legend not reflecting types.
- Tiles page can’t find minimap images after switching to webp.
- Program logs indicate webp, but minimaps are emitted as jpg.
- Need a complete rebuild mode to avoid stale outputs.

## Root Causes (likely)
- Inconsistent pixel→lat/lng transform across grid vs tile pages.
- Overlay JSON lacks reliable `type` field; front-end falls back to filename extension heuristics.
- Image extension mismatch: webp requested, jpg emitted (no webp encoder).
- Browser caching `index.json`/`config.json`; ext mismatch persists until hard refresh.

## Fix Plan

### 1) Marker correctness and legend
- Unify transforms (`pixelToLatLng`, `latToRow`), verify overlay pixel schema and flip handling.
- Ensure overlay JSON includes `type: wmo|m2|mdx|other`; prefer `obj.type` on front-end, fallback to extension.
- Add/refresh legend (WMO=M→orange, M2/MDX→cyan, Other→green) and verify marker visibility.

### 2) Tiles page image loading
- Guarantee `config.minimap.ext` matches emitted files.
- Cache-bust `index.json` and `config.json` fetches.

### 3) WebP encoder integration (or safe default)
- Integrate ImageSharp WebP plugin and compile with `HAS_WEBP`.
- If not integrated, default to `jpg` or ensure runtime fallback also updates `config.minimap.ext` to `jpg`.

### 4) Full pipeline rebuild via script
- `rebuild-and-regenerate.ps1` supports: `-Clean` (wipe outputs) and optional converter integration.
- Order: discover maps → (optional) convert AlphaWDT→LK ADT → analyze → compare-versions → viewer → copy ViewerAssets.

### 5) Harden analyze-lk for ≥0.6.0
- Validate later-era inputs and tile math; add small test corpus.

## Acceptance Criteria
- WMO/M2/MDX show with correct sizes/colors; legend matches.
- Tiles page loads minimaps without 404s; `config.minimap.ext` equals actual filenames.
- Rebuild with `-Clean` yields a consistent viewer.
- Optional: true webp emission when encoder is present; otherwise jpg with matching config.

## Tasks
- Front-end: cache-bust config/index; prefer `obj.type`; legend updates.
- Backend: ensure `effectiveExt` consistency; optionally integrate WebP.
- Script: verify complete pipeline and document usage.
- Tests: quick sanity checks on a couple of maps/versions.

## Update — 2025-10-01

### Findings
- **[MDX/M2 regression]** Markers were suppressed by viewer de-duplication using only `uniqueId`. After splitting WMO vs M2 into different colors, a shared UID between WMO and M2 caused one to be dropped.
- **[Corner ghosts]** Caused by clamping without checking tile membership; fixed by raw-local `[0,1)` gating before clamp.
- **[Map directory drift]** Minimap folders sometimes use internal directory names; added Map.dbc resolver to normalize.

### Changes Implemented
- **Viewer (main.js)**: De-dup key now `uniqueId:type` so WMO and M2 can both render. Pixel-only rendering is allowed; world used to fix corner pixels or fallback.
- **Overlay builder**: Skips out-of-tile placements using `ComputeLocalCoordinatesRaw` to eliminate corner ghosts; emits `world` (normalized) and `worldRaw`.
- **WMO Y-flip**: Applied on server to remove front-end toggles.
- **Map.dbc**: Added `MapDirectoryResolver` via wow.tools DBCD; generator now normalizes map names to canonical directories and writes `display` in `index.json`.

### Next Steps
- **Verify MDX/M2 restored**: Rebuild viewer and confirm cyan markers appear alongside WMO.
- **Viewer UI (optional)**: Show `display` name in map dropdown while fetching by canonical `map`.
- **0.6.0.3592**: Ensure included in `-Versions` and that its minimaps resolve via Map.dbc.
- **Spot check overlays**: Confirm M2 points exist in `overlays/{map}/tile_rX_cY.json` for known tiles.

### Acceptance (update)
- MDX/M2 visible on target tiles; no corner ghosts; WMOs aligned; tiles load across versions with canonical directories.
