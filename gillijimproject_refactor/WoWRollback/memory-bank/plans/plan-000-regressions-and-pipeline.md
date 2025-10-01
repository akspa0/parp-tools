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
