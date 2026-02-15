# Terrain Editing Plan (3.3.5 Big Alpha + MCCV + Import/Export)

Date: 2026-02-14
Project: `src/MdxViewer`

## Objective
Add robust terrain data support for post-0.12 ADTs (focus: 3.3.5):
1. Correct big alpha-map handling (`MCAL` variants)
2. Runtime support for `MCCV` vertex colors
3. UI-exposed import/export tools for heightmaps, alpha masks, and MCCV images

This plan is intentionally phased so we can ship read-only correctness first, then editing workflows.

---

## Current State (What already exists)

## In `MdxViewer`
- `StandardTerrainAdapter` already reads `MPHD` and has `_useBigAlpha` + MCAL decode branches.
- `TerrainRenderer` already blends base+overlay textures using per-layer alpha textures.
- Terrain chunk model currently assumes alpha textures are 64×64 expanded maps.
- No runtime `MCCV` application in terrain shader path.
- No user-facing import/export terrain editing tools yet.

## In `WoWMapConverter.Core` (reusable now)
- `VLM/AlphaMapService.cs`
  - Supports compressed alpha, big alpha, 4-bit packed alpha.
  - PNG conversion helpers (`ToPng`, `FromPng`), compression helper.
- `Formats/LichKing/Mcnk.cs`
  - Extracts `McalRawData`, `TextureLayers`, `MccvData`.
- `VLM/VlmDatasetExporter.cs`
  - Reads layer alpha + MCCV and emits image artifacts.
- `VLM/HeightmapBakeService.cs`
  - Exports 16-bit heightmaps + metadata.
- `Services/MinimapService.cs`
  - Has MCCV utility stubs and conversion helpers.

---

## Ghidra Status (3.3.5 client)
Using the currently loaded program in Ghidra:
- Active function context is symbol-stripped (`FUN_*` naming).
- No direct string hits for `MCAL`/`MCCV` in current indexed string view.
- Practical implication: exact semantic confirmation via current MCP surface is limited without additional symboling / constant-search scripting.

### Ghidra follow-up checklist (required before final write-back tooling)
1. Locate ADT/MCNK parse routine by immediate constants (`0x4D43414C`, `0x4D434356`, `0x4D434C59`).
2. Confirm branch priority for alpha decode:
   - layer compressed flag (`0x200`) first
   - `MPHD` big-alpha flag (`0x4`) path
   - packed 4-bit fallback
3. Confirm `doNotFixAlphaMap` behavior source flag and exact edge-row/column fix logic.
4. Confirm runtime usage of MCCV channel order and modulation space (linear/gamma assumptions).

Until this is complete, we should ship parser behavior matching existing validated pipeline (`AlphaMapService`) and keep write-back operations opt-in/preview.

---

## Target Architecture Changes

## 1) Big Alpha Decode Unification
Replace ad-hoc decode in `StandardTerrainAdapter.ExtractAlphaMaps` with shared `AlphaMapService.ReadAlpha`.

### Why
- Avoid duplicated, divergent decode logic.
- Leverage working VLM path that already handles compressed + big + packed variants.

### Planned changes
- In `StandardTerrainAdapter`:
  - For each overlay layer (`1..3`), call `AlphaMapService.ReadAlpha(...)` with:
    - `mcnk.McalRawData`
    - layer-specific offset (`MclyEntry.AlphaMapOffset`)
    - layer flags
    - `_useBigAlpha`
    - `doNotFixAlphaMap` (derived from chunk/header flags once finalized)
- Preserve 64×64 expanded byte map output (compatible with current `TerrainRenderer`).

---

## 2) MCCV Runtime Support
Add MCCV to terrain chunk data path and shader modulation.

### Planned data model updates
- `TerrainChunkData`:
  - add `byte[]? MccvColors` (expected length `145 * 4`)
- `TerrainChunkMesh`:
  - add packed per-vertex color attribute (normalize to float in shader or CPU)

### Planned renderer updates
- `TerrainMeshBuilder`:
  - map 145 MCCV entries to terrain vertices in same interleaved topology.
- `TerrainRenderer` shader:
  - add vertex color input and `uUseMccv` toggle.
  - multiply final terrain color by MCCV (configurable strength).
- UI toggles in terrain panel:
  - `Use MCCV`
  - `MCCV Strength` (0..1)

### Safety
- If MCCV missing/invalid length, default to white (no visual change).

---

## 3) Import/Export Tooling (Phase 1: Read/Write assets, no brush editor)
Expose deterministic file workflows first.

## Export
- Heightmap export (16-bit PNG + JSON metadata)
- Alpha mask export (per-layer PNGs)
- MCCV export (PNG visualization + raw binary optional)

## Import
- Heightmap import (apply to selected tile/chunks, preview then commit)
- Alpha mask import (layer-specific, tile or chunk scope)
- MCCV import (image-to-vertex resample + apply)

### Reuse targets
- Heightmap logic from `HeightmapBakeService`
- Alpha encode/decode and PNG from `AlphaMapService`
- MCCV image generation/sampling from `VlmDatasetExporter` helper path

---

## 4) UI Exposure Plan (Minimal)
Add a single `Terrain Tools` panel in existing ImGui sidebar.

## Section A: Export
- Tile selector
- Buttons:
  - `Export Heightmap`
  - `Export Alpha Masks`
  - `Export MCCV`

## Section B: Import
- File pickers:
  - Heightmap path
  - Alpha mask path + layer selector
  - MCCV image path
- Buttons:
  - `Preview`
  - `Apply`
  - `Revert Pending`

## Section C: Visualization
- `Show Alpha Mask` (existing)
- `Use MCCV` (new)
- `MCCV Strength` slider (new)

No painting/brush UX in this phase.

---

## Delivery Phases

## Phase 0 — Foundations (1 PR)
- Extract shared terrain image conversion utilities into `MdxViewer` service layer.
- Add immutable DTOs for import/export payloads.

## Phase 1 — Big Alpha Correctness (1 PR)
- Wire `StandardTerrainAdapter` to `AlphaMapService`.
- Add instrumentation logs for per-layer decode mode.
- Verify on representative 3.3.5 tiles.

## Phase 2 — MCCV Render Path (1 PR)
- Carry `MccvData` through adapter → chunk → mesh.
- Add shader support and toggles.
- Visual validation vs known tiles.

## Phase 3 — Export Tools (1 PR)
- Heightmap/alpha/MCCV export from current loaded tiles.
- Add output conventions and metadata sidecars.

## Phase 4 — Import Tools (1–2 PRs)
- Heightmap import/apply.
- Alpha mask import/apply.
- MCCV import/apply.
- Add preview + rollback buffer.

## Phase 5 — Ghidra Closure + Write-Back Hardening
- Resolve remaining semantic details via RE checklist.
- Lock binary write-back behavior to confirmed client semantics.

---

## Validation Matrix
For each tile sample (Alpha, 0.6.0, 3.3.5):
1. Parser mode selected correctly (compressed/big/packed)
2. Overlay blend matches expected in viewer
3. MCCV on/off delta behaves predictably
4. Export→Import roundtrip preserves visual output
5. No seam regression at chunk borders (alpha and color)

---

## Risks / Mitigations
- Risk: mixed alpha encodings in same dataset
  - Mitigation: per-layer decode via flags + robust fallback
- Risk: MCCV channel-order mismatch
  - Mitigation: explicit swizzle tests + toggleable channel debug mode
- Risk: write-back corruption
  - Mitigation: preview buffer + backup + no in-place writes without confirmation

---

## Immediate Next Implementation Sprint
1. Phase 1 implementation (big-alpha decode unification) in `StandardTerrainAdapter`
2. Add MCCV field plumbing in chunk/mesh DTOs (without shader use yet)
3. Add minimal `Terrain Tools` panel with export-only actions
4. Run tile-based regression validation on 3.3.5 sample set
