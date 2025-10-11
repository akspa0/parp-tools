# PM4Faces Export Faces – Minimal Plan (2025-08-17)

- **Objective**: Restore complete faces/surfaces in PM4FacesTool exports (OBJ) to match PoC behavior, and emit a merged render mesh alongside objects/tiles with minimal changes.

- **Current context**:
  - MSCN remap is applied by `src/parpToolbox/Services/PM4/Pm4GlobalTileLoader.cs` in `ApplyMscnRemap()` which:
    - Appends MSCN vertices with axis swap `new Vector3(v.Y, v.X, v.Z)`.
    - Remaps indices across tiles using per-tile thresholds.
  - `src/PM4FacesTool/Program.cs` `ProcessOne()` respects `--no-mscn-remap` via `Pm4GlobalTileLoader.LoadRegion(dir, pattern, !opts.NoMscnRemap)`.
  - Hypothesis: MSCN remap causes missing/stretched faces in merged and tile exports. Disabling it should restore PoC-like completeness.

### Status Update (2025-08-17)
- Implemented object-first assembly using `MSUR.IndexCount` for tiles and merged render mesh via `ExportRenderMeshMerged()`.
- Real-data validation shows coherent tiles and `tiles/render_mesh.obj`; `surface_coverage.csv` has no empty `obj_path` rows.
- Defaults preserved for simplicity; recommended flags: `--no-mscn-remap` and `--ck-min-tris 0`; no new diagnostics added.
- Documentation updated: `src/PM4FacesTool/README.md`, `memory-bank/pm4faces/progress.md`, `systemPatterns.md`, `activeContext.md` verified.

## Plan Steps

- **Step 1: Real-data diagnostic run**
  - Run with MSCN remap disabled and anchors preserved:
    - Flags: `--batch --ck-min-tris 0 --render-mesh-merged --no-mscn-remap`
    - Input: representative tile (e.g., `Stormwind_37_49.pm4`)
  - Validate outputs:
    - `tiles/render_mesh.obj` has full faces.
    - Per-tile OBJs show complete faces.
    - `surface_coverage.csv` has no empty `obj_path`.
    - Optionally scan `objects_index.json` for extremely low-triangle groups.

- **Step 2: Branch on result**
  - If geometry is fixed:
    - Minimal change: default MSCN remap OFF in PM4FacesTool path.
    - Keep a simple opt-in (e.g., retain `--no-mscn-remap` with default true, or add positive `--mscn-remap`); update `--help`.
  - If geometry still broken:
    - Implement minimal per-surface local vertex scope assembly (no projection, no global remap) to prevent cross-surface aliasing.
      - New simple toggle (default OFF), e.g., `--local-scope-assembly`.
      - Apply in `AssembleAndWrite()` and `AssembleAndWriteFromIndexRange()` to strictly build local vertices from current surface set only.
    - Re-run Step 1 validation.

- **Step 3: Validation protocol**
  - Always use real data.
  - Preserve anchors with `--ck-min-tris 0`.
  - Confirm:
    - `surface_coverage.csv` has no empty `obj_path` rows.
    - `tiles/render_mesh.obj` visually has faces.
    - `objects_index.json` has no unexpected ultra-low-triangle groups.

## Non-goals
- No extra diagnostics or complex instrumentation unless requested.
- No wide refactors; only minimal, targeted changes.

## Deliverables
- If Step 2a: default-remap-off change + help text update.
- If Step 2b: minimal local-scope assembly toggle + help text update.
- This plan file under `memory-bank/pm4faces/`.
