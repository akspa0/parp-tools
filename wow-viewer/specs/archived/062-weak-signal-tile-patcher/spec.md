# Feature Specification: Weak Signal Tile Patcher — Pre-Computed Terrain Export

**Feature Branch**: `062-weak-signal-tile-patcher`

**Created**: 2026-06-13

**Status**: Draft

**Input**: User description: "The weak signal amplifier needs a simple tile patcher export function that allows loading the target map in, then have it analyze tiles for the weak signals and try to then just save edited ADT tiles — both alphaWDT and LK ADT should be the target outputs that we build. I'd like to then be able to load those loose files as either alphaWDT or LK ADTs on top of the original target map, so we can pre-compute the weak signal amplification and not slow down the viewer with real-time calculations."

## User Scenarios & Testing

### User Story 1 — Export Weak-Signal-Patched Tiles for a Map (Priority: P1)

As a terrain operator, I want to run a single command that loads a map (Alpha or LK), analyzes every tile for weak signal terrain, amplifies affected tiles, and writes the corrected tiles as loose ADT files to an output directory, so I can pre-compute the weak signal repair offline.

**Why this priority**: This is the core value — converting the real-time in-memory weak signal amplifier into a persistent file output. Without this, every viewer session re-computes the same expensive analysis.

**Independent Test**: Run `terrain-weak-signal-patch --map-path <staged-1.12.1-client>/World/Maps/<mapName> --output-dir <temp-output>`. The command completes and writes patched ADT files to `<temp-output>/World/Maps/<mapName>/`. At least one tile that was identified as weak-signal has a modified heightmap in its output ADT compared to the input.

**Acceptance Scenarios**:

1. **Given** a staged 1.12.1 client with a map that contains weak-signal tiles (e.g., development map), **When** `terrain-weak-signal-patch --map-path <path> --output-dir <out>` runs, **Then** the output directory contains `World/Maps/<mapName>/<mapName>_<x>_<y>.adt` files for every weak-signal tile, and a `weak_signal_patch_report.json` summarizing which tiles were patched.
2. **Given** a map with no weak-signal tiles, **When** the command runs, **Then** it writes zero ADT files and the report shows 0 tiles patched.
3. **Given** a weak-signal tile with height range < 3.0 and normal relief > 0.02, **When** the command processes it, **Then** the output ADT's MCVT heights differ from the input by more than 0.1m in at least one chunk.
4. **Given** the command completes successfully, **When** the output directory is set as a loose overlay root in the viewer, **Then** the viewer loads the patched terrain instead of the original for the patched tiles.

---

### User Story 2 — Export Both AlphaWDT and LK ADT Formats (Priority: P1)

As a terrain operator, I want the patcher to output both Alpha WDT format and LK ADT format from the same analysis, so I can use whichever format matches my target viewer version.

**Why this priority**: The viewer supports both Alpha and LK terrain formats. Exporting both from one run avoids duplicate analysis work.

**Independent Test**: Run the command with `--format both`. The output directory contains both `World/Maps/<mapName>/<mapName>.wdt` (Alpha format with patched tiles embedded) and `World/Maps/<mapName>/<mapName>_<x>_<y>.adt` (LK format, height-only patches via `AdtTerrainWriter`).

**Acceptance Scenarios**:

1. **Given** a map with weak-signal tiles, **When** `--format alpha` is specified, **Then** only an Alpha WDT file is written containing the corrected tiles.
2. **Given** a map with weak-signal tiles, **When** `--format lk` is specified, **Then** only LK ADT `_root.adt` files are written for the corrected tiles.
3. **Given** a map with weak-signal tiles, **When** `--format both` is specified, **Then** both Alpha WDT and LK ADT files are written.
4. **Given** an Alpha WDT output, **When** inspected with `WowViewer.Tool.Inspect`, **Then** the MCVT heights for patched tiles match the corrected values.

---

### User Story 3 — Configurable Detection Thresholds (Priority: P2)

As a terrain operator, I want to override the weak signal detection thresholds via CLI arguments, so I can tune the sensitivity for different maps without code changes.

**Why this priority**: Different maps have different terrain characteristics. Hardcoded thresholds miss some cases and over-flag others.

**Independent Test**: Run with `--min-normal-relief 0.05 --max-height-range 5.0`. The command uses the custom thresholds for detection.

**Acceptance Scenarios**:

1. **Given** `--max-height-range 10.0`, **When** the command runs, **Then** tiles with height range < 10.0 (instead of default 3.0) are considered weak-signal candidates.
2. **Given** `--min-normal-relief 0.05`, **When** the command runs, **Then** only tiles with normal relief >= 0.05 (instead of default 0.02) are flagged.
3. **Given** `--amplification-factor 2.0`, **When** the command runs, **Then** weak-signal heights are amplified by a factor of 2.0 (instead of auto-estimated).

---

### User Story 4 — Load Patched Tiles as Loose Overlay (Priority: P2)

As a terrain operator, I want to point the viewer's loose overlay root at the patcher output directory and see the corrected terrain, without any additional conversion steps.

**Why this priority**: The whole point of pre-computing is to avoid real-time cost. If loading requires extra steps, the value is diminished.

**Independent Test**: Export patches for a map, then launch the viewer with `--loose-overlay-root <patch-output>`. The viewer shows the corrected terrain for patched tiles.

**Acceptance Scenarios**:

1. **Given** patched LK ADTs in `<output>/World/Maps/<mapName>/`, **When** the viewer is launched with `--loose-overlay-root <output>`, **Then** the viewer loads the patched ADTs for the corrected tiles instead of the originals.
2. **Given** a patched Alpha WDT in `<output>/World/Maps/<mapName>/<mapName>.wdt`, **When** the viewer is launched with `--loose-overlay-root <output>`, **Then** the viewer loads the corrected Alpha terrain.
3. **Given** patched tiles exist only for a subset of map tiles, **When** the viewer loads the map, **Then** unpatched tiles load normally from the archive and patched tiles load from the overlay.

---

### Edge Cases

- **Map not found**: If `--map-path` does not point to a valid map directory, the command fails with a clear error.
- **Empty map (no terrain tiles)**: WMO-only maps have no terrain. The command reports 0 tiles and exits cleanly.
- **Mixed Alpha/LK formats**: Some maps may have tiles in Alpha format and others in LK format within the same map directory. The command should handle each tile in its native format.
- **Already-patched tiles**: Re-running the command should produce identical output (idempotent).
- **Seam continuity**: Patched tiles may have height discontinuities at tile edges with unpatched neighbors. Accept this for v1 — the patcher operates per-tile without cross-tile blending.

## Requirements

### Functional Requirements

- **FR-001**: System MUST provide a `terrain-weak-signal-patch` command in `WowViewer.Tool.Converter`.
- **FR-002**: The command MUST accept `--map-path <dir>` pointing to a map directory containing a WDT file.
- **FR-003**: The command MUST accept `--output-dir <dir>` for writing patched terrain files.
- **FR-004**: The command MUST accept `--format <alpha|lk|both>` to control output format (default: `both`).
- **FR-005**: The weak signal detection MUST use the same logic as the viewer runtime: height range within configurable Z band, normal relief threshold, minimum coverage.
- **FR-006**: Default detection thresholds MUST be `--max-height-range 3.0`, `--min-normal-relief 0.02`, `--min-normal-coverage 0.10`.
- **FR-007**: The command MUST support `--amplification-factor <float>` for manual factor override (default: auto-estimated from WDL coarse data and loaded-tile bounds).
- **FR-008**: For LK ADT output, the command MUST use `AdtTerrainWriter.Write()` to patch MCVT heights and MCNR normals in existing ADT files.
- **FR-009**: For Alpha WDT output, the command MUST use `AlphaWdtWriter.Build()` to write a complete WDT with corrected tile data.
- **FR-010**: The command MUST write a `weak_signal_patch_report.json` listing every tile processed, whether it was patched, detection metrics, and amplification factor used.
- **FR-011**: The command MUST copy unpatched tile ADTs to the output directory (full map overlay) so the viewer can load the complete map from the overlay root.
- **FR-012**: The command MUST be idempotent — re-running with the same inputs produces identical output.
- **FR-013**: Weak signal detection logic MUST be extracted from `ViewerApp.cs` into a shared library in `WowViewer.Core.IO` or `WowViewer.Core.Runtime` so both the viewer and converter tool can use it.
- **FR-014**: The output directory structure MUST mirror the game client's virtual file structure: `World/Maps/<mapName>/<mapName>.wdt` and `World/Maps/<mapName>/<mapName>_<x>_<y>.adt`.
- **FR-015**: The command MUST support `--client-root <dir>` for staged client data access.
- **FR-016**: All new code MUST live in `wow-viewer/src/` (libraries) or `wow-viewer/tools/converter/` (command).

### Key Entities

- **WeakSignalTileAnalysis**: Per-tile detection result: tile coordinates, height range, normal relief, normal coverage, whether flagged as weak-signal, severity classification.
- **WeakSignalAmplificationPlan**: Per-tile amplification parameters: factor, anchor height, source height range, WDL reference range.
- **WeakSignalPatchReport**: JSON summary of the patching run: map name, detection thresholds, tiles processed/patched/skipped, per-tile details.
- **PatchedTerrainTile**: A tile with corrected heights and normals, ready for output in either Alpha or LK format.

## Success Criteria

- **SC-001**: The command successfully patches at least 1 tile in the development test map (`World/Maps/development`).
- **SC-002**: The patched LK ADT loads correctly in the viewer when set as a loose overlay root.
- **SC-003**: The patched Alpha WDT loads correctly in the viewer when set as a loose overlay root.
- **SC-004**: Re-running the command on the same map produces bit-identical output (idempotency).
- **SC-005**: The patch report JSON contains accurate detection metrics for every tile.
- **SC-006**: `dotnet build` and `dotnet test` pass on the affected projects.

## Assumptions

- The staged 1.12.1 client at `output/tmp/wowarchive-clients/` contains valid WDT and ADT files for the target map.
- The existing `AdtTerrainWriter` and `AlphaWdtWriter` can produce valid output files without modification (Rule 10 frozen surfaces).
- The weak signal detection logic from `ViewerApp.cs` can be extracted into a shared location without changing its behavior.
- The viewer's existing loose overlay system (`--loose-overlay-root`) can consume the patcher's output directory structure without modification.
- Height-only patching (MCVT + MCNR) is sufficient for v1. Texture, alpha, liquid, and placement patching is out of scope.
- Cross-tile seam blending is out of scope for v1. Per-tile amplification is independent.
- The WDL file for the target map is available for coarse height reference during amplification factor estimation.
