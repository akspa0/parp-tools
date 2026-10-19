# Plan — Spec 277: UniqueId Stratigraphy & Cross-Map Chronological Fork Analysis

## Architectural Approach

```
┌────────────────────────────────────────────────────────┐
│ 1. WowViewer.Core / Archaeology                        │
│    - UniqueIdRecord (unified ADT + SQL placement)      │
│    - UniqueIdStratigraphyEngine (quantiles, layers)    │
│    - CrossMapDivergenceEngine (spatial twin matching)  │
│    - MapStratigraphyProfile & MapDivergenceReport      │
└──────────────────────────┬─────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│ 2. WowViewer.Tool.Inspect │ │ 3. WoWViewer Client       │
│    - map stratigraphy cmd │ │    - Archaeology Panel    │
│    - Whole-client profiler│ │    - Time Machine Scrubber│
│    - Divergence reporter  │ │    - Era/Layer Inspector  │
└───────────────────────────┘ └───────────────────────────┘
```

---

## Phases & Deliverables

### Phase 1: Core Stratigraphy & Divergence Engines (WowViewer.Core)
1. **`UniqueIdRecord` & Models** ([`UniqueIdChronologyModels.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Archaeology/UniqueIdChronologyModels.cs)):
   - Unified record type capable of representing ADT `MDDF` (M2), `MODF` (WMO), and server SQL spawns.
2. **`UniqueIdStratigraphyEngine`** ([`UniqueIdStratigraphyEngine.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Archaeology/UniqueIdStratigraphyEngine.cs)):
   - Quantile computations ($p_{10}, p_{25}, \text{median}, p_{75}, p_{90}$).
   - Layer clustering algorithm using adaptive gap thresholds.
   - Per-tile density and chronological summaries.
3. **`CrossMapDivergenceEngine`** ([`CrossMapDivergenceEngine.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Archaeology/CrossMapDivergenceEngine.cs)):
   - Spatial kd-tree / hash cell matching for twin placements across maps ($\le 0.5\text{ distance}$ and model filename match).
   - $UID_{div}$ computation: highest shared UniqueID before branching occurred.
   - Classification of instance placements into `InheritedOverworld` vs `PostForkAuthored`.

### Phase 2: CLI Inspection Tooling (WowViewer.Tool.Inspect)
1. **`MapStratigraphySupport.cs`**:
   - `inspect map stratigraphy`:
     - `--input <path|dir>`: Profile ADT/WDT placement files.
     - `--population <dir>`: Ingest extracted population dataset partitions.
     - `--divergence --target <mapId> --ref <mapId>`: Run cross-map branching analysis.
     - `--output <path>`: Write JSON report.

### Phase 3: In-Viewer Workbench Integration (WoWViewer)
1. **`ArchaeologyPanelService.Sidebars.cs`**:
   - Integrate `UniqueIdStratigraphyEngine` directly into the live scene.
   - Add "Time Machine" mode with timeline scrubbing, showing the chronological era of visible geometry.
   - Add Cross-Map Divergence summary for instances and development maps.

### Phase 4: Unit Testing & Receipts
1. Unit tests in `tests/WowViewer.Core.Tests/UniqueIdStratigraphyTests.cs`:
   - Synthetic twin matching and divergence cutoff calculation.
   - Quantile accuracy and layer gap detection.
   - Real data verification on Map 451 and 0.5.3 / 3.3.5a population.
