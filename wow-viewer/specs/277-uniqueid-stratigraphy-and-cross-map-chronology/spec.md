# Spec 277 — UniqueId Stratigraphy & Cross-Map Chronological Fork Analysis

**Feature Branch**: `v0.6.1-dev`  
**Created**: 2026-10-18  
**Status**: Specified; in implementation  
**Parent**: [Epic 250 (Reconstruction & Editor Platform)](../250-epic-reconstruction-and-editor-platform/spec.md) & [Epic 252 (World Simulation, Audio & Interaction)](../252-epic-world-simulation-and-audio/spec.md)  
**Governance**: `AGENTS.md` §4 (Code Ownership & Architectural Boundaries), §9 (Governance, Scope Fidelity & Receipts), §10 (God-Class Freeze)

---

## 1. Problem Statement & Background

### 1.1 The Archeological Significance of UniqueIds
In World of Warcraft client data, every placed object is assigned a 32-bit `UniqueId`:
- **ADT `MDDF` Chunk**: Placed M2 doodads (trees, lampposts, debris, furniture).
- **ADT `MODF` Chunk**: Placed WMO structures (buildings, dungeons, bridges, castles).
- **Server SQL Population (`creature` & `gameobject` tables)**: Spawn GUIDs / `SpawnId`.

Because Blizzard's internal world editor (`WoWEdit` / `WorldEdit`) allocated these IDs monotonically (or in sequential designer blocks) during world authoring from 2001 to 2004+, the numeric magnitude of a `UniqueId` acts as a **fossilized relative timestamp**:
- Lower numbers represent earlier placements (foundational geometry, ancient terrain anchors).
- Higher numbers represent later passes (quest hub polish, expansion content, bug fixes).

### 1.2 Cross-Map Branching & Overworld Divergence
When Blizzard created instanced dungeons, battlegrounds, or the development testing map (Map 451 `development`), they typically duplicated an area of the overworld (Map 0 Azeroth or Map 1 Kalimdor). 
- The copied assets **retained their original overworld `UniqueId` values** from the date of duplication.
- Any subsequent assets placed specifically within the instance received much higher, contemporaneous `UniqueId` values.
- Currently, the viewer has no automated method to correlate object positions and models across maps to identify the exact divergence boundary ($UID_{div}$) where an instance branched from the overworld, or to date the different testing eras preserved in Map 451.

### 1.3 Tooling Gaps
1. **No Whole-Client Stratigraphy Profiler**: Existing `map uniqueid-report` in `WowViewer.Tool.Inspect` operates on isolated files or directories without whole-client map cross-referencing or population awareness.
2. **Missing Cross-Map Twin & Divergence Detection**: No mechanism automatically matches objects by spatial coordinates and model path across different maps to calculate divergence boundaries.
3. **In-Viewer Visualization Gaps**: While `SceneObjectFilters` supports basic min/max ID filtering, there is no Epoch Heatmap mode to visually distinguish early alpha objects from beta/release objects in the 3D viewport, nor is there a per-tile age inspector.

---

## 2. User Stories

- **US1 (Whole-Client UniqueID Stratigraphy Profiler)**: As a world researcher, I want a tool to scan all maps in a client (ADTs and population datasets) and output a comprehensive stratigraphy ledger summarizing UID ranges, density, and detected historical layers.
- **US2 (Cross-Map Divergence & Clone Point Identification)**: As an archive researcher, I want to compare any instanced or test map against overworld baseline maps to find identical object placements, determine the exact branching UID, and partition instance objects into "inherited overworld base" vs "post-fork additions".
- **US3 (Interactive "Time Machine" UID Scrubber)**: As a viewer operator, I want an intuitive chronological scrubber in the Archaeology panel that lets me roll back world placements continuously to watch older map states emerge.
- **US4 (Visual Era Heatmap in 3D Viewport)**: As a viewer operator, I want a false-color viewport mode that color-codes placed objects by their historical allocation era (e.g. Early Alpha, Late Alpha, Beta, Release).

---

## 3. Acceptance Criteria

- **AC-1 (Stratigraphy Analysis Engine)**:
  - Implement `UniqueIdStratigraphyService` in `WowViewer.Core` to analyze UID distributions, calculate quantiles (p10, p25, median, p75, p90), and detect layer breaks across map collections.
  - Support ingesting both ADT placement catalogs (`AdtPlacementCatalog`) and SQL population datasets (`PopulationMapData`).
- **AC-2 (Cross-Map Divergence Engine)**:
  - Implement `CrossMapDivergenceService` in `WowViewer.Core` that matches placements between a target map and a candidate reference map using spatial tolerance ($\le 0.5\text{ units}$) and model identity.
  - Calculate the shared UID percentage, the highest shared UID ($UID_{div}$), and generate a classification list marking each object as `Inherited` or `PostFork`.
- **AC-3 (CLI Stratigraphy & Divergence Tool)**:
  - Add `inspect map stratigraphy` command in `WowViewer.Tool.Inspect` supporting whole-client scans and cross-map divergence queries.
  - Produce machine-readable JSON reports (`output/reports/stratigraphy/`).
- **AC-4 (In-Viewer Archaeology Workbench Enhancements)**:
  - Update `ArchaeologyPanelService.Sidebars.cs` to show map era statistics, detected layers, and overworld divergence details.
  - Integrate smooth "Time Machine" slider control.
- **AC-5 (Unit Test Verification)**:
  - Unit tests covering UID distribution quantiles, layer gap detection, and cross-map twin matching with 100% test pass rate.
