# Spec 293 — Restoration Workbench (Layer-Driven ADT Restoration)

**Status:** DRAFT — awaiting operator approval (AGENTS.md §9.1). No code written.
**Origin:** Operator request 2026-11-04 (wording preserved below). Expands the original layer-driven restoration scope into the full Restoration Workbench. Follows Specs 222/232 (phase layers), 281 (museum ADT patcher), 287/289 (PM4 matching).

## Operator requirements (verbatim intent)

> "we were patching ADT data that matched up to other data we had in the 3.0.1 alpha/beta builds, to resolve missing map data in the development map. I want to see that work, but with our layers as the driver for the data, so we match proper things up to proper things perfectly. How can we do that? The viewer and map/tile layers become a major restoration part of the tooling/ui, now."

> "when we overlay entire maps, I'd like it if I could have it not render some of the tiles of the map, selecting the ones we don't want, using the same selection as the spawn map again, but we really need to make that map zoomable and more obvious what we're selecting - give each square a tool tip to show the XX_YY of the selection, so we know what we're selecting is what we intend to select, both in the donor tile and donor map layer selection."

> "let's now use speckit to plan a feature that lets us use donor tiles and donor maps from any version, loading files from any client's data that we have on-disk, the user has to select from the known-good clients list, then we handle reading the data out of that client's maps. This is for the Restoration Workbench that we need to build, now that we have PM4 'workbench' tasks (human curation) stuff sorta figured out. Let's wrap it all up in a single interface, that makes use of both sidebars, to help build complex layered restorations without it being a royal pain to find the settings in the sidebars. This in turn will be how we further refine and clean up the ui. The Texture Masks button should be a left sidebar button, not a toolbar button. The Alpha button needs to be all 3 layers, and a button to do blending adjustments, as a simple pop-up panel like what we initially had for that feature."

## 1. Overview & Problem Statement

Restoration of missing map data in the development map has so far been a batch operation: a tool scans PM4 files, guesses asset identity by shape distance, and writes monolithic ADTs for terrain-less tiles. That path is blind to the operator's judgment and to the authentic donor data already loaded in the viewer's phase-layer stack.

The viewer already holds the authoritative restoration context: a stack of donor-map layers with per-tile donor→target placements, channel masks, transforms, locks, and a persisted project file. The operator wants that stack to **drive** matching and patching — donor placements are authentic (real model names and poses), so matching a donor placement to a PM4 hull yields perfect asset identity instead of a shape guess.

The operator now also wants donor data from **any client version on disk**, chosen from the known-good clients list, and a **single Restoration Workbench interface** that uses both sidebars so complex layered restorations do not require hunting through scattered panels. The same pass refines the UI: Texture Masks becomes a left-sidebar button, the Alpha control covers all three overlay layers, and blending adjustments return as a simple pop-up panel.

## 2. User Stories

### US1 — Layer-driven gap report
As an operator, I want the viewer to tell me which base-map tiles are missing object data that my donor layers can supply, so that I restore only the tiles that need it and only from the layers I chose.

### US2 — Donor↔hull matching with taught pairings
As an operator, I want donor placements matched to PM4 hulls on the target tile, using the taught-pairing database, so that each hull receives the authentic model name and pose rather than a shape guess.

### US3 — ADT export from the resolved layer composition
As an operator, I want to export patched ADTs built from the resolved layer composition (donor placements transformed by the layer, plus PM4-solved placements for uncovered hulls), so that the output matches what I see in the viewer.

### US4 — Zoomable donor tile/map selection with XX_YY tooltips
As an operator, I want to zoom the donor tile/map picker and see each tile's `XX_YY` coordinate in a tooltip, and to exclude tiles I don't want rendered, so that I select exactly the tiles I intend — in both the donor tile picker and the donor map layer selection.

### US5 — Donor data from any client version on disk
As an operator, I want to choose a client from the known-good clients list and have the viewer read donor maps and tiles directly out of that client's data on disk, so that I can restore from any version I have installed without converting or copying files first.

### US6 — One Restoration Workbench across both sidebars
As an operator, I want a single Restoration Workbench that uses the left and right sidebars together — donor sources and layer stack on one side, the selected tile/hull detail and actions on the other — so that building a complex layered restoration does not mean hunting for settings across unrelated panels.

### US7 — UI refinements
As an operator, I want Texture Masks to be a left-sidebar button (not a toolbar button), the Alpha control to cover all three overlay layers, and a blending-adjustments button that opens a simple pop-up panel, so that the texture-mask workflow is where I expect it and the controls match the original feature.

## 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| **AC-001** | A restoration gap report lists, per base-map tile, whether the base has object data, what the enabled layer stack contributes, and whether the tile is a restoration candidate. | Report produced in-viewer for a real map + layer project; counts cross-checked against the layer footprints. |
| **AC-002** | The gap report is driven by the live layer stack (enabled layers, channels, per-tile placements, `UsePlacedTilesOnly`, locks) — disabling a layer or removing a placement changes the report. | Toggle a layer/placement and observe the report change. |
| **AC-003** | Donor placements are matched to PM4 hulls using the existing direct matcher and taught pairings; a matched hull records the donor model path and the solved pose. | Match run on a real tile; matched hulls show donor names; taught rules are reused on a second tile. |
| **AC-004** | Hulls with no donor coverage fall back to PM4 shape-solve placement, clearly marked as inferred rather than donor-authentic. | Report distinguishes donor-matched vs inferred placements. |
| **AC-005** | ADT export writes monolithic WotLK ADTs from the resolved layer composition to a separate output directory; source files are never overwritten; every written file is round-trip verified. | Export run on a real tile; round-trip check passes; source directory unchanged. |
| **AC-006** | Export refuses (and reports) tiles that still have surviving terrain, until an existing-ADT placement reader exists. | Export a terrain-bearing tile; it is skipped with a reason. |
| **AC-007** | The donor tile picker and donor map layer selection are zoomable, and every tile shows an `XX_YY` tooltip on hover. | Operator visual check on a real map. |
| **AC-008** | Every placed donor tile in a layer is listed with an include/exclude checkbox; unchecking it removes that tile from the layer's composition without deleting the mapping, and re-checking restores it. | Toggle a placed tile; the composed map changes; the mapping survives a project save/load. |
| **AC-008b** | Each placed tile exposes per-tile settings (Z offset and Z scale) applied on top of the layer's, with a reset. | Adjust a per-tile Z offset; only that tile moves; reset restores it. |
| **AC-009** | The operator selects a donor client from the known-good clients list; the viewer reads that client's maps/tiles directly from disk and reports the resolved client root and build. | Select a client, load a donor map, confirm the reported root/build and that tiles render from that client. |
| **AC-010** | Donor sources from different client versions can coexist in one restoration project; each layer records which client it came from. | Build a project with two clients; both layers resolve and render. |
| **AC-011** | The Restoration Workbench is a single interface spanning both sidebars: donor sources + layer stack on one side, selected tile/hull detail + actions on the other. | Operator visual check; no restoration setting requires leaving the workbench. |
| **AC-012** | Texture Masks is a left-sidebar button, not a bottom-toolbar button. | Operator visual check. |
| **AC-013** | The Alpha control covers all three overlay layers (L1–L3) in one control. | Toggle each of the three alpha layers from the control. |
| **AC-014** | A blending-adjustments button opens a simple pop-up panel with the original blending controls. | Open the pop-up; adjust blending; scene updates. |
| **AC-015** | All new UI uses `SharedUiWidgets`, registers in `ui-controls.json`/`UI-REFERENCE.md`, and `Verify-UiReference.ps1` passes with 0 errors/warnings. | Script run. |
| **AC-016** | All modified source files respect the ≤2,000-line budget (AGENTS.md §10); new behavior lives in owned service classes, not god-class partials. | Line counts + review. |

Criteria naming visible/runtime behavior (AC-001, AC-003, AC-005, AC-007–014) require an operator run on real client data; build/tests alone do not satisfy them (§9.2).

## 4. Functional Requirements

- **FR-1 Gap detection**: For each tile in the base map, resolve the layer stack's contribution using the existing phase-composition policy and compare it with the base tile's object data. Classify: `base-complete`, `layer-supplied`, `restoration-candidate`, `no-data`.
- **FR-2 Layer authority**: The report and export consume the live `PhaseLayerSettings` stack (enabled state, channels, `TilePlacements`, `UsePlacedTilesOnly`, locks, transforms). No separate configuration source.
- **FR-3 Donor extraction**: For a candidate tile, read the donor tile's MDDF/MODF placements through the layer's donor map and apply the layer transform (tile offset, rotation approximation, Z offset/scale) to obtain target-space donor placements.
- **FR-4 Matching**: Match donor placements to PM4 hulls with the direct matcher and taught pairings; donor identity wins on a match. Unmatched hulls may receive inferred placements, marked as such.
- **FR-5 Export**: Write monolithic ADTs from the resolved composition to a separate output directory, with round-trip verification and the existing terrain-preservation guard.
- **FR-6 Tile selection UI**: Zoomable donor tile/map picker with per-tile `XX_YY` tooltips and per-tile render exclusion, shared by the donor tile picker and the donor map layer selection. Every placed tile in a layer is listed with an include/exclude checkbox and per-tile settings (Z offset, Z scale) that persist with the layer project.
- **FR-7 Persistence**: Restoration selections and exclusions persist with the layer project.
- **FR-8 Donor client selection**: The operator picks a donor client from the known-good clients list; the workbench resolves that client's data source and reads maps/tiles from it directly.
- **FR-9 Multi-client layers**: Each layer records its donor client; layers from different clients can coexist in one project.
- **FR-10 Unified workbench**: A single Restoration Workbench surface uses both sidebars — left: donor clients, donor maps, layer stack, tile selection; right: selected tile/hull detail, match results, export actions.
- **FR-11 Texture Masks placement**: Texture Masks is a left-sidebar button; the bottom toolbar no longer carries it.
- **FR-12 Alpha + blending**: One Alpha control covers all three overlay layers; a blending-adjustments button opens a simple pop-up panel with the original blending controls.

## 5. Key Entities

- **Restoration candidate**: base tile + layer contributions + gap classification.
- **Donor client**: a known-good client entry (root path, build/version) selectable as a donor source.
- **Donor placement**: model path, position, rotation, scale, source donor tile, layer index, donor client.
- **Hull match**: PM4 hull key + donor placement (or inferred placement) + confidence + provenance (`donor` | `inferred`).
- **Tile exclusion set**: per-layer set of donor tiles excluded from rendering/export.

## 6. Out of Scope

- New PM4 decoding or format work.
- Terrain reconstruction (this spec restores object placements only).
- Rewriting the phase-composition engine; the spec consumes it.
- Automatic matching without operator review for donor-authentic output.
- Converting donor client data to another format before use.

## 7. Assumptions

- The known-good clients list already exists and exposes client roots/builds.
- Donor maps from any selected client are loadable through the existing data-source/phase-layer adapters without pre-conversion.
- The taught-pairing database and direct matcher are sufficient for donor↔hull matching; no new matcher is required.
- Export remains limited to terrain-less tiles until an existing-ADT placement reader is implemented (tracked separately).
- The zoomable picker reuses the existing minimap/footprint interaction model.

## 8. Open Items for Operator

1. Should inferred (non-donor) placements be exported by default, or only donor-authentic ones?
2. Should tile exclusions apply to rendering only, or also to export?
3. When a donor client is selected, should the workbench auto-discover its maps, or require the operator to pick a map first?
