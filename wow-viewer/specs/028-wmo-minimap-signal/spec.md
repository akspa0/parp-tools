# Feature Specification: WMO Minimap Ground-Truth Signal

**Feature Branch**: `028-wmo-minimap-signal`

**Created**: 2026-05-29

**Status**: Draft

**Input**: User request to harvest and assemble WMO minimap images from game client DBC chain (WMOAreaTable → AreaTable → Map → BLP) as low-resolution ground-truth signals per unique WMO asset, and store them alongside the per-asset roof renders in the object_visual.zarr.

## Problem Statement

The per-asset roof library currently stores high-resolution rendered roof images and multi-angle perspective captures of each WMO/M2 asset. However, the game client also ships pre-rendered minimap textures for many WMOs (especially dungeons, buildings with interiors, and major landmarks). These minimap BLPs are a valuable low-resolution ground-truth signal: they show the actual artist-authored top-down view of each WMO as it appears in the game's minimap UI, including baked lighting, terrain blending at the base, and zone-appropriate color grading.

Decoding the DBC chain to resolve each WMO's minimap BLP, decoding the BLP, and storing the result in the object_visual.zarr gives downstream models a cheap, zero-cost ground-truth target that the client developers already curated.

## Scope

This feature owns the WMO→minimap resolution path:

1. Add a DBC reader for `WMOAreaTable.dbc` using the existing DBCD/WoWDBDefs infrastructure in `WowViewer.Core.IO`.
2. Add a DBC reader for `AreaTable.dbc` (if not already present) and `Map.dbc`.
3. For each unique WMO asset path, load the WMO root file (using the existing ScreenshotRenderer `LoadWmoData` path), extract the `wmoID` from `MOHD.wmoID`, and walk the DBC chain to resolve the minimap BLP path.
4. Decode the BLP (using existing `SereniaBLPLib` or the `BmpReader` infrastructure).
5. Resize the minimap image to a consistent crop size (128 or 256) and store it in the existing `object_visual.zarr` as `wmo_minimap` and `wmo_minimap_confidence`.
6. Add a Python validation script that produces a side-by-side atlas comparing rendered roof images with the client-shipped minimap textures.

Out of scope for the first slice:
- Rendering the WMO minimap from scratch (we use the shipped BLP as-is)
- Per-group minimaps (we only resolve root-level WMO minimaps)
- Minimap for M2/MDX doodad assets (only WMOs carry a `wmoID`)

## User Scenarios & Testing

### User Story 1 — WMO minimap images are resolved from DBC and stored alongside roof renders (Priority: P1)

A researcher can view the client-shipped minimap BLP for any WMO alongside its rendered roof image, enabling comparison between the synthetic top-down render and the artist-authored minimap.

**Why this priority**: This is the core value — the minimap is a free validated-signal source from the client itself.

**Independent Test**: Run the minimap resolver on a bounded set of WMOs (e.g., all WMOs from `duskwood` on build 3.3.5) and verify that at least one minimap BLP is found and stored in `object_visual.zarr`.

**Acceptance Scenarios**:

1. **Given** a staged game client with `WMOAreaTable.dbc`, `AreaTable.dbc`, `Map.dbc`, and minimap BLPs, **When** the minimap resolver runs, **Then** it emits at least one non-zero minimap image for WMOs that have a wmoID.
2. **Given** a WMO with no matching minimap BLP, **When** the resolver runs, **Then** it writes a zero array with confidence=0.
3. **Given** the minimap signals are written, **When** the object_visual.zarr is inspected, **Then** it contains `wmo_minimap` and `wmo_minimap_confidence` arrays with the same sample count as `roof_rgb`.

---

### User Story 2 — Side-by-side validation atlas compares rendered roofs to client minimaps (Priority: P2)

A reviewer can open a composite image showing the rendered roof image, the client minimap, and an overlay highlighting differences.

**Why this priority**: Validation enables quality triage — some minimaps may be oriented differently than our top-down renders.

**Independent Test**: Run the atlas builder on a build's object_visual.zarr and verify it produces at least one composite image.

**Acceptance Scenarios**:

1. **Given** a valid object_visual.zarr with `wmo_minimap` data, **When** the atlas builder runs, **Then** it emits a PNG showing roof / minimap / difference side by side.
2. **Given** a WMO with zero minimap, **When** the atlas builder runs, **Then** it skips that entry with a log message.

---

## Requirements

### Functional Requirements

- **FR-001**: The system MUST load `WMOAreaTable.dbc`, `AreaTable.dbc`, and `Map.dbc` from the staged game client using DBCD/WoWDBDefs.
- **FR-002**: For each unique WMO asset, the system MUST load the WMO root file's MOHD chunk and extract `wmoID`.
- **FR-003**: The system MUST resolve the WMO minimap BLP path via `WMOAreaTable.wmoID → AreaTable → Map → minimap file path convention`.
- **FR-004**: The system MUST decode the minimap BLP to an RGB uint8 array of consistent crop size (256×256).
- **FR-005**: The system MUST write the minimap image to `object_visual.zarr` as `wmo_minimap` array.
- **FR-006**: The system MUST write a confidence float32 array `wmo_minimap_confidence` where 1.0 = minimap found, 0.0 = not found.
- **FR-007**: The system MUST handle missing minimap BLPs gracefully (write zeros, confidence=0, continue).
- **FR-008**: The validation atlas builder MUST produce a composite PNG per sample with roof | minimap | overlay columns.
- **FR-009**: The resolver MUST run after the object_visual.zarr is built (as a post-processing step).
- **FR-010**: All code MUST live under `wow-viewer/` using existing DBCD/WoWDBDefs infrastructure.

### Key Entities

- **WMOAreaTable.dbc Row**: Maps a `(wmoID, wmoGroupID)` pair to an `AreaTable.dbc` ID. WMO root links to WMOAreaTable via `MOHD.wmoID` with groupID=0.
- **AreaTable.dbc Row**: Named zone/area with parent zone chain and map ID.
- **Map.dbc Row**: Map definition including minimap BLP path templates (e.g. `World/Minimaps/{mapName}/`).
- **Minimap BLP**: Pre-rendered top-down texture for a WMO, typically at 256×256 or 512×512, stored as a BLP in the game client's MPQ.
- **Object Visual Zarr**: The per-asset Zarr datastore at `output/datasets/object_roof_library/object_visual.zarr`.

## Success Criteria

- **SC-001**: The DBC chain resolves at least one valid minimap BLP path per WMO-heavy build.
- **SC-002**: The resolved minimap image is non-zero for at least 10% of WMOs in any given build.
- **SC-003**: The validation atlas can be inspected visually for at least one proof sample.

## Assumptions

- DBCD/WoWDBDefs are already integrated in `WowViewer.Core.IO` or can be called from the data-harvester Python side.
- The existing `SereniaBLPLib` or `BmpReader` infrastructure can decode the minimap BLP files.
- Not all WMOs will have a minimap BLP — only those referenced as interior zones/dungeons typically have them, plus some landmark exterior WMOs.
- The minimap BLP path convention follows the patterns found in `Map.dbc`'s minimap fields rather than a fixed path template.
- The first implementation can be a Python script that calls DBCD via subprocess to parse DBCs, or a C# tool that reads DBCs and BLPs and writes NPZ.

## Relationship to Other Specs

- **Depends on**: `025-object-roof-mask-library-and-minimap-sieve` for the object_visual.zarr store and asset path list.
- **Depends on**: Existing DBCD/WoWDBDefs integration for DBC reading.
- **Enriches**: `object_visual.zarr` with additional signal arrays.