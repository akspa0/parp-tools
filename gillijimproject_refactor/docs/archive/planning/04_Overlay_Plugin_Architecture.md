# Overlay Plugin Architecture Refactor Plan

**Author**: Cascade plan draft

**Last Updated**: 2025-10-04

---

## 1. Purpose

Provide a durable, reusable overlay system that:

- Treats every overlay as a plugin implementing a common lifecycle (discover → load → render → teardown)
- Standardizes data contracts so generators (`WoWRollback.Core`) and viewers (browser, CLI reports, potential web services) share manifests and payloads
- Supports sparse tiles, mixed media (JSON + raster), and future overlays (diffs, ADT grid, time-travel snapshots) without bespoke code paths
- Enables packaging the overlay runtime for reuse in other projects (e.g., WoWToolbox web services) and supports server-side rendering scenarios

---

## 2. Current Pain Points (Baseline Audit)

- **Monolithic manager**: `ViewerAssets/js/overlays/overlayManager.js` mixes fetching, caching, and layer visibility with ad-hoc logic per overlay
- **Diverse data shapes**: Terrain JSON, object overlays, shadow PNG metadata, area IDs, and future diff data lack a shared manifest schema
- **Sparse tile handling**: Shadow overlays and others treat 404s inconsistently; viewer retries repeatedly without manifest guidance
- **No dependency graph**: Area overlays depend on LK ADT accuracy, diff overlays need area context, but there is no central coordination
- **Limited reuse**: CLI generators emit viewer-specific structures; exporting overlays to web services or tests requires re-implementation

---

## 3. Proposed Architecture

### 3.1 Core Concepts

- **Overlay Registry**: JSON manifest describing available overlays for a given map/version. Includes plugin ID, supported tile ranges, required resources, and dependencies.
- **Overlay Plugin Interface (OPI)**: Each overlay implements:
  - `initialize(context)` – receive map/version, data root, shared services (fetch, cache, event bus)
  - `prefetch(tileSet)` – optional prefetch manifest stage for upcoming tiles
  - `loadTile(tileCoord)` – return structured payload / DOM layer descriptor / promise
  - `render(tileCoord, payload)` – draw onto map (Leaflet, WebGL, etc.)
  - `teardown()` – dispose resources when overlay disabled
- **Shared Services**: Provided by runtime
  - Resource fetcher with cache + sparse awareness
  - Tile math utilities (row/col ↔ lat/lng)
  - Event bus for overlay cross-communication (e.g., diffs referencing area boundaries)
  - Diagnostics logger (UI + console)

### 3.2 Data Contracts

- **Overlay Manifest (`overlay_manifest.json`)**
  ```json
  {
    "version": "0.5.3.3368",
    "map": "Azeroth",
    "overlays": [
      {
        "id": "terrain.properties",
        "plugin": "terrain",
        "title": "Terrain Properties",
        "tiles": "sparse",  // none, partial, complete
        "resources": {
          "tilePattern": "overlays/{version}/{map}/terrain_complete/tile_{col}_{row}.json"
        }
        // Note: tile pattern is col_row (not row_col) to match WoW tile naming
      },
      {
        "id": "objects.combined",
        "plugin": "objects",
        "subtype": "combined",
        "resources": {
          "tilePattern": "overlays/{version}/{map}/combined/tile_{col}_{row}.json"
        }
      },
      {
        "id": "shadow.overview",
        "plugin": "shadow",
        "resources": {
          "metadataPattern": "overlays/{version}/{map}/shadow_map/tile_{col}_{row}.json",
          "imagePattern": "overlays/{version}/{map}/shadow_map/{filename}"
        }
      }
    ]
  }
  ```

- **Tile Payload Schema** (plugin-defined but documented)
  - Terrain: JSON layers (properties, liquids, holes, area IDs)
  - Objects: array of markers with `uniqueId`, `changeType`, coordinates, metadata
  - Shadow: PNG overview path + chunk references
  - Diff: `changeType`, `before`, `after`, `magnitude`

### 3.3 Runtime Layers

- **Runtime Core (viewer)**
  - Loads manifest, instantiates plugin modules (ESM imports by plugin ID)
  - Maintains overlay state (visible toggles, opacity, filters)
  - Centralized caching (`Map<string, TileCacheEntry>`) keyed by overlay + tile
  - Handles plugin registration for future frameworks (e.g., React, standalone CLI renderer)

- **CLI Integration**
  - `ViewerReportWriter` writes manifest + overlay payloads
  - New CLI command to validate manifests (`wowrollback overlays validate`)
  - Exporter can serve overlays via local web service using same manifest

### 3.4 Plugin Modules (Initial Set)

- `terrain` – renders properties/liquids/holes/area ID overlays
- `objects` – draws combined/M2/WMO markers, diff states, legend integration
- `shadow` – displays PNG overview, optional chunk layers
- `adtGrid` – red 64×64 grid with tile labels + links (future work)
- `heatmap` (future) – coverage visualization (uniqueID density, diff magnitude)

---

## 4. Implementation Roadmap

### Phase A – Baseline Stabilization (Pre-Refactor)

- Roll back viewer overlays to last stable commit for reference
- Generate current manifests manually to confirm data coverage and missing tiles
- Instrument current viewer with diagnostics to catalog overlay failures

### Phase B – Core Infrastructure

1. **Manifest Generator** (`WoWRollback.Core/Services/Viewer/OverlayManifestBuilder.cs`)
   - Collate overlay availability per map/version
   - Mark sparse overlays to avoid repeated 404 attempts
2. **Viewer Runtime Core** (`ViewerAssets/js/overlay-runtime/`)
   - Implement plugin registry, manifest loader, shared services
   - Provide compatibility shim for existing overlay modules while migrating
3. **Plugin Interface Definition**
   - TypeScript declaration file for plugin lifecycle (optional now, but document thoroughly)

### Phase C – Plugin Migration

- **Terrain Plugin**: Wrap existing layer classes; ensure manifest-driven loads
- **Objects Plugin**: Merge `performObjectMarkerUpdate()` logic into plugin; integrate diff change types from redesign
- **Shadow Plugin**: Use metadata pattern; treat missing tiles as `sparse`
- **Area Plugin**: After LK ADT alignment, expose area boundaries/fill with debug toggle
- **ADT Grid Plugin**: Implement wow.tools-style grid with tile links (per earlier request)

### Phase D – Tooling & Validation

- CLI command `wowrollback overlays manifest --maps ... --versions ...` to regenerate manifests
- Validator verifying manifest references exist, overlay schemas conform, and tile counts match
- Automated tests for plugin lifecycle using Node + headless DOM mocks

### Phase E – Web Service Export (Stretch)

- Package overlay runtime as npm module (`@wowrollback/overlay-runtime`)
- Provide server-friendly loader that returns GeoJSON or canvases
- Document integration points for external projects (WoWToolbox, data viewers)

---

## 5. Dependencies & Integration

- **AreaTable Accuracy**: Ensure `02_CRITICAL_AreaTable_Fix.md` action items complete so terrain/area plugins consume reliable data
- **Diff Overlay Redesign**: Coordinate with `03_Rollback_TimeTravel_Feature.md` updates; diff plugin will rely on new `changeType` schema
- **Shadow Builder**: Enhance `McnkShadowOverlayBuilder` to declare sparse tiles explicitly in manifest
- **ADT Grid**: Requires tile metadata (mapName, row/col) and routing to `tile.html` once per-tile editor is re-enabled

---

## 6. Risks & Mitigations

- **Large Refactor Scope**: Mitigate by migrating one plugin at a time behind compatibility layer
- **Legacy Viewer Support**: Maintain temporary adapter to route manifest-driven runtime through existing UI controls until fully replaced
- **Data Explosion**: Manifest ensures overlays can mark themselves as sparse and avoid redundant downloads; plan future switch to WebP for PNG-heavy layers

---

## 7. Next Steps Checklist

- [ ] Restore stable viewer state (`git checkout -- ViewerAssets/js`) for baseline testing
- [ ] Build overlay manifest prototype for terrain + shadow overlays
- [ ] Draft plugin interface documentation and stub runtime core
- [ ] Schedule working session to migrate terrain plugin to new architecture
- [ ] Update planning docs (`02_CRITICAL_AreaTable_Fix.md`, `03_Rollback_TimeTravel_Feature.md`) with dependencies and cross-references

---

## 8. References

- `WoWRollback/ViewerAssets/js/overlays/overlayManager.js`
- `WoWRollback.Core/Services/Viewer/ViewerReportWriter.cs`
- Planning docs: `02_CRITICAL_AreaTable_Fix.md`, `03_Rollback_TimeTravel_Feature.md`
