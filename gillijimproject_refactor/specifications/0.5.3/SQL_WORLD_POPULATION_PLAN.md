# 0.5.3 SQL-Driven World Population Plan (Alpha-Core)

## Goal

Populate the viewer world with representative **NPCs** and **GameObjects** directly from alpha-core SQL dumps, so we can validate gameplay-like map population **without** implementing full legacy client networking.

Primary UX target:
- Load a map in `MdxViewer`
- Toggle SQL world population on
- See nearby NPC/GO spawns placed into the same world scene as terrain/WMO/MDX doodads

---

## Recommendation: Data-driven first, fake server optional

Use a **direct SQL ingestion path** as the default architecture. A fake server can be added later only if/when we need packet-level behavior testing.

Why this is the right first step:
- We already have SQL parsing + spawn extraction in `Catalog/AlphaCoreDbReader.cs`
- We already track current map id/name in `ViewerApp`
- We already have a robust world instance pipeline in `Terrain/WorldScene.cs`
- It avoids protocol complexity while delivering the exact value we want: visible representative world population

---

## Existing assets we can reuse

- `Catalog/AlphaCoreDbReader.cs`
  - Reads `world.sql` and `dbc.sql`
  - Loads templates and spawn tables (`spawns_creatures`, `spawns_gameobjects`)
  - Resolves creature and gameobject model paths from DBC SQL
- `Catalog/AssetCatalogEntry.cs`
  - Already carries spawn list with map/x/y/z/orientation
- `ViewerApp.cs`
  - Maintains `_currentMapId` after world load
- `Terrain/WorldScene.cs`
  - Centralized instance management, model caching, culling, selection, and render path

Key conclusion: we are not starting from scratch; this is mostly integration and runtime filtering.

---

## Architecture (MVP)

### 1) New world-population domain model

Add a dedicated runtime type (not tied to catalog UI):
- `WorldSpawnRecord`
  - `EntryId`, `SpawnId`, `MapId`, `Type (Creature/GameObject)`
  - `ModelPath`, `Scale`, `DisplayScale`, `EffectiveScale`
  - `PositionWow`, `OrientationWow`
  - optional metadata (`Name`, `Subname`, `Faction`, `NpcFlags`, `GameObjectType`)

Purpose:
- decouple “catalog browsing/export” from “streamable world population”
- keep memory footprint predictable for map streaming

### 2) SQL cache/index service

Introduce `SqlWorldPopulationService` that:
- validates alpha-core root
- parses SQL once (or loads a serialized cache)
- builds map-scoped indexes:
  - `mapId -> List<WorldSpawnRecord>`
  - optional tile buckets for fast AOI queries

Suggested cache artifact:
- `%LOCALAPPDATA%/MdxViewer/sql-world-cache/{hash}.json`
- Hash over `world.sql` + `dbc.sql` file metadata (size + last write time)

### 3) Scene injection layer

Add a focused API in `WorldScene`:
- `SetExternalSpawns(...)` for current map
- `ClearExternalSpawns()`
- keep external lists separate from ADT-derived placements

Render/culling should reuse existing instance path and asset manager.

### 4) Coordinate conversion

Use the same WoW -> renderer conversion already used for placements:
- `rendererX = MapOrigin - wowY`
- `rendererY = MapOrigin - wowX`
- `rendererZ = wowZ`

For orientation:
- start with yaw-only rotation around Z from `spawn_orientation`
- tune sign/offset empirically to match expected facing

---

## Phased implementation

## Phase 0 — Schema + diagnostics (1 short sprint)

Deliverables:
- `WorldSpawnRecord` model
- parser path that extracts only fields required for rendering
- map-level stats logging:
  - creatures total, gameobjects total
  - with model path vs missing model path

Acceptance:
- For a selected alpha-core root, we can print per-map spawn counts and model-resolution rates.

## Phase 1 — Manual map population toggle (MVP)

Deliverables:
- New UI section in world/debug panel:
  - `Enable SQL Spawns`
  - `Load current map spawns`
  - density slider or cap (`Max SQL Spawns`)
  - type toggles (`NPC`, `GameObject`)
- On load:
  - filter by `_currentMapId`
  - build external instances
  - inject into `WorldScene`

Acceptance:
- On any loaded map, user can click once and see representative NPC/GO models appear at SQL spawn coordinates.

## Phase 2 — AOI/tile-aware streaming

Deliverables:
- bucket spawns by tile index to match terrain AOI
- load/unload external instances as tiles stream in/out
- keep frame time stable on dense maps

Acceptance:
- No large stalls when enabling SQL spawns on high-density maps.

## Phase 3 — Quality/UX hardening

Deliverables:
- metadata in selection panel: source=`SQL`, entry id, spawn id, faction/type
- missing model fallback marker (optional impostor/point)
- optional deterministic sampling mode for very dense zones

Acceptance:
- Selection/debug info clearly distinguishes ADT doodads vs SQL runtime spawns.

---

## Concrete file touch plan

### New files (proposed)

- `src/MdxViewer/Population/WorldSpawnRecord.cs`
- `src/MdxViewer/Population/SqlWorldPopulationService.cs`
- `src/MdxViewer/Population/SqlSpawnCoordinateConverter.cs`

### Modified files

- `src/MdxViewer/ViewerApp.cs`
  - add UI toggles + lifecycle hooks
  - trigger service load for `_currentMapId`
- `src/MdxViewer/Terrain/WorldScene.cs`
  - add external spawn instance lists + injection API
  - include them in render/selection/culling paths
- `src/MdxViewer/Catalog/AlphaCoreDbReader.cs` (optional)
  - extract shared parsing helpers for `Population` service

---

## Data mapping contract

### Creature chain

- `creature_template.display_id1..4`
- `CreatureDisplayInfo.ID -> ModelID, scale, textures`
- `mdx_models_data.ID -> ModelName`
- `spawns_creatures.spawn_entry1 + map + pos + orientation`

### GameObject chain

- `gameobject_template.displayId + type + size`
- `GameObjectDisplayInfo.ID -> ModelName`
- `spawns_gameobjects.spawn_entry + map + pos + orientation`

### Runtime spawn resolution rule

For each spawn row:
1. resolve template by entry id
2. resolve model path
3. if model missing, either skip or emit fallback marker
4. transform coordinates to renderer space
5. instantiate into external spawn list

---

## Performance guardrails

- hard cap default: `Max SQL Spawns = 2000` (user-configurable)
- lazy model loading through existing `WorldAssetManager`
- reuse instance structs + frustum culling
- parse SQL once, then serve from in-memory map index or disk cache

---

## Validation checklist

- Spawn count sanity:
  - SQL map spawn count vs injected instance count (plus skipped reasons)
- Position sanity:
  - known NPC hubs visually align to expected locations
- Orientation sanity:
  - facing direction approximately matches expected orientation
- Stability:
  - toggling spawns on/off repeatedly does not leak memory or duplicate instances

---

## Optional fake-server track (deferred)

If needed later for protocol simulation:
- build a tiny local “spawn snapshot” server that streams only create/update packets
- back it with the same `SqlWorldPopulationService`
- keep it isolated from auth/quest/combat emulation

This keeps packet testing possible without blocking the immediate world-population objective.

---

## Immediate next actions

1. Implement `Population` service and map-index cache (Phase 0)
2. Add `WorldScene` external-spawn injection API
3. Add one-button `Load SQL spawns for current map` in `ViewerApp`
4. Validate on one dense map and one sparse map
