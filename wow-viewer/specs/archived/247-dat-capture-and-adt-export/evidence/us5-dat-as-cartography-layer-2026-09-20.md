# Spec 247 Evidence — US5: DAT Folders as Cartography Layers (CODE COMPLETE, UNWITNESSED)

Date: 2026-09-20

## Operator request

> "it'd be nice if we could load DAT files as Cartography layers, since they are effectively the real
> project files for existing tiles"

Added to spec 247 as **US5** (a new user story; see the spec's Requirements section).

## The architectural finding this turned on

A Cartography layer is a `PhaseLayerSettings`, and its donor is identified by exactly one thing:
`MapName`, a `required string`. Every resolution site in the base adapters takes that string and
resolves it **inside the base adapter's own data source**:

| Site | How it resolved a donor before |
|---|---|
| `StandardTerrainAdapter.LoadMapTile` | `World\Maps\<name>\<name>_<y>_<x>` out of the MPQ/CASC data source |
| `StandardTerrainAdapter.OverlayTileExists` | same path probe |
| `StandardTerrainAdapter.GetOccupiedTiles` | the donor map's WDT |
| `AlphaTerrainAdapter` | a **sibling `AlphaTerrainAdapter`** cached per map name |

A DAT folder is none of those: no map name, no WDT, loose files under arbitrary extensions, and read
by a different adapter type entirely. So there was no seam — layers were map-name-only.

The good news is the **data shape already matched**: `AhdrTerrainAdapter.LoadTileWithPlacements`
returns the same `TileLoadResult` that composition consumes, so once a donor can be resolved, every
downstream step (content transform, channel gating, merge) works unchanged.

## Design: locator in the existing field, not a model change

`MapName` now also accepts `dat:<absolute folder>`. A new `DatLayerSource` owns the prefix, the
parsing and a per-folder adapter cache.

Rejected alternative: adding an `ExternalSourcePath` field to `PhaseLayerSettings`. That is a core
model change rippling into `Clone()`, the Cartography project JSON, and the layer UI — and every
resolution site would still need the same branch. The locator approach keeps
[`CartographyProjectStore`](../../../../src/viewer/WoWViewer/Terrain/CartographyProjectStore.cs)
persistence and layer cloning working untouched, because a layer is still just a string.

The cost is a stringly-typed locator. It is contained in one tested parser, and `MapName` is already
a donor *locator* rather than a display name.

**The seam is inert when unused.** Every branch is `if (DatLayerSource.IsDatSource(name))` at the top
of a method; a layer that is not a DAT locator takes exactly the path it took before. This satisfies
AGENTS.md §4 (prefer opt-in adapters over changing shared terrain-loading behaviour).

## Files changed

| File | Change |
|---|---|
| [`DatLayerSource.cs`](../../../../src/viewer/WoWViewer/Terrain/DatLayerSource.cs) | New. Locator build/parse, display name, cached `AhdrTerrainAdapter` per folder, footprint tiles, cache invalidation. Never throws on a missing or unreadable folder. |
| [`StandardTerrainAdapter.cs`](../../../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs) | 4 resolution sites branch to the DAT donor; new `DatLayerHeightDivisor` (default 36) so donor inches land in the base map's yards. |
| [`ViewerApp_PhaseLayers.cs`](../../../../src/viewer/WoWViewer/ViewerApp_PhaseLayers.cs) | "Add DAT folder..." button beside the manual map-name Add, opening the folder picker. |
| [`DatLayerSourceTests.cs`](../../../../tests/WowViewer.Core.Tests/DatLayerSourceTests.cs) | New, 10 tests. |

**No new `ViewerApp` members** (AGENTS.md §10): the picker is opened inline through its singleton,
which defers its own draw, so the button needs no `_wantOpen…` state field. `DatLayerSource` is a
standalone static service, not a god-class member.

## Why this is worth more than a convenience

DAT files are the terrain project files the shipped ADTs were built from, so a DAT layer over the
shipped map is a direct before/after of the same ground.

It also gives the **unresolved DAT axis question a way to be answered**. US3's exporter has a
`--transpose` flag because the DAT grid's row axis runs along ALOC tile Y while the renderer's runs
along its tile X, and nothing measured settles which way a standalone ADT should be written
([US3 receipt](us3-dat-to-lk-adt-2026-09-20.md)). Cartography already has tile offset, cell offset,
quarter-turn rotation and mirrors. Align a DAT donor against its shipped tile by eye once, and the
alignment **is** the answer — which then feeds back into the exporter instead of staying a guess.

## Verification

| Command | Result |
|---|---|
| `dotnet build wow-viewer/WowViewer.slnx -c Debug` | **0 errors** |
| `dotnet test ... --filter "…Ahdr\|…DatToLk\|…DatLayer"` | **Passed! 31/31**, 0 failed, 0 skipped |
| `dotnet test ... --filter "FullyQualifiedName~DatLayerSource"` | **10/10** |

| Criterion | Evidence |
|---|---|
| Ordinary map names are unaffected | `OrdinaryMapNames_AreNotDatSources` over Azeroth / Kalimdor / DeadminesInstance / empty / null — not a DAT source, resolves null, empty footprint |
| A folder round-trips through the locator | `ForFolder_RoundTripsThroughTheLocator` |
| Layer list shows something readable | `DisplayName_UsesTheFolderLeaf_AndLeavesMapNamesAlone` → "DAT: Expansion01" |
| A missing/unreadable donor is displayable, not a crash | `Resolve_OnAMissingFolder_IsADisplayableStateNotAThrow`, `Resolve_OnAFolderWithNoDatFiles_ReturnsNull` |
| Footprint tiles decode correctly | `OccupiedTiles_DecodesTheAdapterTileKeys` — two synthetic v22 files land at (38, 24) and (37, 25), matching the adapter's renderer TileX = ALOC tile Y mapping |

## NOT witnessed — no DAT layer has been composed on screen

This is **code complete and unproven in the viewer**. The tests cover the locator, the donor cache
and the footprint decode; they do **not** cover composition, because that needs a loaded base map
from a real client. Specifically unverified:

1. **Whether a DAT donor actually draws** over a Standard base map.
2. **Alignment.** Almost certainly needs offset/rotation work by hand — that is the point of the
   feature, but it means "it lines up" is not a claim being made.
3. **Textures.** A DAT donor names `Expansion01` tilesets; whether they resolve depends on the base
   map's data source. A donor with unresolved textures will still contribute heights.
4. **Cartography project persistence** round-tripping a `dat:` locator. It should work unchanged
   because the layer is still a string, but it has not been saved and reloaded.

## Limits, stated

- **`StandardTerrainAdapter` only.** `AlphaTerrainAdapter` resolves donors through a sibling
  `AlphaTerrainAdapter` (`_phaseAdapters`, typed to itself), which needs a different change. A DAT
  locator on an Alpha base map will simply fail to resolve — displayable, not a crash — until that is
  done.
- **v22 donors show layer 0 only**, inheriting the open `AMAP` codec (US1).
- The donor adapter is cached per folder for the process lifetime; `DatLayerSource.Forget` exists and
  is called when a folder is re-added, but there is no UI to force a rescan.
- Spec 232, which owns the Cartography layer system, is 4 of 37 tasks done. This builds on primitives
  that exist and compose today, but that lane is far from complete.
