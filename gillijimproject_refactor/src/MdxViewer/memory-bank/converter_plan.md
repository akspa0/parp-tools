# Map/Object Converter & PM4 Tile Viewer — Master Plan

Created: 2025-02-09

## Goal
Build a unified tool that can:
1. **Read** Alpha (0.5.x), LK (3.3.5), and PM4 map/model data
2. **Render** all of these in the viewer
3. **Convert** between formats for real WoW engine import/export
4. **Extract** objects from PM4 pathfinding data

---

## Existing Asset Inventory

### Converters (WoWMapConverter.Core/Converters/)
| Converter | Direction | Status |
|-----------|-----------|--------|
| WmoV14ToV17Converter | Alpha WMO → LK WMO | ✅ Working |
| WmoV14ToV17ExtendedConverter | Enhanced v14→v17 | ✅ Working |
| WmoV17ToV14Converter | LK WMO → Alpha WMO | ✅ Working |
| MdxToM2Converter | Alpha MDX → LK M2 (+.skin) | ✅ Working |
| M2ToMdxConverter | LK M2 → Alpha MDX | ❌ Not built |

### Readers
| Format | Location | Status |
|--------|----------|--------|
| Alpha WDT/ADT | gillijimproject-csharp (WdtAlpha, AdtAlpha, McnkAlpha) | ✅ Working |
| LK ADT (MCNK) | WoWMapConverter.Core.Formats.LichKing.Mcnk | ✅ Working |
| Standard WDT (MAIN/MPHD) | MdxViewer StandardTerrainAdapter | ✅ Just built |
| Alpha WMO v14 | WmoV14ToV17Converter.ParseWmoV14Internal | ✅ Working |
| LK WMO v17 root+group | **Not in viewer** (Warcraft.NET has partial) | ❌ Needed |
| Alpha MDX | MdxViewer MdxFile parser | ✅ Working |
| LK M2 | WoWRollback.PM4Module.M2File (vertices only) | ⚠️ Minimal |
| PM4 | Pm4File in WoWMapConverter.Core + WoWRollback.PM4Module | ✅ Working |

### Writers
| Format | Location | Status |
|--------|----------|--------|
| LK WDT | Wdt335Writer (WoWRollback.Core) | ✅ Working |
| LK ADT | AdtLkFactory (minimal w/ MODF) | ⚠️ Minimal |
| LK M2 | MdxToM2Converter.WriteM2 | ✅ Working |
| LK WMO v17 | WmoV14ToV17Converter (root+groups) | ✅ Working |
| Alpha WDT | WoWRollback.LkToAlphaModule | ✅ Working |
| Alpha ADT | WoWRollback (embedded in WDT) | ✅ Working |

### PM4 Tooling (WoWRollback.Core/Services/PM4/)
| Tool | Purpose |
|------|---------|
| Pm4AdtCorrelator | Correlates PM4 objects with ADT MODF placements |
| Pm4GeometryExporter | Exports PM4 geometry to OBJ |
| Pm4ModfReconstructor | Reconstructs MODF placements from PM4 data |
| Pm4WmoGeometryMatcher | Matches PM4 geometry to known WMO models |
| Pm4GeometryClusterer | Clusters PM4 geometry into distinct objects |
| PipelineCoordinateService | PM4↔ADT coordinate transforms (offset still wrong) |

---

## Phase 1: LK Model Reading (High Priority)

**Goal**: The viewer can render Standard WDT worlds with LK models.

### 1a. M2 Model Reader
- Port/extend `WoWRollback.PM4Module.M2File` into a proper MdxViewer reader
- Parse: MD20/MD21 header, vertices, normals, texture coords, bones, submeshes
- Parse .skin files for render batches
- Load textures from M2 texture refs via IDataSource (MPQ)
- Feed into existing `ModelRenderer` or new `M2Renderer`

### 1b. WMO v17 Reader
- Read split WMO: root file (MOHD, MOTX, MOMT, MOGN, MOGI, MODN, MODS) + group files (MOGP, MOPY, MOVI, MOVT, MONR, MOTV, MOBA)
- Load from IDataSource (MPQ) — root path from MODF, group files = root_000.wmo, root_001.wmo, etc.
- Map to existing `WmoRenderer` data structures (or extend them)

### 1c. Wire into WorldAssetManager
- `WorldAssetManager` currently loads v14 WMO and MDX
- Add detection: if IDataSource is MPQ (LK), try loading as WMO v17 / M2 first
- Fall back to v14/MDX if not found (e.g., converted assets)

---

## Phase 2: Format Conversion Pipeline (Medium Priority)

**Goal**: Bidirectional conversion for real WoW engine compatibility.

### 2a. Export to 3.x (Alpha → LK)
Bundle existing converters into a single pipeline:
1. Read Alpha WDT → enumerate tiles + textures + placements
2. Write LK WDT via Wdt335Writer
3. Write split ADTs via AdtLkFactory (needs enhancement for full terrain data)
4. Convert WMO v14 → v17 via WmoV14ToV17Converter
5. Convert MDX → M2 via MdxToM2Converter
6. Extract and copy BLP textures
7. Optional: build MPQ archive from output

### 2b. Export to 0.5.x (LK → Alpha)
Reverse pipeline:
1. Read Standard WDT+ADTs → terrain + placements
2. Convert to Alpha WDT (WoWRollback.LkToAlphaModule)
3. Convert WMO v17 → v14 via WmoV17ToV14Converter
4. Convert M2 → MDX (new — reverse of MdxToM2)
5. Package textures

---

## Phase 3: PM4 World Support (Medium Priority)

**Goal**: Load the complete PM4 set as a world-space dataset in the viewer, preserving CK24 object identity across tile boundaries and mapping decoded geometry to the correct world positions.

### 3a. Unify PM4 Decode In Core
- Do not treat the current lightweight `WoWMapConverter.Core.Formats.PM4.Pm4File` parser as sufficient for final PM4 support.
- Port or align the richer rollback decoder contract into reusable core types:
	- `Pm4Decoder`
	- `Pm4ChunkTypes`
	- `Pm4FileStructure`
	- `Pm4MapReader`
- Preserve the key PM4 relationships explicitly:
	- `MSUR.PackedParams` → `CK24`
	- `MSUR.MdosIndex` → `MSCN`
	- `MSLK` / `MSPI` / `MSPV` path links
	- `MSVT` / `MSVI` surface mesh links
	- `MPRL` / `MPRR` placement/reference graph data
- Required outcome: one reusable PM4 decode layer that can read all 616 PM4 files from the fixed development dataset without losing CK24 identity or surface-link metadata.

### 3b. Build A Cross-Tile CK24 Registry
- Treat CK24 as the primary object-group key across the full PM4 map set, not just inside a single tile.
- Build a map-level registry that aggregates surfaces, scene nodes, and placement references by CK24 across multiple `development_XX_YY.pm4` tiles.
- Keep tile provenance on every decoded element so cross-tile objects can still be traced back to source PM4s.
- Split responsibilities clearly:
	- `CK24 == 0` → terrain/navmesh/background surface layer
	- `CK24 != 0` → object candidate layer
	- optional `MSVI` gap splitting inside a CK24 group when one CK24 contains multiple repeated instances
- Required outcome: the viewer and later exporters can ask for `all geometry for CK24 X` and get a complete, cross-tile object layer instead of partial tile fragments.

### 3c. Establish One Authoritative World-Coordinate Contract
- Coordinate correctness is the current hard gate.
- Preserve the proven findings from rollback work:
	- CK24 grouping via `MSUR` is already usable
	- MSCN-linked object discovery via `MdosIndex` is already usable
	- coordinate handling is where previous PM4 output became wrong
- Validate and codify the coordinate contract against fixed development data before integrating rendering/export broadly:
	- determine which chunks are already world-space vs tile-local (`MSCN`, `MPRL`, `MSVT`)
	- document any required axis swap and any required tile-origin transform
	- compare transformed PM4 placements against ADT `MODF` / `MDDF` ground truth in `test_data/development/World/Maps/development`
- Required outcome: one documented transform API that converts PM4-derived object anchors into the same world coordinates used by the viewer scene and ADT placements.

### 3d. Classify PM4 Output Into Viewer Layers
- PM4 support should produce distinct renderable/debug layers instead of a single mesh blob.
- Minimum layer split:
	- nav/background surfaces
	- CK24 object groups
	- matched WMO candidates
	- matched M2 candidates
	- unmatched geometry candidates
	- placement/reference markers from `MPRL`
- Viewer integration should start with debug rendering first:
	- wireframe/flat-shaded PM4 surfaces
	- color-by-layer or color-by-CK24 rendering
	- toggleable anchors/bounds for transformed object placements
- Required outcome: visual proof that PM4 layers land in the right world positions before any higher-level conversion/export claims are made.

### 3e. PM4 Object Extraction And Matching
- After world coordinates are trusted, reuse the existing CK24 and MSCN grouping to build object candidates.
- Match grouped geometry against known WMO/M2 libraries.
- Keep unmatched geometry export as a first-class output for manual identification.
- Do not collapse this into only WMO reconstruction; PM4 support should preserve a generic object-class layer whether or not a final asset match exists.

---

## Phase 4: Unified Project System (Low Priority)

**Goal**: A project format that ties everything together.

### 4a. Project Manifest
JSON project file referencing:
- Data sources (Alpha WDT, Standard WDT+MPQ, VLM dataset, PM4 folder)
- Conversion targets (0.5.x, 3.x)
- Asset mappings (which v14 WMO = which v17 WMO)
- Output paths

### 4b. Export Pipeline UI
Viewer panel for "Convert & Export":
- Select source data
- Choose target version
- Preview what will be converted
- Run conversion with progress

---

## Dependencies & Risks

- **M2 parsing complexity**: M2 has bones, animations, particles. Start with static geometry only.
- **WMO v17 group files**: Each group is a separate file in MPQ — need reliable path construction.
- **PM4 coordinate offset**: Known to be wrong; needs careful calibration.
- **ADT writing completeness**: `AdtLkFactory` only writes minimal ADTs — needs terrain heights, textures, alpha maps for real exports.
- **Texture format**: BLP stays the same across versions (mostly), but path references change.

## Recommended Execution Order
1. Phase 1a (M2 reader) — unlocks Standard WDT doodad rendering
2. Phase 1b (WMO v17 reader) — unlocks Standard WDT building rendering
3. Phase 1c (wiring) — completes Standard WDT support
4. Phase 3a (PM4 terrain) — unlocks PM4 visualization
5. Phase 3b (coordinate fix) — makes PM4 overlay useful
6. Phase 2a (export to 3.x) — first conversion pipeline
7. Phase 3c (PM4 objects) — extract objects from PM4
8. Phase 2b (export to 0.5.x) — reverse conversion
9. Phase 4 (project system) — ties it all together
