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

## Phase 3: PM4 as Pseudo-Map Tiles (Medium Priority)

**Goal**: Load PM4 files in the viewer as navigable terrain.

### 3a. PM4TerrainAdapter
- New `ITerrainAdapter` implementation
- Scans folder of PM4 files (development_XX_YY.pm4)
- Maps PM4 coordinates to tile grid
- Extracts MSVT/MSVI mesh geometry
- Produces simplified `TerrainChunkData` (navmesh as terrain surface)
- No textures — use wireframe or solid color based on surface type (MSUR flags)

### 3b. PM4 Coordinate Fix
- Use Pm4AdtCorrelator with known development map data as ground truth
- Compare PM4 MPRL positions with ADT MODF positions
- Calculate and apply the systematic offset

### 3c. PM4 Object Extraction
- MSLK hierarchy defines object grouping
- Use Pm4GeometryClusterer to identify distinct objects
- Match against known WMO models via Pm4WmoGeometryMatcher
- Export unmatched objects as OBJ for manual identification
- Overlay extracted objects on terrain in the viewer

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
