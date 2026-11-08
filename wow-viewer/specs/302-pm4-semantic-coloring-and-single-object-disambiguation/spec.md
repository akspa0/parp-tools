# Spec 302: PM4 Semantic Color-Coding, Single-Object Disambiguation & Null-Safety Fix

## Status
Active & In Implementation (2026-11-08)

## Background & Problem Statement
During operator validation of PM4 whole-object assembly, map propagation, and shape matching:
1. **Propagation Crash**: Applying matches to PM4 hulls triggered an unhandled `System.NullReferenceException` at `WoWViewer.Terrain.WorldAssetPathResolver.NormalizeKey(String path)` during `FlushPendingScenePlacements`. The newly instantiated synthetic `ObjectInstance` had `ModelKey` unassigned (`null`), and `NormalizeKey` unconditionally called `path.Replace(...)`.
2. **Semantic Color-Coding Failure**: In the PM4 Overlay visual legend, selecting `Semantic Surface Role (doodad, floors, walls)` failed to display semantic colors. The viewport wireframes appeared monochrome/yellow rather than reflecting distinct surface roles (`0x03` doodads orange, `0x10` lowest floors green, `0x11` standard floors cyan, `0x13` upper floors magenta). Investigation revealed that `ComputeSemanticCategoryColor` switched on `ck24Type` (which is an IEEE-754 exponent byte, e.g. `0xC1`, `0x40`, NOT a surface class) and ignored `dominantGroupKey` (`MSUR.0x08` SurfaceClass), falling back to generic gray. Furthermore, `Pm4OverlayMeshRenderer` unconditionally overwrote `color = Pm4ColorSelection` (saturated yellow) for all hulls in selected compounds, blinding out semantic colors.
3. **Over-Clustering / Single-Object Disambiguation Regression**: The user reported that an entire ship and its surroundings were merged into a single giant PM4 object: *"we need to determine what a single object is, because now the whole 'ship', which is made up of many objects, not just a single object, is showing up as a single giant pm4 object, which is also not right. You're going backwards instead of forwards."* Transitive spatial clustering within $2.5\text{ yd}$ merged doodads sitting on decks with the ship's floor hulls, and chained gangways, docks, and terrain together. Additionally, cross-tile compound clustering contained a blind fallback that merged all compounds lacking MSCN points across tile seams.

## User Requirements
- **UR-01 Null-Safe Path & Model Resolution**: `WorldAssetPathResolver.NormalizeKey` and `WorldAssetManager` must safely handle null and whitespace path strings without throwing exceptions. `Pm4MapWidePropagationService` must explicitly assign `ModelKey` and guard `FlushPendingScenePlacements`.
- **UR-02 Working Semantic Surface Color-Coding**: `Pm4ColorCodingEngine.ComputeSemanticCategoryColor` must correctly map `DominantGroupKey` (`MSUR.0x08` SurfaceClass) to authoritative semantic colors:
  - `0x03`: Doodad collision surfaces (M2 models: cannons, barrels, crates) -> Coral/Orange (`ColorDoodad`).
  - `0x10`: Lowest floor (ground level WMO) -> Forest Green (`ColorFloorLowest`).
  - `0x11`, `0x12`: Standard/common floor -> Cyan (`ColorFloorCommon`).
  - `0x13`: Upper floor (multi-story WMO) -> Royal Blue / Magenta (`ColorFloorUpper`).
  - `0x14`: Angled / roof surface -> Purple (`ColorFloorAngled`).
  - `0x01`: Terrain linker (MPRL) -> Lime (`ColorTerrainLinker`).
  - `0x02`: MSCN Connector -> Turquoise (`ColorConnector`).
- **UR-03 Preserved Semantic Colors on Selected Objects**: In `SemanticCategory` mode, selected objects and compounds must retain their semantic surface colors rather than being painted 100% solid yellow. Selection must be framed by the yellow 3D bounding box, with optional subtle mesh tinting.
- **UR-04 Single-Object Semantic Disambiguation**:
  - Doodads (`0x03`) must NEVER cluster with WMO structural floors (`0x10..0x16`).
  - Separate doodads must NOT merge with each other unless they share a non-zero `LinkGroupObjectId` or are coincident sub-parts of the same prop.
  - WMO structural floor slices must only cluster when they share `LinkGroupObjectId` OR represent vertically stacked floors with substantial $(X, Y)$ horizontal footprint overlap ($> 0.5\text{ yd}$ in $X$ and $Y$). Side-by-side structures (docks, gangways, buildings) must not transitively chain together.
- **UR-05 Strict Cross-Tile Compound Seam Stitching**: Cross-tile compounds must ONLY stitch across tile boundaries when explicit MSCN connector evidence (coincident MSCN points $\le 0.5\text{ yd}$ or shared quantized connector keys) exists. The blind fallback merging compounds without MSCN points must be eliminated.

## Acceptance Criteria
- **AC-01**: `NormalizeKey(null)` returns `string.Empty` without throwing `NullReferenceException`.
- **AC-02**: Propagation rule application flushes placements cleanly into scene and queue loaders without crashing.
- **AC-03**: `Pm4ColorCodingEngine.ComputeSemanticCategoryColor` returns correct RGB vectors for all surface classes (`0x03`, `0x10`, `0x11`, `0x12`, `0x13`, `0x14`, `0x01`, `0x02`).
- **AC-04**: When `SemanticCategory` is active, viewport mesh rendering displays distinct semantic colors across doodads and floors even when an object or compound is selected.
- **AC-05**: Clustering tests prove that doodads (`0x03`) sitting on a WMO floor (`0x11`) remain separate clusters.
- **AC-06**: Clustering tests prove that separate doodads placed near each other do not merge.
- **AC-07**: Cross-tile compound clustering tests prove that compounds lacking MSCN points do not merge across tile seams solely by AABB proximity.
