# Spec 208 Phase 0 Reconciliation Evidence

**Date**: 2026-09-02  
**Author**: Antigravity  
**Subject**: Architectural audit of Spec 195 transposition engine, channel models, and cross-map boundaries for Spec 208.

---

## T001: Where Map Identity Enters

### Source Inspection of Spec 195 (`ChunkTranspositionService`)
`ChunkTranspositionService.ExtractPayload` accepts:
```csharp
public static ChunkTranspositionPayload ExtractPayload(
    IEnumerable<GlobalChunkCoordinate> coordinates,
    Func<GlobalChunkCoordinate, TransposedChunkRecord?> chunkReader,
    Func<GlobalChunkCoordinate, IEnumerable<TransposedObjectPlacement>?>? placementReader = null)
```
Notice:
1. `ChunkTranspositionPayload` is completely map-agnostic. It stores relative chunk coordinates `(RelativeGx, RelativeGy)` along with raw chunk records (`Heights`, `Normals`, `Layers`, `MccvColors`, `Liquid`, etc.) and relative object placements.
2. In Spec 195 (single-map transposition), `chunkReader` and `placementReader` read from the currently active `EditorSession` on the currently loaded map.
3. For Spec 208 (cross-map sourcing), map identity enters at exactly two boundaries:
   - **Source extraction boundary**: `chunkReader` and `placementReader` must read from the *source* map's data store (e.g. an independent `ITerrainAdapter` / `IArchiveCatalog` / tile reader for $M_{src}$), completely decoupled from the viewer's active display map $M_{tgt}$.
   - **Target texture re-mapping boundary (FR-010 / T103)**: Texture indices in MCLY are local to a map's texture list (or tile-specific texture palette). When transplanting a chunk from $M_{src}$ to $M_{tgt}$, copying raw texture indices causes silent visual corruption. The source layer's texture filename (`l.TexturePath`) must be re-resolved in $M_{tgt}$'s texture palette, allocating a new slot if needed.

The transposition math and payload structure themselves require zero changes to support cross-map operation.

---

## T002: Rotation & Mirror Audit of Spec 195 (FR-005)

### Findings from Source Inspection of `ChunkTranspositionService.TransformPayload`
Spec 195's current implementation in `ChunkTranspositionService.cs` (lines 76–160) exhibits major gaps in rotation and mirroring:
1. **Normal Vectors (`Normals`)**:
   - `Normals = chunk.Normals != null ? (Vector3[])chunk.Normals.Clone() : null;`
   - Normal vectors are cloned without spatial rotation or mirroring. When terrain is rotated $90^\circ$ around the vertical axis, normal vector $(N_x, N_y, N_z)$ must be transformed to $(-N_y, N_x, N_z)$ (or the appropriate coordinate system transformation). Leaving normals un-rotated corrupts diffuse and specular terrain lighting.
2. **Placement Rotations (`p.Rotation`)**:
   - `Rotation = p.Rotation;`
   - Object placement rotations are cloned verbatim. If a building is rotated $90^\circ$ with the region, its yaw must increase by $90^\circ$. Leaving rotation unchanged causes objects to face their original direction while the terrain rotates underneath them.
3. **Placement Relative Positions (`p.RelativePosition`)**:
   - `Vector3 relPos = p.RelativePosition; if (options.HeightOffset != 0f) relPos.Z += options.HeightOffset;`
   - Only the vertical offset is applied. The relative horizontal coordinates $(X, Y)$ within the bounding box of the extracted region are NOT rotated or mirrored.
4. **Internal Chunk Vertex Lattices (MCVT / MCNR)**:
   - While chunk grid coordinates `(RelativeGx, RelativeGy)` are rearranged, the 145 height samples (9x9 outer + 8x8 inner) within each chunk are NOT rotated or mirrored. For a $90^\circ$ rotation, the height matrix inside each chunk must be rotated (transposed and inverted).

### Decision
Per NEXT-DAY-PLAN.md: **"If it does not, that is a defect in 195 to fix there, not to work around here."**  
These spatial transformations must be implemented directly inside `ChunkTranspositionService.TransformPayload` in `WowViewer.Core.Editor`.

---

## T003: Channel Model Reconciliation (FR-003, Constitution II)

### Current Situation
Two channel models exist in the codebase:
1. `ChunkTranspositionOptions` (Spec 195): A collection of individual booleans (`IncludeHeights`, `IncludeTextures`, `IncludeHoles`, `IncludeLiquid`, `IncludeVertexShading`, `IncludeM2Placements`, `IncludeWmoPlacements`).
2. `PhaseDataChannel` (Spec 203): A `[Flags]` enum defining granular channels:
   - `Heightmap` (1 << 0)
   - `Normals` (1 << 1)
   - `TextureLayers` (1 << 2)
   - `VertexColors` (1 << 3)
   - `Holes` (1 << 4)
   - `Shadows` (1 << 5)
   - `Liquid` (1 << 6)
   - `Doodads` (1 << 7)
   - `WorldObjects` (1 << 8)
   - `AreaId` (1 << 9)
   - Group presets: `Terrain`, `Texturing`, `Objects`, `All`.

### Reconciliation
We must NOT create a third channel model. `PhaseDataChannel` is the comprehensive, unified channel standard.  
`ChunkTranspositionOptions` is updated to hold:
```csharp
public PhaseDataChannel Channels { get; set; } = PhaseDataChannel.All;
```
The legacy boolean properties are retained as getters/setters over `Channels.HasFlag(...)` for seamless backward compatibility.

---

## T004: Partial Tile Offset Granularity

### Confirmation
- A world tile is $16 \times 16$ chunks ($533.33333$ yards).
- One chunk is $1/16$ tile ($33.33333$ yards, $145$ vertices).
- The global chunk lattice `GlobalChunkCoordinate(Gx, Gy)` operates with integer coordinates in $[0, 1023] \times [0, 1023]$.
- An offset $(\Delta Gx, \Delta Gy)$ allows placement at any $1/16$ tile increment.
- Continuous sub-chunk offset ($< 33.33$ yards) would require non-aligned resampling of heightmaps, normal fields, alpha maps, and hole masks across arbitrary chunk borders, creating interpolation artifacts and performance overhead.
- Chunk-granular ($1/16$ tile) offset exactly preserves raw vertex fidelity, avoids resampling, and aligns with standard ADT chunk architecture. It fully satisfies the operator's requirement for partial-tile placement.

---

## T005: Proposal Lifecycle & Datastore Boundary (FR-011)

### Decision
1. **Proposal Phase**: When the operator configures a transplant in the UI (source map, source tile, target coordinates, transform, channel mask), the proposal lives as a staged `ChunkTranspositionPayload` in the active `EditorSession`. The viewer renders this payload as an interactive preview layer.
2. **Commit Phase**: When the operator clicks "Apply", `EditorSession` executes the transposition into the target map's active terrain chunks, pushing a reversible command onto the undo/redo stack (FR-008).
3. **Datastore / Export Phase**: The modified target map can be exported either directly to ADT format or as `ARRY/ENDS` raw blobs for ingestion into the client datastore. Provenance metadata (source map, source tile, transform, timestamp) is attached to the exported tile manifest.
