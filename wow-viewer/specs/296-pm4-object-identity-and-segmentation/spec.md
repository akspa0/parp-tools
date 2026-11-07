# Spec 296 — PM4 Object Identity, Topological Segmentation & Conflict Decoupling

## 1. Overview & Context

- **Spec ID**: 296
- **Title**: PM4 Object Identity, Topological Segmentation & Conflict Decoupling
- **Epic**: Epic 253 (PM4 Navmesh Research) & Epic 251 (Viewer UX & Code Health)
- **Status**: In Progress
- **Created**: 2026-11-06
- **Origin**: User identified critical failure in PM4 matching and selection: `CK24` is not an object identifier, but the placement height $Z$ float (`MSUR._0x1C`). On real maps, multiple objects at the same elevation were collapsed into a single mega-mesh, click selection highlighted entire multi-building plateaus, and batch analysis reported false-positive conflicts where distinct objects were reported as conflicting with themselves (e.g., `CK24 0x42DBCF vs 0x42DBCF`).

---

## 2. Problem Statement

1. **`CK24` is a Float $Z$ Slice, Not an Object ID**:
   Format research established that `MSUR._0x1C` is an IEEE-754 32-bit `float` holding ADT placement $Z$ in yards. Grouping by `CK24` worked only on synthetic single-building test tiles where each model had a unique altitude. On real terrain, every structure on a plateau shares the exact same elevation.
2. **Mega-Mesh Merging & Bounded Selection Failure**:
   Topological connectivity splitting was disabled by default, causing disparate buildings to merge into a single `Pm4OverlayObject`. In the viewport, clicking any surface selected the entire height group rather than the clicked building component.
3. **Shape Match Proposal Collisions & Self-Conflicts**:
   `Pm4ShapeMatchAnalysisService` assigned `HullKey = hull.Ck24`, discarding `ObjectPartId` whenever `Ck24 != 0`. When two distinct buildings at the same height matched the same candidate, the conflict detector reported `CK24 0x... vs 0x... claiming <Model> (1.3 yds apart)`, confusing users and breaking review workflows.
4. **Footprint Extraction Pollution**:
   `ExtractCompositeSurfaceVertices` unioned all surfaces in the tile where `part.Ck24 == pm4Obj.Ck24`, feeding a 200-yard multi-model polygon cloud to the descriptor extractor instead of an isolated model footprint.

---

## 3. Goals & Requirements

### Functional Requirements (FR)

- **FR-001 (Concrete Object Identity)**: Define a first-class `Pm4ObjectIdentity` struct encapsulating `(TileX, TileY, ComponentId, LinkGroupObjectId, PlacementZ, Centroid, BoundingSize, SurfaceClass)`.
- **FR-002 (Unambiguous 64-bit Hull Key)**: Replace `uint HullKey` with a composite key encoding `(TileX, TileY, ComponentId)` so that multiple objects at the same elevation never collide under one identifier.
- **FR-003 (Topological Island Segmentation)**: Partition all PM4 surface groups (both zero and non-zero $Z$) into disjoint topological connected components using floor vertex sharing (`MSVT`/`MSVI`) and navmesh adjacency links (`MSLK`).
- **FR-004 (Bounded Viewport Selection)**: Clicking a surface in the 3D viewport selects and outlines *only* the specific clicked component (`ObjectPartId`), not every object on the tile sharing that elevation.
- **FR-005 (Conflict Disambiguation & Human-Readable Display)**:
  - Update `Pm4ProposalConflict` and conflict rendering to display exact component IDs and physical elevations: `[<Tile>] Obj #<A> (Z=<zA> yd) vs Obj #<B> (Z=<zB> yd) claiming <Model> (<Dist> yds apart)`.
- **FR-006 (Isolated Footprint Extraction)**:
  - `ExtractCompositeSurfaceVertices` gathers triangles only for the selected component (or parts sharing an explicit non-zero `LinkGroupObjectId`), never unioning disparate components across the tile.
  - `ExtractTileHullInputs` extracts hull inputs for all distinct components on the tile without deduplicating by `Ck24`.
- **FR-007 (Audit Ledger & Batch Summary Clarity)**: Update batch JSON and Markdown exports to record `ComponentId`, `LinkGroupObjectId`, and `PlacementZ (yd)` for full auditability.

---

## 4. Success Criteria (SC)

- **SC-001**: Solution builds with 0 errors and passes all existing unit tests.
- **SC-002**: Two objects at the exact same elevation ($Z$) have distinct `UniqueHullId` values and are never flagged as self-conflicts.
- **SC-003**: In batch conflict reporting, conflicts between distinct parts of the same height tier explicitly state both part IDs and elevation in yards.
- **SC-004**: Clicking a surface in the 3D viewport colors and bounds only the clicked component.
- **SC-005**: All UI controls registered in the reference inventory pass with 0 errors and 0 warnings.

