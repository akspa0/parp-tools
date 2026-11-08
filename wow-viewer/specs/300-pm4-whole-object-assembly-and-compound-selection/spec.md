# Spec 300 — PM4 Whole-Object Assembly, Compound Hull Selection & Unified Spatial Matching

## 1. Overview & Problem Statement

In the WoWViewer PM4 navmesh subsystem, collision surfaces in PM4 files (`MSUR`) have historically been partitioned at the root level by `CK24` (placement $Z$ elevation in yards). This causes complex visual game objects (such as buildings, pirate ships, bridges, watchtowers, and multi-story structures) to be fragmented horizontally into dozens of disconnected micro-hulls (`Pm4OverlayObject`), each containing only a single flat elevation slice (e.g. 7 surfaces at $Z = 6.9\,\text{yd}$).

When an operator clicks on a PM4 mesh in the 3D viewport:
1. The raycast selects only that single horizontal $Z$ placement slice (e.g. `[22_38] Obj #13 (Z=6.9yd)`).
2. The selection wireframe and bounding box highlight only that flat slice, leaving the rest of the ship or building unselected.
3. Shape-matching and proposal engines evaluate candidate models against the proportions of a flat 2D slice rather than the true 3D physical object, producing erratic aspect ratios, misaligned centroids, and false shape mismatches.
4. GLB exports of selected PM4 objects export only the clicked slice rather than the complete structure.

This specification unifies contiguous, overlapping, and link-connected PM4 hulls into **Compound PM4 Whole Objects** (`Pm4CompoundObject`). Clicking any constituent hull in the viewport selects the entire compound object, renders a bounding box around the whole structure, displays aggregate physical metrics (total triangles, surfaces, 3D bounds, volume centroid), and evaluates spatial shape matching against the authentic whole object.

---

## 2. User Stories

- **US-001 (Whole-Object Selection)**: As an operator viewing terrain with PM4 navmesh overlays, clicking anywhere on a structure (such as a pirate ship, tower, or building) must select the whole object—including all decks, hull, walls, cabin, and masts—highlighting all constituent hulls in the viewport and enclosing them within a single unified bounding box.
- **US-002 (Accurate Physical Dimensions & Sizing)**: As an operator pairing PM4 hulls with game models, the selected PM4 object must expose the true 3D dimensions (length, width, height) of the complete object rather than an elevation slice, enabling accurate proportional filtering and direct model matching.
- **US-003 (Compound GLB Export)**: As an operator exporting PM4 navmesh geometry for external 3D analysis, clicking "Export Selected PM4 to GLB" must export all constituent hulls of the selected compound object as named sub-nodes sitting on the terrain.
- **US-004 (Human-in-the-Loop Teaching & Curation)**: As an operator teaching a pairing rule between a resident/donor model (e.g. `PIRATESHIP.WMO`) and PM4 navmesh geometry, the pairing rule must store the compound object's authentic 3D centroid and bounds.

---

## 3. Functional Requirements (FRs)

- **FR-001 (Compound Whole-Object Model)**:
  - Define `Pm4CompoundObject` representing a cohesive physical object comprising one or more constituent `Pm4OverlayObject` instances on a tile.
  - Compute aggregate properties: `CompoundId`, `TileX`, `TileY`, `WorldBoundsMin`, `WorldBoundsMax`, `WorldCentroid`, `Length`, `Width`, `Height`, `TotalTriangles`, `TotalLines`, `TotalSurfaces`, and `ConstituentHulls`.
- **FR-002 (Whole-Object Assembly Engine)**:
  - Implement `Pm4CompoundObjectService.AssembleTileCompoundObjects(tileX, tileY, rawObjects, pm4File)`:
    - Run Disjoint Set Union (Union-Find) clustering across all hulls on a tile.
    - Cluster hulls when their 3D world-space bounding boxes intersect or are adjacent within $\le 2.5\,\text{yd}$ (continuous spatial volume).
    - Cluster hulls when they share a non-zero `MSLK.GroupObjectId` or have mutual MSLK neighbor links.
    - Cluster hulls when they share vertices in `MSVT`.
  - Maintain a fast lookup from hull key `(tileX, tileY, ck24, objectPart)` to its owning `Pm4CompoundObject`.
- **FR-003 (Compound Selection & Viewport Highlighting)**:
  - When `TryPickPm4ObjectByRay` intersects any hull, resolve its owning `Pm4CompoundObject`.
  - Store the active compound object in `Pm4OverlayScene._selectedCompoundObject`.
  - Update `Pm4OverlayMeshRenderer` to highlight ALL constituent hulls of `_selectedCompoundObject` in the selection color (purple wireframe).
  - Update `WorldScene` selection bounding box renderer to compute the oriented/axis-aligned bounding box from `compoundObject.WorldBoundsMin` to `compoundObject.WorldBoundsMax`.
- **FR-004 (Unified Cockpit & Sidebar Presentation)**:
  - Update viewport hover tooltip: `PM4 Object: [<Tile>] Whole Obj #<Id> (<N> hulls, <T> tris, <L>x<W>x<H> yd)`.
  - Update PM4 sidebar: display compound object header, constituent hull breakdown, true 3D dimensions, and nearest resident model distance.
- **FR-005 (Compound Shape Matching & GLB Export)**:
  - Update `Pm4GlbExporter` and `ExportSelectedPm4ToGlb` to export all constituent hulls of the selected compound object.
  - Update `MatchesTaughtShape` and proposal generation to measure against the compound object's combined prism and 3D bounds.

---

## 4. Success Criteria (SCs)

- **SC-001**: Clicking any part of `PIRATESHIP.WMO`'s collision geometry on tile `(22, 38)` in `development.wdt` selects the entire ship (all decks, hull, and masts) simultaneously in purple wireframe.
- **SC-002**: The selection bounding box wraps the entire physical object ($38\,\text{yd} \times 14\,\text{yd} \times 28\,\text{yd}$) rather than a flat $Z=6.9\,\text{yd}$ plane.
- **SC-003**: The PM4 sidebar displays the compound object summary with total hulls, total triangles, and true 3D dimensions.
- **SC-004**: Exporting the selected PM4 object to GLB creates a GLB containing all constituent collision hulls.
- **SC-005**: Full solution compilation succeeds with 0 errors, UI reference check passes with 0 errors/warnings, and all unit tests pass.
