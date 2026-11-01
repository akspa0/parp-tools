# Spec 289: PM4 3D Footprint Extrusion, Synthetic Wall Reification & Direct MPQ Asset Matching

## 1. Overview & Problem Statement

The PM4 navmesh format records collision geometry generated from world terrain, buildings, and placed doodads. While the viewer provides the only visual inspection tool in the world for these files, previous object-restoration efforts relied on "Rosetta" cross-map matching—finding donor ADTs from later clients (e.g., 3.3.5 or 4.3.4) that preserved original `MDDF`/`MODF` placements. On unauthored or novel development tiles (such as `development` / `ProgrammerIsle`), no donor maps exist, rendering Rosetta-based restoration impossible.

Furthermore, empirical testing in Spec 288 and 287 revealed two key geometric insights:
1. **Doodad Surfaces are Walkable Tops**: PM4 surfaces for doodads (`MSUR.0x00 == 0x03`, group key 3) represent the walkable *top surfaces* of objects where player characters and NPCs walk. Placing models with their origin at the PM4 centroid caused objects to hover with the PM4 mesh trapped underneath at their base. The correct placement formula requires top-surface alignment:
   $$P_Z = Z_{\text{pm4\_top}} - (H_{\text{model\_top}} \cdot s)$$
2. **3D Envelope Reification**: A PM4 object segment encodes:
   - The top walkable cap $(X, Y)$ perimeter and elevation $Z_{\text{top}}$ via `MSUR`/`MSPV`.
   - The ground-level boundary via `MPRL` terrain contact points (linked through `MSLK.0x04`) and `MSUR.0x1C` Placement-Z markers.
   - The bounding extents $(W \times D \times H)$ where $H = Z_{\text{top}} - Z_{\text{base}}$.

By extruding the top surface perimeter down to $Z_{\text{base}}$, we reify the missing vertical walls of the object. This gives a closed 3D prism and scale-invariant shape signature that can be queried directly against the client's raw MPQ model archives (`H:\CLIENTS`), enabling authentic asset identification and placement **without any donor map or Rosetta cross-reference**.

---

## 2. Requirements & User Stories

### User Story 1: Top-Surface Magnetization & Placement Fix
As an operator curating objects against PM4 collision data, I want magnetized and duplicated M2 doodads to have their top surface align with the PM4 mesh, so that the PM4 collision geometry sits cleanly on top of the object rather than trapped beneath its base on the ground.

### User Story 2: 3D Footprint & Synthetic Wall Extrusion
As an operator inspecting PM4 navmesh objects, I want the viewer to compute and visualize the full 3D extruded prism (top surface + synthetic vertical walls down to MPRL terrain contact or Placement-Z), so that I can see the true physical envelope and footprint of the missing structure or doodad.

### User Story 3: Zero-Donor Geometric Model Matching
As an operator working on tiles without donor ADTs, I want to match PM4 3D footprints directly against the client's `.m2` and `.wmo` model geometry by comparing dimensionless aspect ratios ($W/D$, $H/\max(W,D)$) and silhouette similarity, so that authentic placements can be identified purely from PM4 geometry.

---

## 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| **AC-001** | Top-Aligned Magnetization | Magnetizing an M2 model to a PM4 doodad hull sets $P_Z = Z_{\text{centroid}} - (H_{\text{local\_top}} \cdot s)$; the resulting transformed `BoundsMax.Z` equals or closely bounds the PM4 surface height. |
| **AC-002** | 3D Footprint Reification | A pure algorithm in `WowViewer.Core.PM4` extracts the 2D convex hull / boundary of top surfaces, resolves $Z_{\text{base}}$ from MPRL (doodads) or Placement-Z (WMOs), and extrudes vertical wall quad geometry. |
| **AC-003** | Scale-Invariant Shape Signatures | Geometric descriptors compute $W, D, H$, aspect ratio $W/D$, height ratio $H/\max(W,D)$, and footprint polygon area. |
| **AC-004** | Direct MPQ Asset Matching | Given a PM4 3D footprint and an indexed list of model candidates, the matcher ranks candidates by shape distance without requiring ADT or Rosetta references. |
| **AC-005** | Sidebar Workbench Visualizer & Controls | The PM4 Sidebar workbench provides a toggle to display extruded 3D footprints and walls, and an interactive match trigger with results disclosure. |
| **AC-006** | UI Reference & Governance Conformance | All new UI elements use `SharedUiWidgets`, `Verify-UiReference.ps1` passes with 0 errors/warnings, and all modified files respect the $\le 2,000$ line limit (AGENTS.md §10). |

