# Spec 297 — Automated Known-Good Pairing Harvester & Streamlined Matching Workbench

## 1. Problem Statement

When reconstructing lost alpha maps or donor zones (e.g. restoring Coldwind Pass in Dragonblight or development land via donor tiles and map layers), the scene contains authentic placed objects (WMOs and M2 doodads) sitting alongside their authoritative PM4 navmesh hulls. Currently, turning these into taught pairings requires tedious manual 1-by-1 clicking, teaching, and saving across dozens of models.

Furthermore, the PM4 user interface is fragmented across multiple disjoint panels (PM4 Sidebar, Review Queue, Shape Match, Manual Curation, and Raw Chunks SceneGraph), with raw binary reverse-engineering dumps (`MSHD`, `MSLK`, raw bytes, regional peer lists) cluttering the view.

## 2. User Requirements

- **FR-001 (Automated Donor Pairing Harvester)**: The system must provide a one-click automated harvester that scans all placed models on loaded / donor tiles, finds their corresponding spatial PM4 hull, validates geometric agreement (centroid proximity, footprint overlap, and height), and automatically registers them as verified `TaughtPairing` rules.
- **FR-002 (One-Click Multi-Tile Propagation)**: The system must allow propagating all harvested / taught rules across all loaded tiles to immediately populate matching unoccupied PM4 hulls.
- **FR-003 (Streamlined Unified Workbench UI)**: Consolidate the disparate PM4 matching surfaces into a single, cohesive, clutter-free workbench panel:
  - High-level summary (loaded tiles, placed models, PM4 hulls, active learned rules).
  - Prominent one-click batch action buttons (`Harvest Pairings from Placed Models`, `Apply Learned Rules to All Tiles`, `Export Manifest`).
  - Active selection card showing the clicked model and nearest PM4 hull side-by-side with clear `Pair & Teach` and `Snap & Align` actions.
  - Clean list of active learned rules showing model name, matched hull dimensions, and count of matching unoccupied hulls found on the map.
- **FR-004 (Bury Raw Forensics Clutter)**: The low-level binary chunk inspector (`MSHD`, `MSLK`, raw byte tables) must be tucked into a collapsed "Advanced Forensics & Raw Chunks" drawer at the bottom of the workbench, so it never distracts from the curation workflow.
- **FR-005 (Persistence & Non-Destructive Operation)**: Harvested pairings must save directly to `output/projects/pm4_pairings/<map>.pairings.json` and export to `<map>.manifest.json` without altering source files or creating duplicate models.

## 3. Success Criteria

- **SC-001**: One-click harvest accurately identifies and pairs all valid placed objects on loaded donor tiles whose centroids are within spatial agreement ($\le 15\,\text{yd}$) of a conforming PM4 hull.
- **SC-002**: Applying harvested rules across all loaded tiles spawns conforming models into unoccupied matching hulls with correct yaw and scale.
- **SC-003**: The consolidated UI provides a clean, single-screen workflow for harvesting, manual pairing, and propagation without navigating between 4 separate panels.
- **SC-004**: Full solution builds with 0 errors and passes all unit tests and `Verify-UiReference.ps1`.
