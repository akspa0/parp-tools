# Project Vision & Immediate Technical Goal (2024-07-21)

## Vision
- Build tools that inspire others to explore, understand, and preserve digital history, especially game worlds.
- Use technical skill to liberate hidden or lost data, making it accessible and reusable for future creators and historians.

## Immediate Technical Goal
- Use PM4 files (complex, ADT-focused navigation/model data) to infer and reconstruct WMO (World Model Object) placements in 3D space.
- Match PM4 mesh components to WMO meshes, deduce which model is where, and generate placement data for ADT chunks.
- Output reconstructed placement data as YAML for now, with the intent to build a chunk creator/exporter later.

---

# Mode: PLAN

# Active Context: PM4/WMO Mesh Comparison (2024-07-21)

### Goal
Implement and test functionality to extract and compare renderable mesh geometry from PM4/PD4 navigation files and corresponding WMO group files, focusing on connected component analysis for PM4 meshes and canonical WMO group file handling.

---

## New Context: Development Map Asset Correlation (2024-07-21)

- **Source:** The PM4 files being parsed are from World of Warcraft's development map (Map ID 451), which likely contains references to assets that were cut or changed during development.
- **Asset Dataset:** The user possesses a set of recreated assets and genuine assets from multiple WoW versions (0.5.3, 1.12, 2.4.3, 3.3.5, and retail).
- **Strategy:** For each ADT, reconstruct the list of referenced assets by cross-referencing PM4/ADT placements with the multi-version asset dataset. This enables identification of assets that were present, cut, or changed across versions, and allows mesh comparison even for missing/cut assets.
- **Historical Mapping:** This approach supports historical mapping of asset usage and changes, providing archival value and deeper analysis of WoW's development history.

---

## NEW SECTION: ADT Placement Reconstruction from PM4, WMO/M2, and WDL (2024-07-21)

### Objective
Reconstruct ADT placement data (model and WMO placements) for the development map (ID 451) by synthesizing information from:
- PM4 mesh data (as a proxy for placements, despite its limitations)
- Real WMO/M2 asset geometry (from the multi-version dataset)
- WDL files for low-res terrain (already used for terrain mesh reconstruction)

### Approach
- **Mesh Matching for Placement Inference:**
  - For each PM4 mesh component (island), attempt to match it to a WMO/M2 asset from the dataset (across all available versions).
  - Use geometric similarity (bounding box, vertex/triangle count, rough shape) to find the best fit.
  - Where a match is found, record a "synthetic placement" (asset file, position, orientation, scale) in a reconstructed ADT-like structure.
  - Where no match is found, flag for manual review or as "unknown/cut asset."
- **Integrate WDL Terrain:**
  - Use the WDL mesh as a spatial reference to help align and validate placements.
  - Optionally, use WDL to fill in missing terrain for incomplete ADTs.
- **Output:**
  - Generate a reconstructed ADT placement file (or set of files) for the development map, with as much fidelity as possible.
  - Annotate each placement with confidence level (e.g., "matched by mesh," "manual guess," "unknown").
  - Optionally, visualize the reconstructed placements over the WDL terrain for QA.

### Next Steps
1. Deepen WMO data parsing to extract placement and geometry data needed for matching.
2. Implement mesh matching utility to compare PM4 mesh components to WMO/M2 assets and suggest best-fit matches.
3. Write logic to generate synthetic ADT placement records from mesh matches.
4. Integrate with WDL terrain to ensure spatial consistency.
5. Develop tools to visualize the reconstructed placements for validation.
6. Update documentation and memory bank to reflect this workflow and rationale.

### Blockers & Open Questions
- Incomplete WMO parsing for robust geometry/placement extraction.
- Mesh matching accuracy and heuristics (bounding box, shape, etc.).
- Incomplete ADT set for the development map.
- How to handle ambiguous or partial matches?
- What format should the synthetic ADT output take?

---

### Recent Progress
- Finalized WMO APIs (`WmoGroupMesh.cs`) for loading and exporting WMO group geometry to OBJ.
- Finalized PM4 state-0 extraction (`Pm4MeshExtractor.cs`) to export geometry to OBJ.
- Confirmed both WMO and PM4 OBJ files can be generated for test assets (`ND_IronDwarf_LargeBuilding.wmo` and `development_00_00.pm4`).
- Pivoted away from filename-based filtering in PM4 mesh extraction; now focusing on connected component analysis.

### Current Focus: PM4 Component Analysis Strategy

*   **Observation:** PM4 OBJ output contains geometry for multiple assets; direct comparison to WMO is not meaningful.
*   **Strategy:** Analyze extracted PM4 mesh for connected components, isolate the largest, and compare to WMO mesh.
*   **UniqueID Correlation:** Only possible for `development_00_00.pm4` due to available ADT data; do not generalize.
*   **WMO Group Handling:** Always parse root for group info, then parse/merge group files for geometry. Never concatenate root+groups.
*   **Asset Reference Recovery:** For each ADT, extract all placements and attempt to resolve asset references against the multi-version dataset (prioritize genuine, fall back to recreated). Note missing assets for archival purposes.
*   **Mesh Comparison:** When comparing PM4 mesh groups to WMO/MDX assets, allow for version selection or fallback, and document discrepancies.

### Next Steps
1. Implement `MeshAnalysisUtils` for connected component analysis and largest component extraction.
2. Update PM4 mesh extraction tests to use this logic and output OBJ for the largest component.
3. Compare largest PM4 component to WMO mesh visually and programmatically.
4. Document canonical WMO group file handling in tools/docs.
5. Inventory and plan consolidation of tools/tests/utilities in a dedicated branch.
6. Continue research into MSCN/MSLK and doodad decoding.
7. Address build errors (notably uint/int index types in PM4 extraction).
8. Implement or update utilities to parse and index the asset dataset by version, automate matching of ADT/PM4 references to available assets, and output reports/logs on missing or unmatched assets.
9. Integrate asset matching logic into the mesh extraction and comparison pipeline.

### Blockers
- Build errors in PM4 mesh extraction (uint/int index types).
- Validation of "largest component" hypothesis across more test cases.
- Ongoing research into MSCN/MSLK and doodad decoding.
- Need for robust asset matching and reporting utilities.

---

# Plan: Consolidate Tools, Tests, and Utilities (2025-04-16)

## Goal
Unify all mesh tools, explorers, analysis utilities, and tests into a single, well-documented suite, while keeping WoWToolbox.Core as a pure library of chunk definitions and data structures. This will reduce duplication, clarify project structure, and make future development more maintainable.

## Steps
1. Inventory & Documentation: Catalog all current tools, tests, explorers, and mesh utilities. Document in a central markdown file.
2. Gap & Redundancy Analysis: Identify overlaps, redundancies, and missing pieces.
3. Consolidation Roadmap: Propose unified structure, plan merges/refactors, ensure non-core logic is outside Core.
4. Implementation in a Separate Branch: Migrate/refactor in a new branch, keep main branch stable.
5. Documentation & Onboarding: Update docs and onboarding guides as suite is consolidated.

## Execution
- This plan will be executed in a dedicated branch to allow for a cleaner implementation and to preserve the current working state for ongoing work or reference.

---

## NEW FOCUS: Mesh Analysis and Comparison (2024-07-21)

### Objective
Develop robust logic to analyze and compare extracted meshes from PM4 and WMO files. Move beyond basic vertex/triangle count checks to implement geometric and shape-based comparison metrics.

### Requirements
- Define what constitutes a mesh "match" (identical geometry, similar shape, bounding box overlap, centroid distance, etc.).
- Implement comparison metrics: vertex/triangle count, bounding box, centroid, surface area, and (optionally) Hausdorff or other shape distances.
- Support tolerance for translation, rotation, and scale differences where appropriate.
- Provide detailed diagnostic output for mismatches (e.g., which vertices/triangles differ, by how much).

### Next Steps
1. Design a mesh comparison API/interface (input: two MeshData objects, output: result object with match/mismatch, score, diagnostics).
2. Implement basic geometric comparisons (vertex/triangle count, bounding box, centroid).
3. Add advanced shape similarity metrics as needed.
4. Integrate with test project and validate on real data.
5. Document rationale and design in memory bank.

---

# NEW SECTION: Direct PM4 OBJ vs. WMO Mesh Data Comparison (2024-07-21)

## Objective
Compare mesh data from PM4-extracted OBJ files (already generated by the test suite/tooling) directly to mesh data loaded from WMO binaries (using existing WMO loaders), without converting WMO data to OBJ or re-extracting PM4 mesh data from binary. This comparison will be used to infer and generate WMO placement data for future restoration of model placements on the development map.

## Workflow
- **PM4 Meshes:** Use OBJ files output by the test suite/tooling as the source of PM4 mesh data. Load these using a simple OBJ loader into a common mesh structure (e.g., MeshData).
- **WMO Meshes:** Use existing WMO loaders (e.g., WmoGroupMesh, WmoRootLoader) to extract mesh data directly from WMO binaries, merging group meshes as needed.
- **Comparison:** For each PM4 OBJ mesh, compare it against every WMO mesh using existing mesh comparison utilities (MeshComparisonUtils, etc.), computing similarity metrics (vertex/triangle count, bounding box, centroid, shape similarity, etc.).
- **Reporting:** Output a report of best matches, similarity scores, and diagnostics for each comparison. Use these results to infer/generate WMO placement data for restoration.

## Rationale
- **No WMO-to-OBJ conversion:** Avoids duplicating work and leverages perfected PM4 OBJ extraction.
- **No PM4 re-extraction:** Uses already-generated OBJ files for PM4 meshes.
- **Direct, robust comparison:** Ensures the most accurate and efficient workflow for filling gaps in PM4 data and supporting future placement restoration tools.

---
