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

# Active Context: WMO v14 Converter Texture Issues (2025-04-22)

## Problem
- While the geometry conversion from v14 to v17 WMO format is working correctly, textures are not being properly applied in the exported OBJ/MTL files.
- We made significant progress fixing the triangle index handling in the `ExportMergedGroupsAsObj` method, ensuring vertex indices are properly calculated without truncation.
- However, texture mapping is still missing, resulting in models with correct geometry but no textures.

## Current Status
- The converter successfully exports merged geometry from all groups in the v14 WMO file.
- Triangle index handling has been fixed in the `ExportMergedGroupsAsObj` method.
- Material ID to PNG mapping is being created but may not be correctly linked to the MTL file.
- The exported MTL file references texture files, but they may not be properly extracted or the paths may be incorrect.

## Next Steps
1. Verify the texture extraction process in `ExtractAndConvertTextures` method.
2. Ensure BLP to PNG conversion is working correctly.
3. Double-check the MOTX/MOMT chunk processing to verify material ID to texture mapping.
4. Review the WmoGroupMesh.SaveToObjAndMtl method to ensure it's using the material ID to PNG mapping correctly.
5. Analyze file paths in the MTL file to ensure they correctly reference the exported textures.

## Recent Discoveries
- The issue appears to be related to texture path handling rather than a fundamental problem with the converter.
- Geometry conversion is working properly, suggesting that the core logic for chunk parsing and mesh assembly is correct.
- The materialIdToPng dictionary may not be populated correctly or may use incorrect paths.

---

# Active Context: WMO v14 Group Mesh Extraction (2025-04-19)

## Problem
- v14 monolithic WMOs (e.g., Ironforge) do not store explicit mesh data in group files. Instead, geometry must be assembled from raw chunk data (MOVT, MONR, MOTV, MOPY, MOVI, etc.).
- Previous attempts to use synthetic MVER logic and legacy group file parsing failed, as the embedded data does not match the expected standalone group file format.

## New Approach
- Directly parse and decode all relevant geometry subchunks from the group data.
- Assemble mesh structures (vertices, normals, UVs, triangles, materials) from the decoded arrays.
- Remove all synthetic MVER logic and do not use the legacy group file parser for v14 embedded data.
- Add granular debug output for each chunk and mesh, including parsed counts and warnings for missing/malformed data.

## Current Status
- v14-specific group chunk parser implemented: parses MOVT, MONR, MOTV, MOPY, MOVI, and assembles mesh data.
- OBJ output now depends on correct chunk decoding and mesh assembly, not on legacy loader compatibility.
- Logging and error handling improved; missing geometry is now traceable to missing or malformed subchunks.

## Next Steps
1. Refine mesh assembly logic for edge cases and additional subchunks (MOBA, MLIQ, etc.).
2. Add batch/group export support for multiple groups per WMO.
3. Further improve error handling and debug output.
4. Test and validate on a wider range of v14 WMOs.

## Blockers & Discoveries
- Some geometry may still be missing due to incomplete chunk parsing or undocumented subchunk formats.
- Need for robust handling of optional/missing subchunks.
- Further research required for less common subchunks and edge cases.

---

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

### Current Focus: Standalone OBJ-to-WMO Matcher Tool (2024-07-21)

- WMO mesh export and coordinate system are now correct (+Z up).
- PM4 OBJ clusters are used as input for matching, not re-segmented in the main pipeline.
- A new, standalone tool will batch-compare OBJ clusters to all WMO assets in `test_data/335_wmo/World/wmo` (recursively).
- The tool will use rigid registration (translation + rotation, scale=1) to find the best match for each OBJ cluster.
- All placement extraction and matching logic will be in this standalone tool, not in the main PM4/WMO code.
- **Mesh extraction and OBJ output for mesh+MSCN boundary are correct, but the test process currently hangs after output and does not exit.**

## Why
- Avoids breaking the main mesh extraction/conversion code.
- Enables robust, automated matching and placement extraction for restoration and archival.

## Next Steps
1. Debug and resolve the test process hang after mesh+MSCN boundary output.
2. Once resolved, proceed with mesh comparison and placement inference.
3. Implement the standalone analyzer tool in C#.
4. Use MathNet.Numerics for SVD-based registration.
5. Output YAML/JSON/CSV with placement and match data.
6. Validate on real data and iterate.
7. Document the workflow and update the memory bank after tool validation.

### Blockers
- **Test process hang after mesh+MSCN boundary output.** The process does not exit and must be manually cancelled. Need to ensure all resources are disposed and the test method completes.
- Build errors in PM4 mesh extraction (uint/int index types).
- Validation of "largest component" hypothesis across more test cases.
- Ongoing research into MSCN/MSLK and doodad decoding.
- Need for robust asset matching and reporting utilities.

### Recent Discoveries (2025-04-17)
- Visual inspection now confirms that the MSCN chunk in PM4 files represents the exterior (boundary) vertices for each object.
- Plan: Update the PM4 exporter to annotate/export MSCN points as "exterior vertices" in output formats (OBJ, YAML, etc.) and validate this in future exports.

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

## NEW: Batch WMO-to-OBJ Exporter (2025-04-18)

- Implemented a batch WMO-to-OBJ exporter as an xUnit test in `WmoBatchObjExportTests`.
- Recursively scans a WMO binary directory, exports each WMO to OBJ, and writes to `/output/wmo/` while preserving folder structure.
- Uses caching: skips export if OBJ already exists, enabling fast, repeatable test runs.
- This exporter is now the canonical way to generate OBJ caches for WMO assets for mesh comparison and analysis.
- Integrated into the test suite for automation and reproducibility.
- Enables robust, repeatable workflows for PM4/WMO mesh comparison and placement inference.

---

## NEW: WMO v14 Monolithic Mesh Extraction Blocker (2025-04-18)

- Current loader for v14 monolithic WMOs only finds a single geometry set (e.g., 688 vertices, 573 triangles for Ironforge), which is far too small for a city-scale WMO.
- Exported OBJ files are invalid/corrupt and crash mesh viewers (e.g., Meshlab), confirming incomplete or misaligned geometry extraction.
- The likely cause is a chunk alignment or traversal bug: the loader does not scan the entire MOMO chunk for all geometry sets, and may miss or misinterpret sub-chunks.
- **Next step:** Implement a robust MOMO chunk scanner that logs all sub-chunks (ID, offset, size) and extracts all geometry sets (MOPV/MOPT pairs), merging them into a single mesh for OBJ export.
- This is critical for supporting large, monolithic WMOs like Ironforge and for enabling valid mesh comparison and placement workflows.

---

# Correction: WMO v14 File Handling and User Feedback (2024-07-21)

## Key Correction
- Previous assumptions or code that treated WMO v14 files as lacking mesh data were incorrect. v14 WMOs contain all the same mesh and placement data as v17 and later, but organized in a single monolithic file rather than split group files.
- The correct approach is to scan the monolithic file for all relevant geometry and placement chunks (MOGP, MOPY, MOVI, MOVT, etc.) and decode them directly, not to rely on version-based heuristics or legacy split-file logic.

## User Feedback and Policy Update
- User feedback highlighted the cost of incorrect assumptions: time and energy wasted chasing non-existent problems due to AI hallucination and version-based exclusion.
- **New Policy:**
  - Always inspect the raw chunked data in any WMO file, regardless of version or organization.
  - Never assume the absence of mesh or placement data based on file version or structure.
  - All mesh extraction and analysis must be chunk-driven, not version-driven.
  - User input about file structure or data presence must be prioritized and validated by direct inspection of the file, not dismissed due to terminology or internal heuristics.

## Implementation
- Refactor all mesh extraction and analysis tools to operate directly on the chunked data in WMO files, with no version-based exclusions or synthetic data hacks.
- Update documentation and memory bank to reflect this corrected understanding and approach.
- Maintain respectful, precise, and user-centered communication in all future interactions.

---

# Update (2024-07-21): Liquid Handling Removal and New Focus

## Liquid Handling Removal
- All code and dependencies related to liquid handling (including DBCD) have been removed from the project. Liquid support will be implemented in a separate project in the future.

## New Technical Focus: WMO Texturing (v14/v17)
- The current technical focus is on robust handling of WMO chunk data related to texturing, supporting both v14 (mirrormachine) and v17 (wow.export) formats.
- The goal is to ensure all text strings (texture names, material names, etc.) and relevant data are extracted and mapped for full WMO reconstruction in v17 and/or OBJ+MTL with proper texturing.
- Investigation is underway to:
  - Review wow.export for v17 chunk/texturing support.
  - Crosswalk v14 (mirrormachine) knowledge to v17 structures.
  - Enumerate all relevant chunks and string fields.
  - Design a unified data model for texturing.
  - Update parsers and export logic accordingly.

## Next Steps
- Complete the investigation and mapping of texturing-related chunk data.
- Propose and implement a unified approach for handling and exporting texturing data.
- Document all new patterns and findings in the memory bank.

---

## Update (2024-07-21): WMO Texturing Model Integration Debugging

- Encountered a persistent CS0149 'Method name expected' error when integrating the unified WMO texturing model (MOTX.ReadStrings) into the pipeline.
- Resolved by introducing a local alias for MOTX (using MOTXStruct = WoWToolbox.Core.WMO.MOTX;) in the factory, eliminating ambiguity and ensuring correct static method resolution.
- The unified model and loader now fully support both v14 and v17+ WMO files, and are integrated into the converter pipeline.

---

# NEW SECTION: WMO v14 Texturing Deep-Dive (2025-04-22)

## Problem
- Previous attempts at v14 WMO OBJ/MTL export resulted in only one texture being applied, or only a single pixel of each texture mapped to surfaces—indicating a misunderstanding of the v14 texturing pipeline.
- The mapping from chunk data (MOTX, MOMT, MOPY, MOTV, etc.) to correct OBJ/MTL export is not fully understood, leading to broken or incomplete material assignments.

## Current Technical Focus
- **Goal:** Fully comprehend the canonical texturing flow for WMO v14 before making any further code changes.
- **Reference Implementations:**
  - `wow.export` (modern v17 WMO parser/exporter, robust OBJ/MTL export logic)
  - `mirrormachine` (C++ v14↔v17 converter, v14 parsing logic)
- **Plan:**
  1. Review how v14 WMO stores and references textures and materials (MOTX, MOMT, MOPY, MOTV, MOVT, etc.).
  2. Compare with v17 (wow.export) and v14 (mirrormachine) reference implementations.
  3. Document the step-by-step mapping from triangle to material to texture to UV for v14.
  4. Synthesize a minimal, correct mapping table for a small v14 WMO as a test case.
  5. Only after this, propose a robust fix/refactor for the texturing pipeline.
- **Why:** Avoid repeating past mistakes and ensure that all surfaces in exported OBJ/MTL files use the correct textures and UVs, matching the original WMO as closely as possible.

---

## Current Focus (2024-07-21)
- Main focus: v14→v17 WMO conversion pipeline.
- v17 WMO writer skeleton is complete.
- Real serialization implemented for MOMT (materials) and MOGI (group info) chunks.
- WmoMaterial and WmoGroupInfo classes are mapped to the correct v17 binary layouts.
- Next steps: implement real serialization for MOHD (root header), MOBA (batch info), MOGP (group header), and integrate the writer into the conversion pipeline.
- Deep-dive analysis of chunk mapping and reference implementations (wow.export, mirrormachine) is ongoing.
- Validation with wow.export, mirrormachine, and noggit-red is planned to ensure output compatibility and fidelity.

---
