# Active Context: ADT/PM4 Correlation & Analysis

**Goal:** Accurately parse PD4 and PM4 files, understand their structure, integrate ADT parsing (correctly handling split files like `_obj0.adt` using `Warcraft.NET`) to correlate PM4/ADT UniqueIDs and extract object placement data (including FilePath, FileDataId, readable Flags), resolve discrepancies, and export usable OBJ files and analysis reports.

**Current Focus:** Implement directory-based processing in `AnalysisTool` to handle the `development` dataset, generating `pm4_unique_ids.txt` for all PM4s and `correlated_placements.yaml` for ADT/PM4 pairs.

**Discoveries Since Last Update:**
*   **ADT/PM4 Link:** Identified `m_destructible_building_index` from MDOS (via MDSF) as the UniqueID linking PM4 data to ADT object placements.
*   **`Warcraft.NET` ADT Split File Handling:** Confirmed `Warcraft.NET` uses separate classes for ADT split files (e.g., `TerrainObjectZero` for `_obj0.adt`, `Terrain` for base `.adt`). Our previous custom `ADTFile` was incompatible with this structure.
*   **Old Parser (`WCAnalyzer.bak`):** Found relevant models (`Placement.cs`) but decided to build new service on `Warcraft.NET`.
*   **Warcraft.NET Property Names:** Corrected usage to `Filenames` for `MMDXChunk`/`MWMOChunk`.
*   **Listfile Pathing:** Determined correct relative path from `AnalysisTool` output directory to listfile location.
*   **YAML Rotation Anchors:** Confirmed `*o0` syntax is standard YAML optimization for zero rotation, not a data error.

**Preliminary Visualization Insights (User Observation - From previous runs):**
*   Combining PD4 geometry chunks into one OBJ caused misalignment.
*   Individual MSPV, MSVT chunks visually matched source geometry when correctly transformed.
*   MSLK nodes often near geometric corners.

**Recent Changes:**
*   **ADT Loading Refactor:**
    *   Removed custom `WoWToolbox.Core/ADT/ADTFile.cs`.
    *   Updated `AnalysisTool/Program.cs` to load base `.adt` files into `Warcraft.NET.Files.ADT.Terrain.Wotlk.Terrain` and `_obj0.adt` files into `Warcraft.NET.Files.ADT.TerrainObject.Zero.TerrainObjectZero`.
    *   Updated `AdtService.ExtractPlacements` to accept these `Warcraft.NET` objects and access `MDDF`/`MODF` from `TerrainObjectZero` and `MMDX`/`MWMO` from `Terrain`.
*   **Placement `FilePath` Added:** Populated `FilePath` via MMDX/MWMO lookup.
*   **Listfile Integration:** Implemented loading (`Program.cs`) and usage (`AdtService.cs`) of community listfile (`community-listfile-withcapitals.csv`).
*   **FileDataId Lookup:** Added logic to `AdtService` to determine `FileDataId` based on `NameIdIsFiledataId` flag (direct lookup) or reverse path matching against listfile (case/slash insensitive).
*   **Placement `Name` Populated:** Set obsolete `Name` property from `Path.GetFileName(FilePath)`.
*   **Flag Decoding:** Added `FlagNames` list to `Placement` model and implemented logic in `AdtService` to populate it based on `MDDFFlags`/`MODFFlags` enums.
*   **Added Missing MDDF Flags:** Added definitions for `Unk4`, `Unk8`, `Unk10`, `Unk100`, `AcceptProjTextures` to the `MDDFFlags` enum in `Warcraft.NET` and updated `AdtService` to check and report them.
*   *(Older changes for PM4/PD4/Build listed in previous versions)*

**Next Steps:**
1.  ~~Implement `AdtService.cs`~~ (Done)
2.  ~~Integrate into AnalysisTool~~ (Done)
3.  ~~Add YAML Output~~ (Done)
4.  ~~Add FilePath Lookup~~ (Done)
5.  ~~Add Listfile Integration & FileDataId Reverse Lookup~~ (Done)
6.  ~~Add Flag Name Decoding~~ ✓ (Completed)
7.  ~~Refactor ADT Loading for Split Files~~ ✓ (Completed using `Warcraft.NET` native classes)
8.  **Implement Directory Processing:** Modify `AnalysisTool` to scan input directory, process all PM4s (outputting `pm4_unique_ids.txt`), process ADT/PM4 pairs (outputting `correlated_placements.yaml`), and generate a summary report. Target: `I:\parp-scripts\WoWToolbox_v3\original_development\development`.
9.  **Visualize & Test:** Verify outputs from directory processing (`correlated_placements.yaml`, `pm4_unique_ids.txt`, summary report) and OBJ outputs (PM4/PD4).
10. **Analyze Logs:** Examine MSLK analysis logs, MDSF logs (focus on `Unk12`).
11. **Interpret `Unk01`:** Analyze node anchor visualization.
12. **Leverage ADT/PM4 Link:** Perform detailed analysis using UniqueIDs and FilePaths/FileDataIds from YAML.
13. **Investigate Destruction:** Analyze `MDOS.destruction_state` correlation with PM4 face generation.
14. **Investigate Faces:** Refine face generation if needed.
15. **Re-enable Asserts:** Gradually re-enable commented assertions in tests.
16. **(Future Goal) WMO Geometry Parsing & Matching:** Investigate using `Warcraft.NET` to parse WMO geometry and compare against PM4 geometry to identify assets.

**--- (Context Updated: ADT loading refactored for split files; Next: Directory processing) ---**
