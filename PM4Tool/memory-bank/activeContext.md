# Active Context: Interpreting PD4 MSLK Structure

**Goal:** Accurately parse PD4 and PM4 files, understand their structure (especially shared chunks like MSLK, MSUR), resolve discrepancies, and export relevant data. Key focus is understanding the **differences in MSLK structure between PM4 and PD4, driven by file scope.**

**Current Focus:** Analyzing the detailed MSLK entry data from the updated PD4 logs (`6or_garrison...`) to understand the patterns and meaning of fields within "Node Only" groups (`Unk00`, `Unk01`, `Unk10`).

**Recent Changes:**
*   **PM4 MSLK JSON Export:** Added logic to `PM4FileTests` to generate a hierarchical JSON file (`_mslk_hierarchy.json`) representing the Mixed Group structure.
*   **PD4 MSLK Analysis Tool Update:** Modified `MslkAnalyzer` to log detailed information for every entry within each group (node and geometry).
*   **PD4 Log Regeneration:** Re-ran the updated `MslkAnalyzer` on PD4 logs to generate detailed analysis files.
*   **PD4 MSLK Structure Interpretation:**
    *   Confirmed PD4 uses separate "Node Only" / "Geometry Only" groups (based on `Unk04`), unlike PM4's "Mixed Groups".
    *   **Hypothesis:** This difference stems from PD4 representing a single object (WMO) while PM4 represents a multi-object collection (map tile), making explicit node-geometry linking via group ID unnecessary in PD4.
    *   Preliminary analysis of PD4 "Node Only" groups (limited sample) showed potentially consistent `Unk00=0x01`, variable `Unk01`, and presence of `Unk10` (MSVI link).
*   **Memory Bank Update:** Updated `systemPatterns.md`, `techContext.md`, and `progress.md` with the latest findings regarding MSLK structural differences, the file scope hypothesis, and preliminary PD4 node analysis.

**Next Steps:**
1.  **Analyze More PD4 MSLK Nodes:** Examine the detailed output in the *newly generated* `6or_garrison..._mslk_analysis.log` files for more "Node Only" groups. Confirm `Unk00`/`Unk01` patterns and value ranges. (Current step)
2.  **Interpret PD4 Node Semantics:** Determine the meaning/purpose of different `Unk01` values and the role of nodes in a single-object context.
3.  **Investigate PD4 `Unk10` Links:** If patterns emerge, plan how to cross-reference node `Unk10` values with `MSVI` data.
4.  **Revisit PM4/PD4 General Tasks:** Address other items like MSCN alignment, MDSF research, MSUR analysis based on MSLK insights.

**--- (Context Updated: PD4 MSLK structure difference linked to file scope, preliminary node analysis done, detailed logs generated) ---**

## Recent Findings & Decisions
*   **MSLK `Unknown_0x04` Behavior:** Confirmed difference PM4 (Mixed Groups link node/geom) vs PD4 (Node Only/Geom Only groups). **Hypothesis:** Driven by multi-object (PM4) vs single-object (PD4) file scope.
*   **PD4 MSLK Node Preliminary:** `Unk00` often `0x01`, `Unk01` varies, `Unk10` (MSVI link) present.
*   **Analysis Tool:** Updated `MslkAnalyzer` to provide detailed entry logs.
*   **PM4 Visualization:** Added JSON export for hierarchy.
*   **Chunk Correlations:** Documented known direct/indirect links (`MSUR`->`MDOS`/`MSVI`, `MSLK`->`MSPI`/`MSVI`) and lack of implemented links (`MSLK`<->`MSCN`/`MSUR`).
*   PD4 MSLK.Unk10 Analysis: (Status unchanged) Confirmed likely `MSVI` index. Purpose TBD.
*   PM4 Test Build: Resolved persistent build issues by moving analysis logic to a separate console application (`WoWToolbox.AnalysisTool`).
