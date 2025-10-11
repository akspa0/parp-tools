# Analyze-Links Implementation Plan

**Author:** Cascade

**Created:** 2025-07-11 02:20 (local)

---

## Objective
Develop a unified CLI/tool that extracts *all* inter-chunk references present in a PM4 file (or directory) and emits **raw-only CSVs** to facilitate forensic analysis of missing polygons and other geometry issues.

## Key Requirements
1. **Input Modes**
   * Single PM4 file
   * Directory of PM4 files (recursive search)  
     Optional flags: `--out <dir>` (default `./links`), `--verbose`.
2. **Output**
   * One CSV per link type (MSLK, MSUR→MSVI/MSPV, MPRL→MSPI, MPRR, etc.).
   * Header row always present.
   * Columns reflect **directly-stored fields only** – no derived data.
3. **Coverage**
   * Handle every parsed chunk; gracefully skip unknown or empty chunks.
4. **Integration**
   * Live in `WoWToolbox.Core.v2/Services/PM4/AnalyzeLinksTool.cs`.
   * Exposed via `Pm4BatchTool analyze-links` command.
5. **Testing**
   * Use real sample data (`development_*` tiles).
   * Smoke tests: CSV row counts == chunk entry counts.

## Implementation Roadmap

| Step | Description | Owner | Status |
|------|-------------|-------|--------|
| 1 | **Survey existing analyzers** (`MslkLinkScanner`, `MslkHierarchyAnalyzer`, `MprrHypothesisAnalyzer`) for reusable logic. | Cascade | ⏳ |
| 2 | **Define DTOs** for each link record if needed. | Cascade | ⏳ |
| 3 | Implement generic `LinkCsvDumper` (reflection-based). | Cascade | ⏳ |
| 4 | **Per-chunk scanners**<br/>• MSLK — parent/child, tile XY, flags.<br/>• MSUR ↔ MSVI/MSPV — vertex/index refs.<br/>• MPRL ↔ MSPI — index ranges.<br/>• MPRR — raw fields potentially referencing geometry.<br/>• Catch-all for additional link fields. | Cascade | ⏳ |
| 5 | **Integrate CLI**: add `analyze-links` sub-command to `Pm4BatchTool`. | Cascade | ⏳ |
| 6 | **Validation**: run on sample directory, verify CSVs, ensure no derived columns. | Cascade | ⏳ |
| 7 | **Documentation & Memory Bank**: update `progress.md` and reference this plan when milestones complete. | Cascade | ⏳ |

## Deliverables
* `AnalyzeLinksTool.cs` service class
* `Pm4BatchTool analyze-links` command wired to service
* Generated CSVs in `<outDir>/<Tile>/<ChunkName>.csv`
* Updated unit/integration tests (if any)
* Memory-bank updates (`progress.md`, `activeContext.md`)

---

*End of plan. Begin implementation in next chat once approved tasks are prioritized.*
