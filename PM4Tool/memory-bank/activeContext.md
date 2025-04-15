# Mode: PLAN

# Active Context: WMO Mesh Extraction Pipeline

**Goal:** Parse PM4/PD4 files and reconstruct WMO group/root geometry as OBJ, using MSUR/MSVI/MSVT and node metadata (unk00/unk01).

**Current Focus:**
- Pipeline clusters and groups MSUR surfaces by (unk00, unk01) and unk00, exporting OBJs for WMO group and root geometry.
- Visual validation confirms strong correspondence to in-game assets, with some missing geometry (likely in MSCN/MSLK).

**Next Steps:**
1. Parse/integrate missing geometry from MSCN/MSLK.
2. Refine grouping logic as more is learned.
3. Automate mapping of unk00/unk01 to WMO filenames.

**Context Updated: Pipeline reconstructs most WMO group geometry; focus is now on missing details and further automation.**
