# Progress

## PM4 Mesh Extraction Pipeline

### What Works
* MSUR surfaces are clustered and grouped by (unk00, unk01) and unk00, with OBJ exports for WMO group/root geometry.
* Visual validation confirms strong correspondence to in-game assets, with some missing geometry (likely in MSCN/MSLK).
* Core PM4/PD4 loading, chunk definitions, and OBJ export logic are stable.

### Next Steps
* Parse/integrate missing geometry from MSCN/MSLK.
* Refine grouping logic as more is learned.
* Automate mapping of unk00/unk01 to WMO filenames.

### Known Issues
* Some geometry is still missing, likely defined in MSCN/MSLK.
* Doodad property decoding and MPRR index target identification remain open research items.

**Status: Pipeline reconstructs most WMO group geometry; further work needed for full fidelity and automation.** 