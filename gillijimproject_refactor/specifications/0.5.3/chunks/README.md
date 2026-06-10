# Chunk Research — Build 0.5.3.3368

This folder mirrors the 0.7.0 chunk-oriented layout for **0.5.3.3368**.

## Subfolders
- `ADT/`
- `WDT/`
- `WMO/`
- `MDX/`

## Research stance
- 0.5.3 is treated as a **monolithic map era** where WDT root parsing and ADT-like terrain/object chunks are intertwined.
- Evidence is drawn from direct Ghidra assertion/token strings in the current binary session.
- Any unresolved field remains marked `???`.

## Core evidence cluster used
- Root WDT: `MARE`, `MAOF`, `MVER`, `MAIN`, `MPHD`, `MDNM`, `MONM`
- Embedded terrain/object: `MHDR`, `MCIN`, `MTEX`, `MCNK`, `MCLY`, `MCRF`, `MDDF`, `MODF`
- WMO group-domain tokens and MDLX (`XLDM`) check also confirmed and documented in their subfolders.
