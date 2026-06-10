# ADT Chunks — Build 0.5.3.3368

This folder documents ADT-like chunks as they appear in the **0.5.3 monolithic WDT parse domain**.

## Evidence policy
- **Confirmed**: explicit token/assertion strings in the current 0.5.3 binary session.
- **Inferred**: behavior inferred from parser-stage clustering and 0.6.0/0.7.0 lineage.
- **Unknown (`???`)**: requires direct decompile loop/body confirmation.

## Chunks documented
- `MHDR.md`
- `MCIN.md`
- `MTEX.md`
- `MCNK.md`
- `MCLY.md`
- `MCRF.md`
- `MDDF.md`
- `MODF.md`
- `MCVT.md` *(not directly observed by string in this pass)*
- `MCNR.md` *(not directly observed by string in this pass)*
- `MCAL.md` *(not directly observed by string in this pass)*
- `MCLQ.md` *(not directly observed by string in this pass)*

## Notes
- In 0.5.3 these are treated as embedded terrain/object chunks under map loading, not fully separated into later ADT file workflows.
