# WMO Chunks — 0.5.3.3368

## Confirmed by parser assertion strings
- `MOGP` (`0x008A2854`)
- `MOBA` (`0x008A286C`)
- `MOTV` (`0x008A28C0`)
- `MOVT` (`0x008A28F8`)
- `MOPY` (`0x008A2914`)
- `MLIQ` (`0x008A2930`)

## Notes
- These are high-value confirmations that the 0.5.3 client still uses an IFF-style chunk parser with explicit token assertions.
- Function entrypoint addresses are still unresolved in this pass due missing xref-level navigation in current MCP surface.

## Next Deep-Dive Targets
- Resolve `MOGP` header offsets and field semantics
- Confirm whether `MOTV` appears once or multiple times per group in this build
- Recover MOBA/MOPY entry widths directly from decompiled parser loops
