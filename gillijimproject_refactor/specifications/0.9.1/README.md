# WoW 0.9.1.3810 (macOS) — Ghidra Findings

This folder contains chunk-format findings from direct Ghidra analysis of the macOS WoW client `0.9.1.3810`.

## Files
- `ADT_MCNK_0.9.1.3810.md`
- `ADT_MCLQ_0.9.1.3810.md`
- `WMO_MOGP_MOTV_MLIQ_0.9.1.3810.md`
- `WMO_Root_Chunk_Order_0.9.1.3810.md`

## Highlights
- `MCLQ` is active in ADT and wired through `CMapChunk::CreateLiquids`.
- `MH2O` was not observed in this binary pass.
- WMO parsing is strict-order and assertion-heavy in this build.
- `MOGP` optional chunks are flag-gated; `MLIQ` is gated by `0x1000`.
