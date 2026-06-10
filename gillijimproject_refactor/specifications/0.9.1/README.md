# WoW 0.9.1.3810 (macOS) — Ghidra Findings

This folder contains chunk-format findings from direct Ghidra analysis of the macOS WoW client `0.9.1.3810`.

## Files
- `ADT_MCNK_0.9.1.3810.md`
- `ADT_MCLQ_0.9.1.3810.md`
- `ADT_Unknown_Field_Resolution_0.9.1.3810.md`
- `WMO_MOGP_MOTV_MLIQ_0.9.1.3810.md`
- `WMO_Root_Chunk_Order_0.9.1.3810.md`
- `Map_Load_Freeze_Analysis_0.9.1.3810.md`
- `Parser_Profile_0.9.1.3810_Binary.md`
- `Parser_Profile_0.9.1.3810_Field_Map.md`
- `WMO_Group_MLIQ_Contract_0.9.1.3810_Binary.md`
- `MDX_Binary_Contract_0.9.1.3810.md`
- `Parser_Profile_0.9.1.3810_KnownUnknowns.md`

## Highlights
- `MCLQ` is active in ADT and wired through `CMapChunk::CreateLiquids`.
- `MH2O` was not observed in this binary pass.
- WMO parsing is strict-order and assertion-heavy in this build.
- `MOGP` optional chunks are flag-gated; `MLIQ` is gated by `0x1000`.
