# Project Brief

## Mission
WoW format preservation, analysis, and visualization across **Alpha 0.5.3**, **0.6.0**, and **LK 3.3.5** clients.

## Primary Project: MdxViewer
A high-performance .NET 9 / OpenGL 3.3 world viewer that renders terrain, WMOs, MDX models, liquids, and DBC-driven overlays from any supported WoW build. This is the main focus of active development.

## Scope
- **MdxViewer**: Real-time 3D rendering of WoW worlds from MPQ data (0.5.3, 0.6.0, 3.3.5)
- **Read/Write**: WDT, ADT, WMO, M2/MDX, BLP formats for all supported versions
- **Convert**: Modern → Alpha (retroporting) and Alpha → Modern (analysis)
- **Tools**: DBC crosswalk generation, PM4 reconstruction, ADT merging, texture conversion

## Current Reality (Feb 2026)
**MdxViewer is the primary deliverable.** It supports Alpha 0.5.3 monolithic WDTs, 0.6.0 split ADTs (including WMO-only maps), and partial 3.3.5 support. Terrain, WMOs, MDX models, liquids, and DBC overlays all render correctly. Performance is being optimized with persistent caching, throttled I/O, and expanded AOI streaming.

## Success Criteria
1. **Multi-version rendering** — 0.5.3, 0.6.0, 3.3.5 worlds render correctly
2. **Smooth streaming** — No frame drops during tile loading
3. **Correct liquids** — MCLQ, MH2O, and MLIQ all render at proper heights with correct types
4. **WMO-only maps** — Instance dungeons and battlegrounds load from WDT MODF

## Next Milestones
- Fix 3.3.5 ADT loading freeze
- MDX animation support
- Vulkan render backend research
