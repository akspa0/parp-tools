# Progress — MdxViewer Renderer Reimplementation

## Status: Planning Complete → Ready for Phase 0

## What Works Today

| Feature | Status |
|---------|--------|
| MDX model loading + rendering | ✅ Per-geoset, multi-layer materials, textured |
| WMO v14 loading + rendering | ✅ Groups, doodad sets, textured |
| BLP2 texture loading | ✅ DXT1/3/5, palette, JPEG |
| MPQ data source | ✅ Listfile, nested WMO archives |
| DBC integration | ✅ DBCD, replaceable texture resolution |
| Camera | ✅ Free-fly WASD + mouse look |
| ImGui UI | ✅ File browser, model info, visibility toggles |
| GLB export | ✅ MDX + WMO |

## What's Missing (by phase)

| Phase | Description | Items | Status |
|-------|-------------|-------|--------|
| 0 | Foundation (constants, blend, materials, render queue, frustum, shaders) | 6 | ✅ Complete |
| 1 | MDX Animation (keyframes, bones, geoset anim, playback, UI) | 7 | ⏳ Not started |
| 2 | Particles (emitters, physics, rendering) | 5 | ⏳ Not started |
| 3 | Terrain (WDT/ADT loading, mesh, texture layers, lighting, AOI) | 8 | ⏳ Not started |
| 4 | World Scene (composition, placements, fog, day/night) | 6 | ⏳ Not started |
| 5 | Liquids (mesh, rendering, detection) | 3 | ⏳ Not started |
| 6 | Detail Doodads (generation, rendering) | 2 | ⏳ Not started |
| 7 | Polish (perf, debug overlays, extras) | 3 | ⏳ Not started |

## Key Risks

- ~~**ADT parser**: May need to write one~~ → RESOLVED: Reuse `gillijimproject-csharp` (WdtAlpha, AdtAlpha, McnkAlpha)
- **Alpha ADT format**: Non-interleaved vertices (81 outer then 64 inner) — McvtAlpha.ToMcvt() handles reorder
- **Floating-point precision**: Large world coordinates need camera-relative rendering
- **Texture memory**: Full map tile with all layers could be heavy
- **Target map**: Need Alpha-format WDT from MPQ; test_data/development has split Cata ADTs (not suitable)

## Implementation Order (terrain-first strategy)

1. ~~Phase 0 — Foundation~~ ✅
2. **Phase 3 — Terrain rendering** ← NEXT
3. Phase 4 — World Scene (compose terrain + models + WMOs)
4. Phase 1 — MDX Animation
5. Phase 2 — Particles
6. Phase 5-7 — Liquids, Detail Doodads, Polish

Reference implementation first, then extend in our own direction.

## Recent Changes

- 2026-02-06: Created itemized renderer plan (`renderer_plan.md`)
- 2026-02-06: Created active context and progress tracking
- 2026-02-06: Phase 0 COMPLETE — 6 foundation files building (0 errors)
- 2026-02-06: Discovered existing Alpha parsers in gillijimproject-csharp — eliminates Phase 3 parser work
- 2026-02-06: Updated plan: Phase 3 now only needs rendering adapter bridge, not new parsers
- 2026-02-06: Decided terrain-first strategy: Phase 3 → Phase 4 → Phase 1 → rest
