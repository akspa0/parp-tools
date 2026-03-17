# Renderer Plan (Reference)

## Completed Phases
- Phase 0: Foundation (constants, blend, materials, render queue, frustum, shaders)
- Phase 1: MDX Animation (keyframes, compressed quats, bones, geoset anim, playback)
- Phase 2: Particles (PRE2 complete — emitter, physics, atlas, billboard; RIBB pending)
- Phase 3: Terrain (Alpha adapter, mesh gen, texture layering, lighting, AOI streaming)
- Phase 4: World Scene (placements, fog, day/night, ViewerApp integration)
- Phase 5: Liquids (MCLQ mesh + type detection; MH2O broken for 3.3.5)

## Remaining (not active)
- Phase 1 gaps: Material/texture animation (TXAN, UV scrolling, global sequences)
- Phase 2 gap: Ribbon emitters (RIBB)
- Phase 5 gap: MH2O (3.3.5 liquid format) — currently falls back to MCLQ
- Phase 6: Detail Doodads (per-chunk grass/foliage + distance fade)
- Phase 7: Polish (GPU instancing, texture atlas/array, LOD, debug overlays, skybox metadata)
- Skybox: backdrop pass works; DBC/WMO-driven metadata not yet first-class

See `implementation_prompts.md` for ready-to-use Copilot prompts for each feature.
