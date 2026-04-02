# Immediate Next Steps

## Reattach and capture continuity

1. Reattach x64dbg to the same Win32 `WoW.exe` 3.3.5.12340 process context.
2. Confirm debugger sanity with command breakpoint:
   - `0x0077f600` (`maxLOD` handler)
3. Re-arm minimal M2 chain breakpoints:
   - `0x0083cc80`
   - `0x00835a80`
   - `0x0083cb60`
   - `0x00838490`
   - `0x00836600`
4. If `DebugRun` keeps re-pausing in system DLL frames, stabilize first:
   - continue until WoW module user-code resumes, or
   - temporarily disable noisy first-chance/system-event breaks in x64dbg before world-path capture pass

## World-path acquisition runbook

1. Prioritize `0x0083cb60` and read stack-local skin buffer (`[ebp-0x104]`) each hit.
2. Classify each hit path as:
   - UI (`interface\\...`)
   - Portrait-flow side effects (portrait texture path triggers)
   - World (`world\\...`, `doodads\\...`, `expansion...`)
3. Continue until at least one world-path model is captured.
4. For the first world-path hit, collect full chain evidence:
   - profile index at `0x0083cd2a`
   - built `%02d.skin` at `0x0083cb60`
   - load success at `0x0083cd32`
   - init/rebuild at `0x00838490` and callback hits (`0x00824510`, `0x00832ea0`)
   - combiner output at `0x00836600` and return handle at `0x00836dab`

## Hidden-path experiments

1. Toggle and record runtime impact for:
   - `M2BatchDoodads`
   - `M2BatchParticles`
   - `M2ForceAdditiveParticleSort`
   - `M2UseClipPlanes`
   - `M2UseZFill`
2. Probe high optimization mode parser behavior:
   - `M2Faster` values: `0`, `1`, `2`, `3`, and decimal forms with non-zero hundreds digit
   - `M2FasterDebug` values: `0`, `1`, `2`, `3`
3. Record resulting `DAT_00d3fcf4` bit changes at runtime where possible.
4. During world-path hits, capture model-load flag words entering `FUN_0081c390` and compare with UI-path captures.

## wow-viewer follow-through (library target)

1. Keep this session as evidence input only; do not treat active `MdxViewer` behavior as design ownership.
2. Feed confirmed native findings into `wow-viewer` M2 seams:
   - explicit runtime flag-word contract
   - strict skin choose/load/init stage contract
   - combiner-family-to-effect contract
   - section-classification contract preserving unresolved `0x20` and `0x40`
3. Stage first wow-viewer implementation slice after world-path captures are in hand.

## Subsystem verification follow-up

1. Rendering and shaders (live)
   - breakpoint `FUN_00876d90` callsites during world load to capture real effect filenames requested at runtime
   - compare `M2_BuildCombinerEffectName` outputs against actual effect creation hits in `FUN_00876be0`
   - capture whether `M2UseClipPlanes` or `M2UseZFill` changes effect selection or only state bits
2. Liquids (live)
   - capture liquid type ids entering `FUN_008a1fa0`/`FUN_008a28f0` and verify fallback-to-water behavior on missing type
   - verify `WaterRipples` path activation in `FUN_0079e1a0` under `waterRipples` and `waterLOD` toggles
3. Particles (live)
   - capture branch conditions in `FUN_008214e0` to confirm when runtime uses direct submission vs merged `ParticleBatch`
   - toggle `M2BatchParticles` and `M2ForceAdditiveParticleSort`, then sample resulting particle ordering/batch behavior
4. Lighting (live)
   - capture `mapObjLightLOD` transitions and resulting light-pool behavior in world scenes
   - compare script-side `SetLight`/`AddLight` effects with world-map DBC-driven lighting state
5. LIT proof closure
   - keep standalone `.lit` support classified as unconfirmed until a positive native file-open/path anchor is found
   - if still negative after targeted file-open trace coverage, record explicit negative-evidence bounds in canonical doc
