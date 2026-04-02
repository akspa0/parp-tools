# Win32 M2 Anchor Map

## Confirmed static anchors (Ghidra)

- 0x00835a80 - M2_FormatSkinFilename_02d
  - Builds exact %02d.skin path from model basename.
- 0x00835a20 - M2_FormatAnimFilename_04d_02d
  - Builds external animation path %04d-%02d.anim.
- 0x0083cc80 - M2_ChooseAndLoadSkinProfile
  - Chooses profile, loads skin payload, allocates texture array.
  - Emits: Failed to choose/load skin profile.
- 0x00838490 - M2_InitializeSkinProfileAndRebuildInstances
  - Validates and initializes profile payload; rebuilds live instances.
  - Emits: Corrupt skin profile data / Failed to initialize model skin profile.
- 0x00836600 - M2_BuildCombinerEffectName
  - Maps state to Diffuse_* and Combiners_* families.
- 0x00402760 - M2_RegisterRuntimeFlags
  - Registers M2UseZFill, M2UseClipPlanes, M2UseThreads, M2BatchDoodads,
    M2BatchParticles, M2ForceAdditiveParticleSort, M2Faster, M2FasterDebug.
- 0x0053c430 - M2_NormalizeModelPathAndProbeSkins
  - Normalizes .mdl/.mdx to .m2 and probes 00.skin..03.skin.

## Why these anchors are first-line

- They cover the entire critical chain for invisible M2s:
  - path/profile ownership
  - skin payload acceptance
  - live instance rebuild
  - combiner/effect family routing

## Runtime capture order

1. 0x0083cc80
2. 0x00838490
3. 0x00836600
4. 0x00835a80 and 0x00835a20 as supporting evidence

## Open runtime gap (current)

- First confirmed hits are now captured for the full choose/load/init/combiner chain, but only for UI-model context.
- Remaining gap is world-path capture under the same chain after debugger reattach.

## Live-confirmed chain (Apr 01 continuation)

- `0x0083cc80` / `0x0083cd2a`
  - profile selection observed live with index `1`, quality bucket path `0x40`.
- `0x00835a80` / `0x0083cb60`
  - exact formatter output confirmed:
    - `interface\\glues\\models\\ui_mainmenu_northrend\\ui_mainmenu_northrend01.skin`
- `0x0083cd32`
  - post-load success path confirmed (`EAX=1`).
- `0x00838490` / `0x00838561`
  - skin-init completion observed with model flags transition from `...01` to `...03`.
- `0x00824510` and `0x00832ea0`
  - callback rebuild loop observed live after skin-init completion.
- `0x00836600` / `0x00836dab`
  - combiner/effect path returns non-null handle in this runtime.

## Hidden or secondary path anchors worth tracking

- `0x004048b8`
  - startup callsite into `M2_RegisterRuntimeFlags`; immediately followed by M2 cache init call path.
- `0x004021c0` and `0x00402210`
  - `M2Faster` and `M2FasterDebug` callbacks; write high optimization bits through shared runtime flag helper.
- `0x00402410`, `0x00402470`, `0x004024d0`
  - live callbacks for doodad batching, particle batching, and additive-particle sorting bits.
- `0x0081c0d0`
  - M2 init routine with thread-path setup and fallback bit logic (`0x40`) that appears difficult to reach in normal startup flow.
- `0x0053c520`, `0x0053e810`, `0x0053e930`, `0x0053eaa0`
  - repeated `M2_NormalizeModelPathAndProbeSkins` prewarm chain (aggressive warm-up behavior, not the main model load path).

## Subsystem expansion anchors (Apr 02 continuation)

### Rendering/shader anchors

- `0x00780f50`
  - world render initialization; reloads `MapObj.wfx`, `MapObjU.wfx`, `Model2.wfx`, `Particle.wfx`, `ShadowMap.wfx`.
- `0x00876d90`
  - `Shaders\Effects\%s` loader (`ShaderEffectManager.cpp`).
- `0x00876be0`
  - effect object create/cache by effect name.
- `0x00872d30`
  - effect shader bind (`Shaders\Vertex` + `Shaders\Pixel`).
- `0x008728c0`
  - combiner argument table write for effect setup.
- `0x0068a9a0` and `0x00684c40`
  - shader-capability target selection and capability logging.

### Liquid anchors

- `0x008a3e00`, `0x008a3f70`, `0x008a4070`, `0x008a4190`
  - liquid material shader family constructors (`ProcWater`, `Water`, `WaterNoSpec`, `Magma`).
- `0x008a1fa0`
  - material bank lookup with water fallback.
- `0x008a28f0`
  - settings bank lookup with water fallback.
- `0x00793d20`
  - WMO liquid runtime setup and per-type material/settings bind.
- `0x007cefd0` and `0x007cf790`
  - `MapChunkLiquid.cpp` allocation/recycle seams.
- `0x0079e1a0`
  - `WaterRipples` shader and texture setup.

### Particle anchors

- `0x00821100`
  - merged particle emitter batch path (`ParticleBatch`).
- `0x008214e0`
  - direct particle submission path and model-linked particle log.
- `0x0081f330`
  - particle effect-name table init (`Particle`, `Particle_Unlit`, projected variants).
- `0x00979170`
  - `CParticleEmitter2_idx` pool setup.

### Lighting anchors

- `0x008bdfc0`, `0x008bdfd0`, `0x008bdfe0`, `0x008be100`, `0x008be1a0`
  - `LightSkybox.dbc`, `LightParams.dbc`, `Light.dbc`, `LightIntBand.dbc`, `LightFloatBand.dbc` table seams.
- `0x0079e7c0`
  - world-map pool setup includes `WLIGHT` and `WCACHELIGHT` objects.
- `0x004e6c60`, `0x004e6d60`, `0x004e6e60`, `0x004e6be0`
  - debug/script light controls (`AddLight`, `AddCharacterLight`, `AddPetLight`, `ResetLights`).

### LIT status anchors

- positive `.lit`/`.LIT` path anchors: none recovered in this pass
- only `Unlit` effect-mode names recovered (`Particle_Unlit`, `Projected_*_Unlit`)
- practical classification: no positive standalone `.lit` file loader evidence yet
