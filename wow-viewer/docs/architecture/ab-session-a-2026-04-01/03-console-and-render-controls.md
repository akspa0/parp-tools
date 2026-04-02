# Console and Render Controls (Discovered)

## Console API anchors

String anchors indicate active console plumbing:

- ConsoleExec
- ConsoleAddMessage
- closeconsole
- consolelines

## World/terrain render command registration

From decompiled registration flow (FUN_00780f50 and FUN_0078e400):

- showDetailDoodads
- maxLOD
- showCull
- setShadow
- waterRipples
- waterParticulates
- showShadow
- showLowDetail
- showSimpleDoodads
- detailDoodadAlpha
- characterAmbient

Related cvars discovered:

- terrainAlphaBitDepth
- farclip
- nearclip
- objectFade
- objectFadeZFill
- mapShadows
- extShadowQuality
- specular
- mapObjLightLOD
- particleDensity
- waterLOD

## Notable behavior snippets

- FUN_0078d610 toggles terrain LOD and logs:
  - Terrain LOD enabled.
  - Terrain LOD disabled.
- FUN_0078da50 validates terrainAlphaBitDepth and only accepts 4 or 8.
- FUN_0077f700 (detailDoodadAlpha command path) clamps alpha reference to 0..255.

## M2 runtime controls

From M2 registration anchor (0x00402760):

- M2UseZFill
- M2UseClipPlanes
- M2UseThreads
- M2BatchDoodads
- M2BatchParticles
- M2ForceAdditiveParticleSort
- M2Faster
- M2FasterDebug

These are high-value knobs for isolating invisibility caused by pass order/batching/sorting state.

Registration nuance from disassembly (`0x00402760`):

- `M2UseZFill`, `M2UseClipPlanes`, and `M2UseThreads` are registered with null callback pointers.
- `M2BatchDoodads`, `M2BatchParticles`, `M2ForceAdditiveParticleSort`, `M2Faster`, and `M2FasterDebug` register non-null callbacks.
- practical implication: the first three options look like runtime toggles but may behave startup-only unless another path re-applies them.

## Win32 runtime-flag word mapping (confirmed)

Shared runtime flag storage and helpers:

- global word: `DAT_00d3fcf4`
- getter: `FUN_0081c0b0`
- setter: `FUN_0081c0c0`
- high-bit OR helper: `FUN_0081c060` (applies `param & 0xe000`)

Callback writes confirmed from decompilation:

- `FUN_00402410` (`M2BatchDoodads`)
  - sets or clears bit `0x20`
  - emits `Doodad batching enabled/disabled.`
- `FUN_00402470` (`M2BatchParticles`)
  - sets or clears bit `0x80`
  - emits `Particle batching enabled/disabled.`
- `FUN_004024d0` (`M2ForceAdditiveParticleSort`)
  - sets or clears bit `0x100`
  - emits `Sorting all particles as though they were additive.` or `Sorting particles normally.`

Initial registration fold from `M2_RegisterRuntimeFlags` (`0x00402760`):

- bit `0x1`: `M2UseZFill`
- bit `0x2`: `M2UseClipPlanes`
- bit `0x4`: `M2UseThreads`
- bit `0x20`: `M2BatchDoodads`
- bit `0x80`: `M2BatchParticles`
- bit `0x100`: `M2ForceAdditiveParticleSort`
- bit `0x8`: forced on by return expression (`uVar1 | 8`)

## Hidden and likely-dead branches

`M2Faster`/`M2FasterDebug` path (`FUN_004021c0`, `FUN_00402210`, `FUN_00402100`):

- mode parser maps to high optimization masks:
  - mode `1` -> `0xe000`
  - mode `2` or `3` -> `0x2000` baseline
  - special decimal parser branch can produce `0x6000` and `0xe000`
- a non-obvious gate in `FUN_00402100` can zero the requested mode when an internal capability check fails (`FUN_0047d230(2)`).

Likely dead fallback in normal startup path:

- in `FUN_0081c0d0`, bit `0x40` fallback is only applied if `(flags & 0x8) == 0`.
- startup flow at `0x004048b8` uses `M2_RegisterRuntimeFlags`, whose return always includes `0x8`.
- consequence: this fallback branch appears unreachable in normal startup-driven init unless flags are injected differently by another path.
- downstream impact: `FUN_00824550` tests `(flags & 0x40)` in a combinable-doodad gate; if `0x40` is never set in startup flow, that branch degenerates to the non-`0x40` behavior by default.

## Hidden prewarm chain (non-primary load path)

`M2_NormalizeModelPathAndProbeSkins` (`0x0053c430`) is called in repeated warm-up patterns from:

- `FUN_0053c520`
- `FUN_0053e810`
- `FUN_0053e930`
- `FUN_0053eaa0`

Observed behavior includes repeated probe calls and explicit `00.skin` through `03.skin` checks.
This appears to be aggressive prewarm or cache-probe behavior distinct from the strict choose-load-init path used for active runtime model ownership.

Known trigger context recovered in this pass:

- `FUN_006e7d60` and `FUN_006e7e00` call into the prewarm chain through `FUN_0053eaa0` during player-object update flows.
- practical implication: this probe path is not just dead utility code; it can still run during normal client object-update behavior, but it is still separate from strict skin ownership semantics used by the primary model runtime.

## Lifecycle diagnostics still present

- `Model2: M2Initialize called more than once`
  - emitted from `FUN_0081c0d0` guard path.
- `Model2: M2Destroy never called`
  - emitted from `FUN_0081c870` teardown guard path.

These are useful anchors for detecting hidden repeated-init or teardown-order bugs during longer runtime capture sessions.

## Secondary M2 render path: portrait pipeline

The Win32 pass also surfaced a distinct portrait-focused M2 path in `FUN_00619580`:

- emits `BEGIN SetPortraitTexture`
- acquires M2 runtime state via `FUN_0081c080`
- resolves model through M2 cache calls before rendering to a portrait texture target
- uses dedicated setup and callback wiring (`FUN_00616bc0`) separate from world-scene submission

Implication for parity work:

- do not treat portrait capture behavior as direct evidence for world doodad path correctness
- if portrait and world diverge, compare them as two pipelines sharing model state but not identical submission/state setup

## Hard load rejection and hidden cache flags

`FUN_0081c390` (Win32 `CM2Model` cache-open path) shows strict rejection gates still in place:

- invalid extension path logs:
  - `Model2: Invalid file extension: %s`
- not-found path logs:
  - `Model2: File not found: %s`
- canonicalization still rewrites `.mdl` and `.mdx` to `.m2` before open.

Hidden flag behavior visible in this path:

- `param_2 & 0x10` changes hash keying behavior (basename-focused vs broader path handling)
- `param_2 & 0x8` controls whether the loaded model is linked into the normal cache list
- `param_2 & 0x40` sets an extra model-state bit after load

Implication for parity work:

- loader-flag mismatches can produce cache or lookup behavior differences even when parser logic appears equivalent.

## Subsystem deep-dive anchors (Apr 02 continuation)

### Rendering and shader effect ownership

- `FUN_00780f50` explicitly reloads:
  - `MapObj.wfx`
  - `MapObjU.wfx`
  - `Model2.wfx`
  - `Particle.wfx`
  - `ShadowMap.wfx`
- `FUN_00876d90` loads `Shaders\Effects\%s` and routes into `ShaderEffectManager.cpp` parse/bind work.
- `FUN_00876be0` allocates or reuses effect records by name hash.
- `FUN_00872d30` binds vertex/pixel shader names per effect.
- `FUN_008728c0` writes combiner argument tables for the active effect.
- `FUN_0068a9a0` + `FUN_00684c40` log shader capabilities and selected targets (`pixelShaderTarget`, `vertexShaderTarget`).
- `FUN_0078de60` confirms `specular` is pixel-shader gated (`Specular not enabled.  Requires pixel shaders.`).

### Liquid runtime ownership

- `FUN_008a3e00` -> `vsLiquidProcWater%s` + `psLiquidProcWater%s`
- `FUN_008a3f70` -> `vsLiquidWater` + `psLiquidWater`
- `FUN_008a4070` -> `vsLiquidWaterNoSpec` + `psLiquidWaterNoSpec`
- `FUN_008a4190` -> `vsLiquidMagma` + `psLiquidMagma`
- `FUN_008a1fa0` and `FUN_008a28f0` perform DBC-driven liquid material/settings lookup with water fallback.
- `FUN_00793d20` shows WMO liquid type fallback handling and binds material/settings/runtime state.
- `FUN_007cefd0` / `FUN_007cf790` are dedicated `MapChunkLiquid.cpp` ownership for chunk-liquid buffers/instances.
- `FUN_0079e1a0` builds `WaterRipples` shader path; `FUN_0079d460` gates ripple emission through runtime toggles.

### Particle runtime ownership

- `FUN_00821100` is the merged emitter batch path (`ParticleBatch`) with strict state compatibility and buffer-cap checks.
- `FUN_008214e0` handles direct submission path and logs `Particle: model=%s` when binding model-driven particle state.
- `FUN_0081f330` initializes effect handles:
  - `Particle`
  - `Particle_Unlit`
  - `Projected_ModMod`
  - `Projected_ModMod_Unlit`
  - `Projected_ModAdd`
  - `Projected_ModAdd_Unlit`
- `FUN_00979170` manages `CParticleEmitter2_idx` allocation pools.
- `FUN_0078d860` enforces `particleDensity` bounds (`0.1..1.0`).

### Lighting ownership

- DBC seams are explicit:
  - `FUN_008bdfc0` -> `LightSkybox.dbc`
  - `FUN_008bdfd0` -> `LightParams.dbc`
  - `FUN_008bdfe0` -> `Light.dbc`
  - `FUN_008be100` -> `LightIntBand.dbc`
  - `FUN_008be1a0` -> `LightFloatBand.dbc`
- `FUN_0079e7c0` allocates `WLIGHT` and `WCACHELIGHT` world-map light object pools.
- debug/script light command path remains live:
  - `FUN_004e6c60` `AddLight`
  - `FUN_004e6d60` `AddCharacterLight`
  - `FUN_004e6e60` `AddPetLight`
  - `FUN_004e6be0` `ResetLights`
  - `FUN_00960d20` `SetLight`
  - `FUN_00960dd0` `GetLight`
- `FUN_0078ded0` enforces `mapObjLightLOD` range `0..2`.

### LIT file-support status (current evidence)

- no positive `.lit` or `.LIT` path string found in this Win32 pass
- no `%s.lit` formatter seam recovered
- `Unlit` appears as effect mode naming (`Particle_Unlit`, `Projected_*_Unlit`), not as a standalone file loader
- active light data path in this pass is DBC + shader/effect routing, not direct `.lit` file ingestion

Current reading:

- treat standalone `.lit` file support as unconfirmed and currently unsupported by recovered evidence
- do not add a speculative `.lit` loader seam to `wow-viewer` until a positive native anchor is recovered
