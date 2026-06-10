# Research: Renderer Improvements Convergence

## Decision: Create a new convergence owner instead of mutating specs 030-032 into one of the old feature packs

**Rationale**:
- Specs 030, 031, and 032 each capture valid renderer slices, but they do not share one proof owner.
- Mutating one of them into the new owner would blur what is historical source material versus what is now the active roadmap.
- A new convergence feature keeps provenance intact and makes future routing explicit.

**Alternatives considered**:
- Update spec 032 to absorb 030 and 031: rejected because it would hide the original terrain and WMO slice boundaries.
- Keep all three plans independent: rejected because that preserves the current overlap and drift problem.

## Decision: Keep specs 030-032 as source-slice references and point them at spec 036

**Rationale**:
- Those feature packs contain detailed Ghidra-derived constraints and subsystem notes that are still useful.
- The repo needs a visible handoff note so future work opens the convergence plan first.

**Alternatives considered**:
- Delete or archive 030-032 immediately: rejected because the details are still implementation inputs.

## Decision: Keep M2 recovery work outside this convergence plan

**Rationale**:
- Spec 035 is already the active owner for current M2 route and parity recovery.
- Convergence of 030-032 is about terrain, WMO, lighting, sky/fog, liquid, and viewer host sequencing.
- Pulling M2 parity fully into this feature would break the “one phase at a time” guardrail and blur proof surfaces.

**Alternatives considered**:
- Fold spec 035 into 036 immediately: rejected because it would expand the scope too far and remove a currently useful regression-recovery lane.

## Decision: Sequence lighting and fog foundations before terrain/WMO pipeline convergence

**Rationale**:
- Terrain shading, WMO interior fog, water tint, sky clear color, and shadow behavior all depend on a stable lighting-state source.
- Building geometry or pass orchestration first would force later rewrites when the lighting source changes.

**Alternatives considered**:
- Start with terrain topology first: rejected because several downstream visual behaviors depend on shared lighting/fog contracts.
- Start with WMO pass architecture first: rejected because interior fog and exterior lighting ownership would still be unsettled.

## Decision: Add a 3.3.5 runtime-controls evidence lane from Ghidra before deeper parity slicing

**Rationale**:
- New Ghidra extraction from staged `wow.exe` `3.3.5.12340` exposed concrete runtime control surfaces that are currently under-modeled in spec 036:
  - `M2_RegisterRuntimeFlags` (`0x00402760`) with toggles for `M2UseZFill`, `M2UseClipPlanes`, `M2UseThreads`, `M2BatchDoodads`, `M2BatchParticles`, and `M2ForceAdditiveParticleSort`.
  - Terrain/video option registration (`0x0078e400`) and handlers for `terrainLOD` (`0x0078d610`), `mapObjLightLOD` (`0x0078ded0`), `terrainAlphaBitDepth` (`0x0078da50`), `MaxLights` (`0x0078d6b0`), `projectedTextures` (`0x0078dcf0`), and `waterLOD` (`0x0078d8b0`).
  - Liquid shader family loading showing distinct magma and water paths (`vsLiquidMagma`/`psLiquidMagma`, plus `psLiquidWater`, `psLiquidWaterNoSpec`, `psLiquidProcWater*`).
  - Fog override parsing from frame/environment config (`fogNear`, `fogFar`, `FogColor`) in `0x0095f800`.
- Convergence planning without these runtime controls leaves missing dependency gates for validation and can cause false parity conclusions.

**Alternatives considered**:
- Keep spec 036 purely high-level and defer runtime controls to implementation: rejected because the missing controls are phase-ordering inputs, not optional implementation detail.
- Move all control-surface tracking into spec 035: rejected because most extracted controls are terrain/WMO/liquid/lighting convergence concerns rather than M2-only parity concerns.

## Decision: Keep M2 in spec 036 as a bounded dependency surface, not a parity-owner transfer

**Rationale**:
- `M2_ChooseAndLoadSkinProfile` (`0x0083cc80`) and `M2_InitializeSkinProfileAndRebuildInstances` (`0x00838490`) show runtime initialization/failure gates that affect world-scene confidence and diagnostics.
- These findings should inform renderer convergence phase gating (especially telemetry and diagnostics), while full M2 parity and behavior recovery remain owned by spec 035.

**Alternatives considered**:
- Exclude M2 entirely from spec 036: rejected because world-render diagnostics would miss key runtime-control interactions.
- Absorb full M2 parity into spec 036: rejected because this violates one-phase-at-a-time scope control and existing ownership boundaries.

## Decision: Add a telemetry-first validation contract for runtime controls

**Rationale**:
- Current phase guidance is mostly screenshot-comparison oriented.
- Ghidra evidence now supports deterministic runtime telemetry checkpoints, including active values for terrain LOD state, map-object light LOD, alpha-bit depth, projected textures, water/material path, fog parameters, and M2 optimization flags.
- Telemetry-first checkpoints reduce ambiguity before visual parity sweeps.

**Alternatives considered**:
- Keep only visual proof checkpoints: rejected because visual-only deltas are hard to root-cause when multiple runtime toggles are drifting.

## Cross-Reference: spec 038 — 3.0.1 Renderer Performance Research Slice

`specs/038-m2-301-renderer-perf-research/spec.md` + `research.md` extend this convergence plan with deeper **3.0.1.8303** Ghidra evidence. Where this plan focuses on 3.3.5 runtime controls (lines 43-51 above), spec 038 documents the same control family in 3.0.1 with explicit function addresses, default values, and callback behavior:

- The 3.0.1 graphics-options registry (`FUN_006ee8e0`) has **25 cvars** vs the 3.3.5 list — the extras are `groundEffectDensity`, `groundEffectDist`, `objectFade`, `objectFadeZFill`, `horizonfarclip`, `footstepBias`, `bspcache`, `worldPoolUsage`. The 3.0.1-only cvars should be gated behind `M2BuildProfile == Build301` (spec 037 owns the `M2BuildProfile` enum).
- The 3.0.1 `waterLOD` cvar (`FUN_006edfb0`) is **frozen at 0** in 3.0.1.8303. Any 3.0.1 staging must keep `waterLOD == 0`.
- The 3.0.1 per-batch alpha-cull algorithm (`FUN_00788fb0` + `FUN_00789440`) is **not** mentioned in this plan's 3.3.5 inventory. It is the recommended first implementation slice in spec 038 — adds an `M2BuildCullPolicy` service that consults `model_alpha × transparency × color` against a cull threshold constant for every M2 batch.
- The 3.0.1 3-tier SmallCull (`FUN_006edb30` + `FUN_006f2a00`) and DistCull (`FUN_006ee4a0`) are confirmed in 3.0.1 with explicit precompute tables; 3.3.5 evidence (this plan, line 48) confirms the cvars exist but the algorithm is not decompiled there. Spec 038's recommended first slice takes both.
- The 3.0.1 master render flag word `DAT_00edfae0` (init `0x7104b73`) is the cleanest single-source-of-truth state model in any build recovered so far. Spec 038 recommends mirroring it as a follow-on slice.

When implementing Phase 1 (Lighting Foundation) or any later phase that touches cull/lighting, the spec 038 research pack should be read first to avoid re-decompiling the same 3.0.1 functions.
