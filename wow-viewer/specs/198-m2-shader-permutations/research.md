# Phase 0 Research: M2 and WMO Shader Permutation System

**Date**: 2026-09-01
**Source**: `Wow.exe`, MoP Beta 5.0.1.15464, image base `0x00400000`, via GhidraMCP.

## R1 — Shader selection is by integer index into two ordered name tables

**Decision**: Model the permutation as an **index into an ordered table**, not as a parsed
name or a flag set.

**Evidence**: Two consecutive pointer arrays of shader name strings live at `0x00eb4b10`.
The pixel table runs from `0x00eb4b10` for 35 entries, terminated by a NULL word at
`0x00eb4b9c`; the vertex table begins immediately after at `0x00eb4ba0` with
`Diffuse_T1`. Both are dense arrays of `char*`, so the selector is an ordinal.

**Alternatives considered**: name-string lookup (rejected — the names exist only as a
table's contents, nothing in the data references them by text); a flag/bitfield decode
(rejected — the entries are not combinatorial, e.g. `Illum` and `Guild` are not products
of any flag axis).

## R2 — The pixel shader table, in index order

Decoded from the pointer array at `0x00eb4b10` and the string bodies at
`0x00d72864`–`0x00d72bcc`.

| # | Name | # | Name |
|---|---|---|---|
| 0 | `Combiners_Opaque` | 18 | `Combiners_Opaque_Alpha_Alpha` |
| 1 | `Combiners_Mod` | 19 | `Combiners_Opaque_Mod2xNA_Alpha_3s` |
| 2 | `Combiners_Opaque_Mod` | 20 | `Combiners_Opaque_AddAlpha_Wgt` |
| 3 | `Combiners_Opaque_Mod2x` | 21 | `Combiners_Mod_Add_Alpha` |
| 4 | `Combiners_Opaque_Mod2xNA` | 22 | `Combiners_Opaque_ModNA_Alpha` |
| 5 | `Combiners_Opaque_Opaque` | 23 | `Combiners_Mod_AddAlpha_Wgt` |
| 6 | `Combiners_Mod_Mod` | 24 | `Combiners_Opaque_Mod_Add_Wgt` |
| 7 | `Combiners_Mod_Mod2x` | 25 | `Combiners_Opaque_Mod2xNA_Alpha_UnshAlpha` |
| 8 | `Combiners_Mod_Add` | 26 | `Combiners_Mod_Dual_Crossfade` |
| 9 | `Combiners_Mod_Mod2xNA` | 27 | `Combiners_Opaque_Mod2xNA_Alpha_Alpha` |
| 10 | `Combiners_Mod_AddNA` | 28 | `Combiners_Mod_Masked_Dual_Crossfade` |
| 11 | `Combiners_Mod_Opaque` | 29 | `Combiners_Opaque_Alpha` |
| 12 | `Combiners_Opaque_Mod2xNA_Alpha` | 30 | `Guild` |
| 13 | `Combiners_Opaque_AddAlpha` | 31 | `Guild_NoBorder` |
| 14 | `Combiners_Opaque_AddAlpha_Alpha` | 32 | `Guild_Opaque` |
| 15 | `Combiners_Opaque_Mod2xNA_Alpha_Add` | 33 | `Combiners_Mod_Depth` |
| 16 | `Combiners_Mod_AddAlpha` | 34 | `Illum` |
| 17 | `Combiners_Mod_AddAlpha_Alpha` | | |

The vertex table at `0x00eb4ba0` begins `Diffuse_T1`, `Diffuse_Env`, `Diffuse_T1_T2`,
`Diffuse_T1_Env`, … (16 entries; full ordering to be decoded in T003).

## R3 — OPEN: where the per-batch index comes from

**Status**: **Not measured. Must be resolved before any selection code is written.**

The tables prove selection is ordinal. They do not prove what supplies the ordinal. The
plausible source is a field on the M2 skin-profile batch record, but this project has twice
shipped decoders that assigned meaning to an unmeasured field — `MCXH` was invented outright,
and `Ck24Type` was an exponent band read as a type tag. Spec FR-002 exists because of that
history.

**Resolution path** (T002): find the consumer that indexes `0x00eb4b10`, and walk back to the
value it indexes with. `FUN_00577790` references `Diffuse_T1` and is the first candidate,
though its address range suggests a generic shader-load routine rather than the M2 selector.

**Until resolved**: no batch may be assigned a permutation. The registry, the fallback, and
the reporting (FR-003, FR-004) are all implementable without it and are sequenced first, so
the unknown blocks only the selection step.

## R4 — Era gating

**Decision**: Apply permutation selection only where the model data carries the selector
established in R3; every other era continues through the existing single program.

**Rationale**: FR-006, and the constitution's Real-Data Validation principle. 0.5.3 and LK
models predate this table. Our 0.5.3 support is the project's primary lane and must not
regress.

## R5 — Programs are compiled once, keyed by permutation

**Decision**: A registry keyed by (vertex index, pixel index) holding compiled programs, with
lazy compilation on first request and a permanent fallback entry.

**Rationale**: FR-005. Also bounds the work — the corpus decides which of the 16 x 35 pairs
occur, and SC-006 makes the remainder enumerable rather than estimated.

**Alternatives considered**: compiling all pairs at startup (rejected — 560 programs, most
never used, and a startup cost on a renderer that already has a load-budget defect per spec
153); über-shader with runtime branching (rejected — it moves the combiner cost to per-pixel
branching, which is the opposite of the stated CPU goal).

## R6 — Existing infrastructure this builds on

- `M2Renderer` / `MdxRenderer` / `WmoRenderer` all implement `IGpuInstancedModelRenderer`
  or `IGpuInstancedWmoRenderer`, and instanced submission is live from `WorldScene`. Spec 136
  established that path; FR-008 forbids disturbing it.
- Shaders are inline GLSL strings inside the renderer classes today; there is no shader asset
  pipeline, and this spec does not introduce one.
- `M2MaterialPassProfile` already classifies blend/depth state into `Opaque`/`Cutout`/
  `Blended`. That classification governs pipeline state and stays; the permutation governs the
  combiner. They are orthogonal and must not be merged.
