# Phase 0 Research: MCAL Alpha Map Decode Correctness

**Date**: 2026-09-01

## R1 — Four decoders exist, and they disagree

Measured by searching `src/` and `tools/` for alpha decode routines on 2026-09-01.

| # | Implementation | Selection method |
|---|---|---|
| 1 | `Core.IO/Lk/Mcal.cs` — `GetAlphaMapForLayer`, `GetAlphaMapForLayerRelaxed` | compressed flag, then `bigAlphaDefault`, then **span inference** from the next layer's offset (`>= 4096` → big, `>= 2048` → 4-bit, else compressed) |
| 2 | `Core.IO/Maps/AdtMcalDecoder.cs` | own `ReadCompressedAlpha` with explicit `maxLength` |
| 3 | `viewer/Terrain/StandardTerrainAdapter.cs` — `DecodeLayerBySpan`, plus a second fallback loop, plus `SynthesizeCataclysm400ResidualAlpha` | span-based, then fabrication |
| 4 | `viewer/Terrain/Vlm/AlphaMapService.cs` — `ReadBigAlpha` | "ported from Warcraft.NET" |

**Decision**: consolidate to one owner in `WowViewer.Core.IO`. Constitution II requires it
("one canonical owner per format surface"), and it is a precondition for measuring anything —
four decoders means "our alpha" has four different meanings.

## R2 — Harvest and renderer already disagree, by construction

`VlmDatasetExporter.cs:1908` calls `GetAlphaMapForLayer(layer, false)` — big alpha hardcoded
**false**. `StandardTerrainAdapter.cs:91` derives it from the WDT: `_useBigAlpha =
(_mphdFlags & _adtProfile.BigAlphaFlagsMask) != 0`.

On any map whose `MPHD` sets `0x4` or `0x80`, the harvested alpha and the rendered alpha are
decoded by different rules from the same bytes. This is the operator's stated concern and it
is confirmed, not hypothetical.

## R3 — The fabrication, and why it looks the way it does

`SynthesizeCataclysm400ResidualAlpha` fills any undecoded layer 1-3 with
`clamp(255 - Σ other layers, 0, 255)` over a full 64x64. Where the other layers are near
zero the value stays 255, producing a **fully opaque chunk-sized block**. It is gated on
`useBigAlpha`, which is why the artifact appears on Cata/MoP maps and never on 0.5.3.
`StitchCataclysm400ChunkEdges` then blends those blocks into neighbours, spreading invented
data across seams.

**Decision**: delete it (FR-002), but **after** US1/US2. Removing it first would trade wrong
terrain for missing terrain and destroy the only current signal that decode is failing.

## R4 — Span inference is the design flaw, not a bug in it

Decoders 1 and 3 choose the encoding by measuring the gap to the next layer's offset. This
cannot be correct in general: the gap is a consequence of the encoding, not evidence for it.
A compressed layer that happens to occupy 2100 bytes is indistinguishable from a 4-bit layer
by span alone, and the last layer has no following offset at all.

**Decision**: rule selection comes from era profile + format flags (FR-004). The format
already carries the needed signals — `MCLY` compression flag, `MPHD` big-alpha bits, MCNK
`doNotFixAlphaMap` (`0x8000`).

**Alternative considered**: keep span inference as a tiebreak. Rejected — it is exactly the
"plausible fallback that appears to succeed" FR-005 forbids, and it is how the current state
became unmeasurable.

## R5 — OPEN: the native decoder is not in the chunk or render-state code

**Status**: still not isolated. Second search pass 2026-09-01. Recording what is now
**ruled out**, so this is not re-searched a third time.

### Ruled out

| Target | What it actually is |
|---|---|
| `+0x124` instruction search | 30+ unrelated hits program-wide. Dead end, twice. Do not retry. |
| `FUN_00ba36d0` (MapChunk) | per-layer **texture readiness** check — iterates `+0x94` count over the `+0xa0` MCLY array, 16-byte stride |
| `FUN_00ba80f0` (MapChunk) | `CMapChunk` **destructor** |
| `FUN_00badab0` (RenderChunkState) | state **destructor** — frees two per-layer arrays at `+0x64`/`+0x74` and `+0x84` |
| `FUN_00bad4c0` (RenderChunkState) | terrain **draw loop** |
| `FUN_00bacdc0` (RenderChunkState) | projected/decal batch renderer |
| `FUN_00babb70` (RenderChunkState) | terrain **shader selector** — see R7 |
| `FUN_00ba9a00` (RenderChunkState) | builds `+0x84`, which is **`m_envTexture`**, not alpha |
| `+0x84` write search | `m_envTexture`, confirmed by the assert at `MapRenderChunkState.cpp:0x10b` |

### Current hypothesis

The render-chunk state holds **texture handles**, not decoded alpha. `FUN_00bb2030`
(`MapTexture.cpp`) takes a name plus a flags block and returns a handle. The per-chunk alpha
is most likely created as a **procedural/callback-filled texture** — which would explain why
no chunk-side or render-state-side code dereferences the MCAL pointer at `+0x124` at all: the
decode happens inside a texture fill callback, reached indirectly through a function pointer.

**Next search** (if T101 is attempted again): find the texture-creation call that passes a
fill callback rather than a filename, and follow the callback. `FUN_00bb1850` and
`FUN_00bb1a50`/`FUN_00bb1b60` in `MapTexture.cpp` are the entry points. Budget this
deliberately — two passes have now failed, and R5's fallback exists precisely so the spec is
not blocked on it.

**Consequence for the spec**: US2 proceeds on the file-side proof described below. The rule is
backed by exact-payload accounting across the corpus (SC-002) with every exception enumerated
(SC-006), and `AdtAlphaDecodeRule.Justification` records that provenance honestly rather than
implying the client was read.

## R7 — Terrain shader selection (measured, belongs to a future terrain-shader spec)

Not required by this spec, but measured while searching and worth not re-deriving.

`FUN_00babb70` (`MapRenderChunkState.cpp:0x3a5`) computes the terrain shader index:

```
index = capBits
      + ((hasCap1) + ((envTexture != 0) + shadowFlag*2
        + (renderMode*2 + (layers-1 + param4*4)*6)*2)*2)*2
```

Asserted constraints: `nTextures <= ShaderConstants::MAPCHUNK_MAX_TEXTURES` where
**MAPCHUNK_MAX_TEXTURES = 4**, and `layers < 4` (asserted again at `0x167`, `0x17b`, and in
the draw loop at `0x1d0`). The draw loop `FUN_00bad4c0` uploads `layers - 1` as a shader
constant.

Render-chunk-state layout established along the way:

| Offset | Meaning |
|---|---|
| `+0x24` | per-layer flags array, uint stride 4; bit `0x80` and bit `0x400` are both tested |
| `+0x38` | chunk render mode, values 0-6 |
| `+0x39` | layer count |
| `+0x64`, `+0x74` | two per-layer resource arrays |
| `+0x84` | `m_envTexture` — from a DBC row via `+0x34`, or from a layer with flag `0x400` |

## R6 — Existing machinery to build on, not replace

- `AdtMcalDecodeProfile` (`LegacySequential`, `LichKingStrict`, `Cataclysm400`) already
  exists as the era axis.
- `AdtFormatProfile.BigAlphaFlagsMask` (`0x4 | 0x80`) already encodes the `MPHD` bits per era.
- `AdtMcalAlphaEncoding` (`BigAlpha`, `BigAlphaFixed`, …) and `AdtMcalSummary` already exist
  as reporting types and are close to what FR-003 needs.

This spec establishes what these profiles should *contain* and makes one decoder consult
them. It does not introduce a parallel mechanism.
