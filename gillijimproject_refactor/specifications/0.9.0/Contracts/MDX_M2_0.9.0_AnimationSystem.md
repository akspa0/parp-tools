# MDX / M2 Animation System — 0.9.0 Deep Dive

## Scope
- Document the animation data contract for 0.9.0 MDX/M2: sequence headers (SEQS) and per-track keyframe chunks used by geosets, materials, emitters, etc.
- Capture loader strictness (size/overrun guards) and field meanings to realign tooling where current animation handling is known to be incorrect.

## High-level model
- Animation is sequence-driven: `SEQS` defines named time ranges and metadata; individual sections (GEOA, MTLS layers, PRE2 tracks, ribbons, lights, etc.) carry per-property keyframe subchunks (K* tags) that reference those sequences.
- No evidence of MD20-specific animation paths in 0.9.0; animations are MDX/MDL style keyframe blocks embedded under each section.

## SEQS chunk (FUN_007a9c70)
- Stride check: `(sectionSize - 4) == count * 0x8C`; otherwise fatal "Invalid SEQX section detected".
- Count: first u32 of the chunk.
- Record size: 0x8C bytes per sequence. Parsed with hard bounds checks per field.
- Field map (offsets within each 0x8C record):
  - +0x00: `animId` (u32) — primary sequence id.
  - +0x04: `subId` (u32) — secondary id/index.
  - +0x08..0x0F: zeroed.
  - +0x50: `startTime` (u32).
  - +0x54: `endTime` (u32).
  - +0x58: `moveSpeed` (float).
  - +0x5C: `flags` (u32) — sequence flags (not yet mapped; keep logged).
  - +0x60..0x6B: bbox min (3 * float).
  - +0x6C..0x77: bbox max (3 * float).
  - +0x78: `blendTime` (float) — cross-fade duration.
  - +0x7C: `playbackSpeed` (u32) — likely frame rate / rate divisor.
  - +0x80: `frequency` (u32) — likely loop frequency or rarity.
  - +0x84: `pad/unk` (u32).
  - +0x88: `pad/unk` (u32).
- Allocation: resizes sequence array at +0x3b8 with capacity tracked at +0x3b4; older entries freed/zeroed if count shrinks.
- Strictness: each field read via guarded helpers (`FUN_007f40a0`, `FUN_007f4060`, `FUN_007f41a0`, `FUN_007f4470`), aborting on buffer underrun.

## Generic keyframe readers (helpers)
These helpers are used across track subchunks (e.g., KMTA/KMTF in MTLS, KPLN/KP2G/KP2E in PRE2, GEOA color/alpha tracks):
- `FUN_007f40a0` (u32 read) — advances cursor by 4; underrun fatal.
- `FUN_007f4060` (u32 read) — identical guard semantics, separate callsite id.
- `FUN_007f41a0` (float read) — reads a 32-bit float with underrun guard.
- `FUN_007f4470(dst, count)` — bulk-copy `count` u32s (often used for float3/float4 payloads) with length validation before advancing cursor.

## Track subchunk pattern (observed)
- Subchunk tags vary by property (e.g., `KMTA` alpha, `KMTF` tex frame, geoset anim color/alpha, PRE2 `KPLN`/`KP2G`/`KP2E`).
- Common layout per subchunk:
  1) u32 `count` (number of keys).
  2) For each key: u32 `time`, then payload of 1–3 floats (or float3) depending on property; read via `FUN_007f41a0` / `FUN_007f4470`.
  3) Bounds checks per subchunk: if `count * stride` exceeds remaining bytes, section reader raises a fatal error (e.g., "TexLayer" or "ParticleEmitter2").
- Interpolation hints: No explicit interpolation enum was surfaced in these handlers; behavior likely follows classic MDX scheme (linear/Bezier/step) determined by property-specific flags. Keep logging unknown flag bits and avoid assuming Hermite unless verified.

## Known property tracks (examples)
- MTLS layer (`FUN_007a7740`):
  - `KMTA`: alpha keys — time + 1..3 floats; count stored in layer fields (+0x10/+0x11) with pointer at +0x0E.
  - `KMTF`: texture frame/transform — time + 1..3 floats; count stored at +0x05 with pointer at +0x06.
- PRE2 emitter (`FUN_00795870`):
  - `KPLN`: longitude keys — time + float3.
  - `KP2G`: gravity keys — time + float3.
  - `KP2E`: emission rate keys — time + float3.
- GEOA (geoset anim): color/alpha tracks not yet fully mapped, but expect the same time + float/float3 packing with the guarded readers above.

## Loader strictness to replicate
- All track readers use guarded cursor helpers; any overrun triggers a fatal callback via the section-specific error string. Do not allow soft failures.
- Section-level size checks (e.g., SEQS stride, per-subchunk remaining-bytes checks) must be mirrored to avoid silent truncation.

## Implementation guidance (fixing current tooling)
1) Rebuild SEQS parsing exactly: enforce `(size-4) == count*0x8C`; zero/resize arrays when counts shrink; reject underruns.
2) Standardize a shared `AnimStream` helper matching `FUN_007f40a0/60/41a0/4470`: cursor + length, guarded reads, fatal on underrun.
3) For each property track subchunk, require `count` keys and validate `remaining >= count*stride`. Keep stride definitions per property (alpha: 1 float, color: 3 floats, tex frame: 1 float, vector tracks: 3 floats).
4) Preserve sequence metadata fields (ids, start/end, bbox, blendTime, playbackSpeed/frequency) and feed them to the runtime animation state machine; stop inferring fields from later expansions.
5) Log and mask unknown flags in sequence records and per-track structures; do not repurpose them until renderer consumption is confirmed.

## Open questions (to close later)
- Exact interpolation mode selection for each track type (linear vs Hermite vs step) is not explicit in the recovered handlers; needs renderer-side xrefs. Hypothesis: classic MDX property flags drive interpolation (1 = none/step, 2 = linear, 4 = Hermite), but this must be confirmed against render-time sampling.
- SEQS `flags`, `playbackSpeed`, and `frequency` fields are not consumed in the loaders; must confirm renderer usage. Hypothesis from WC3-era layouts: `flags & 0x1` → non-looping, `frequency`/`playbackSpeed` pair → rarity/selection weighting. Keep logging and avoid behavioral coupling until verified.
- GEOA track payloads (color/alpha) and any global-sequence semantics were not fully recovered; expect classic MDX global-sequence behavior but verify. No dedicated global-sequence chunk was seen in 0.9.0, so any global timing would have to be embedded per-track.

## Verification plan (renderer xrefs to close the gaps)
- Locate render-time sampling functions that read the sequence table at +0x3b8; track uses of offsets +0x5C (flags), +0x7C (playbackSpeed), +0x80 (frequency) to confirm meaning. Expect branches choosing loop vs non-loop and rarity selection.
- Find interpolation dispatch by scanning for tables keyed by track header flags in K* subchunk consumers (e.g., MTLS `KMTA/KMTF`, PRE2 `KPLN/KP2G/KP2E`, GEOA color/alpha). Confirm if bitmasks map to step/linear/Hermite.
- GEOA: decompile geoset animation handler to recover color/alpha track layouts and check for global-sequence id usage. If absent, conclude only per-sequence tracks exist in 0.9.0.
- Add assertions in tooling: log any nonzero SEQS `flags`, non-default `frequency`/`playbackSpeed`, and unknown interpolation bits; fail-fast on out-of-range interpolation ids once mapped.
