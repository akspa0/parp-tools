# Phase 1 Data Model: Unified M2 Format Version Profiles

**Feature**: 105-format-version-profiles | **Date**: 2026-07-15

Entities and their era-dependent behaviour. Every layout fact cites evidence or is marked provisional (FR-011).

---

## M2EraProfile *(new — `WowViewer.Core/M2/M2EraProfile.cs`)*

The single canonical answer to "what is this M2's layout?" Replaces the inert viewer-side `M2Profile`
and makes explicit what the era readers currently know only implicitly.

| Field | Type | Notes |
|---|---|---|
| `EraId` | enum | `Era100`, `Era1121`, `ThreeX`, `FourX`, `Mdlx` |
| `VersionField` | uint | `0x100` for **both** era-100 and era-1121 — the reason resolution is non-trivial |
| `SequenceStride` | int | `0x44` / `0x6C` / `0x40` / `0x40` |
| `SequenceTimeBase` | enum | `StartEnd` (≤BC) or `Duration` (Wrath+) |
| `TrackAddressing` | enum | `FlatWithRanges` (1.0.0) or `Nested` (Wrath+) |
| `TrackStride` | int | `0x1C` (old, has interp ranges) or `0x14` (Wrath+) |
| `BoneStride` | int | `0x6C` for era-100 |
| `Evidence` | string | Ghidra address, wiki ref, or `PROVISIONAL: <reason>` |

**Validation**: every instance MUST carry non-empty `Evidence`. A provisional fact must be
distinguishable at the point of use, not merely in a comment (FR-011).

**Relationships**: resolved by `M2ModelReaderDispatcher`; consumed by the era readers and (via the
resolved track/sequence types) the sampler.

---

## M2SequenceDefinition *(modify — `WowViewer.Core/M2/M2SequenceDefinition.cs`)*

**Today**: carries `Duration` only. **This is actively wrong for era-100**, where `Duration` is
populated from the field at `+0x04` that is really `start` (research R1).

| Field | Change | Era-100 source | Wrath+ source |
|---|---|---|---|
| `Start` | **NEW** | `start` @ `+0x04` | `0` (constant) |
| `Duration` | meaning preserved | `end@+0x08 − start@+0x04` | `duration` @ `+0x04` |
| `MoveSpeed` | **offset fix (era-100)** | `+0x0C` (was misread from `+0x08`) | `+0x08` |
| `Flags` | **offset fix (era-100)** | `+0x10` (was misread from `+0x0C`) | `+0x0C` |

**Era-100 record layout** — `M2Sequence`, stride `0x44`
*(evidence: Ghidra `FUN_0070f960`; corroborated by wowdev.wiki's documented `≤ BC` layout)*:

```
+0x00 id             u16
+0x02 variationIndex u16
+0x04 start          u32   <-- our reader currently reads this as `duration`
+0x08 end            u32   <-- our reader currently reads this as `moveSpeed` (as a float!)
+0x0C movespeed      f32
+0x10 flags          u32   (bit0 clear => looping)
```

**Invariant**: `Duration > 0` for any sequence that animates. `Start = 0` for every Wrath+ sequence.

**Why `{Start, Duration}` and not `{Start, End}`** (research R2): `Duration` keeps its exact present
meaning in both eras (`end − start` **is** the duration), so no existing consumer changes behaviour,
and the sampler's rule `Start + (elapsed mod Duration)` collapses to today's `elapsed mod Duration`
when `Start = 0`. The 3.x/4.x no-regression guarantee is structural.

**Era-1121 note**: stride `0x6C`, but field offsets are currently the Wrath ones (`moveSpeed@0x08`).
Suspected wrong for the same reason as era-100, but **unverified — no 1.12.1 client traced**. Marked
`PROVISIONAL`; **not** fixed by analogy (R1).

---

## M2TrackDefinition&lt;T&gt; *(modify — `WowViewer.Core/M2/M2AnimationBlocks.cs:24`)*

**Today**: models only the Wrath nested form. `M2TrackSampler.TryReadSequenceSlice` indexes
`TimestampArray`/`ValueArray` as arrays of 8-byte `{count,offset}` refs (`ArrayReferenceSize = 0x08`).
Pointed at a 1.0.0 track, it reads the **first timestamp value** as if it were a `{count,offset}` pair.

| Field | Change | Notes |
|---|---|---|
| `Interpolation` | unchanged | |
| `GlobalSequenceIndex` | unchanged | `0xFFFF` ⇒ none (already normalized to `-1`) |
| `TimestampArray` | meaning now mode-dependent | `Nested`: outer array of refs. `FlatWithRanges`: one flat array. |
| `ValueArray` | meaning now mode-dependent | as above |
| `Addressing` | **NEW** | `Nested` \| `FlatWithRanges`; set by the reader from the profile, never sniffed (FR-002) |
| `InterpolationRanges` | **NEW** | `M2TrackArrayReference`; empty for `Nested`. Absent from the codebase today. |

**Era-100 track layout** — `M2Track`, stride `0x1C` *(evidence: Ghidra `FUN_0070f6d0`)*:

```
+0x00 interpType     u16
+0x02 globalSequence u16   (0xFFFF = none)
+0x04 interpRanges   M2Array  -> 8-byte M2Range { u32 first, u32 last }  INCLUSIVE
+0x0C timestamps     M2Array  -> flat u32[]
+0x14 values         M2Array  -> flat T[]
```

Wrath+ drops `interpRanges` and nests `M2Array<M2Array<T>>` per sequence ⇒ stride `0x1C` → `0x14`.

---

## M2InterpolationRange *(new)*

| Field | Type | Notes |
|---|---|---|
| `First` | uint | first key index — **inclusive** |
| `Last` | uint | last key index — **inclusive** |

**Indexed by sequence index** — verified: the client passes the value it also uses to index the
sequence array (`animId * 0x44`), so `interpRanges[i]` describes `sequences[i]`. Our existing
`sequenceIndex` is the correct value (Ghidra `FUN_0070f960`).

**Validation**: `Last ≤ First` ⇒ single key, no interpolation. Empty range array ⇒ the whole flat
array is the range (FR-003).

---

## Sampling contract *(behaviour, `WowViewer.Core.Runtime/M2/M2TrackSampler.cs`)*

Unified across eras:

```
sampleTime = Start + (elapsed mod Duration)          // Start = 0 for Wrath+ => today's expression

if Addressing == Nested:
    slice = TimestampArray[sequenceIndex]            // today's path, unchanged
else:  // FlatWithRanges
    if InterpolationRanges.Count == 0: [first, last] = [0, timestamps.Count - 1]
    else:                              [first, last] = InterpolationRanges[sequenceIndex]
    if last <= first: return values[first]            // single key
    if globalSequence != none: sampleTime = globalTimer[globalSequence]
    k0 = bracket(sampleTime, timestamps[first..last])
    k1 = k0 + 1
    if timestamps.Count <= k1: return values[k0]      // NB: TOTAL count, not `last` -- FR-004
    t  = (sampleTime - ts[k0]) / (ts[k1] - ts[k0])
    return lerp(values[k0], values[k1], t)            // slerp for quaternions
```

**Two deliberate fidelity choices:**

- **The clamp tests `timestamps.Count`, not `last`** — reproduced from the client, not "fixed"
  (FR-004). When `k0 == last` and more keys follow, the client interpolates across the boundary into
  the next sequence's first key. Unreachable on well-formed data (time is clamped to the sequence
  span; the last key sits at the end, giving `t = 0`). Do not treat a diff here as a bug without a
  reproduction.
- **No key cache.** The client seeds a 3-way search (forward scan / binary search / backward scan)
  from a cached key index passed in `out[0]`. All three branches provably converge on the same key —
  it is purely a performance device. A stateless bracketing search is equivalent (FR-006).

---

## Explicitly not modelled

Real client behaviours, recorded rather than flattened away (spec Assumptions):

- **Per-bone animation state**: each bone carries its own `(time, animIndex)`; `-1` means "inherit the
  parent bone's". Our runtime is one-sequence-per-model.
- **Cross-animation blending**: a second `(time, animIndex)` pair drives a second sampler call whose
  result is blended with the first.

Neither blocks the P1 slice. Both are why a future spec may revisit this contract.
