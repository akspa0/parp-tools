# MDX Animation System — Build 0.5.3.3368

## Scope
This document explains how animation works in the 0.5.3 MDX pipeline: sequences, keyframe tracks, interpolation, and runtime evaluation behavior.

## Evidence Base
- `BuildModelFromMdxData` @ `0x00421fb0` (loading order and animation gate flags)
- Section/type symbols seen in 0.5.3 binary:
  - `MDLSEQUENCESSECTION`
  - `MDLGLOBALSEQSECTION`
  - `MDLBONESECTION`
  - `MDLTEXANIMSECTION`
  - `MDLCAMERASECTION`
  - `MDLEVENTSECTION`
- 0.7.0 parser corroboration for chunk roles:
  - `SEQS` used for sequence timing metadata
  - `BONE` used for transform animation tracks
  - `TXAN` used for texture animation entries

## High-Level Model
MDX animation is a layered system:

1. **Sequence layer**
   - Defines timeline windows (start/end ms), playback flags, and per-sequence metadata.
2. **Track layer**
   - Stores typed keyframes (`translation`, `rotation`, `scale`, `alpha`, etc.).
3. **Object layer**
   - Bones, attachments, cameras, geoset anims, texture anims, emitters, and other nodes each own one or more tracks.
4. **Evaluation layer**
   - Runtime chooses effective time, samples each track, builds transforms/parameters, and applies hierarchy.

## Loader Integration (0.5.3)
Animation-related stages inside `BuildModelFromMdxData`:

1. Load model/global properties.
2. Load textures/materials/geosets/attachments.
3. If `flags & 0x100` is **not** set:
   - load animation (`BONE` + sequence data)
   - load ribbon emitters
4. Load emitter2, matrices, optional hit-test/lights, extents, pivots, cameras.

### Important flags
- `0x100` — disable animation setup path (`no_anim`).
- `0x20` — complex-model/hit-test path toggle (context-dependent in loader).
- `0x200` — skip lights.

## Sequence System

## Role
`SEQS`/`MDLSEQUENCESSECTION` provides the animation timeline catalog.

Each sequence conceptually includes:
- sequence name
- `startTimeMs`
- `endTimeMs`
- playback behavior flags (looping/non-looping semantics)
- optional movement/sync/extents metadata

### Runtime meaning
- Sequence duration is `endTimeMs - startTimeMs`.
- Track keys are resolved against sequence time unless the track is bound to a global sequence.

## Global Sequences

`MDLGLOBALSEQSECTION` indicates time domains that are independent of currently active sequence.

When a track references a global sequence ID:
- it evaluates against global clock modulo global duration,
- not against local sequence start/end.

This is typically used for always-running effects (idle UV panners, continuous glows, etc.).

## Key Track Structure

The observed MDX family uses a common key-track pattern.

Conceptual layout:

```c
struct MdlKeyTrack<T> {
    uint32 keyCount;
    uint32 interpolationType;   // none/linear/hermite/bezier
    uint32 globalSequenceId;    // 0xFFFFFFFF => local sequence time
    Key<T> keys[keyCount];
};

struct Key<T> {
    uint32 timeMs;
    T value;
    // Tangents are present for spline modes
    T inTan;   // optional
    T outTan;  // optional
};
```

### Common `T` types
- `vec3` for translation and scale
- `quat` for rotation
- `float` for alpha/intensity/emission-like scalar properties
- `vec3` for color-style tracks
- `uint32`/`int32` for visibility/toggle-like tracks (format-dependent)

## Interpolation Modes

Typical mode mapping in MDX-family formats:
- `0` = none/step
- `1` = linear
- `2` = Hermite spline
- `3` = Bezier spline

### Sampling behavior
- **Step**: hold previous key value.
- **Linear**: blend directly between surrounding keys.
- **Hermite/Bezier**: use key tangents for curved interpolation.

For quaternions, rotation interpolation should use normalized quaternion interpolation semantics (`nlerp`/`slerp` family) rather than raw component lerp in production-quality decoders.

## Bone Animation (Skeleton)

`BONE` / `MDLBONESECTION` drives articulated animation.

Each bone conceptually has:
- node identity + parent index
- flags/inheritance controls
- translation track
- rotation track
- scale track

### Hierarchical solve
Per frame:
1. Evaluate local TRS for each bone at effective time.
2. Convert local TRS to matrix.
3. Combine with parent matrix unless blocked by inheritance flags.
4. Use resulting bone matrices for skinned vertex transforms and attached nodes.

## Other Animated Domains

Animation tracks are not only for bones:

- **Texture animation** (`TXAN` / `MDLTEXANIMSECTION`): UV translation/rotation/scale tracks.
- **Geoset animation** (`GEOA` / geoset anim section): visibility/alpha/color over time.
- **Camera animation** (`CAMS` / `MDLCAMERASECTION`): camera position/target/FOV-like channels.
- **Attachment/Event/Emitter domains**: visibility and effect-parameter tracks.

## Runtime Evaluation Algorithm

Use this decoding/evaluation model:

```text
Input: sequenceIndex, absoluteTimeMs

1) seq = sequences[sequenceIndex]
2) localTime = absoluteTimeMs - seq.start
3) if seq is looping:
       localTime = localTime mod seq.duration
   else:
       localTime = clamp(localTime, 0, seq.duration)
4) For each animated track:
       if track.globalSequenceId != 0xFFFFFFFF:
           t = absoluteTimeMs mod globalSeqDuration[track.globalSequenceId]
       else:
           t = localTime
       sample track at t with its interpolation mode
5) Build node transforms/properties from sampled values
6) Resolve hierarchy (bones/nodes), then render/apply
```

## Practical Parser Guidance

When implementing a 0.5.3 decoder:

1. Parse sequence table before decoding node track payloads.
2. Keep a reusable generic `KeyTrack<T>` reader.
3. Validate that `keyCount`, timestamps, and payload sizes fit chunk bounds.
4. Support empty tracks (constant defaults).
5. Preserve unknown fields and raw bytes where field map is not yet proven.

## Known Unknowns (0.5.3-Specific)

The animation model itself is clear, but some 0.5.3 field-level details are still pending direct decompile mapping:

- exact packed structure of `MDLSEQUENCESSECTION` records
- exact packed structure of `MDLBONESECTION` base node headers
- definitive quaternion packing/compression behavior in this specific build
- full field map for texture/camera/event sub-record payloads

## Confidence
- Architecture (sequence + tracks + hierarchical evaluation): **High**
- Loader gating behavior (`0x100`, staged loading): **High**
- Generic key-track semantics/interpolation model: **Medium-High**
- Byte-accurate 0.5.3 per-record field maps: **Low-Medium**
