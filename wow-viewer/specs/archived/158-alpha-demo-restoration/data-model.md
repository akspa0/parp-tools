# Data Model: Alpha Demo Restoration

Entities new to this spec. `WtfLine`/`WtfLineKind` (Spec 159) are reused unmodified as the input to
`PortCommandRequest` below, not redefined here.

## PortCommandRequest

Built from a Spec-159-classified `WtfLine` of kind `PortCommandCandidate`. The execution-side
interpretation of that already-parsed data.

| Field | Type | Notes |
|---|---|---|
| Kind | `Worldport` \| `Teleport` | From `WtfLine.HasMapIdArg` — true → Worldport, false → Teleport |
| MapId | `int?` | Present only for `Worldport` |
| Position | `Vector3` | X/Y/Z from `WtfLine.NumericArgs` |
| SourceLine | `WtfLine` | Retained for error reporting — a failed command must be traceable to its
| | | original text (FR-007) |

**Validation rules**: `Kind = Teleport` requires `MapId = null`; `Kind = Worldport` requires `MapId` set.
A `PortCommandRequest` is never constructed from a `WtfLine` whose `CoordinatesPlausible` is false without
that fact being surfaced to whatever reports the outcome — implausible coordinates are still attempted
(per spec.md, this feature does not decide the coordinates are wrong), but the report notes the
plausibility flag so a human reviewing results has it.

## PortCommandOutcome

| Field | Type | Notes |
|---|---|---|
| Request | `PortCommandRequest` | |
| Status | `Applied` \| `MapLoadFailed` \| `NoCurrentMap` \| `Unrecognized` | `NoCurrentMap` is the
| | | teleport-with-nothing-loaded edge case from spec.md |
| Detail | `string?` | Failure reason, when applicable |

**State transitions**: `Applied` is terminal and successful. Every other status leaves camera state
unchanged from before the command was attempted (FR-005) — there is no partial-apply state.

## CameraFollowTarget

| Field | Type | Notes |
|---|---|---|
| ModelInstanceId | opaque reference | Whatever this project's existing scene-object identity already is for a placed model (WMO/M2 `ObjectType`+`ObjectIndex`, matching `SceneObjectPickHit`'s existing shape — no new identity scheme) |
| BoneReference | `int` (bone index) or `KeyBoneId` | Prefer `KeyBoneId` (e.g. "head") when the model declares one; fall back to a raw bone index otherwise |
| Pipeline | `Legacy` \| `Modern` | Which bone-matrix source resolves this target (`MdxAnimator.BoneMatrices` vs `M2BonePoseState.Matrices`) |

**Validation rules**: A `CameraFollowTarget` whose referenced model is no longer present in the scene
resolves to "no transform available" rather than throwing — the camera falls back to free-fly (FR-011,
edge case) rather than the resolution failing loudly.

## M2Era100Attachment

Mirrors the real 1.0.0 client's attachment record (`test_data/native-research/1.0.0-decomp/feat_has_attachment.c`),
parsed from the already-documented offsets in `M2Era100Constants.cs`.

| Field | Type | Notes |
|---|---|---|
| AttachmentId | `int` | The slot identity (e.g. hand) — exact enumeration confirmed against real data during Phase 5a, not assumed here |
| BoneIndex | `int` | Which bone this attachment rides on |
| Pivot | `Vector3` | Local offset from the bone, per the real client's `GetAttachmentWorldTransform` algorithm |

**Relationships**: A `CameraFollowTarget` and a torch's `DynamicPointLight` can both resolve against the
*same* underlying bone-pose data for the same model instance — they are independent consumers of the
same already-computed transform, not coupled to each other.

## DynamicPointLight

| Field | Type | Notes |
|---|---|---|
| Position | `Vector3` | Updated every frame from its source (an `M2Era100Attachment`'s resolved world transform, typically) |
| Color | `Vector3` (RGB) | |
| Radius | `float` | Falloff distance; exact curve is an implementation detail (research.md §7) |
| SourceId | opaque reference | What this light is attached to, so removing/un-equipping that source removes exactly this light (FR-015) and nothing else |

**State transitions**: A `DynamicPointLight` exists only while its source is equipped/present. There is no
"disabled but retained" state — removal is deletion, not a flag flip, per FR-015's "no residual glow"
requirement.
