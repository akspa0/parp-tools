# Data Model: Minimap, Fog Coverage, and Doodad Batches

The following records are contracts for the implementation. They describe state and ownership;
they do not require a particular class layout.

## MinimapInteractionState

| Field | Meaning |
|---|---|
| `SurfaceId` | Stable identity for docked or fullscreen map surface |
| `PointerDown` | Screen position at the start of the current left-button gesture |
| `LastPointer` | Last position used to update pan |
| `PanOrigin` | Pan offset before the current drag |
| `IsDragging` | Whether movement crossed the configured click threshold |
| `PendingTarget` | Integer map tile or precise map coordinate awaiting confirmation |
| `PendingClickCount` | Number of valid same-target clicks in the current sequence |
| `LastClickTime` | Timestamp used by the confirmation window |
| `MapBounds` | Valid map coordinate range used to reject invalid teleports |

Invariants:

- `IsDragging` and a teleport confirmation cannot be active for the same completed gesture.
- A teleport only executes when `PendingClickCount` reaches three for the same valid target.
- A pan update is expressed in minimap map-space units, not renderer world units.

## EffectiveFogCoverageWindow

| Field | Meaning |
|---|---|
| `Source` | Existing active lighting source label |
| `FogStart` / `FogEnd` | Resolved renderer fog range |
| `WorldRadius` | Conservative normal detailed-coverage radius derived from `FogEnd` |
| `TileBoundsPolicy` | Bounds-intersection policy, including tile edge protection |
| `Hysteresis` | Retain/release margins and last stable window |
| `Revision` | Changes whenever effective fog or policy changes |

`FogEnd` is the source value. `WorldRadius` is derived state and must never become an independent
hardcoded lighting truth.

## TileResidencyRecord

| Field | Meaning |
|---|---|
| `Tile` | ADT tile coordinate |
| `Exists` | Indexed client tile exists |
| `Selected` | Chosen for normal detailed submission priority |
| `Retained` | Kept in the camera-centered residency window |
| `WithinFogCoverage` | Tile bounds can contribute within effective fog range |
| `Preloaded` | Protected by an explicit camera-path lease |
| `DiagnosticFullLoad` | Protected by explicit stress mode |
| `CpuDecoded` / `GpuReady` | Existing load stages |
| `Reason` | Deterministic admission/eviction reason |

Normal submission requires `Exists`, `WithinFogCoverage`, and readiness, plus the existing content
visibility tests. `Preloaded` and `DiagnosticFullLoad` are reported exceptions, not replacements for
normal coverage.

## DoodadBatchKey

| Field | Meaning |
|---|---|
| `AssetIdentity` | Canonical model/asset key |
| `RendererBackend` | Native, legacy, or adapted backend identity |
| `RenderPass` | Opaque, alpha-test, transparent, or other pass |
| `MaterialState` | Texture/material/shader compatibility identity |
| `FadeClass` | Compatible fade/alpha state |
| `AnimationClass` | Shared/static versus placement-local animation state |
| `EffectClass` | Particle/ribbon/effect requirements |
| `WmoContext` | WMO/group/placement context when it affects correctness |

Two placements may share a batch only when all key fields are compatible. Unsupported fields route to
the fallback path and record a reason.

## DoodadAssetBatch

| Field | Meaning |
|---|---|
| `Key` | `DoodadBatchKey` |
| `ImmutableGeometry` | Shared uploaded geometry/material resource |
| `Instances` | Placement transforms and allowed per-instance values |
| `UniqueAssetCount` | Number of unique asset resources represented |
| `FallbackCount` | Placements excluded from this batch |
| `SubmissionCount` | Actual draw submissions for this bucket |

## FrameResidencyDiagnostics

The frame record must separately expose:

- active fog source/start/end and derived coverage radius;
- selected, retained, resident, drawable, preloaded, and excluded tile counts/identities;
- unique doodad assets, compatible batches, instances, fallbacks, animation updates, and draws;
- per-stage streaming, visibility, preparation, submission, and GPU/driver timing;
- invariant failures with the tile/asset and admission/submission reason.
