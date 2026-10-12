# Data Model: Precise Object Selection

## PickableGeometry

CPU-side triangle data for one model key, served by `WorldAssetManager` (Phase 2). Explicitly separate
from the existing `WmoMeshSummary`/`MdxCollisionMeshSummary`, which retain only counts/bounds/samples
(research.md §1).

| Field | Type | Notes |
|---|---|---|
| Vertices | `Vector3[]` | Model-local space |
| Indices | `int[]` | Triangle list; length always a multiple of 3 |
| Availability | `Available` \| `Unsupported` \| `NotLoaded` \| `Failed` | `Unsupported` covers model kinds that structurally cannot supply it (research.md §3) — a normal state, not an error |

**Validation rules**: `Available` requires a non-empty, well-formed index array. Every other state carries
no geometry and routes the caller to bounding-volume fallback (FR-002). A negative result is cached, so a
repeat pick against an unavailable model does not re-read the file (research.md §2).

## PreciseHitResult

| Field | Type | Notes |
|---|---|---|
| Distance | `float` | Along the pick ray from its origin |
| WorldPoint | `Vector3` | The intersection point |
| TestKind | `Precise` \| `BoundingVolume` | Which test produced this hit — required by spec.md's Hit Result entity so a consumer can tell whether a hit is mesh-accurate or approximate |
| ObjectRef | existing scene-object identity | Reuses `SceneObjectPickHit`'s existing `ObjectType`+`ObjectIndex` shape; no new identity scheme |

**State transitions**: None — a hit result is immutable, produced fresh per pick.

## ConfirmedMatch

Persisted (Phase 3). Identifiers, paths, and provenance only — never client asset bytes (FR-016).

| Field | Type | Notes |
|---|---|---|
| Pm4Identity | `(int TileX, int TileY, uint Ck24, int ObjectPart)` | The existing PM4 object key shape already used by `SelectPm4Object` |
| PlacementIdentity | build label, map/tile, MDDF-or-MODF, UniqueId (or equivalent), asset path | Enough to re-find the placement in a future session without re-deriving it |
| ConfirmedAtUtc | `DateTimeOffset` | |
| Reason | `string` | The user's stated basis for confidence (FR-011) — required, not optional |
| Status | `Confirmed` \| `Retracted` | Derived from the event history, not stored as a mutable flag |

**Validation rules**: A `ConfirmedMatch` is only ever created by an explicit user action — no code path
may construct one from a matcher score or proximity threshold (FR-012, SC-009).

**State transitions**: `Confirmed` → `Retracted` via an explicit retraction event. Retraction appends
history; it never deletes the original confirmation (FR-013). A retracted match's PM4 object returns to
unconfirmed/candidate status.

## MatchCandidate

Never persisted as a confirmation; surfaced for review only.

| Field | Type | Notes |
|---|---|---|
| Pm4Identity | as above | |
| Origin | `SharedFingerprint` \| `Proximity` \| `SharedTile` | Why it was surfaced |
| RelatedConfirmedMatch | `ConfirmedMatch?` | For `SharedFingerprint`, the confirmed match whose fingerprint it shares (FR-015) |
| Rejected | `bool` | A durably-recorded rejection, so a false candidate is not re-surfaced as new each session (spec.md edge case) |

**Relationships**: A `MatchCandidate` becomes a `ConfirmedMatch` only through an explicit user
confirmation — there is no automatic promotion path, by design (FR-012).

## WorldCursorState

| Field | Type | Notes |
|---|---|---|
| HasHit | `bool` | False when the cursor is over sky/nothing |
| WorldPoint | `Vector3?` | Null when `HasHit` is false |
| SurfaceKind | `Terrain` \| `Object` \| `None` | `Object` only becomes reachable once Phase 2 lands (FR-008) |

**State transitions**: Recomputed every frame the cursor or camera moves; never persisted, never
interpolated across frames (spec.md edge case requires no stale position when the camera moves).
