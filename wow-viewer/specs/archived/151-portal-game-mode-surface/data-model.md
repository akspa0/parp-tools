# Data Model: Portal-Aware Rendering, Game Mode, and Simple Surface

## WmoPortalVisibilityDecision

Owned by the runtime/world visibility layer and consumed by the WMO renderer.

| Field | Type | Meaning |
|---|---|---|
| `Mode` | `WmoVisibilityMode` | `Disabled`, `Exterior`, `Interior`, or `ConservativeFallback`. |
| `SourceGroupIndex` | `int` | Group containing the camera when known; `-1` otherwise. |
| `VisibleGroupIndices` | `IReadOnlyList<int>` | Bounded, deduplicated groups admitted for this frame/pass. |
| `VisitedPortalCount` | `int` | Number of portal links tested. |
| `MaxDepthReached` | `int` | Maximum traversal depth reached. |
| `FallbackReason` | `string?` | Stable diagnostic reason when the decision is fail-open. |
| `UsedPortalClip` | `bool` | Whether transformed portal-volume clipping was trusted. |

Invariants: all indices are in range, all vectors/rectangles used to build the decision are finite,
and a fallback never returns fewer groups than the current conservative behavior.

## GameModeState

Owned by a pure viewer-runtime game-mode service and projected onto `Camera` by the shell.

| Field | Type | Meaning |
|---|---|---|
| `Enabled` | `bool` | Explicit opt-in state; default `false`. |
| `CharacterKey` | `string?` | Selected model/runtime identity. |
| `HeadAnchor` | `Vector3` | World-space camera anchor, always finite when active. |
| `AnchorSource` | `HeadAnchorSource` | Recognized attachment, model height fallback, or editor fallback. |
| `MoveIntent` | `Vector2` | Forward/strafe intent before acceleration. |
| `RunRequested` | `bool` | Whether the run modifier is active. |
| `Grounded` | `bool` | Whether the body has a valid ground contact. |
| `JumpQueued` | `bool` | One-shot jump request consumed by the integrator. |
| `FallbackReason` | `string?` | Missing model/anchor/collision reason exposed to the UI. |

Invariants: enabling/disabling does not overwrite the saved editor camera pose; invalid model data
cannot produce NaN/Infinity; input and physics are bounded by the configured step limits.

## PhysicsBodyState

| Field | Type | Meaning |
|---|---|---|
| `Position` | `Vector3` | Body/world position. |
| `Velocity` | `Vector3` | World velocity. |
| `Radius` | `float` | Horizontal collision radius. |
| `Height` | `float` | Body height used for grounding/head offset. |
| `WalkSpeed` / `RunSpeed` | `float` | Configurable horizontal speeds. |
| `Gravity` | `float` | Positive magnitude applied downwards. |
| `JumpSpeed` | `float` | Upward impulse. |
| `GroundNormal` | `Vector3` | Last accepted ground normal. |
| `Contact` | `PhysicsContactKind` | None, terrain, WMO bounds, or fallback. |

Invariants: `dt` is clamped to a finite maximum; velocity and displacement are finite; jump is
accepted only from grounded state; unavailable collision is explicit rather than silently treated as
precise collision.

## ViewerSurfaceProfile

| Value | Default UI behavior |
|---|---|
| `SimpleInteractive` | Core load/camera/game/audio/region controls; no raw workbench refresh; concise status. |
| `AdvancedExplorer` | Existing tabbed/dockable data explorer, overlays, inspectors, and forensic controls. |

## DiagnosticProfile

| Value | Policy |
|---|---|
| `Interactive` | Essential errors/warnings/counters only; no raw payload logging or per-frame verbose refresh. |
| `Forensic` | Existing detailed logs and explicitly requested inspection routes. |

## Relationships

```text
ViewerSurfaceProfile
  -> selects DiagnosticProfile
  -> exposes GameModeState controls
  -> selects overlay/audio presentation

GameModeState
  -> owns PhysicsBodyState
  -> projects HeadAnchor/Yaw/Pitch onto Camera
  -> leaves editor Camera pose untouched while disabled

WmoPortalVisibilityDecision
  -> admits WMO groups/doodads/liquids for a render pass
  -> reports bounded counters to WorldRenderFrame
```
