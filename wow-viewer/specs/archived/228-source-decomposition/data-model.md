# Design Model: Source Decomposition — God-Class Split

This is an in-process C# refactoring plan. It creates no network API or persisted schema.

## First extraction boundary — selection

| Entity | Owner | Responsibility | Boundary rule |
|---|---|---|---|
| `WorldSceneSelectionSnapshot` | `WowViewer.Core.Runtime/World/Selection` | Immutable candidate list and policy inputs for one hover/click evaluation. | Contains values and stable identifiers only; never a `WorldScene` reference. |
| `WorldSceneSelectionCandidate` | `WowViewer.Core.Runtime/World/Selection` | Normalized bounds, kind, stable scene identifier, selection point, and depth/distance inputs. | No renderer/ImGui object references. |
| `WorldSceneSelectionRequest` | `WowViewer.Core.Runtime/World/Selection` | Ray or brush mode, distance limit, PM4-depth policy, and current pick position. | Validates finite vectors, non-negative limits, and explicit mode. |
| `WorldSceneSelectionResult` | `WowViewer.Core.Runtime/World/Selection` | No-hit or the selected stable identifier with its reason and distance. | No UI strings or mutable scene state. |
| `WorldSceneSelectionService` | `WowViewer.Core.Runtime/World/Selection` | Deterministic candidate filtering/ranking that can be unit-tested. | Receives a snapshot; does not call back into the viewer. |
| `WorldScene` adapter seam | `src/viewer/WoWViewer/Terrain/Scene/Selection` | Maps existing resident WMO/MDX/liquid/PM4 data to the snapshot and maps the result back to current public methods. | Is the only viewer-specific bridge; it may not expose a new UI route. |

## Validation rules

- Candidate identifiers are stable for one frame and unique within a snapshot.
- A request with a non-finite vector or negative range returns a typed invalid/no-hit result;
  it never manufactures a selection.
- Ranking and WMO-container fall-through preserve current ordering before any feature work.
- PM4 depth preference and hover range remain explicit request/policy fields, not reads from a
  parent `WorldScene` object.
- Viewer-only `HoveredAssetInfo` formatting and Inspector ownership stay outside the Core Runtime
  service until Spec 227 approves their authoritative UI home.

## Relationships

```text
WorldScene resident state
        │ explicit snapshot
        ▼
WorldSceneSelectionService ──► WorldSceneSelectionResult
        ▲                              │ explicit mapping
        └──────── no parent reference ─┘
                                       ▼
                            existing WorldScene public route
```
