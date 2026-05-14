# WoWAlphaViewer Parity Matrix Template

## Purpose

Track feature-by-feature migration from legacy `MdxViewer` into layered `WoWAlphaViewer` ownership.

## Usage Rules

1. Every migrated capability must map to exactly one target layer.
2. Every row must include objective proof artifacts.
3. Mark `status=done` only when proof is attached and reviewed.

## Status Vocabulary

- `missing` — not started
- `in-progress` — implementation started, proof not complete
- `blocked` — waiting on prerequisite layer or dependency
- `done` — implemented and proven
- `intentional-drop` — intentionally not ported (must include reason)

## Matrix

| Legacy Surface | Capability | Target Layer | New Owner Path | Status | Proof Artifact | Notes |
|---|---|---|---|---|---|---|
| `ViewerApp` startup flow | deterministic app bootstrap | Layer 1 | `wow-viewer/src/viewer/WowViewer.App` | missing | | |
| world load bootstrap | client root + map open | Layer 2 | `wow-viewer/src/viewer/WowViewer.App` + `WowViewer.Core.IO` | missing | | |
| `WorldScene` frame staging | runtime stage graph | Layer 3 | `wow-viewer/src/core/WowViewer.Core.Runtime/World` | missing | | |
| AOI culling + ranges | visibility/LOD discipline | Layer 4 | `wow-viewer/src/core/WowViewer.Core.Runtime/World` | missing | | |
| terrain draw path | terrain shader baseline | Layer 5 | `wow-viewer/src/viewer/WowViewer.App` | missing | | |
| liquid draw path | liquid shader baseline | Layer 5 | `wow-viewer/src/viewer/WowViewer.App` | missing | | |
| sky + zone light behavior | skybox + lighting baseline | Layer 6 | `wow-viewer/src/core/WowViewer.Core.Runtime` + app consumer | missing | | |
| model viewer workspace | M2 consumer parity | Layer 7 | runtime + app | missing | | |
| model viewer workspace | MDX consumer parity | Layer 7 | runtime + app | missing | | |
| world model rendering | WMO consumer parity | Layer 7 | runtime + app | missing | | |
| converter UX panel | guided LK↔Alpha conversion | Layer 8 | `wow-viewer/src/viewer/WowViewer.App` + converter adapter | missing | | |

## Layer Completion Checklist

### Layer 0
- [ ] module registry exists
- [ ] layer readiness command exists
- [ ] baseline build proof captured

### Layer 1+
- [ ] all scoped rows moved to `done`
- [ ] proof artifacts linked
- [ ] regression guard added where applicable

