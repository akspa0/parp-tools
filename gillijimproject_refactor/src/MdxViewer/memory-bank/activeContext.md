# Active Context — MdxViewer / AlphaWoW Viewer

Last updated: 2026-07-04
Compatibility lane only. Older milestones live in `memory-bank/archive/2026-07-04-older-history.md`.

## Role

- Legacy viewer host.
- Use for bounded hotfixes, validation capture, and archaeology.
- New ownership belongs in `wow-viewer`.

## Current slice

- Spec 080 Phase A partial source slice landed.
- Bottom display bar owns terrain/world toggles.
- Top toolbar is launcher strip.
- PM4 Object Match and PM4/WMO Correlation now render from `DrawUI()` and `Tools`.
- Proof level = source-only. `MdxViewer.sln` still fails on broad missing refs outside this slice.

## Still-valid compatibility routes

- Runtime-backed M2 path is default successful viewer route.
- Validation capture keeps doodads visible and uses build-aware object-mask policy.
- PM4 compatibility surface consumes `wow-viewer` region-id seam for grouping, export, and debug.
- Alpha terrain adapter ownership moved to `wow-viewer`; this lane should consume shared contracts, not grow new parsers.

## Boundaries

- Do not claim full M2 parity.
- Do not treat terrain-restore heuristics as broadly runtime-proven.
- Route new renderer/runtime ownership to `wow-viewer`.
