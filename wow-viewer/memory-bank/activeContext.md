# Active Context — wow-viewer

Last updated: 2026-08-04

**This file is a dashboard, not a log.** It says what is live, what changed last, and where the
detail lives. Findings belong in the workstream file, not here — see "Memory bank layout" in
`coding_standards.md`. If a section here grows past a screen, it belongs somewhere else.

## Workstreams

| Workstream | State | Detail |
|---|---|---|
| PM4 decode | **active** — placement solved and confirmed in the viewer; now decoding the scene graph | [workstream-pm4-decode.md](workstream-pm4-decode.md) |
| Terrain / minimap ML | **idle** — nothing training; one curation decision pending before GPU time | [workstream-terrain-ml.md](workstream-terrain-ml.md) |
| Tile archaeology | **parked** | [weak-signal-tile-archaeology.md](weak-signal-tile-archaeology.md) |

## Now — PM4 scene graph (spec 131)

Branch `131-pm4-scene-graph-doodads`.

PM4 placement is **fixed and visually confirmed** (2026-08-04): tiles aligned, tents correctly
identified, previously-rotated walls and buildings correct. That unblocked the scene-graph work.

Three things were established in the same push, each with a falsifiable test:

1. **A keyed (non-zero) CK24 is one placed WMO.** 47 tiles have no WMO placements and none of them
   carries a keyed object; keyed count matches WMO count exactly on 136/179 tiles.
2. **CK24 0 is not an object** — it is the per-tile remainder, exactly one per tile, holding
   everything that is not a keyed WMO.
3. **That remainder splits into per-doodad components by mesh connectivity**, and
   **`MSLK.GroupObjectId` is the per-doodad identity**: 95.1% of 20,113 components land on an MDDF
   placement, and 3,343 of GroupObjectId's 3,345 pure components are unique on their tile.

**Next**: component coverage. 34.4% of components have no MSLK link at all, so GroupObjectId names
only a minority. The anchor-only MSLK entries (`MspiFirstIndex < 0`, 53% of 1.27M links) are the
next place to look.

New ground truth this session: user-supplied screenshots of **Blizzard's WoW Editor 1.9.0** rendering
this data, with the Karazhan Crypts WMO loaded for comparison. Decorative M2s have no nav polygons
beneath them — that is what makes 0.339 components per placement the expected result rather than a
shortfall. Details and the other readings in the workstream file.

## Test state

`WowViewer.Core.PM4.Tests`: **102 passed, 1 failed** —
`Pm4RegionObjectGrouperTests.AnalyzeDirectory_DevelopmentCorpus_NonEmptyRegionsHaveObjects`,
**pre-existing**, confirmed failing at baseline.

## Durable constraints

- `gillijimproject_refactor` is read-only. New code lives in `wow-viewer`.
- The user runs training, capture, client-backed proof, and all heavy/GPU work. Hand over the exact
  command; never launch it.
- No DepthAnything / multi-head / shared-weight model paths.
- Constitution IV: per-signal evidence — a strong signal must never mask a dead one.

## Incidental

`pm4 inspect` and `pm4 audit` accept `--output` and silently ignore it; the other `pm4` report
commands honour it.
