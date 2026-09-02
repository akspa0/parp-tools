# Feature Specification: WLW / MCLQ Liquid Convergence

**Feature Branch**: `209-wlw-mclq-convergence`

**Created**: 2026-09-02

**Status**: Draft — diagnosis not yet done

**Input**: Operator, 2026-09-02: "WLW data and MCLQ data do not nicely converge on each other, often
causing weird gaps in the MCLQ data where WLW's overlap, which isn't ideal."

## Why this is its own spec

It arrived alongside the cross-map transplant work (spec 208) but has nothing in common with it. It
is a **liquid representation** problem: the tile carries more than one description of the same water
and they disagree, so merging them leaves holes. It needs its own evidence and has a different owner.

Splitting it also keeps 208 honest — a transplant that carries liquid would otherwise inherit this
defect and be blamed for it.

## What is known, and what is not

**Known**: `TerrainTileTensorPack` carries **four** liquid representations, and they are separate
signals in the harvest stream:

| representation | signals |
|---|---|
| MH2O (WotLK+) | `mh2o_surface_height`, `mh2o_depth`, `mh2o_type_mask`, `mh2o_presence_mask` |
| MCLQ (inline, legacy) | `mclq_surface_height`, `mclq_type_mask`, `mclq_presence_mask` |
| WLW/WLM (world liquid) | `wl_liquid_mask`, `wl_liquid_height` |
| Unified | `unified_liquid_mask`, `unified_liquid_height` |

There is already a `BuildUnifiedLiquid` step, so convergence is *attempted* today. The operator's
report is that it produces **gaps in MCLQ where WLW overlaps** — a merge artifact, not an absence of
merging.

## Two concrete mechanisms, derived from source (2026-09-02)

Reading `AdtTensorPackBuilder.BuildUnifiedLiquid` narrows this considerably, and **rules one thing
out**: precedence is WL (lowest) -> MCLQ -> MH2O, and each source only ever *overwrites where it has
coverage*. The unified array is a **union**. It can add coverage; it cannot remove it. **So the merge
is not dropping cells**, and "gaps" are therefore either a wrong *height* or a source that never
supplied coverage in the first place.

Both remaining mechanisms are **coastline-specific**, which matches the operator's report that the
Wetlands coast is the worst area.

### Mechanism A — MCLQ partial-presence quads poison the interpolated height

The 129x129 -> 257x257 upsample admits a destination pixel when **any** of the four source corners
has presence:

```csharp
if (!mclqPresence[iy, ix] && !mclqPresence[iy, ix + 1]
    && !mclqPresence[iy + 1, ix] && !mclqPresence[iy + 1, ix + 1])
    continue;

float h = BilinearInterpolate(
    mclqHeight[iy, ix], mclqHeight[iy, ix + 1],
    mclqHeight[iy + 1, ix], mclqHeight[iy + 1, ix + 1], fx, fy);
```

The gate is "any corner present"; the interpolation then uses **all four heights unconditionally**,
including corners with **no presence**, whose height value is whatever the array holds there —
plausibly zero. Blending a real water height against a non-present corner **drags the surface toward
that value**.

Partial-presence quads occur exactly at the **edge of a water body**. A coastline is the highest
possible density of them. This would render as a water surface that sags or steps as it approaches
land — and where WL* covers the same area at its own height, the two disagree visibly.

### Mechanism B — `KeepOnlyAboveTerrain` culls WL* precisely at the shoreline

WL* coverage is filtered by `WlLiquidRasterizer.KeepOnlyAboveTerrain(wlMask, wlHeight, height257, …)`
before it reaches the merge. At a coast the terrain surface rises **through** the water plane, so
cells near the shoreline are the ones most likely to be culled. If MCLQ does not cover those same
cells — or covers them with a height corrupted by Mechanism A — the result is a gap at the waterline
rather than in open water.

### What the measurement must separate

These are distinguishable, and the report is built to distinguish them rather than to confirm either:

- **A** predicts unified cells whose height came from a **partially-present** MCLQ quad, concentrated
  at coverage boundaries, with height differing from the WL* height at the same cell.
- **B** predicts cells where WL* had coverage **before** `KeepOnlyAboveTerrain` and none after, with
  no MCLQ coverage to replace it.
- Neither predicts cells that are covered by a source and absent from unified — the union rules that
  out. **If the report finds such cells, both hypotheses are wrong** and something outside
  `BuildUnifiedLiquid` is at fault, most likely the renderer.

**Do not fix before the report separates them.** Spec 205 is the precedent: the obvious reading of the
field was wrong, the wiki's threshold did not hold for this client, and the first measurement came
back **inverted** because of a reader bug. Liquid in this codebase has burned two assumptions already.

### A confound to control

`ReadWlFiles` runs only in `AdtTensorPackBuilder.Build(adtPath, …)`. The `BuildFromBytes` path passes
`null, null` for WL*, so **any tool that builds packs from archive bytes measures with no WL* at all**
and will report zero WL coverage regardless of the truth. The report must use the path that actually
loads WL*, or it will produce a confident null result — the exact failure mode
`feedback_verify_detector_power_before_null_results` warns about.

## Subject

**Azeroth, the Wetlands coast.** Operator-supplied, chosen because those tiles are unchanged in 23
years, so the measurement is reproducible across clients and any difference is not era drift.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The disagreement is measured before anything is changed (Priority: P1)

A report says, per tile, where WLW and MCLQ both claim liquid, where only one does, and how far apart
their surfaces are where both do.

**Why this priority**: It is the whole first phase. Every candidate fix implies a different
measurement, and the report distinguishes them.

**Independent Test**: Run the report on a map with known overlapping WLW and MCLQ and read off the
four populations.

**Acceptance Scenarios**:

1. **Given** a tile with both sources, **When** the report runs, **Then** it counts cells covered by
   MCLQ only, WLW only, both, and neither.
2. **Given** cells covered by both, **When** the report runs, **Then** it reports the distribution of
   surface-height difference, not just a mean (per `feedback_calibrate_contrast_not_just_mean`).
3. **Given** a gap is observed, **When** it is attributed, **Then** the report states whether the
   unified array or the rendered mesh is where it appears.

---

### User Story 2 - Convergence produces no holes (Priority: P2)

Where either source describes liquid, the unified result describes liquid.

**Why this priority**: The actual goal, but it cannot be specified properly until US1 says which
disagreement is producing the holes.

**Acceptance Scenarios**:

1. **Given** a cell covered by exactly one source, **When** unified is built, **Then** that cell is
   covered.
2. **Given** a cell covered by both with differing heights, **When** unified is built, **Then** the
   winner follows a stated, tested precedence rule rather than an incidental one.
3. **Given** unified is rebuilt, **When** compared against the previous behaviour, **Then** cells that
   change are counted and attributed.

### Edge Cases

- WLW at a coarser resolution than MCLQ's per-chunk grid.
- A cell where the two sources disagree about liquid *type*, not just height.
- Ocean, where MH2O is legitimately depth-only (measured in spec 205) and a height comparison is
  meaningless.
- Tiles carrying MH2O and MCLQ and WLW simultaneously.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: A per-tile report MUST classify every cell as MCLQ-only, WLW-only, both, or neither.
- **FR-002**: Where both cover a cell, the surface-height difference MUST be reported as a
  distribution.
- **FR-003**: The report MUST state whether an observed gap is in the unified array or in the rendered
  mesh.
- **FR-004**: Unified liquid MUST cover every cell either source covers, unless a stated rule
  deliberately excludes it — and that rule MUST be named in the output.
- **FR-005**: Precedence where sources conflict MUST be explicit and tested, not incidental to
  evaluation order.
- **FR-006**: A change to convergence MUST report how many cells changed against the previous
  behaviour.
- **FR-007**: Ocean depth-only layers MUST NOT be treated as a height disagreement (spec 205).

## Success Criteria *(mandatory)*

- **SC-001**: The four coverage populations are reported for a real map, with counts.
- **SC-002**: Zero cells covered by a source but absent from unified, except those excluded by a named
  rule.
- **SC-003**: The precedence rule is stated and has a test per branch.
- **SC-004**: The operator confirms the reported gaps are the ones seen in the viewer — i.e. the
  measurement finds the actual complaint, not a different one.

## Out of Scope

- **MH2O decode.** Spec 205 owns it and it is fixed.
- **Liquid rendering appearance** — palettes and era scoping belong to the minimap/liquid work.
- **Cross-map transplant** — spec 208.

## Dependencies

- **Spec 205** — MH2O decode, and the precedent that liquid assumptions in this codebase need
  measuring rather than reading.

## Assumptions

- **The gaps are reproducible on a specific map and tile.** The operator has seen them; the first task
  is to name a tile so the report has a subject.
