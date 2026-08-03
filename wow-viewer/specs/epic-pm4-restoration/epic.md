# Epic: PM4 Restoration — decode, dataset, matching

**Status**: specs complete, implementation not started
**Created**: 2026-08-03
**Branch**: `130-pm4-remaining-decode` (contains specs 128, 129 and 130 — they are stacked linearly)
**Member specs**: [128](../128-pm4-negative-bsp-matching/spec.md) ·
[129](../129-pm4-zarr-dataset/spec.md) · [130](../130-pm4-remaining-decode/spec.md)

## Read this first if you are starting cold

**The PM4 stack already exists and is the oldest, deepest part of this codebase.** It was built
before the rest of the viewer and refactored as the first piece of the current wow-viewer era. Do
not write a new PM4 parser. Do not hand-parse chunks. Do not assume a coordinate convention.

What exists:

- `src/core/WowViewer.Core.PM4/` — 12 model files, 14 services, **13 research analyzers**
- `Pm4CoordinateService` — the coordinate transforms. **Use it.**
- `Pm4ObjectSegmentBuilder`, `Pm4SegmentSignalExtractor`, `Pm4AssetMatchScorer`,
  `Pm4ReplacementPlacementSynthesizer`, `Pm4FingerprintMatcher`, `Pm4SurfaceCorrelationMatcher`
- A **30-command CLI** under `WowViewer.Tool.Inspect`:

```
dotnet tools/inspect/WowViewer.Tool.Inspect/bin/Debug/net10.0/WowViewer.Tool.Inspect.dll pm4
```

`inspect`, `unknowns`, `cross-tile`, `hierarchy`, `linkage`, `mscn`, `mshd`, `audit`,
`export-segments`, `match-assets`, `synthesize-placements`, `correlate-models`, `dump-collision`,
`fingerprint-scan`, `build-wmo-surface-db`, `match-surfaces`, and more.

**The corpus**: `test_data/development/World/Maps/development` — 616 PM4 files, 309 non-empty.

### The mistake that must not be repeated

While researching these specs, a PM4 chunk was hand-parsed in Python outside `Pm4CoordinateService`,
its axis order assumed rather than read. The result was a confident, entirely wrong conclusion that
tile data was stacked above and below the map. The canonical decoder had the correct transform all
along.

The hazard is real and measured. In `development_00_00.pm4`:

| chunk | axis 1 | axis 2 | axis 3 |
|---|---|---|---|
| MSVT | 168–501 | 31–450 | −12–133 |
| MSCN | 169–499 | 31–450 | −12–133 |
| MPRL | 31–364 | 5–40 | 168–499 |

MPRL's third axis is MSVT's first. **Two chunks of one file, different axis order.** This is why
FR-001/FR-002 in spec 130 forbid reimplementation and require corpus-wide evidence.

## Why these three are one epic

They form a dependency chain, and each was specced separately because each is independently
testable — but implementing one without the others strands it.

```
130 (decode)  ──produces──>  129 (dataset)  ──feeds──>  128 (matching)
     │                            │                          │
     └── object identity          └── stored signals          └── restoration placements
         connective geometry          at 3 nesting levels         for the 4.0.0 dev map
```

**The end goal**: restore object placements for a 4.0.0-era development map whose placements are
otherwise lost. 128 is the deliverable; 129 makes it tractable; 130 makes it correct.

## Recommended order

**130 first.** It produces the decode the other two consume, and its object-grouping outcome is
currently ~5% resolved — building a dataset or a matcher on a 5%-resolved grouping bakes in a known
defect.

**129 second**, once grouping is better. It can technically be built on today's decode, but its row
layout is object-primary, so it depends on object identity being trustworthy.

**128 last.** It consumes both.

One deviation worth considering: 130's US1 (viewer selects whole objects) is the fastest visible
win and the acceptance test for the grouping decode. Doing it early gives a human-checkable signal
that grouping is right, before 129 commits that grouping to stored form.

## Measured baselines

All produced by `pm4 unknowns` and `pm4 cross-tile` over the 616-file corpus. These are the numbers
success is measured against — do not re-estimate them, re-run the tools.

### Solved (zero misses)

| relationship | fits |
|---|---|
| MSUR.Msvi window → MSVI | 518,092 |
| MSVI → MSVT | 1,930,146 |
| MSLK.Mspi window → MSPI | 598,882 |
| MSPI → MSPV | 2,418,205 |
| MDSF.MsurIndex → MSUR | 2,684 |
| MDSF.MdosIndex → MDOS | 2,684 |

### Open

| relationship | fits | misses |
|---|---|---|
| **MSLK.GroupObjectId → MPRL.Unk04** | **65,819** | **1,206,977** |
| MSLK.RefIndex → MSUR | 1,268,782 | 4,553 |
| MSLK.RefIndex → MPRL | 417,148 | 856,187 |
| MPRR.Value1 → MPRL | 6,778,712 | 7,200,518 |
| MPRR.Value1 → MSVT | 8,740,189 | 5,239,041 |
| MDOS.buildingIndex → MDBH | 1 | 24 |

### Corpus structure

- 1,229 distinct CK24 object keys; **266 (21.6%) span 2+ tiles**
- **CK24 = 0 spans 291 tiles and is a null sentinel, not an object** — exclude it
- Genuine cross-tile objects span 3–8 tiles
- Tile 0_0: 16 CK24 groups / 4,110 surfaces. Tile 0_1: 2 groups / 230 surfaces
- Tile 0_0 is the only tile with a populated destructible payload and is **explicitly
  unrepresentative** of the general mismatch population

## The two insights that shaped these specs

**1. The viewer bug and the missing geometry are one problem.** Clicking a surface selects a surface
rather than the whole object because `MSLK.GroupObjectId → MPRL.Unk04` is ~5% resolved. Surfaces are
the largest unit the decode can justify. This is not a UI defect.

**2. PM4 is the negative BSP.** It holds no object geometry yet records more per placed object than
the model file does: the surfaces players and NPCs walk on. An object's PM4 footprint is the shape
of the hole it makes in the walkable world. Current matching compares aggregate scalars (footprint
area, plane-distance stats), which cannot separate a round tower from a square keep of equal base
area — but their negative space is completely different.

Supporting context: WMO is Blizzard's answer to multi-nested Quake 3-style BSPs, one WMO group being
roughly one Q3 map, and the earliest WMOs were authored in Radiant. This data descends from an
engine family where the walkable surface set *was* the level representation.

## The leading lead

**MSPV/MSPI is a second geometry stream larger than the decoded surface mesh** — 2,418,205 index
fits against MSVI→MSVT's 1,930,146 — attached to the same link records. Its windows resolve
perfectly but their meaning does not: `indicesOnly=399,183`, `both=199,699`, `trianglesOnly=0`.

That is the right shape for the connective geometry that would close a surface set into a sealed
negative volume. **It is a lead, not a premise, and may be eliminated.**

**MPRR is the largest undecoded surface in the format**: 327,744 bytes in tile 0_0, and neither
candidate domain explains it.

## Tracking

| # | spec | status | gate |
|---|---|---|---|
| 130 | remaining decode | **planned** (spec + plan + research + contracts) | grouping beats 65,819 / 1,206,977; one object reconstructed and measured against its real asset |
| 129 | zarr dataset | specced | corpus stats reproduced from the store match the analyzers exactly |
| 128 | negative-BSP matching | specced | correct-match rate beats scalar-only scoring on the same segment set |

130 has `plan.md`, `research.md`, `data-model.md`, `contracts/` and `quickstart.md` as of 2026-08-03.
Next step is `/speckit.tasks` on 130. Implementation: not started.

## What planning 130 changed about this epic

Four findings from 130's Phase 0 research revise things stated above. See
[130/research.md](../130-pm4-remaining-decode/research.md).

**The April 2025 "lost geometry" was not lost** (R7). A screenshot of `output_development_00_00.obj`
showing 15,096 vertices / 7,382 faces is fully explained: vertices = MSPV (8,778) + MSVT (6,318);
faces = Σ(MSUR.IndexCount − 2) = 15,602 − 2×4,110, i.e. every MSUR surface as a **triangle fan**.
Both reproduce exactly. `WorldScene.BuildCk24ObjectTriangles` already does that fan, and the current
stack has strictly more (CK24, TypeFlags, coordinate resolution, cross-tile). **MSPI contributed no
faces in that export** — the connective geometry was never decoded in any era, so the leading lead
below has no shortcut.

**MSPV shares MSVT's coordinate frame** (R8, measured on tile 0_0). MSPV (169.60–498.79, 31.84–363.85,
0.85–134.55), MSVT (168.11–501.55, 31.00–450.70, −12.08–133.74) and MSCN all share axis order.
**MPRL is the permuted chunk** — the hazard table above is an MPRL property, not a general one. The
nesting hypothesis is eliminated for the second geometry stream.

**MSCN is a co-equal candidate for the connective geometry** (R10b). Prior art calls it the
per-object *exterior boundary*; `MSUR._0x18` already indexes into it; it exceeds MSVT in count and
shares its frame. "Exterior boundary" fits "what seals a surface set" more literally than a second
index stream does. MSPV/MSPI stays a lead, not a premise — as this epic already said.

**~60 files of prior PM4 object-assembly work sit unported on `main`** (R9), in
`parpToolbox/src/parpToolbox/Services/PM4/` (including `Pm4CrossTileObjectAssembler.cs` and a
56 KB `Pm4GroupingTester.cs`) with documentation in `PM4Tool/docs/pm4/` and a vendored MirrorMachine
(`bsptreegenerator.cpp`) relevant to the negative-BSP thesis. `AGENTS.md` line 345 already designates
these as extraction inputs. 130 gained a prior-art harvest as its Phase 1. It also surfaced a live
contradiction: prior art reads `MSLK`'s `0x10` field as an **MSVI anchor index**, where the current
stack reads it as an MSUR `RefIndex` — and the 99.64% figure is a bounds test that does not settle it.

## Standing rules for this epic

1. **Use the C# stack.** Never reimplement chunk parsing or coordinate handling.
2. **Corpus-wide evidence.** No claim from a single file or tile — tile 0_0 especially.
3. **Confidence travels with the claim.** The decoder already publishes confidence levels
   (MSLK TypeFlags "medium, partial, not corpus-closed"; MSUR GroupKey "low"; MSLK GroupObjectId
   explicitly not a confirmed object identity). Do not flatten these into equally-authoritative
   facts.
4. **Negative results count.** Eliminating a candidate domain with evidence is progress and must be
   recorded so the search is not repeated.
5. **Sentinels are not data.** CK24 = 0 is a null key, not an object spanning 291 tiles.
