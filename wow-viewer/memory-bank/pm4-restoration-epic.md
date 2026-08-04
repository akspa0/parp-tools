# PM4 Restoration Epic — pointer

Recorded: 2026-08-03. Status: three specs written; **130 planned** (plan + research + contracts);
implementation not started.

**Full epic doc**: `specs/epic-pm4-restoration/epic.md` — read that before touching PM4 work.

## One-line summary

Restore object placements for a 4.0.0-era development map whose placements are lost, by decoding
what PM4 actually is: the NEGATIVE BSP — the walkable surfaces every placed object carves out.

| spec | branch | what |
|---|---|---|
| 130 | `130-pm4-remaining-decode` | undecoded fields, connective geometry, object identity |
| 129 | `129-pm4-zarr-dataset` | stored queryable form at map/tile/object levels |
| 128 | `128-pm4-negative-bsp-matching` | match on negative-space structure, not aggregate scalars |

Branches are stacked linearly; `130` contains all three. Order: 130 -> 129 -> 128.

## The three things most likely to be got wrong

1. **The PM4 C# stack already exists and is the oldest part of the codebase** — 13 research
   analyzers, 14 services, a 30-command CLI under `WowViewer.Tool.Inspect pm4`. Never hand-parse
   PM4 chunks. A Python parse outside `Pm4CoordinateService` during spec research produced a
   confident and entirely wrong conclusion; the axis order differs BETWEEN CHUNKS OF ONE FILE
   (MPRL's third axis is MSVT's first).
2. **The viewer selecting a surface instead of an object is not a UI bug.**
   `MSLK.GroupObjectId -> MPRL.Unk04` is 65,819 fits / 1,206,977 misses (~5%). Surfaces are the
   largest unit the decode can justify.
3. **CK24 = 0 spans 291 tiles and is a null sentinel**, not an object. 1,229 distinct keys, 266
   (21.6%) genuinely span 2+ tiles — which is why the dataset is object-primary, not tile-keyed.

## Leading lead

MSPV/MSPI is a second geometry stream LARGER than the decoded surface mesh (2,418,205 vs 1,930,146
index fits) attached to the same link records, with its window meaning unresolved
(indicesOnly=399,183, both=199,699, trianglesOnly=0). Right shape for the connective geometry that
seals a surface set into a negative volume. A lead, not a premise.

MPRR is the largest undecoded surface in the format (327,744 bytes in tile 0_0); neither candidate
target domain explains it.

## Prior art is IN THIS REPO, unported (found 2026-08-03)

`AGENTS.md` line 345 already names these as extraction inputs. All on `main`:

- `parpToolbox/src/parpToolbox/Services/PM4/` — ~60 files of object-assembly work from 2025-07-25..31,
  including `Pm4CrossTileObjectAssembler.cs` (21 KB) and `Pm4GroupingTester.cs` (56 KB, a
  grouping-rule harness). Its own commit messages say "semi-functional" and "broken" — hypotheses,
  not authority.
- `PM4Tool/docs/pm4/` — written findings. `PM4Tool/docs/apps/mirrormachine/bsptreegenerator.cpp` —
  BSP generation, relevant to the negative-BSP thesis.
- Spec 130 gained a prior-art harvest as its Phase 1 because of this.

The public `akspa0/parp` repo (`scripts/pm4tool`, `scripts/old/pm4_to_obj`) is **not** where the
geometry is — those are JSON/SQLite tools that emit no faces. Last commit 2025-03-10.

## Three things the 130 planning measured and settled

1. **The April 2025 "lost geometry" was never lost.** Screenshot of `output_development_00_00.obj`
   at 15,096 verts / 7,382 faces decodes exactly: verts = MSPV(8,778) + MSVT(6,318);
   faces = Σ(MSUR.IndexCount − 2) = 15,602 − 2×4,110 → **MSUR surfaces are triangle fans**.
   `WorldScene.BuildCk24ObjectTriangles` already does this. **MSPI produced zero faces** — the
   connective geometry was never decoded in any era.
2. **MSPV/MSVT/MSCN share one coordinate frame; MPRL is the permuted chunk.** Measured on tile 0_0.
   The nesting hypothesis is eliminated for the second geometry stream.
3. **MSCN is a co-equal connective-geometry candidate** — prior art calls it the per-object exterior
   boundary, `MSUR._0x18` already indexes into it, and it exceeds MSVT in count.

Also: `pm4 inspect` and `pm4 audit` accept `--output` and silently ignore it.

## Related

- [[weak-signal-tile-archaeology]] — the terrain-side sibling, complete and parked. Same discipline:
  measure with the project's own tooling, verify detector power before trusting a null.

## Coordinate frames may be REGION-scoped (user observation + hover data, 2026-08-03)

After the MSVT world-coordinate finding, misplacement persists in the viewer, but not uniformly —
and the hover overlay shows the discriminator is likely `MSHD.Field04` (exposed as `region`).

Observed, via viewer hover tooltips:

| object | file | region | CK24 | symptom |
|---|---|---|---|---|
| red props | `development_01_00.pm4` | **6** | 0x000000 | belong on the tile to the RIGHT of 0_0 |
| yellow object | `development_00_00.pm4` | **146** | 0x41D4B1 | correct TILE, but **polar opposite** position, possibly 180 deg out |
| M2 group | `development_00_00.pm4` | **146** | 0x000000 | polar opposite, moves WITH the yellow object |

**The pattern**: the two region-146 objects are wrong *together and in the same way*, while the
region-6 object is wrong differently. That is the signature of a per-region frame, not a single
global transform error. The user's reading: "there are layers of data encoded that we do not decode
properly, and they can all contain different coordinate cardinal directions."

**Why this is credible.** `MSHD.Field04` is already known to be a scene/region division —
`Pm4RegionObjectGrouper` groups by it, `Pm4MshdHeader.RegionId` names it, and `IsEmptyStubRegion`
treats `RegionId == 1` specially. It has 227 distinct values corpus-wide and tile-coordinate packing
was already ruled out for it. A region id that selects a coordinate frame would explain why it has
many values, why it is not tile-derived, and why it has resisted interpretation.

**The test, and it needs no new decode.** Group MSVT bounds by `MSHD.Field04` and compare each
region's occupied range against the tile band implied by the filename. If regions fall into a small
number of families — identity, negated, axis-swapped, 180-degree rotated — the frame is
region-scoped and the family table is the fix. If every region behaves identically, the hypothesis
is dead and the residual error is elsewhere.

Run it per region AND per tile, because a region spans tiles: the interesting case is one region
behaving consistently across many files.

**Status**: hypothesis only. `pm4 bounds-audit` already reads every file and computes MSVT bounds,
so adding a `--by-region` grouping is the cheapest path to testing it.

### CONFIRMED: regions are real semantic units spanning many tiles (user, 2026-08-03)

The region hypothesis above is no longer speculative. `MSHD.Field04` identifies **whole authored
areas spanning multiple ADT tiles**, not per-file bookkeeping.

- **Region 245 is an entire prototype zone**, laid out around **2006**, a test for what became
  **Sholazar Basin** — and it is a prototype of what *could* have been, not what shipped. It is
  assembled from **Feralas** and base-vanilla assets **plus assets added later in Wrath of the Lich
  King** (Grizzly Hills flowers planted in its garden). That asset mix dates the region internally
  and proves regions are authored content, not runtime partitioning artefacts.
- **Region 73** is a different region in the same area. Regions neighbour each other spatially.
- 227 distinct Field04 values corpus-wide is therefore plausibly ~227 authored areas.

**Open question the user raises**: do these values index something **outside the PM4**? A master
region table, a server-side list, or a DBC. The working theory is that a region is the unit the
**server loads and unloads for pathfinding** — which would explain why it is not tile-derived, why
it spans tiles, and why it would need a stable id shared with something else.

That reading also makes region a strong candidate for the coordinate-frame key: a server that swaps
navigation data in and out per region has every reason to store each region in its own frame.

### The phased-object hypothesis for the rotation bug (user, 2026-08-03)

**This is the most promising explanation for the residual misplacement, and it is testable.**

The weird rotation only appears on **00_00** — and 00_00 is the **only tile in the corpus with the
phased/destructible payload populated**, in both the ADT and the PM4. The spec already records this
as measured fact without connecting it to placement: MDBH/MDOS/MDSF are populated on exactly one
tile, 2,684 MDSF entries, and tile 0_0 is "explicitly unrepresentative of the general mismatch
population".

The user's reading: the rotation may be a mechanism for **swapping collision geometry in and out for
phased objects**. A phased object needs more than one collision state, so its geometry may be stored
in an alternate orientation or frame that only makes sense once the phase state is applied.

**Why this is worth taking seriously over a plain transform bug**: it explains the *localisation*. A
global transform error should affect every tile equally. This one concentrates on the single tile
that has a feature no other tile has.

**The test**: check whether the misplaced objects on 00_00 are reachable through
`MDSF.MsurIndex -> MSUR` (2,684 fits, 0 misses — a verified edge). If misplaced surfaces are
disproportionately MDSF-referenced, the phase payload is implicated. If misplacement is independent
of MDSF membership, it is not.

**Caution**: 00_00 is also the tile where every coordinate error cancels (both indices are 0, so
world equals local). Two unrelated reasons for 00_00 to be special are now on the table, and they
must not be conflated. Separate them before concluding.

### Viewer performance — PM4 renders everything at once (user, 2026-08-03)

The PM4 overlay currently builds and draws **all** loaded objects regardless of camera position:
9,207 objects on the development map, with visibility counted but not used to cull work. It is very
slow.

**Required**: render per tile, driven by the existing **ADT Detail Tiles** slider
(`_terrainManager.DetailedTileCountOverride` / `EffectiveDetailedTileCount`), so PM4 honours the
same distance budget terrain already does. The overlay build is already per-tile
(`BuildPm4TileObjects` runs per file and `Pm4MaxLinesPerTile` / `Pm4MaxTrianglesPerTile` budgets
exist), so the work is in gating upload/draw by tile distance rather than restructuring the build.

Note wall rendering roughly doubles the triangle count, so this got more urgent this session.
