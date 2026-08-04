# Implementation Plan: PM4 Remaining Decode — Connective Geometry and Object Identity

**Branch**: `130-pm4-remaining-decode` | **Date**: 2026-08-03 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/130-pm4-remaining-decode/spec.md`

**Epic**: [epic-pm4-restoration](../epic-pm4-restoration/epic.md) — 130 produces the decode that 129
and 128 consume.

## Summary

The walkable-surface mesh is fully decoded. What is missing is **object identity** (which surfaces
belong to one original asset) and **connective geometry** (what closes a surface set into a sealed
negative volume). The spec's central claim is that the viewer selecting a surface instead of an
object is not a UI defect but the visible symptom of the first gap.

This plan implements that decode in nine phases, all in C# through the existing PM4 stack:

1. **A prior-art harvest.** ~60 files of PM4 object-assembly work sit unported in
   `parpToolbox/src/parpToolbox/Services/PM4/` on `main`, including a cross-tile object assembler and
   a grouping-rule tester, with matching documentation in `PM4Tool/docs/pm4/`. `AGENTS.md` line 345
   already designates these as extraction inputs. Harvesting hypotheses before writing new analyzers
   is cheaper than rediscovering them.
2. **An evidence register** that carries confidence, supporting evidence, and *eliminations* with
   every published interpretation — so FR-008/FR-012 negative results are first-class instead of
   being lost, and so 129/128 can consume findings without re-deriving them.
3. **A grouping-rule harness** that evaluates any candidate surface→object rule corpus-wide and
   reports fits, misses, and ungrouped counts per file and in total, against the recorded baseline.
4. **A candidate rule set**, measured — not asserted.
5. **A canonical object-identity service** producing a tile-independent object id, a per-surface
   assignment table, and an explicit `ungrouped` marking, exported as the artifact 129 consumes.
6. **Viewer whole-object selection** driven by that table, including cross-tile parts and a visible
   ungrouped fallback.
7. **A geometric discriminator for the connective geometry** — evaluating MSPV/MSPI *and* MSCN as
   co-equal candidates, because the existing mode counters are structurally incapable of answering
   the question they are posed against (research.md R3) and prior art identifies MSCN as the
   per-object exterior boundary (R10b).
8. **One reconstruction measured against its real asset**, with and without the connective stream,
   and sealedness reported numerically — the one non-statistical gate in the spec.
9. **The remaining open questions**, principally MPRR: a full-domain sweep plus sentinel-delimited
   run structure, with every eliminated domain recorded.

Phase 0 produced ten findings. Six shape the plan enough to state here, because they change what
success means:

- **The 65,819 / 1,206,977 baseline does not measure grouping.** It measures whether a link's
  `GroupObjectId` happens to be a valid `MPRL.Unk04` value — reference resolution, not partitioning.
  It is kept and reported unchanged for continuity, but SC-001 is additionally measured with a
  metric that is actually about grouping. Improving one does not imply improving the other.
- **`MSLK.RefIndex → MSUR` is 99.64% resolved** (1,268,782 fits / 4,553 misses) — but that is a
  *bounds* test, not a semantic one, and prior art reads the same field as an **MSVI anchor index**
  instead (R10c). The leading grouping rule routes through this edge, so Phase 4 must settle which
  reading the corpus supports rather than inheriting the current one.
- **`trianglesOnly = 0` is zero by construction, not by evidence.** The triangles test is strictly
  weaker than the indices test, so it can never fire alone. The published number is real output from
  a test that cannot answer the question.
- **The April 2025 mesh is fully accounted for, and nothing was lost from it** (R7). Its 15,096
  vertices are MSPV + MSVT concatenated; its 7,382 faces are Σ(MSUR.IndexCount − 2) — every surface
  as a triangle fan. Both reproduce exactly. The current stack already does that fan and more.
  Critically, **MSPI contributed no faces in that export**: the connective geometry was never decoded
  in any era, so US3 has no shortcut.
- **MSPV shares MSVT's coordinate frame** (R8, measured). The nesting hypothesis is eliminated for
  the second geometry stream — MPRL is the permuted chunk, not MSPV. Phase 7's question is purely
  topological.
- **MSCN is a co-equal connective-geometry candidate** (R10b). Prior art calls it the per-object
  exterior boundary; `MSUR._0x18` already indexes into it; it exceeds MSVT in count and shares its
  frame. "Exterior boundary" is a more literal fit for "what seals a surface set" than a second index
  stream is. The spec permits this — MSPV/MSPI is recorded as a lead that may be eliminated.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: `WowViewer.Core.PM4` (12 models, 14 services, 13 research analyzers — the
canonical decoder), `WowViewer.Core.IO` (WMO/M2 readers for the SC-006 asset comparison),
`WowViewer.Tool.Inspect` (30-command CLI, thin wrappers), `WoWViewer` (viewer app, selection only).
No new external package.

**Storage**: JSON reports under `output/pm4-decode/` (analyzer output, evidence register, per-surface
object assignment table). No database. The assignment table is the contract Spec 129 reads.

**Testing**: xUnit in `tests/WowViewer.Core.PM4.Tests` (14 existing test files). Unit tests use
synthetic chunk sets; every *claim* is validated on the 616-file corpus, never on synthetic data.

**Target Platform**: Windows 11 / PowerShell 7. CLI:
`dotnet tools/inspect/WowViewer.Tool.Inspect/bin/Debug/net10.0/WowViewer.Tool.Inspect.dll pm4 <cmd>`

**Project Type**: Shared library (`WowViewer.Core.PM4`) + CLI tool (`WowViewer.Tool.Inspect`) +
desktop viewer (`WoWViewer`, consumer only)

**Performance Goals**: A corpus-wide grouping-rule sweep over 616 files must complete in one pass
per corpus read, not one per rule — read once, evaluate all rules. `pm4 unknowns` already reads the
whole corpus in a single pass and is the throughput reference.

**Constraints**: FR-001 forbids reimplementing chunk parsing or coordinate handling — all reads go
through `Pm4ResearchReader` and all transforms through `Pm4CoordinateService` / `Pm4PlacementMath`.
FR-002 forbids single-file claims. `gillijimproject_refactor` is read-only. Client roots are
configured, never hardcoded (Constitution VI) — this binds Phase 8, which needs real WMO/M2 assets.

**Scale/Scope**: 616 PM4 files, 309 non-empty. ~1.27M MSLK entries, ~518K MSUR surfaces, ~2.4M MSPI
indices, ~1.93M MSVI indices. Tile 0_0 alone holds 81,936 MPRR entries (327,744 bytes at stride 4).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Repo Independence | PASS | All new code under `wow-viewer/src/core/WowViewer.Core.PM4`, `wow-viewer/tools/inspect`, `wow-viewer/src/viewer`. No path outside `wow-viewer/`. |
| II. Library-First | PASS | Every analyzer and the identity service live in `WowViewer.Core.PM4`. CLI commands are thin wrappers matching the existing `RunPm4Unknowns` shape. The viewer consumes the library and adds only selection UI. |
| III. Real-Data Validation | PASS with a noted dependency | The 616-file development corpus at `test_data/development/World/Maps/development` is staged real client data and is the spec's declared working corpus. **Phase 8 (SC-006) additionally needs real WMO/M2 geometry**, which must come from a configured client root, not the staged corpus. That dependency is called out in Phase 8 rather than discovered there. |
| IV. Model Architecture / Per-Signal Evidence | N/A — with intent honoured | No ML in this feature. The principle's durable intent (a strong signal must never mask a dead one) maps onto FR-002/FR-004: each candidate rule and each field interpretation is reported with its own fits/misses against its own baseline, never rolled into an aggregate score. |
| V. Streaming-First Dataset Pipeline | N/A | Spec 129 owns the Zarr store. This feature emits JSON that 129 reads. |
| VI. No Game Client Path Assumptions | PASS | Phase 8 takes the asset root as a CLI argument resolved at runtime. No client path in source. |
| Read-Only Reference Codebase | PASS | No writes to `gillijimproject_refactor`. |
| Format Reader Ownership | PASS | FR-001 restates it. No new PM4 parser; `Pm4ResearchReader` is the single reader. |
| One Phase at a Time | PASS | Nine phases, each ending in a validation gate (corpus-wide from Phase 2 on; Phase 1's gate is an inventory, since harvesting makes no claims). Phase N+1 does not start until Phase N's gate is met. |
| Spec Docs Are Source of Truth | PASS | `docs/architecture/pm4-object-identity.md` is written in Phase 5, in the same commit as the service. |
| Bite-Sized Plans | PASS | Every phase decomposes to ≤10 single-concern steps; see tasks.md. |

**Result: no violations. Complexity Tracking is empty.**

## Project Structure

### Documentation (this feature)

```text
specs/130-pm4-remaining-decode/
├── plan.md                          # This file
├── research.md                      # 10 findings, all measured or code-derived
├── prior-art-inventory.md           # written by implementation Phase 1 — harvested hypotheses
├── data-model.md                    # entities and their relationships
├── quickstart.md                    # the exact commands
├── contracts/
│   ├── README.md                    # Contract index and stability policy
│   ├── evidence-register.md         # Finding / confidence / elimination record schema
│   ├── grouping-rule.md             # Rule interface + evaluation report schema
│   ├── object-identity.md           # Per-surface assignment table — the Spec 129 contract
│   ├── geometry-stream.md           # MSPV/MSPI window interpretation + discriminator schema
│   └── cli-commands.md              # New `pm4` subcommands, flags, and output shapes
├── checklists/
│   └── requirements.md              # Existing — spec quality gate, already passed
└── tasks.md                         # /speckit.tasks output — NOT created by /speckit.plan
```

**Terminology note**: "Phase 0/1/2" in the Spec Kit workflow means research / design / tasks. Every
numbered phase below is an **implementation** phase. The two numberings are unrelated; below, "Phase"
always means the implementation one.

### Source Code (repository root)

```text
wow-viewer/
├── src/core/WowViewer.Core.PM4/
│   ├── Models/
│   │   ├── Pm4DecodeEvidenceModels.cs        # NEW — finding, confidence, elimination
│   │   ├── Pm4GroupingModels.cs              # NEW — rule descriptor, evaluation report
│   │   ├── Pm4ObjectIdentityModels.cs        # NEW — object id, surface assignment row
│   │   └── Pm4GeometryStreamModels.cs        # NEW — window interpretation, discriminator result
│   ├── Research/
│   │   ├── Pm4DecodeEvidenceRegister.cs      # NEW — Phase 2
│   │   ├── Pm4GroupingRuleEvaluator.cs       # NEW — Phase 3
│   │   ├── Pm4GroupingRules.cs               # NEW — Phase 4, the candidate rule set
│   │   ├── Pm4ConnectiveGeometryAnalyzer.cs  # NEW — Phase 7, MSPV/MSPI and MSCN
│   │   ├── Pm4MprrAnalyzer.cs                # NEW — Phase 9
│   │   └── Pm4ResearchUnknownsAnalyzer.cs    # EXTENDED — Phase 9, existing edges kept intact
│   ├── Services/
│   │   ├── Pm4ObjectIdentityService.cs       # NEW — Phase 5, canonical grouping
│   │   └── Pm4NegativeVolumeBuilder.cs       # NEW — Phase 8, reconstruction + sealedness
│   └── Matching/
│       └── Pm4ObjectSegmentBuilder.cs        # EXTENDED — Phase 5, consume identity service
├── tools/inspect/WowViewer.Tool.Inspect/
│   └── Program.cs                            # EXTENDED — new pm4 subcommands (thin wrappers)
├── src/viewer/WoWViewer/
│   ├── Terrain/WorldScene.cs                 # EXTENDED — Phase 6, object-keyed selection
│   └── ViewerApp_ClickSelection.cs           # EXTENDED — Phase 6, pick resolves to object
├── tests/WowViewer.Core.PM4.Tests/
│   ├── Pm4DecodeEvidenceRegisterTests.cs     # NEW
│   ├── Pm4GroupingRuleEvaluatorTests.cs      # NEW
│   ├── Pm4GroupingRulesTests.cs              # NEW
│   ├── Pm4ObjectIdentityServiceTests.cs      # NEW
│   ├── Pm4ConnectiveGeometryAnalyzerTests.cs # NEW
│   ├── Pm4NegativeVolumeBuilderTests.cs      # NEW
│   └── Pm4MprrAnalyzerTests.cs               # NEW
└── docs/architecture/
    └── pm4-object-identity.md                # NEW — Phase 5
```

**Read-only during this feature** (extraction inputs per `AGENTS.md` line 345 — harvested in Phase 1,
never edited, never referenced by a project file, since Constitution I forbids `wow-viewer` depending
on anything outside itself):

```text
parpToolbox/src/parpToolbox/Services/PM4/     # ~60 files of prior object-assembly work
PM4Tool/docs/pm4/                             # prior written findings
PM4Tool/docs/apps/mirrormachine/              # BSP/WMO generation reference
```

**Structure Decision**: The existing three-layer split is kept exactly as-is — research analyzers and
services in `WowViewer.Core.PM4`, thin CLI wrappers in `WowViewer.Tool.Inspect`, viewer as a pure
consumer. Nothing new is introduced structurally, because the stack already has the right shape and
FR-001 exists specifically to stop this feature from growing a parallel one. Two files are extended
rather than replaced: `Pm4ResearchUnknownsAnalyzer` (its published edges are the baseline every
success criterion is measured against, so they must keep emitting identical numbers) and
`Pm4ObjectSegmentBuilder` (Spec 128 already consumes its output).

## Implementation status (updated 2026-08-03)

Work started out of phase order because two things turned out to be reachable immediately and one
turned out to be blocking. Recorded here so the next session does not re-derive it.

### Done, out of order

- **Phase 7 (connective geometry) is effectively resolved for MSPV/MSPI.** It is a **vertical planar
  quad mesh — the walls between the MSUR floors**. Corpus-wide: 98.05% of windows hold exactly 4
  indices, 99.6% coplanar, and zero of 598,790 faces have Z as their dominant normal, against MSUR's
  91.7% Z-dominant. Polyline and triangle-list eliminated. Shipped as `pm4 connective-geometry` with
  6 detector-power tests satisfying the gate. **MSCN remains untested** as the co-equal candidate.
- **US1 partially delivered**: walls render in the viewer and select with their object.
- **Phase 9 (MPRR) partially done.** Structural hypothesis **eliminated** — no chunk's entry count
  matches the sentinel-delimited run count (best: MPRL, 5/502 files). New structure found: **94% of
  3,171,410 runs are exactly length 3 (75.5%) or length 7 (18.5%)**, so MPRR is small fixed-shape
  records, not a bulk index stream. No domain explains Value1 by bounds; MPRL is worst of nine.

### Coordinate frames — RESOLVED against ADT ground truth

**MSVT is stored in ADT placement space**: a distance-from-origin coordinate, exactly like a raw
MDDF position, with its two horizontal fields in the **opposite order** to MDDF's
(`MSVT.X == MDDF.rawY`, `MSVT.Y == MDDF.rawX`). The conversion is a per-axis subtraction with **no
axis swap**:

```text
placement = (17066.666 - MSVT.X, 17066.666 - MSVT.Y, MSVT.Z)
```

**Evidence** — over the 179 development tiles holding both a PM4 and a correctly named `_obj0.adt`,
**55,978 of 60,560 (92.4%)** MDDF/MODF positions fall inside their paired PM4's MSVT footprint. The
unswapped alternative scores **412 of 60,560 (0.7%)** and is eliminated.

The earlier "MSVT is already in absolute world coordinates" reading came from a bounds fit, which
proves only which BAND a value lies in — it cannot see a reflection about the map centre, because
reflecting a band yields a band. Its raw measurement still reproduces (309/309 files have X inside
the band of the filename's SECOND number and Y inside the band of its FIRST); what was wrong was
reading a **distance-from-origin band as a map tile index**. The map tile index is `31 - band`.

`Pm4PlacementMath.ConvertPm4VertexToWorld` is **correct and must not be touched**. It emits an
intermediate space the viewer finishes with `renderer = (MapOrigin - world.Y, MapOrigin - world.X,
world.Z)`; composing the two reproduces the transform above, so its axis swap is cancelled by the
renderer's. The 7 `PlacementMath_*` tests defending it are right.

### Region-scoped frames — REFUTED by `pm4 bounds-audit --by-region`

Over **1,895 CK24 objects** in 309 files spanning **207 regions**:

| measurement | result |
|---|---|
| objects resolving to the canonical frame | 1,877 / 1,895 |
| regions spanning more than one file | 62 |
| ...of those with mixed frames | 1 |
| objects with zero whole-tile displacement | 1,892 / 1,895 |

There is no frame family table to build. Crucially, `MSHD.Field04` is a **per-file** header value, so
"objects in one region fail identically" and "objects in one file fail identically" were always the
same observation — the hypothesis was unfalsifiable on the evidence that motivated it. Only regions
spanning several files can distinguish the two, and they agree.

**The residual misplacement is the per-object MPRL-scored fitter**, which the viewer runs per CK24
group:

- `ResolveCoordinateMode` picks `TileLocal` for data that is never tile-local and then adds tile
  offsets to already-absolute coordinates. 18 objects. **The human tents are one of them**:
  `development_01_00.pm4` (region 6) resolves `TileLocal/.UV.` and moves from canonical tile (0,1)
  to (1,-1). All 3 ADT placements for that tile sit inside its canonical footprint, so ground truth
  confirms canonical is right and the fitter is wrong.
- `TryComputeWorldYawCorrectionRadians` rotates **974 of 1,895 objects (51%)** by 15–45°, fitted
  against MPRL packed angles. **Now disproven** — see below.

### The yaw correction is wrong — `pm4 yaw-evidence`

The transform test above compares a placement *point* against a *box*, and the correction rotates
geometry about its own centroid, which moves neither. MODF is the way out: it carries a world
bounding box, so rotating a non-square object inside it ejects vertices. Objects are matched to a
box by **centroid containment**, which a centroid rotation cannot change — matching on best fit
would have selected for the unrotated reading and then concluded in its favour.

Over 1,066 objects matched to a WMO box, of which **127 have a box able to see a rotation** (proven
per object by a deliberate 45° control):

| geometry | mean fraction of vertices inside its WMO box |
|---|---|
| canonical, no yaw | **93.3%** |
| canonical + fitted yaw | 88.2% |
| full resolved solution | 89.5% |
| 45° control (known wrong) | 79.0% |

**yaw hurts 96, helps 3, tie 28.** The fitted yaw moves geometry in the same direction as the
known-wrong control. Worst cases are real WMOs whose collision fits perfectly without it —
`WG_GATE01.WMO` drops 100% → 50%, `WG_WALL01.WMO` 100% → 82%, `WALLPIECE01.WMO` 66% → 47%.

The 401 matched objects whose box *cannot* see a rotation are excluded from that headline and
reported separately, so "no difference" is never confused with "cannot tell".

### The fix — landed

The clincher was a differential the user spotted: **MSCN nodes render at the tents while the MSUR
mesh does not**, in the same file. `EnsurePm4MscnData` and `EnsurePm4MspvData` place points with
`(MapOrigin - p.X, MapOrigin - p.Y, p.Z)` — the canonical transform applied raw, with no fitter — and
the mesh was the only path going through `ResolvePlacementSolution`. MSPV, MSVT and MSCN share one
chunk frame, so that difference was never legitimate.

`WorldScene.ResolveCk24CoordinateModeResolution` now returns a constant canonical resolution;
`WorldScene.ResolvePlacementSolution` uses the identity planar transform and zero yaw, keeping the
real world centroid as the pivot because selection and connector merging need it. **`Pm4PlacementMath`
is deliberately untouched** — the fitter still exists for callers that want to explore it, all 16
`PlacementMath_*` tests still pass, and the render path simply no longer asks it to fit anything.

That unblocks Phases 5, 6 and 8, subject to visual confirmation in the viewer.

Phases 5, 6 and 8 stay blocked until that lands, since object identity, viewer selection and
reconstruction all sit on placement being right. Phases 2, 3, 4 and the rest of 9 are index-domain
work and were never blocked.

### Regions are confirmed authored areas, not bookkeeping

`MSHD.Field04` identifies **whole authored zones spanning many ADT tiles**. Region 245 is an entire
prototype zone laid out around 2006 as a test for what became Sholazar Basin, assembled from Feralas
and vanilla assets plus Wrath of the Lich King assets — which dates it internally. Region 73
neighbours it. 227 distinct values plausibly means ~227 authored areas.

This raises a new decode question worth adding to the open list: **does Field04 index something
outside the PM4** — a master region table, a DBC, or a server-side list? The working theory is that
a region is the unit the server loads and unloads for pathfinding, which would explain why it is
not tile-derived, why it spans tiles, and why it would need a stable shared id.

### A competing explanation for the rotation: phased objects

The rotation appears only on **00_00**, the **only tile with the phased/destructible payload
populated** (MDBH/MDOS/MDSF, 2,684 MDSF entries) — a fact this spec already records without
connecting it to placement. The rotation may be a mechanism for swapping collision geometry between
phase states, which would explain the *localisation* that a global transform error cannot.

**Test**: are misplaced 00_00 surfaces disproportionately reachable via `MDSF.MsurIndex -> MSUR`
(verified, 2,684 fits / 0 misses)? **Caution**: 00_00 is also the one tile where every coordinate
error cancels, so there are now two independent reasons for it to look special. Separate them.

### Viewer performance is now a gating concern

The overlay builds and draws all 9,207 objects regardless of camera position and is very slow; wall
rendering roughly doubled the triangle count. It must cull per tile using the existing ADT Detail
Tiles budget. The build is already per-tile, so the work is gating upload and draw by distance, not
restructuring. This blocks practical visual verification of any placement fix.

### Revised order

**0.5 (new, first): resolve coordinate frames** — `--by-region` bounds audit, plus the MDSF/phasing
test, keeping the two 00_00 explanations separate. **0.6: per-tile PM4 culling**, because placement
fixes cannot be verified visually until the viewer is usable. Then 1 (prior-art harvest), 2, 3, 4 as
written. Phase 7 needs only its MSCN half. Phase 9 needs the length-3/length-7 record shapes decoded
and should add "does Field04 reference something outside the PM4" to its open list.

## Phases

Each phase ends with a **gate**. A phase is done when its gate is met on the corpus, not when its
code compiles.

### Phase 0 — Research *(complete — see research.md)*

Ten findings. R1–R6 are derived from the code and the project's own analyzer output. R7–R10 came
from recovering the prior art the user identified, and are the reason a harvest phase now precedes
everything else. R7 and R8 include new measurements taken through the canonical stack on
`development_00_00.pm4`.

**Gate**: met. No NEEDS CLARIFICATION remains.

---

### Phase 1 — Prior-art harvest *(FR-002, FR-008; enables 3, 4, 7)*

Inventory and extract the PM4 work already in this repository but never ported into `wow-viewer`.
Nothing is ported wholesale — FR-001 forbids a second decoder — and nothing is trusted. Each
harvested idea becomes either a **candidate rule** (Phase 4), a **candidate interpretation**
(Phase 7), or a **finding** in the register (Phase 2), and is then measured corpus-wide like any
other hypothesis.

Sources, all on `main`, all designated extraction inputs by `AGENTS.md` line 345:

| source | what to extract |
|---|---|
| `parpToolbox/…/PM4/Pm4CrossTileObjectAssembler.cs` (21 KB) | cross-tile assembly approach → FR-006 |
| `parpToolbox/…/PM4/Pm4GroupingTester.cs` (56 KB) | a grouping-rule harness already exists — compare against the Phase 3 design before building |
| `parpToolbox/…/PM4/Pm4MsurObjectAssembler.cs` (41 KB) | surface→object rules → Phase 4 candidates |
| `Pm4SpatialClusteringAssembler`, `Pm4SmartGrouper`, `Pm4MprlObjectGrouper`, `Pm4RefinedHierarchicalObjectAssembler` | further grouping candidates |
| `MslkHierarchyAnalyzer.cs`, `Pm4MslkPatternAnalyzer.cs`, `MSUR_FIELDS.md` | field semantics → register findings |
| `MscnRemapper.cs`, `Pm4HierarchicalContainerDecoder.cs` | MSCN handling → Phase 7 |
| `PM4Tool/docs/pm4/*.md` | the written findings, including R10a and R10b |
| `PM4Tool/docs/apps/mirrormachine/` (`bsptreegenerator.cpp`, `WMO_exporter.cpp`) | BSP/WMO generation, against the epic's negative-BSP thesis |

**Why this is first**: two of the things later phases were going to build from scratch already exist
in some form here — a cross-tile assembler and a grouping-rule tester. Reading them costs a fraction
of rediscovering them, and the two contradictions they surface (R10a, R10c) change what Phase 4
measures.

**Gate**: an inventory document listing every extracted hypothesis, its source file, and which later
phase will test it. Every hypothesis enters the register as `Open` with its provenance. **No decode
claim is made in this phase** — harvesting is not validating.

---

### Phase 2 — Evidence register *(FR-007, FR-008, FR-012)*

A durable, versioned record of every field interpretation: its status (`open` / `partial` /
`resolved` / `eliminated` / `no-semantic-meaning`), its confidence, its corpus-wide evidence, and —
for eliminations — what ruled it out. Loaded and written by every later phase.

This is first because it is the smallest thing that makes negative results survive, and because
nine questions have been open long enough that some will end as eliminations rather than answers.
Without it those eliminations get re-searched.

**Gate**: the nine open questions from `Pm4ResearchUnknownsAnalyzer` round-trip into the register
with their current status, evidence, and confidence preserved verbatim, and a written finding
reloads byte-identically.

---

### Phase 3 — Grouping-rule harness *(US2, FR-002, FR-003, FR-004)*

`Pm4GroupingRuleEvaluator`: given the corpus and a set of rules, report for each rule — per file and
in total — surfaces grouped, surfaces left ungrouped, object count, objects spanning multiple tiles,
and the size distribution of the resulting objects. One corpus read, all rules evaluated.

The size distribution is not decoration: a rule that puts every surface in one object scores a
perfect "grouped" count and is worthless. The distribution is what exposes that.

**Gate**: the harness reproduces the recorded baseline edge
(`MSLK.GroupObjectId → MPRL.Unk04` = 65,819 fits / 1,206,977 misses) exactly when run as a rule, and
`pm4 unknowns` still emits every one of its existing numbers unchanged.

---

### Phase 4 — Candidate grouping rules *(US2, SC-001, SC-003)*

Implement the candidate set — research.md R2's seven rules plus whatever Phase 1 harvested — run it
corpus-wide, and record the outcome of every rule in the register, including the losers with the
evidence that eliminated them.

**Two extra obligations from the prior art**, both of which change what is measured rather than
merely adding rules:

- **Settle `MSLK.RefIndex`** (R10c). The current stack reads it as an MSUR index; prior art reads the
  same offset as an MSVI anchor. The 99.64% figure is a bounds test and does not decide it. Rule G3
  routes through this edge, so evaluate both readings and report which the corpus supports.
- **Route, do not drop, `MspiFirstIndex == -1` records** (R10a). Prior art says these are doodad
  placements carrying group/object ids and anchors. If so they are grouping signal, not noise. Test
  whether they carry it; count them either way.

**Gate**: every candidate has corpus-wide fits/misses/ungrouped reported; at least one rule beats the
baseline on the grouping metric; ungrouped surfaces are counted and characterised, never absorbed;
each eliminated candidate is in the register with its eliminating evidence; the RefIndex reading is
resolved or explicitly recorded as undecided with the evidence for both sides.

---

### Phase 5 — Canonical object identity *(FR-003, FR-006, FR-007, FR-011)*

`Pm4ObjectIdentityService` applies the winning rule and emits the per-surface assignment table.
Phase 1's harvest of `Pm4CrossTileObjectAssembler` informs the cross-tile half rather than being
ported into it. The table is:
`(file, surfaceIndex) → objectId`, plus `ungrouped` where membership is undetermined, plus the
confidence of each assignment. The object id is **tile-independent** so cross-tile membership is
expressible. Sentinel keys (CK24 = 0, which spans 291 tiles) are excluded by an explicit, named
policy rather than by accident.

`Pm4ObjectSegmentBuilder` is switched onto this service so Spec 128 inherits the improvement rather
than keeping a second, divergent notion of an object. `docs/architecture/pm4-object-identity.md`
lands in the same commit.

**Gate**: the table covers every non-empty file; every surface is either assigned or explicitly
`ungrouped`; no surface is silently dropped; the cross-tile object count is reported and reconciles
with `pm4 cross-tile`; re-running produces a byte-identical table.

---

### Phase 6 — Viewer whole-object selection *(US1, FR-005, FR-006, SC-002)*

The viewer's PM4 selection key is `(tileX, tileY, ck24, objectPart)` with a tile-scoped group key —
which is why cross-tile selection cannot work today regardless of decode quality. Selection is
re-keyed onto the Phase 4 object id: the pick still hits a rendered part, then resolves to its
object and selects every part of it, in every tile. A part whose object is undetermined selects
alone and is visibly marked ungrouped — never guessed into a neighbouring group.

**Gate**: clicking any surface of a multi-surface object selects the whole object, verified on
objects of known extent including at least one spanning multiple tiles; the same click twice gives an
identical selection; an ungrouped surface selects alone and is visibly marked as such.

---

### Phase 7 — Connective geometry: MSPV/MSPI **and** MSCN *(US3, FR-009, SC-005)*

Two candidate sources, evaluated as co-equals rather than in sequence.

**MSPV/MSPI** — the spec's named lead. Research finding R3: the existing mode counters cannot
discriminate, because `trianglesMode` implies `indicesMode`, so `trianglesOnly` is zero by
construction and the published `0` carries no information. The existing counters stay (they are a
published baseline) and a real geometric discriminator is added alongside: the corpus-wide
window-size histogram bucketed by TypeFlags family, whether windows close on themselves, whether
consecutive index triples form degenerate triangles, and whether a window's MSPV points are collinear
or coplanar. Candidate interpretations — polyline, closed polygon, triangle list, triangle fan/strip
— each get their own corpus-wide fits and misses.

**MSCN** — promoted to a co-equal candidate by R10b. Prior art states MSCN holds the per-object
*exterior boundary* vertices, which is a more literal description of "what closes a surface set into
a sealed negative volume" than a second index stream is. Three facts already in the tree support
testing it seriously: `MSUR._0x18` **already indexes into MSCN** (a per-surface link to a boundary
vertex — the right shape); MSCN exceeds MSVT in count (9,990 vs 6,318 in tile 0_0), so it is not a
subset of the mesh; and it shares MSVT's frame (R8), so the two are directly co-locatable. Prior art
also flags MSCN↔MSLK as unresolved, so this is a lead, not an answer.

**Simplification from R8**: neither candidate needs frame resolution. MSPV, MSVT and MSCN measurably
share one coordinate frame; MPRL is the permuted chunk. The question here is topological only.

Elimination of either candidate is a valid outcome and is recorded with its evidence.

**Gate**: every candidate interpretation, for both sources, has corpus-wide fit and miss counts; the
window-size histogram is published per TypeFlags family; `MspiFirstIndex < 0` records are counted
separately and tested against the R10a doodad-placement reading; and **the discriminator is shown
able to separate interpretations on constructed cases before any corpus claim is made**.

---

### Phase 8 — Reconstruction measured against a real asset *(US3, FR-010, SC-006)*

`Pm4NegativeVolumeBuilder` rebuilds one object from surfaces alone, then with whichever connective
source Phase 7 established, and reports the difference in the resulting volume plus how sealed each
is. The result is compared against the corresponding real WMO or M2.

The surface-only baseline is not new work: R7 established that MSUR surfaces triangulate as fans over
their MSVI window, that this is what the April 2025 export did, and that
`WorldScene.BuildCk24ObjectTriangles` already implements it. Lift that into the library rather than
rewriting it.

`PM4Tool/docs/apps/mirrormachine/bsptreegenerator.cpp` and `WMO_exporter.cpp`, harvested in Phase 1,
are reference material for the sealed-volume comparison — the epic's thesis is that PM4 is the
negative BSP, and MirrorMachine generates the positive one.

This is the only criterion in the spec that is physical rather than statistical, and it exists
because a rule can fit the corpus and still produce objects that are physically absurd.

**Dependency, called out rather than discovered**: this needs real WMO/M2 geometry from a configured
client root (Constitution VI — passed as a CLI argument, never hardcoded). If the corresponding
asset for a chosen object cannot be located, the phase reports which objects were attempted and why
each was rejected, rather than substituting a weaker comparison.

**Gate**: at least one object reconstructed both ways; the volume difference quantified; sealedness
reported numerically against the real asset; the object's identity and asset provenance recorded.

---

### Phase 9 — MPRR and the remaining open questions *(US4, SC-004, FR-008, FR-012)*

MPRR is the largest undecoded surface in the format and neither candidate domain explains it. The
existing analyzer only tests `Value1` against MPRL and MSVT — but a full-domain sweep helper
(`AddMismatchDomainFits`) already exists in the same file for `MSLK.RefIndex`. Reuse it for `Value1`
and `Value2`, and add the structural hypothesis the current test cannot see: MPRR is a sequence of
**sentinel-delimited runs** (`Value1 == 0xFFFF` is already modelled as `IsSentinel`), so the target
domain may be per-run and local rather than per-chunk and global.

Then close out the remainder: `MPRL.Unk14`/`Unk16`, the `MSHD` header fields, `MSLK.RefIndex`
semantics for the 4,553 non-fitting entries, `MSLK.TypeFlags`/`Subtype` corpus closure, and
`MDOS.buildingIndex → MDBH` (1 fit / 24 misses, on the one tile that has the payload and is
explicitly unrepresentative).

**Gate**: each of the nine open questions is either resolved with corpus evidence, narrowed with
specific domains eliminated and recorded, or documented as unresolvable with the reason. A field
established to carry no semantic meaning is recorded as such with its evidence, not left open.

## Risks

| Risk | Why it matters here | Mitigation |
|---|---|---|
| A grouping rule wins on the metric but produces physically absurd objects | The spec calls this out explicitly; it is the reason SC-006 exists | Phase 2 reports object *size distribution*, not just coverage; Phase 7 is a physical gate that a statistical winner must also pass |
| Optimising the MPRL-resolution baseline instead of grouping quality | The two are different questions and the headline number measures the first | Both metrics reported side by side, always; research.md R1 records why they differ |
| Tile 0_0 findings generalised to the corpus | It is the only tile with a destructible payload and is explicitly unrepresentative | FR-002 enforced in the harness: every report is per-file *and* total, so a single-tile effect is visible as one |
| A repeat of the hand-parse failure that produced a confidently wrong conclusion | It already happened once during spec research, on axis order | FR-001 is structural: no new reader, no new transform; Phase 6/7 geometry goes through `Pm4PlacementMath` |
| A null result from a test that could not have found the thing | Already present in the tree — R3 is exactly this | Phase 7's gate requires demonstrating discriminator power on constructed cases *before* any corpus claim |
| Phase 8 blocked by asset availability | Needs a configured client root, unlike every other phase | Declared as a Phase 8 dependency up front; phases 1–7 and 9 do not depend on it and are not blocked by it |
| **Prior art treated as authority instead of hypothesis** | Its own commit messages say "semi-functional" and "the functionality is broken"; it ran on a different reader | Phase 1's gate explicitly forbids decode claims. Every harvested idea enters the register as `Open` and is measured corpus-wide like any other candidate |
| **Porting `parpToolbox` code instead of its ideas** | It carries a second PM4 reader; importing it would violate FR-001 and Constitution I | Harvest interpretations only, reimplement against `Pm4ResearchReader`. No project reference is added outside `wow-viewer/` |
| **Anchoring on MSPV/MSPI because the spec named it the lead** | R10b makes MSCN at least as good a candidate, and confirmation bias is cheap | Phase 7 evaluates both as co-equals and reports both, whichever wins |
| **Inheriting the current `MSLK.RefIndex` reading unexamined** | R10c: the 99.64% figure is a bounds test, and grouping rule G3 depends on the reading | Phase 4's gate requires the contradiction resolved or explicitly recorded as undecided with evidence for both sides |

## Complexity Tracking

No constitution violations. Table intentionally empty.
