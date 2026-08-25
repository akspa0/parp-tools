# Spec 189 — PM4/PD4 complete field map

**Status:** open
**Branch:** v0.5.3-dev
**Supersedes nothing.** Spec 185 owns naming and the wiki draft; spec 188 owns behaviour
characterisation. This spec owns the **ledger**: every field in both formats, its status, and what
would change that status. Nothing here is allowed to be "done because it has a name".

---

## Context

Two years of PM4 work produced a decode that is right about most of the bytes and wrong in a
characteristic way about the rest. The characteristic way is this: a field acquires a name, the name
reads as an explanation, and nobody measures it again. `AttributeMask` was a window length.
`CK24` was a slice of a float. `_0x18` indexes `MSLK`, not
`MSCN`. `MprlToAdtPlacement` returned its argument unchanged. `MSCN` was "an unordered node cloud with
no index consumer" — because every test asked about the point SET and none asked about ORDER.

The second half of the pattern is worse. Because a named model leaves no room for what it does not
explain, anything that does not fit gets filed as an artefact: stray boxes, misses, outliers,
"remainder" buckets. Twice this month the user has pointed at something dismissed as noise and been
right that it was structure — the vertical stretch in the `_0x1C == 0` bucket, and `MSUR._0x00 == 0x10`
marking undersides rather than a storey index.

The failure runs the other way too, and this spec caught an instance of it on its first day.
`GroupObjectId` was written off here as a misnomer on the strength of a 0.676 distinct ratio. Measured
properly it is a group key whose groups are simply tiny, and 0.676 is exactly what mostly-pairs
produces. Rejecting a name too fast is the same error as accepting one: in both cases a summary stood
in for a measurement.

So this spec exists to make the unknown parts **enumerable and tracked**, rather than implicitly
closed by vocabulary.

## Current ledger

Measured with `pm4 field-sweep`, 120 files, all three built-in controls passing.

### `MSLK` — the chunk this spec most suspects

The user's read: *"we're not handling the MSLK fully, probably ignoring random other data, assuming
the boxes we find are mistakes, but are likely some sort of id in a node graph system."*

| field | behaviour | status |
|---|---|---|
| `MspiFirstIndex` / `Count` | wall-quad window; negative = open passage | MEASURED |
| `RefIndex` | names a neighbouring surface, 98.76% reciprocal | MEASURED |
| `LinkId` | tile address, `(second << 8) \| first`, 98.74% vs 0.31% control | MEASURED |
| `SystemFlag` | constant 32768 across 486,819 records | MEASURED |
| `_0x02` | constant 0 | MEASURED |
| `_0x00` "TypeFlags" | **bitfield; bit 0 = carries no geometry** (0.0% vs 100.0%, no exceptions) | MEASURED |
| `_0x01` "Subtype" | **not a taxonomy** — 19 values that separate on nothing; counts decay like a counter | MEASURED as not-a-category; what it counts is open |
| `_0x04` "GroupObjectId" | **a group key with TINY groups**: 283,066 pairs, 250,192 singletons | PARTIAL — grouping confirmed, what a pair MEANS is open |

**Tested 2026-08-25, and the first reading was wrong in an instructive way.** `_0x04` was recorded as
"near-unique per record", which made `GroupObjectId` look like a misnomer. It is a group key; the groups
are simply tiny. 250,192 singletons plus 283,066 pairs yields a distinct ratio of ~0.65 — precisely the
0.676 that read as near-unique. A summary statistic was mistaken for a structure.

The half-edge reading is refuted: of the 283,066 pairs, 7.11% are adjacent in the array, 0.38% share a
`RefIndex` against a 0.01% control, and **0.08%** point at each other. They are not two ends of an edge.
What they do share is `TypeFlags`, at **99.82%**.

That test has since been run and refuted: over 283,066 pairs, 77.09% are two anchors, 22.91% are two
wall records, and one-of-each is **0.00%**. A pair never binds geometry to an anchor — which follows
from the shared `TypeFlags`, now that bit 0 is known to decide anchor-vs-geometry. What a pair denotes
remains open.

`_0x00` and `_0x01` are settled enough to move off the unknown list. `_0x00` is a bitfield whose bit 0
means "no geometry", splitting 443,882 records at 0.0% carries-wall from 386,351 at 100.0% with no
exceptions; only 10 of 32 combinations appear, and bits 1 and 2 never co-occur. `_0x01` is the more
useful negative result: its 19 values are statistically identical on every geometric measure and each
co-occurs with all 10 `_0x00` values, so it is not a category at all. Its counts decay like a counter.
What it counts is the open question, and reading it as an enum was the error.

### `MPRL`

| field | behaviour | status |
|---|---|---|
| position | height is `Y`, horizontal pair `Z` then `X`; terrain contact | MEASURED |
| `Unk02` / `Unk06` | constant 65535 / 32768 | MEASURED |
| `Unk00` | index-like, 95 distinct, 5% zero | UNKNOWN |
| `Unk04` | **index-like, 3,932 distinct, 0.278 per-file ratio** | **UNKNOWN — most structured unknown in the format** |
| `Unk14` | enum, 14 values, 24% zero | UNKNOWN |
| `Unk16` | enum, 2 values, 69% zero | UNKNOWN |

`Unk04` carries a live wrong assumption: `Pm4ObjectPositionDecoder` converts it to a heading with
`Unk04 * 2π/65536` and feeds that into object placement. The sweep says it is index-like, not angular.
That conversion has no evidence behind it and is running in production code.

### `MSUR`

| field | behaviour | status |
|---|---|---|
| `_0x01`, `_0x02`, `_0x14`, `_0x18`, `_0x1C`, normal | see wiki §3–§5 | MEASURED |
| `_0x00` | surface class; `0x03` doodad, `0x10` **underside faces** | MEASURED (0x10 identified 2026-08-25) |
| `_0x03` | constant 0 | MEASURED |
| `_0x10` | behaves as a signed plane distance | PARTIAL — convention unconfirmed |

### Everything else

| chunk | status |
|---|---|
| `MPRR` | UNKNOWN. Structure only: 4n+3 run constraint, sentinel `0xFFFF` |
| `MSHD._0x00` / `._0x08` | PARTIAL — clamped world-unit spans, matched to nothing |
| `MSHD._0x04` | PARTIAL — `== 1` marks an empty tile; otherwise unexplained |
| `MVER` high byte `0x30` on PM4 | UNKNOWN |
| `MCRC` (PD4) | UNKNOWN — zero in the reference file |
| `MDOS` / `MDSF` | MEASURED as a per-surface destruction state; payload chunks PARTIAL |

## Requirements

- **FR-001** — A single command emits the ledger above from live data, so it cannot drift from the
  code. Status is derived where derivable (constant / enum / near-unique / float) and annotated where
  not.
- **FR-002** — `MSLK._0x04` is tested as a node identifier: density, sparsity, whether `RefIndex`
  indexes it, stability across files for repeated geometry. The current name is treated as unexamined.
- **FR-003** — `MSLK._0x00` and `_0x01` value meanings are characterised against geometry the way
  `MSUR._0x00` was — orientation, height within object, relationship to walls and floors.
- **FR-004** — `MPRL.Unk04` is tested against the heading interpretation currently in
  `Pm4ObjectPositionDecoder`. If it fails, that conversion is removed rather than left running.
- **FR-005** — Anomalies are recorded as open items, not dismissed. Anything that looks like an
  artefact gets a line in the ledger until it is explained or measured away.
- **FR-006** — The wiki draft and this ledger are updated in the same change as any new measurement.

## Success criteria

- **SC-001** — Every field of both formats appears in the ledger with exactly one of MEASURED /
  PARTIAL / UNKNOWN, and UNKNOWN explicitly covers "has a confident name nobody has tested".
- **SC-002** — No status is claimed without a control.
- **SC-003** — `MSLK._0x04` has a measured role or a documented set of eliminated hypotheses.
- **SC-004** — No production code path depends on an UNKNOWN field's assumed meaning without that
  dependency being listed here. `Pm4ObjectPositionDecoder`'s heading is the first entry.

## Assumptions

- Corpus sweeps and viewer sessions are user-run.
- The development corpus is representative for `MSLK`/`MSUR`; `MDBH`-family chunks are not, appearing
  on essentially one tile.

## Out of scope

- Decoding `MPRR`, which spec 185 tracks under its structural constraint.
- Renaming code fields — spec 185 FR-003 owns that; this spec supplies evidence to it.
- Generation from source geometry, which is spec 184.

## Known production code depending on unmeasured meaning

| site | dependency | risk |
|---|---|---|
| `Pm4ObjectPositionDecoder` | `MPRL.Unk04` as a heading, `* 2π/65536` | sweep says index-like, not angular |
| `AdtPm4MaskBuilder` | corner-relative coordinate space, `.X`/`.Z` horizontal | space never verified |
