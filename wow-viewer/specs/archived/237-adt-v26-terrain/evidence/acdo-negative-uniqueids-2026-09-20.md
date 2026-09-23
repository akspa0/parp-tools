# Spec 237 Evidence — ACDO Negative uniqueIds Are Real: Two Allocators (MEASURED)

Date: 2026-09-20

## Operator question

> "v26 has objects with negative uniqueID's, not sure if that's right?"

Screenshot: `AZSHARAROCK02.M2`, `UniqueId: -210`, on the v26 corpus loaded as a DAT folder.

**Answer: it is right.** The value is real on-disk data, not a misread, not corruption, and the field
really is a uniqueId. v26 uses **two independent id allocators**, one counting up and one counting
down.

This replaces the speculative note in
[v22-v23-filename-tile-location-2026-09-19.md](v22-v23-filename-tile-location-2026-09-19.md), which
recorded the observation as an unmeasured operator suspicion.

## Measurement — v26 corpus, all 5,309 ACDO records

Field read as **int32** at `ACDO` +0x2C across all 700 files:

| | Count | Range | Distinct | Density |
|---|---|---|---|---|
| **Positive** | 4,096 (77.15%) | 63,418,942 … 63,423,379 | 4,096 | 92.3% of a 4,438 span |
| **Negative** | 1,213 (22.85%) | −1,233 … −2 | 1,213 | **98.5%** of a 1,232 span |
| **Total** | 5,309 | | **5,309** | **0 duplicates** |

Both populations are **dense, monotonic, sequential blocks**. That is the signature of two counters,
not of noise, sentinels or sign-bit corruption. Every one of the 5,309 ids is distinct, so both
allocators feed one namespace and the field's name is earned.

Note the negative block starts at **−2**, not −1 — consistent with −1 being reserved as
"none/invalid".

## They are not separable by file, record shape or model type

| Split | Negative | Positive |
|---|---|---|
| Files containing any | 9 | 13 |
| **Files containing both** | **4** | |
| Record size | 56 B × 1,213 | 56 B × 4,085, 60 B × 11 |
| Model kind | 1,208 M2, 5 WMO | 4,068 M2, 28 WMO |

Four tiles hold objects from **both** allocators, so this is not one tool versus another, nor one
authoring pass versus another at file granularity. All 11 of the 60-byte records (the ones with
trailing values, all WMOs) are positive.

## Reading: a local allocator alongside the world-DB one

The positive block sits at ~63.42 million — a plausible value for a shared, long-running uniqueID
service late in a project's life. The negative block counts **down from −2** in a tight, nearly
gapless run.

Counting down from zero into negatives is the standard shape of an id handed out **locally, without
consulting the shared service** — a placeholder for an object that was never committed to the world
database. The operator's original suspicion ("objects not meant to ship in the client or be tracked")
is supported by the shape of the data, though nothing here proves intent.

## Revision comparison

| Corpus | Revision | ACDO | Negative | Range |
|---|---|---|---|---|
| `test_data/v22_adts/unknown` (700 files) | v26 | 5,309 | **1,213** | −1,233 … 63,423,379 |
| Expansion01 (4 files) | v22 | 849 | **0** | 681,015 … 728,199 |
| IcecrownCitadel (3 files) | v23 | 0 | — | no `ACDO` at all |

**Negative ids are v26-only in everything held.** v22 uses a single positive allocator three orders of
magnitude lower (681K vs 63.4M), which fits v22 being far earlier in development. With one v22 sample
of four files, that is an observation, not a rule.

## Why the viewer displays it as signed

[`AdtAhdrObjectDefinition.UniqueId`](../../../../src/core/WowViewer.Core/Maps/AdtAhdr/AdtAhdrTile.cs) is
typed `uint`, so −210 is held as 4,294,967,086. The 2026-09-19 note predicted that a negative would
therefore "surface as a large unsigned value" — it does not, because
[`AhdrTerrainAdapter`](../../../../src/viewer/WoWViewer/Terrain/AhdrTerrainAdapter.cs) does
`unchecked((int)obj.UniqueId)` when building the placement, and `MddfPlacement.UniqueId` is `int`. The
Inspector reading `-210` is therefore **correct**, by way of a cast rather than by design.

The stored type is still the wrong shape for a field the data proves is signed.

## Consequence for DAT → LK ADT export (spec 247 US3)

`DatToLkAdtConverter` passes the id straight through as `unchecked((int)…)`. On disk, LK `MDDF`
uniqueId is a **uint32**, so a DAT −210 is written as **4,294,967,086**. That is legal and still
unique, but the "uncommitted object" signal is lost and the value is implausible as a real world id.

Not an issue for the current exports — the v22 corpus has no negative ids — but it will be the moment
a v26 tile is exported. Options, none taken here: pass through (current), remap the negative block
into a high positive range and record the mapping, or drop those placements. **This needs an operator
decision.** The export manifest now reports the count so it cannot pass unnoticed.

## Verification

| Action | Result |
|---|---|
| int32 read of `ACDO` +0x2C, 700 v26 files | 5,309 records, 1,213 negative, 0 duplicates |
| Density of each block | negatives 98.5%, positives 92.3% — both sequential |
| File / record-size / model-kind splits | 4 files contain both; no clean separator |
| Same read on v22 (4 files) and v23 (3 files) | v22: 849 records, 0 negative; v23: no `ACDO` |

**Not claimed**: why the two allocators exist, what the negative ids meant to the authoring tool, or
that the v22 result generalises beyond four files. No code behaviour was changed by this measurement
beyond the manifest count.
