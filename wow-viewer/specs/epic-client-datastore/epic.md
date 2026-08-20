# Epic: Client Datastore — one Zarr store, many builds, processed once

**Status**: specs complete, implementation not started
**Created**: 2026-08-19
**Member specs**: [179](../179-patch-chain-resolver/spec.md) · [180](../180-multi-build-datastore/spec.md) ·
[181](../181-incremental-processing/spec.md) · [182](../182-adaptive-encoding/spec.md) ·
[183](../183-datastore-viewer-load/spec.md)

**Background** (superseded draft, kept for rationale — **not** requirements):
[`background/165-unified-zarr-datastore.md`](background/165-unified-zarr-datastore.md).

---

## Read this first if you are starting cold

**The primary consumer is ML/AI, not the viewer.** This is a training corpus that a viewer can also
read — not the reverse. Where the two pull in different directions, the training surface wins. That
is why viewer loading (183) is last.

**The point is not saving disk. It is not redoing work.** Storage dedupe saves disk, which is cheap.
**Derivation dedupe saves harvest and processing, which is not.** If build B's ADT is byte-identical
to build A's already-processed ADT, every tensor derived from it is already computed and the correct
amount of work is **zero**. Across many same-era 0.x builds that share the overwhelming majority of
their content, that is the difference between processing every build and processing every build's
delta.

### The approach is already validated

A prototype was measured: **~160 GB of builds → ~40 GB of Zarr (~4:1)** — achieved *with* duplicate
copies per build-version and at Blosc **lz4 clevel-1, no shuffle specified**
(`data-harvester/src/harvester/v25/dataset.py:62`), close to the weakest setting available.

So 4:1 is the **floor**, not the target, and two gains remain unclaimed: cross-build deduplication
(180/181) removes the duplicate copies that 4:1 was measured *with*, and per-type encoding (182)
attacks the near-minimum compression.

### The builder resolves patches wrong, measurably

Two MPQ patch-priority implementations exist, and the dataset builder inherited the weaker one.

| Case | `MpqArchiveCatalog` (viewer) | `NativeMpqService` (**builder**, 8 call sites) |
|---|---|---|
| `patch.mpq`, `patch-N.mpq` | ✅ | ✅ |
| `patch-[a-z].mpq` | ✅ `500 + letter` | ❌ **all → 1099** |
| `patch-<locale>-N.mpq` | ✅ | ❌ **all → 1099** |
| Deletion markers | ✅ skipped + counted | ⚠️ skipped silently, no provenance |

`.OrderBy()` is a **stable** sort, so everything at 1099 keeps *filesystem enumeration order* — not
patch precedence, and not stable across machines.

**Measured on `H:\CLIENTS\WoW335\modernwow\`: 9 archives tie at 1099.**

| Class | Archives |
|---|---|
| Letter | `Patch-D.mpq`, `Patch-N.MPQ`, `patch-P.MPQ`, `Patch-T.mpq`, `Patch-Y.mpq` |
| Locale | `patch-enUS.MPQ`, `patch-enUS-2.MPQ`, `patch-enUS-3.MPQ`, `patch-enUS-3-orig.MPQ` |

`patch-enUS-3-orig.MPQ` is a **backup that parses as a live patch**. Archive admission must be a
recorded, overridable decision — never an incidental consequence of a filename glob.

*(Checked and cleared: the `*.mpq` + `*.MPQ` double-enumeration in `NativeMpqService` is de-duplicated
by `.Distinct(OrdinalIgnoreCase)`. Not a bug — do not "fix" it.)*

### The mechanism is one step from existing

`tools/harvest` **already computes `InputSha256`** and writes it into a manifest
(`Program.cs:642,689`) — and never compares it to anything. Input hashing exists here as a provenance
habit; it has never once been used to avoid work.

---

## Design decisions already made — do not relitigate

**Content-addressed storage, NOT a stored diff format.** A diff format needs a reconstruction path,
and a bug there corrupts every version downstream. Content-addressing gets every benefit *and* the
diffs, without the bug class: shared bytes stored once automatically, differences are set operations
over per-build manifests, every file reconstructs by direct lookup, content hashes self-verify. It is
**simpler** than what was originally asked for, not more complex.

**Nearest-build comparison is a scan optimization, not a dedupe requirement.** Content-addressing
matches by hash regardless of build proximity — content shared with a build five versions away
dedupes exactly as well as the immediate predecessor. Choosing the nearest processed build only
speeds up *finding* what changed. A wrong "nearest" guess must cost time, never a missed dedupe.

**Full decode is deliberate, and is a preservation requirement.** ADTs and the assets they reference
are decoded completely so the MPQ corpus survives as portable Zarr. A store of opaque blobs is only
as durable as its parser; a store of named typed arrays is readable years from now by anyone with
standard Zarr tooling. **Every array must open with no code from this repository.** This store is
intended as a node in a larger system, so the consumer may not be this project.

## The hazard that shapes 181

**A stale derivation cache in an ML corpus does not fail loudly — it silently trains on wrong data,
and surfaces months later as an unexplained result.** Two rules follow, and they are strict:

1. The reuse key is the **complete input set plus the processing version** — never one file's hash.
2. **If the input set cannot be enumerated with certainty, redo the work.** Uncertain reuse is
   prohibited outright.

Recomputing costs time. A wrong reuse costs trust in the corpus.

## Dependency chain

```
179 patch resolver ──> 180 multi-build store ──┬──> 181 incremental processing
                                               ├──> 182 adaptive encoding
                                               └──> 183 viewer load
```

179 must land first: everything downstream inherits whatever it resolves, correctly or not.

## Measured baselines

| Metric | Today | Target |
|---|---|---|
| Patch-priority implementations | 2 | 1 |
| Archives tied at rank 1099 on modernwow | 9 | 0 |
| Zarr codec defaults | 2 | 1 |
| Corpus compression | ~4:1 (160→40 GB), duplicated + lz4-1 | **beat 4:1** |
| Uses of `InputSha256` to avoid work | 0 | derivation reuse |
| Builds per store | 1 | many, add/remove cheaply |

## Hard constraints

1. **No Blizzard containers as output** (Constitution VII). MPQ is an input format only. There is no
   "export a build back to MPQ" path, now or later.
2. **The Zarr store is the only on-disk artifact** (Constitution V). The reuse index lives *inside*
   the datastore, never as a sidecar — a cache separable from the store it describes can go stale
   against it.
3. **Incremental output must equal from-scratch output.** Reuse may never change results.
4. **Real-data validation** (Constitution III) against `H:\CLIENTS`, ≥3 builds, with commands, build
   identity, and hashes recorded.

## Tracking

| # | Spec | Status | Gate |
|---|---|---|---|
| 179 | Canonical patch-chain resolver | Draft | 9 tied archives → 0; one implementation remains |
| 180 | Multi-build content-addressed store | Draft | Every file byte-identical to source; remove reclaims only unique content |
| 181 | Incremental processing | Draft | Second build costs work ∝ its unique content; output identical to from-scratch |
| 182 | Adaptive per-type encoding | Draft | Beats 4:1; ratio **and** decompression throughput reported per type |
| 183 | Viewer: Load Zarr Datastore | Draft | Same render as loading the original client; no extraction to disk |
