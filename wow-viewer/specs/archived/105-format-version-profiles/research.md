# Phase 0 Research: Unified M2 Format Version Profiles

**Feature**: 105-format-version-profiles | **Date**: 2026-07-15

Resolves every NEEDS CLARIFICATION in `plan.md` Technical Context. Each decision states what was
chosen, why, and what was rejected.

---

## R1. The sequence record is misread today (NEW — found during planning)

**This was not known when the spec was written. It converts FR-005a from "a design gap" into "a live bug."**

Ghidra `FUN_0070f960` computes sequence time as:

```c
seq   = animId * 0x44 + *(int*)(data + 0x20);   // sequences array
start = *(int*)(seq + 4);
end   = *(int*)(seq + 8);
if ((*(byte*)(seq + 0x10) & 1) == 0)           // flags@0x10 bit0 clear => looping
    t = start + ((elapsed + offset) % (end - start));
```

So 1.0.0's `M2Sequence` (stride `0x44`) is:
`{id u16@0x00, variationIndex u16@0x02, start u32@0x04, end u32@0x08, movespeed f32@0x0C, flags u32@0x10, …}`

Our readers do this instead:

| File | Stride | `+0x04` | `+0x08` | `+0x0C` | `+0x10` | Verdict |
|---|---|---|---|---|---|---|
| `M2ModelReader` (3.x/4.x) | `0x40` | duration | moveSpeed | flags | frequency | **correct** (Wrath) |
| `M2Era100ModelReader` | `0x44` | duration | moveSpeed | flags | frequency | **WRONG** — every field from `+0x04` is shifted 4 bytes |
| `M2Era1121ModelReader` | `0x6C` | duration | moveSpeed | flags | frequency | **SUSPECT** — same Wrath offsets, non-Wrath stride |

The strides were corrected per era; the **field offsets inside the record were copied from Wrath and never revisited**. `M2Era100Constants.SequenceStride` even carries the comment *"0x44 — 68 B, same as 1.12.1 V100"*, recording that the size was checked while the interior was not.

**Independent corroboration**: wowdev.wiki documents `M2Sequence` as carrying `startTimestamp`/`endTimestamp` for `≤ BC` and a single merged `duration` from Wrath onward. The Ghidra reading and the public documentation agree. This is not a novel claim.

**This explains the user's "frame 2030/3333, static" report.** With `duration` actually reading `start`, a
later sequence's global-timeline start offset is displayed as a duration. The number was never a duration.

- **Decision**: Treat the sequence record layout as era-dependent data, fix era-100 against the Ghidra evidence, and mark era-1121 **provisional** (FR-011) until a 1.12.1 client is traced.
- **Rationale**: The evidence is direct for 1.0.0 and only inferential for 1.12.1. Per FR-011 and the project's history of guessed-layout bugs, an inference must be labelled, not silently applied.
- **Rejected**: Fixing era-1121 by analogy. 1.0.0 and 1.12.1 already proved they differ despite sharing version `0x100` — analogy is exactly the reasoning that produced this bug.

---

## R2. Sequence time base — normalize to `{Start, Duration}`

**Decision**: `M2SequenceDefinition` carries **`Start` and `Duration`**. Era readers populate it:

- **Wrath+**: `Start = 0`, `Duration = duration@0x04`
- **1.0.0 (≤BC)**: `Start = start@0x04`, `Duration = end@0x08 - start@0x04`

The sampler then uses one rule for every era:

```
sampleTime = Start + (elapsed mod Duration)
```

**Rationale**: This is the decisive design choice of the feature, because it closes the era gap **without branching in the sampler**. For Wrath, `Start = 0` reduces the expression to `elapsed mod Duration` — **byte-identical to today's `ResolveSampleTime`**. The 3.x/4.x no-regression gate (FR-015) is therefore satisfied *by construction*, not by testing luck. `Duration` also remains the honest meaning of the field in both eras (`end - start` **is** the duration), so no caller reading `.Duration` changes behaviour or meaning.

**Rejected**:
- *An era enum on the sampler branching between two time rules.* Adds a branch on the hot path and a second thing to keep in sync, to express something two data fields already express. Violates SC-004's shrink bar in spirit.
- *Storing raw `end` and computing per-sample.* Leaks the era's encoding into every consumer.
- *Leaving `Duration` alone and adding `Start` as optional.* `Duration` is currently **wrong** for era-100 (it holds `start`), so "leaving it alone" preserves the bug.

---

## R3. Track addressing mode — discriminated on the track, set by the reader

**Decision**: `M2TrackDefinition<T>` gains an explicit addressing mode plus an interpolation-range reference:

- `Nested` (Wrath+): `TimestampArray`/`ValueArray` are outer arrays of `{count,offset}` refs indexed by sequence. Today's only behaviour.
- `FlatWithRanges` (1.0.0): `TimestampArray`/`ValueArray` address one flat array each; `InterpolationRanges` slices them per sequence with inclusive `{first,last}` bounds.

The reader sets the mode from the resolved profile. The sampler switches on the mode, never on bytes (FR-002).

**Rationale**: The two eras use different *addressing*, not different strides — no offset arithmetic bridges them, so the distinction must be represented, not computed. Putting the mode on the track (rather than threading an era through every call) keeps the sampler's signature unchanged and keeps each track self-describing.

**Rejected**:
- *Separate `M2Era100TrackDefinition` type.* Forks the sampler and every consumer; grows code (SC-004).
- *Normalizing 1.0.0 into the nested form at read time* (synthesizing per-sequence arrays). Tempting — it would need no sampler change at all — but it **materializes N copies of slice metadata** and, worse, it cannot reproduce the client's documented cross-range clamp (spec Edge Cases / FR-004), because that behaviour depends on the flat array's *total* count, which the nested form structurally discards. Rejected on fidelity, not cost.

---

## R4. Deterministic era resolution without core → viewer reference

**The tension**: FR-012 requires killing the trial-parse. Build context would disambiguate 1.0.0 from 1.12.1 (both report `0x100`), but it lives viewer-side, and Constitution I forbids core referencing the viewer.

**Decision**: **Dependency inversion.** Core defines the profile contract and an optional build hint on its read API. The viewer (which already has `_dbcBuild`) supplies the hint. Core never references the viewer; the viewer depends on core, as it already does.

Resolution order:

1. **Explicit build hint**, when supplied → deterministic.
2. **Structural discriminator**, when absent → deterministic *and evidence-backed*: the 1.0.0 header carries one more array across `0x74–0xAC` than v256, so the two eras' header sizes differ by 8 bytes, observable as a different first-data offset. **This must be confirmed against both a real 1.0.0 and a real 1.12.1 model before it is relied on** — it is currently reasoned, not measured. Marked provisional until then.
3. **Neither resolves** → fail loudly, naming the version and what was ambiguous (FR-013).

**Why this is not the trial-parse renamed**: a trial-parse *attempts a full parse and infers from failure* — it fails open, and misroutes whenever the wrong layout happens to produce plausible offsets. A structural discriminator *reads one declared field and decides* — it fails closed. The difference is that step 3 exists.

**Rejected**:
- *Keep the trial-parse.* FR-012; and it is a latent silent-misroute hazard.
- *Move build detection into core.* Core would need archive/DBC awareness to identify a build — a much larger dependency inversion than passing a string the caller already has.
- *Make the hint mandatory.* Breaks every existing core caller and the focused tests, for a case the discriminator can usually handle.

---

## R5. Where the profile lives

**Decision**: `WowViewer.Core` (contract types) and `WowViewer.Core.IO` (resolution + the era table). `FormatProfileRegistry` stays in the viewer, owning ADT/WMO/MDX only; its `M2Profile` records and `ResolveModelProfile` are deleted.

**Rationale**: Constitution II ("one canonical owner per format surface") is satisfied *per surface*: after this change the M2 surface has exactly one owner in core, where the readers can reach it. Constitution I is satisfied because nothing in core points outward.

**Rejected**:
- *Migrate ADT/WMO/MDX too.* FR-018/FR-020 — user-decided out of scope; those records carry real varying values inside the Terrain Alpha Risk Area.
- *Keep M2 profiles in the viewer and have core call up.* Constitution I violation, and it is the exact structure being removed.

---

## R6. Whether the shrink bar (SC-004) is actually achievable

Checked explicitly, because the plan is invalid if the bar cannot be met.

| Change | Δ lines (est.) |
|---|---|
| Delete 5 inert `M2Profile` records + `ModelRootMagic` + `ResolveModelProfile` | **−95** |
| Delete `ValidateModelProfile` M2-stride validation in `WarcraftNetM2Adapter`¹ | **−30** |
| Delete trial-parse `ValidateLayout` call path in `DetectEra` | **−15** |
| Add era profile table + contract | +60 |
| Add interpolation-range reference + addressing mode | +25 |
| Add `Start` to `M2SequenceDefinition` | +5 |
| **Net** | **≈ −50** |

¹ **Constraint check**: FR-017 says `WarcraftNetM2Adapter` is untouched. Deleting the M2 profile records it validates against *does* touch it. **Resolution**: FR-017's intent is "do not restructure the legacy MDX fallback path." Removing a validation call that checks against records proven inert (identical strides everywhere — it can only ever pass) is not a restructure and cannot change behaviour. **Flagged for user confirmation in Phase 1 rather than assumed** — if the user reads FR-017 strictly, the adapter keeps a local copy of the records and the net becomes ≈ −20, still a shrink.

**Conclusion**: SC-004 is achievable under either reading. The bar holds.

---

## R7. Baseline capture ordering

**Decision**: Phase 0 captures the 3.x/4.x baseline **before any shared type is touched**, into a committed artifact.

**Rationale**: FR-016. `M2TrackDefinition` and `M2SequenceDefinition` are shared by the working 3.x/4.x path. A baseline captured after they change measures the new code against itself and proves nothing. This ordering is load-bearing and is why baseline capture is its own phase with its own gate rather than a step inside Phase 1.

**Rejected**: *Rely on the existing focused tests.* `M2Era1121ModelReaderTests` (9/9) never touched the era-100 path — the memory bank records that those "passing" tests gave false confidence for exactly this area. Tests that pass today do not pin sampled animation values across 3.x models.

---

## Open items carried to Phase 1

- **R4 step 2** (header-size discriminator) is reasoned, not measured. Must be confirmed against real 1.0.0 and 1.12.1 models; until then the build hint is the only proven path and the discriminator ships marked provisional.
- **R1 era-1121 sequence offsets** are suspect-by-inference only. Needs a 1.12.1 client in Ghidra. Ships marked provisional; **not** fixed by analogy.
- **R6 FR-017 interpretation** needs a one-line user confirmation.
