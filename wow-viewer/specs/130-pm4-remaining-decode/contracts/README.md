# Contracts: PM4 Remaining Decode

**Feature**: [../spec.md](../spec.md) | **Plan**: [../plan.md](../plan.md)

This feature has no network API. Its contracts are **on-disk JSON artifacts and CLI surfaces** —
the boundary across which Spec 129 (zarr dataset), Spec 128 (matching), and the viewer consume this
decode without re-deriving it (FR-011).

| contract | what it fixes | consumed by | phase |
|---|---|---|---|
| [evidence-register.md](./evidence-register.md) | how a finding, its confidence, and its eliminations are recorded | every later phase; humans | 2 |
| [grouping-rule.md](./grouping-rule.md) | the rule interface and the evaluation report shape | Phase 4 rules | 3 |
| [object-identity.md](./object-identity.md) | **the per-surface object assignment table** | Spec 129, Spec 128, viewer | 5 |
| [geometry-stream.md](./geometry-stream.md) | connective-geometry interpretations (MSPV/MSPI **and** MSCN) and their discriminator | Phase 8 | 7 |
| [cli-commands.md](./cli-commands.md) | new `pm4` subcommands, flags, exit codes | operators, CI | 2–9 |

Phase 1 (prior-art harvest) produces `../prior-art-inventory.md`, not a contract — it makes no
claims and nothing consumes it programmatically.

## Stability policy

**`object-identity.md` is the load-bearing one.** Spec 129's row layout is object-primary, so a
change to `Pm4ObjectId` or to the assignment table's shape invalidates a built dataset. It carries a
`schemaVersion` and changes to it are breaking changes requiring a version bump and a note in the
epic.

The other four are research artifacts. They may gain fields freely; removing or repurposing a field
requires a version bump.

## Rules that apply to every contract here

1. **A number without a corpus behind it is not publishable.** Every fits/misses pair carries
   `fileCount`. A `fileCount` of 1 is legal to record and never sufficient to conclude from (FR-002)
   — tile 0_0 is the standing example.
2. **Confidence travels with the claim** (FR-007). No artifact states an interpretation as a bare
   fact. `confidence` is a required field wherever an interpretation appears.
3. **Eliminations accumulate and are never deleted** (FR-008). Merging a new run into an existing
   register adds eliminations; it never drops them.
4. **Determinism.** Every artifact is deterministically ordered and must re-serialize byte-identical
   on an unchanged corpus.
5. **`corpusSignature` on every document.** `<directory>@<fileCount>` plus a content hash, so
   evidence gathered on one corpus is never silently compared against another.

## Serialization

`System.Text.Json` with `WriteIndented = true`, matching every existing `pm4` report command
(`RunPm4Unknowns`, `RunPm4Mshd`, `RunPm4CrossTile`). Enums serialize as their string names, not
integers — an evidence register whose statuses are `0`/`3`/`4` is unreadable to the humans it is for.
