# Plan — Epic 253 PM4/PD4 Navmesh Decoding, Matching & Generation

**Status**: Implementation approach **not yet selected** (operator directive 2026-09-23: triage first).

## Dependencies

```text
P-02 field map ──> P-03 terminology ──> P-04 grouping readout
P-01 decode ──> P-07 negative-BSP matching
            └─> P-08 dataset
P-05 disambiguation ──> P-06 rotation (needs a corpus with differing rotations)
P-09 generation (independent; PD4 first)
```

185, 188 and 189 were one research thread split by date; P-02–P-04 are its merged residue.

## Design documents adopted by reference

| Item | Adopted design |
|---|---|
| P-01 | [130 plan](../archived/130-pm4-remaining-decode/plan.md) · [research](../archived/130-pm4-remaining-decode/research.md) · [data-model](../archived/130-pm4-remaining-decode/data-model.md) |
| P-02–P-04 | [189 spec](../archived/189-pm4-complete-field-map/spec.md) · [188 spec](../archived/188-pm4-field-semantics/spec.md) · [185 spec](../archived/185-pm4-pd4-format-documentation/spec.md) |
| P-05, P-06 | [065 plan](../archived/065-pm4-correlation-to-world-assets/plan.md) |
| P-07, P-08 | [128 spec](../archived/128-pm4-negative-bsp-matching/spec.md) · [129 spec](../archived/129-pm4-zarr-dataset/spec.md) · [old PM4 epic](../archived/epic-pm4-restoration/epic.md) |
| P-09 | [184 spec](../archived/184-pm4-generation-from-geometry/spec.md) |
| P-10 | [149 plan](../archived/149-pm4-region-audio-controls/plan.md) |

## Standing rules (memory-bank feedback)

A named field is unexamined until measured; pair purity with distinctness; prove a detector can see
the effect before reporting a null. Python owns any Zarr storage (P-08); C# emits blobs.
