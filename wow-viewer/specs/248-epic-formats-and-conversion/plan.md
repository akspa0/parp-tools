# Plan — Epic 248 Formats, Readers, Writers & Conversion

**Status**: Implementation approach deliberately **not yet selected** (operator directive 2026-09-23:
flag wanted/unwanted first, then choose the approach). This plan records only facts that constrain
any approach.

## Dependencies between backlog items

```text
F-05 survey library ──┬─> F-04 chunk survey ──> F-03 modern→legacy conversion
                      └─> F-10/F-11/F-12 conformance fixes
F-06 MAI2 flow ───────────> F-03 (conversion report integration, 244 FR-010)
F-07 CDN streaming ───────> F-09 tier A/B coverage
F-01 AMAP codec (independent) · F-02 capture (independent)
F-20 converter unification ──> F-28 converter harness (validator targets one converter)
F-23 profile unification ──> F-24 animation addressing
```

## Design documents adopted by reference

These archived documents remain the design detail for their items. They are authority **only** for
the item that adopts them here.

| Item | Adopted design |
|---|---|
| F-03 | [243 plan](../archived/243-modern-to-legacy-map-conversion/plan.md) · [data-model](../archived/243-modern-to-legacy-map-conversion/data-model.md) · [contracts](../archived/243-modern-to-legacy-map-conversion/contracts/) · [research](../archived/243-modern-to-legacy-map-conversion/research.md) |
| F-01, F-02 | [247 spec](../archived/247-dat-capture-and-adt-export/spec.md); [DAT format doc](../../docs/architecture/adt-v26-format.md) |
| F-04 | [245 scoping note](../archived/245-modern-chunk-completeness-survey/evidence/modern-write-support-state-2026-09-20.md) |
| F-05, F-10–F-12 | [240 plan](../archived/240-format-conformance/plan.md) · [research](../archived/240-format-conformance/research.md) |
| F-07–F-09 | [238 plan](../archived/238-casc-data-source/plan.md) · [239 plan](../archived/239-modern-client-assets/plan.md) |
| F-20–F-24 | [235 plan](../archived/235-legacy-mdx-m2-rendering/plan.md) · [105 research](../archived/105-format-version-profiles/research.md) |
| F-27 | [209 convergence evidence](../archived/209-wlw-mclq-convergence/evidence/) |
| F-28 | [221 Phase 0 baseline](../archived/221-converter-validation-harness/evidence/phase0-baseline.md) |
| F-29 | [197 plan](../archived/197-workspace-profiles-editor-and-mop-adt-pipeline/plan.md) · [Ghidra 5.0.1 research](../archived/197-workspace-profiles-editor-and-mop-adt-pipeline/research-ghidra-5.0.1.md) |
| F-30 | [199 plan](../archived/199-mcal-decode-correctness/plan.md) |

## Standing constraints

- Reader freeze (AGENTS.md §4). F-25 and F-30 touch reader behaviour: each needs its verified-bug
  evidence recorded before code changes.
- One owned service per feature (AGENTS.md §10). F-03 is specified as `ModernToLegacyMapConversionService`
  in 243's plan, surfaced in the CLI and the Editor.
- Receipts per §9.2. Client-load and visual proof are operator-owned.
