# Implementation Plan: Rosetta Calibration Corpus for PM4 Object Identification

**Feature Branch**: `190-rosetta-calibration-corpus`  
**Status**: In Progress  
**Spec**: [spec.md](spec.md)

## Summary
The Rosetta Calibration Corpus builds a synthetic, offline-only labeled reference tileset that places every enumerable client asset at known grid coordinates on an ADT canvas with terrain text labels and comprehensive manifests. This reference library enables deterministic PM4 geometry lookup and companion ADT synthesis.

This plan details the implementation across 4 phases:
- **Phase 1 (US1)**: Labelled Rosetta Tileset Generator (MCAL multi-texture painting, pedestal heightfields, full-grid Alpha WDT up to 4096 tiles, LK ADT/WDT/WDL).
- **Phase 2 (US2)**: Labelled Reference Library Builder (decode synthetic tiles back through `Pm4ObjectSegmentBuilder`, extract signature features, verify $\ge 99\%$ self-test accuracy).
- **Phase 3 (US3)**: Deterministic PM4 Object Lookup Engine (replace heuristic matching with exact reference library search).
- **Phase 4 (US4)**: Companion ADT Synthesizer for Orphan PM4 Tiles (generate minimal compliant companion ADTs for tiles lacking `_obj0.adt`).

---

## Technical Architecture

```
                       ┌────────────────────────┐
                       │ Client Data Discovery  │ (0.5.3 Alpha or 3.3.5+ LK)
                       └───────────┬────────────┘
                                   │ Enumerates M2/MDX + WMO
                                   ▼
                       ┌────────────────────────┐
                       │ RosettaTilesetGenerator│
                       │ - Designkit grouping   │
                       │ - Pedestal mesh (MCVT) │
                       │ - MCAL / MCLY text     │
                       └─────┬────────────┬─────┘
                             │            │
             (Alpha Monolithic)          (LK Standalone)
                             ▼            ▼
                     ┌──────────────┐   ┌──────────────┐
                     │AlphaWdtWriter│   │ LkAdtWriter  │
                     │(up to 4096 T)│   │ + LkWdtWriter│
                     └───────┬──────┘   └──────┬───────┘
                             │                 │
                             └────────┬────────┘
                                      │
                                      ▼
                       ┌────────────────────────┐
                       │ Decoded Synthetic ADTs │
                       └──────────────┬─────────┘
                                      │
                                      ▼
                       ┌────────────────────────┐
                       │Pm4ObjectSegmentBuilder │ (Standard PM4 Pipeline)
                       └──────────────┬─────────┘
                                      │
                                      ▼
                       ┌────────────────────────┐
                       │   ReferenceLibrary     │ (Ground Truth DB)
                       └──────────────┬─────────┘
                                      │
             ┌────────────────────────┴────────────────────────┐
             ▼                                                 ▼
   ┌───────────────────────┐                         ┌───────────────────────┐
   │ Real PM4 Lookup (US3) │                         │ Companion Synth (US4) │
   └───────────────────────┘                         └───────────────────────┘
```

---

## Phases & Deliverables

### Phase 1: Generator Enhancements & Alpha WDT Full-Map Support (Completed)
- [x] Analytical font rendering and quincunx lattice antialiasing.
- [x] Client era auto-detection (`.mdx`/`.mdl` vs `.m2`).
- [x] Multi-texture layer painting via `RosettaAlphaPainter` (`MCLY` + `MCAL` 4-bit uncompressed nibble maps, 1024×1024 resolution) for sharp labels in Alpha WDT and LK formats.
- [x] Museum pedestal mesh generation in `MCVT` heightmap (flat raised plateau with beveled slope).
- [x] Remove 512-tile Alpha WDT limitation, allowing continent-scale maps up to 4096 tiles.
- [x] Synthetic minimap model/WMO pins render at the object band center used by the generated placements.
- [x] Complete focused test coverage in `RosettaTilesetGeneratorTests.cs`.

### Phase 2: Reference Library Builder (US2)
- [ ] Pipeline adapter decoding synthetic Rosetta tiles back into standard `AdtPlacementCatalog`.
- [ ] Signature extraction for 100% of placed objects.
- [ ] `ReferenceLibrary` model and serializer.
- [ ] Automated self-test runner asserting $\ge 99\%$ top-1 identification accuracy.

### Phase 3: Deterministic PM4 Lookup Engine (US3)
- [ ] Lookup engine querying `ReferenceLibrary` for real decoded PM4 segments.
- [ ] Tri-state result classification: `Identified`, `Ambiguous`, `NoReference`.
- [ ] Integration with Spec 176 Reconciliation workbench.

### Phase 4: Companion ADT Synthesizer (US4)
- [ ] Scan PM4 tiles lacking companion `_obj0.adt`.
- [ ] Synthesize minimal valid companion ADTs with provenance tracking.
