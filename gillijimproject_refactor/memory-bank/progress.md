# Progress

## Spec Consolidation (2026-06-14)
- Archived dead/done: 005 (legacy PM4), 020 (subsumed by 056), 026 (no tasks), 036 (superseded by 056), 059 (Cata M2 done)
- Marked research complete: 030/031/032/038/040 → consumed by 056
- Fixed stale status: 025→Complete, 060→Complete, 043→stale tasks noted

## Direction Change (2026-06-14)
- Replaced `wow-engine-modernization-plan-2026-05-14.md` (509 lines, engine program framing) with 35-line viewer-first + UE bridge doc
- Updated 28 references across AGENTS.md, architecture docs, memory bank, and 10 specs

## 046 PM4 Asset Matching (2026-06-08 checkpoint)
- **26/42 tasks done.** All C# lib landed in `Core.PM4/Matching/`: Pm4ObjectSegmentBuilder, Pm4SegmentSignalExtractor, Pm4AssetMatchScorer, Pm4ReplacementPlacementSynthesizer, Pm4SegmentExportService. 4 inspect CLI commands live. 14 tests pass. 7 smoke proofs in output/tmp/.
- **16 remaining**: Python/Zarr tooling, schema validation, known-tile validation, polish/doc, viewer TypeFlags filter
- Biggest gap: Python/Zarr signal-store lane completely unimplemented

## V18 Focused Terrain Training (2026-06-06 checkpoint)
- Spec 047 rewritten as final V18 terrain system design. Focused on 0_5_3_3368 + 3_3_5_12340, minimap RGB → height/normal.
- Curation manifest: 6763 audited → 4096 kept. Tiny manifest (21 rows) available.
- Tooling landed: early-stop patience (8), rotating bucket fractions (0.10), roof-aware terrain masks, loader-pressure guardrail, strict build balancing, offline-supervised-eval vs deployment-proof split.
- **Next**: launch full height + normal training runs, then infer_v18_focus.py on checkpoints.

## Viewer UI (2026-06-10/12)
- **044** dockable shell: 8/8 P1 tasks verified landed. US4 (cursor-as-model) deferred P2.
- **060** UI cleanup: 20/20 complete. Runtime Stats dedup (5→1), status bar cleanup, capture UI-hide default, SceneInspector removal. ImGui migration note saved.
- **049** UI consolidation: categorized Tools menu, floating window extraction, sidebar consolidation — defined, not yet started.

## M2 / Format Work (2026-06-05/11)
- **048** 1.12.1 era-aware MD20 reader: Ghidra traced, implemented, 7 tests pass on real Bear.mdx. Archived.
- **059** Cata v0x109 M2: era tag + dispatch + tests, done. Archived 2026-06-14.
- **057** Archive catalog scanner: WoWArchiveCatalog.Scan() done. 10 tests pass.
- **058** PM4 Scene Graph panel: dockable panel, type-bucketed tree, 82% done.
- **043** Chunked MDLX reader: foundational code landed for 0.5.3/0.7.0/0.8.0. Tasks.md stale (predates 1.12.1→MD20 discovery).

## PM4 Research (2026-06-08/09)
- Segment builder rewritten: CK24+TypeFlags grouping, 4110→18 segments on dev tile
- Scorer rewritten: type-profile matching (typed overlap 35% + type profile 15% + shape 50%)
- TypeFlags color mode + CK24Type-vs-TypeFlags mismatch mode + 3px neon lines
- PM4 Info docked panel, Export Report button, tool window extraction
- MSCN/MSPV = collision containment (not navmesh). Trees → per-branch MSCN, WMOs → ~50% containment walls, M2 → top-of-model only.

## Previous Landmarks
- Platform-independent path normalization merged (2026-06-02)
- M2 3.0.1 embedded-profile no-draw fix (layer-0 missing-texture fallback)
- 14 pre-existing test failures in Core.Tests (stale ChunkedFileReader fixtures)
