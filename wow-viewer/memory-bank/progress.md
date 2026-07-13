# Progress — wow-viewer

Last updated: 2026-07-13

## 2026-07-13 — Pivot to Spec 103 (revive v7); image-only law established

- **New governing law** captured in Spec 103: input is one image, every signal is generated from it, no
  ground-truth signal at inference, validation is label-free. This resolves the long-running "missing signals"
  problem — the model generates missing signals; it never assumes them.
- **Direction:** revive the real single-model **v7** (`MultiChannelUNetV7`, no stages) on the current clean
  signals, synthetic-first to de-risk. Quick-and-dirty (no object-mask loss gating; WDL-prior dropout).
  Spec 103 = spec.md + plan.md + tasks.md + checklist (all via Spec Kit).
- **V24 / Spec 094 dropped** as non-functional. `wdl_height_33` prohibited; the WDL prior is the verified
  `height257[::16]` / `[8::16]` transform, derivable from existing `height_257` (no reharvest).
- **Spec 102 M0 object-mask work paused/superseded** but preserved: full-stack simple M0 trainer + image-only
  inference committed; strict fragment-trace tests 42/42 green. Not the active path.

## Key facts for the next session

- v7 source (read-only ref): `gillijimproject_refactor/src/WoWMapConverter/scripts/{v7_model,train_v7,v7_losses,infer_v7}.py` (added 2026-04-14; V7.7 detail head 2026-04-19).
- v7 input = 13 ch: minimap RGB (0-2), normal RGB (3-5), WDL prior (6, the trestle base), aux (7-12). Output = height.
- Existing V18 store `output/datasets/v18/3_3_5_12340.zarr` has `minimap_rgb` + `height_257` + `normal_xyz` (no WDL arrays — derive the prior).
- Next agent work (no training): Spec 103 Phase 2 — pin v7's exact contract from `train_v7.py`, port the model + 13-channel assembler into `spec103/`, CPU sanity. Then the USER runs synthetic capture + training.

## Durable boundaries

- `gillijimproject_refactor` read-only (port from, never edit). C# WDL reader + AlphaWdtWriter frozen.
- The USER runs all training/capture/heavy jobs (AGENTS RULE 0). Staged clients only; never `H:\CLIENTS`.
- Older M0 strict-target detail: `memory-bank/archive/2026-07-13-spec102-strict-target-detail.md`.
