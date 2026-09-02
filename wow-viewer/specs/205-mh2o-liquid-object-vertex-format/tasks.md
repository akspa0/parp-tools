# Tasks: MH2O LiquidObject Vertex-Format Resolution

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Research**: [research.md](research.md)
| **Created**: 2026-09-01

Diagnosis is complete and measured. These tasks are implementation.

Legend: `[ ]` open · `[x]` done · `[-]` in progress · **(operator)** = user-run.

---

## Phase 1 — Verify the chain before building on it (GATE)

- [ ] **T101** Confirm `LiquidObject.dbc` and `LiquidMaterial.dbc` exist in
      `C:\WoW4-data\MoPBeta` and record their row count and field count from the WDBC header.
- [ ] **T102** Confirm the field offsets for `LiquidObject.LiquidTypeID`,
      `LiquidType.MaterialID` and `LiquidMaterial.LVF` **against these DBCs**, not the wiki
      (research R7). `DbcLiquidTypeTable` currently reads only `Type` at `0x38`.
- [ ] **T103** Check the resolution against the corpus answer, which is already known: **id 42 must
      resolve to a depth-only format; ids 2325, 2333 and 2372 must resolve to height-bearing ones.**
      If the chain disagrees, the chain is being read wrong — stop and re-derive.

**Phase 1 exit**: the chain reproduces the measured populations. Nothing is built on an unverified
offset.

---

## Phase 2 — Resolve, and never silently flatten

- [ ] **T201** Add `DbcLiquidObjectTable` (id → `LiquidTypeID`) and `DbcLiquidMaterialTable`
      (id → `LVF`) beside the existing `DbcLiquidTypeTable`.
- [ ] **T202** Extend `DbcLiquidTypeTable` to expose `MaterialID`.
- [ ] **T203** In the **render-path** decoder `Mh2oChunk.Parse`, treat `liquid_object_or_lvf >= 42`
      as a LiquidObject id and resolve it through the chain (FR-001, FR-002). Values below 42 keep
      their current meaning exactly (FR-003).
- [ ] **T204** Add the `default` both decoders lack: an unresolved format is **counted, logged with
      its id, and the fallback stated** (FR-004). This is the change that stops the next such
      encoding from hiding.
- [ ] **T205** Degrade cleanly when a DBC is missing — current behaviour, reported, no crash
      (FR-009).
- [ ] **T206** **(operator)** Re-run `inspect adt liquid-formats --client "C:\WoW4-data\MoPBeta"
      --map HawaiiMainLand --limit 80`. SC-001: zero unresolved layers, against 17,461 today.

---

## Phase 3 — Read the heights that are being thrown away

- [ ] **T301** Decode per-vertex heights for height-bearing layers (FR-005).
- [ ] **T302** Leave depth-only layers flat at their declared level (FR-006). **Ocean is not a
      defect** — 17,317 of the 17,461 measured layers are ocean and they render correctly today.
- [ ] **T303** Unit-test against fixture payloads: one layer per resolved format, plus an
      unresolved id.
- [ ] **T304** **(operator)** SC-002: the 144 river layers decode with varying heights whose spreads
      match the measured 11.90 (id 2325), 70.15 (id 2333) and 163.25 (id 2372). SC-003: ocean
      levels unchanged.

---

## Phase 4 — One decoder

- [ ] **T401** Consolidate `Mh2oChunk` and `AdtLiquidReader` onto a single implementation
      (FR-007, Constitution II). Research R5: they are independent today and both were wrong.
- [ ] **T402** If two entry points must remain, enforce agreement by test rather than by intent.
- [ ] **T403** SC-005: both paths return identical heights for the same ADT.

---

## Phase 5 — Verify

- [ ] **T501** **(operator)** SC-004: fly a river crossing at least three chunk boundaries; no step
      at the seams.
- [ ] **T502** **(operator)** SC-006: 0.5.3, LK and Cata liquid unchanged.
- [ ] **T503** Retain the float-plausibility probe as a **reporting cross-check only** (research
      R6). It misclassified 18 of 6,194 ocean layers; a disagreement between it and the DBC answer
      is informative, but it must never be the decode path (FR-008).

---

## Notes

- **Fix the render-path decoder first.** `StandardTerrainAdapter` calls `Mh2oChunk.Parse`;
  `AdtLiquidReader` serves harvest and the converter. Fixing the wrong one produces a change with no
  visible effect — the same trap as `MdxRenderer` vs `M2Renderer` on 2026-09-01.
- **Harvested MoP liquid data has been wrong too**, not just the render. Any dataset built from MoP
  tiles carries flat rivers at incorrect heights.
- The verification command already exists:
  `tools/inspect/WowViewer.Tool.Inspect/AdtLiquidFormatSupport.cs`.
- All 17,461 measured layers are 8×8. Sub-rectangle handling (`x_offset`/`y_offset`) is **untested
  by this corpus** — keep the existing code, do not simplify it on evidence that cannot see it.
- Known test baseline: `WowViewer.Core.Tests` has **9 pre-existing failures**. Compare to 9.
- The viewer holds its exe open while running; close it before rebuilding or MSBuild fails with
  MSB3027.
