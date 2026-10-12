# Tasks: MCAL Alpha Map Decode Correctness

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Created**: 2026-09-01

Phases are sequential. **Do not reorder them** — the ordering is the point, and plan.md
records why (baseline before consolidation; consolidation before deletion).

Legend: `[ ]` open · `[x]` done · `[-]` in progress · **(operator)** = user-run, agent
prepares the command and stops.

---

## Phase 1 — Instrument first, change no decode rule

Goal: turn "alpha looks wrong" into a number, before anything moves.

- [ ] **T001** Add `AdtAlphaEncoding`, `AdtAlphaDecodeRule`, `AdtAlphaDecodeFailure`,
      `AdtLayerDecodeOutcome`, `AdtChunkDecodeOutcome`, `AdtAlphaDecodeReport` per
      [data-model.md](data-model.md). Enforce contract C1 in the constructor — an outcome
      carrying both alpha and a failure, or neither, throws.
- [ ] **T002** Unit-test C1's invariant directly: assert that no valid outcome can carry
      manufactured alpha. This is the structural guard against the defect being removed.
- [ ] **T003** Wrap the **existing** decode paths so they emit outcomes without changing what
      they decode. Today's rules, today's results, now observable.
- [ ] **T004** Add `alpha-decode-sweep` to `WowViewer.Tool.Inspect`: walk a configured client
      root, decode every MCNK, aggregate an `AdtAlphaDecodeReport` per build, write it out.
- [ ] **T005** **(operator)** Run the sweep across 0.5.3, LK, Cata and MoP builds. Record the
      baseline: per-era rule counts, failure counts, and the fully-accounted-chunk rate.
      **This number is the gate for Phase 3** — consolidation must not make it worse.

**Phase 1 exit**: the current failure rate is measured, per era, from real data.

---

## Phase 2 — Establish the rule

- [-] **T101** Isolate the client MCAL consumer in Ghidra. **Two passes have failed; see
      [research.md](research.md) R5 for the ruled-out list — do not re-search the +0x124
      offset or the MapChunk/MapRenderChunkState functions already eliminated.** Remaining
      lead: the alpha texture appears to be created as a procedural/callback-filled texture,
      so follow the texture-creation calls in MapTexture.cpp (FUN_00bb1850, FUN_00bb1a50,
      FUN_00bb1b60) that pass a fill callback rather than a filename. **Time-box this.**
      R5 fallback exists so the spec is not blocked on it.
- [ ] **T102** Record the per-era rule in research.md: which encoding is selected from which
      flags. **If T101 does not resolve, record that as the finding** and proceed to T103 on
      file-side evidence, labelling the rule's provenance accordingly (FR-005).
- [ ] **T103** Express the established rule as `ResolveRule(...)` per contract C2 — era +
      flags in, rule out, no offsets, no payload length.
- [ ] **T104** Unit-test `ResolveRule` against the era matrix, including `doNotFixAlphaMap`
      set and unset, and the 0.5.3 no-alpha case (C6).

**Phase 2 exit**: the rule exists as code and its evidence is written down, including if that
evidence is "corpus accounting" rather than "the client does this".

---

## Phase 3 — One decoder

- [ ] **T201** Promote `AdtMcalDecoder` to the canonical owner: consume an
      `AdtAlphaDecodeRule`, return `AdtLayerDecodeOutcome`, never infer from spans.
- [ ] **T202** Delegate `Mcal.GetAlphaMapForLayer` / `…Relaxed` to it, or delete `Mcal` if it
      has no remaining role.
- [ ] **T203** Delete `DecodeLayerBySpan` and the second fallback loop from
      `StandardTerrainAdapter`; route the renderer through the canonical decoder.
- [ ] **T204** Delegate or delete `AlphaMapService.ReadBigAlpha`.
- [ ] **T205** Verify SC-001: a repository search for alternative MCAL decode routines returns
      only delegations.
- [ ] **T206** **(operator)** Re-run the sweep. Compare against T005. The fully-accounted rate
      must be **equal or better** per era; any regression stops the phase.

**Phase 3 exit**: one decoder, and it is measurably no worse than four.

---

## Phase 4 — Delete the fabrication

Blocked on T206. Do not start early to "fix the screenshot" — the blocks are currently the
only visible signal that decode fails.

- [ ] **T301** Delete `SynthesizeCataclysm400ResidualAlpha`.
- [ ] **T302** Reduce `StitchCataclysm400ChunkEdges` to operate only on decoded data, or
      delete it if it exists solely to blend synthesized blocks.
- [ ] **T303** Verify SC-003: no code path produces an alpha map the decoder did not read.
- [ ] **T304** **(operator)** Load Mogu'shan Palace and capture. SC-004: the blocky
      full-chunk patches are absent. Expect some layers to now be *missing* rather than wrong
      — that is the intended honest state, and T206's report says how many.
- [ ] **T305** **(operator)** Capture a 0.5.3 reference scene. SC-007: pixel-identical to
      before the change.

**Phase 4 exit**: nothing is invented, and the remaining decode gap is visible.

---

## Phase 5 — Harvest parity

- [ ] **T401** Remove the hardcoded `bigAlpha: false` at `VlmDatasetExporter.cs:1908`; take
      era inputs the way the renderer does (FR-006).
- [ ] **T402** Route the harvest through the canonical decoder.
- [ ] **T403** Test C4 directly: same chunk, same era inputs, byte-identical alpha from the
      renderer's call path and the harvest's call path.
- [ ] **T404** **(operator)** SC-005: harvest and render a sample of tiles on a big-alpha map;
      compare the arrays byte-for-byte.
- [ ] **T405** Enumerate which tiles in any existing harvested corpus would change under the
      canonical decoder, so the rebuild decision is the operator's (US4, and explicitly not a
      rebuild).

**Phase 5 exit**: harvested alpha and rendered alpha are the same bytes, and the cost of the
old corpus being wrong is a known list.

---

## Notes

- `AdtMcalDecodeProfile`, `AdtFormatProfile.BigAlphaFlagsMask`, `AdtMcalAlphaEncoding` and
  `AdtMcalSummary` already exist. This work fills them in; it does not build a parallel
  mechanism (research.md R6).
- Known test baseline: `WowViewer.Core.Tests` carries **9 pre-existing failures** unrelated to
  this feature. Compare against 9, not 0.
