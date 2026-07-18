# Phase 0 Research: V50-Native Height-First Terrain Model

## Decision 1 — `mcnk_flags_16`'s zero coverage is a confirmed wiring gap, not an era limitation

**Finding**: `RawArraySerializer.cs` (the harvest-stream writer) unconditionally serializes
`pack.McnkFlags16` under the key `mcnk_flags_16` (lines 223/273/352). `AdtTensorPackBuilder.cs`
(the LK/retail-format builder) populates it via `ReadMcnkFlags` (line 1503) and assigns it on
construction (lines 200/405). **`AlphaTensorPackBuilder.cs`** — the builder used for every 0.5.3
Alpha-format tile, i.e. all of `0_5_3_3368` — reads per-chunk MCNK flags internally (`lc.McnkFlags`,
used only to resolve liquid type at lines 266/294) but never assigns them to the output
`TerrainTileTensorPack.McnkFlags16` property (constructed at line 122). The property stays `null`,
so every downstream consumer (including `v50_build_dataset.py`'s `_cmd_build`, which zero-fills any
signal absent from the harvest-stream blob) writes zeros.

**Decision**: Fix `AlphaTensorPackBuilder.cs` to assign the flags it already reads onto the output
pack's `McnkFlags16`, matching the shape/convention `ReadMcnkFlags` establishes for the LK path. No
new reader is written (constitution II) — the data is already being parsed, just discarded.

**Alternatives considered**: Reclassifying `mcnk_flags_16` as era-unavailable (FR-002's fallback
path) was considered and rejected — the flags are demonstrably present and already parsed in the
Alpha reader; declaring them unavailable would be documenting a bug as a limitation.

## Decision 2 — `minimap_rgb_1024` under-coverage is a suspected concurrent-archive-access race, not file locking

**Finding**: `NativeMpqService`'s file reads use `FileShare.Read` (confirmed at three call sites),
which permits concurrent cross-process reads — ruling out simple OS-level file locking as the cause
of the two parallel `synthetic-minimap` subprocesses (256px and 1024px, launched as separate
`Popen` calls in `_cmd_build`) interfering with each other. The more likely mechanism is *within* a
single `synthetic-minimap` process: `Program.cs`'s per-tile composition loop runs under
`Parallel.ForEach` (added in the Spec 109 Phase 8-era parallelization). `NativeMpqService` holds
several plain (non-concurrent) `Dictionary`/`HashSet` fields — notably `_scannedArchives`, mutated
by a lazy per-archive fallback scan inside the read path — accessed from every worker thread in that
`Parallel.ForEach` without visible synchronization. Concurrent mutation of a plain `Dictionary` from
multiple threads is a classic silent-corruption/intermittent-exception source, and 1024px synthesis
(more decode work per tile, wider concurrency window) would be expected to hit it more often than
256px — matching the observed 40–92% vs ~100% coverage gap.

**Decision**: Treat this as a hypothesis to confirm empirically in Phase 1, not a certainty to fix
blind. Implementation step: run 1024px-only synthesis for one map with `Parallel.ForEach` intact
versus a sequential fallback, diff the skip/failure counts. If the race is confirmed, the fix is
synchronizing (or making thread-local) `NativeMpqService`'s mutable scan-cache fields, not touching
the two-process design (which is unaffected by `FileShare.Read`).

**Alternatives considered**: Serializing the two resolutions back into one sequential pass would
trivially remove the suspected race but regress the wall-clock win Phase 8 delivered; rejected
unless the in-process fix proves intractable.

## Decision 3 — the manifest template must be *generated from*, not hand-synced with, the frozen catalog

**Finding**: `v50_configs/v50-manifest-template-0_5_3_3368.json` is currently a hand-authored file
that has drifted from `docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`'s frozen
signal table — it still declares `mddf_mask`, `modf_mask`, `object_filtered_mask`,
`model_focus_mask` (all explicitly marked dropped in that doc's "Dropped Signals" section) and
`mccv_rgb` (not listed for 0.5.3 in the doc's client-specific signal matrix at all — MCCV is a
WotLK+/4.x-era signal). Any future catalog edit will re-drift the same way unless the template is
mechanically derived.

**Decision**: A new script (`v50_generate_manifest_template.py`) parses the frozen catalog table
(the existing markdown doc, or a parallel machine-readable copy of it — see data-model.md) and
emits the manifest template, so the template can never declare a signal the catalog doesn't. The
catalog markdown stays the human-authored source of truth per constitution ("Spec Docs Are Source
of Truth"); the generator is the only writer of the JSON template.

**Alternatives considered**: Hand-editing the existing template to drop the four dead signals and
mccv_rgb was considered as a smaller diff, but rejected — it fixes today's drift without preventing
tomorrow's, which is the actual defect (a process gap, not just a data gap).

## Decision 4 — era-unavailability is a new, explicit `UnavailableSignal` reason, not a boolean

**Finding**: `harvester.v50.contracts.UnavailableSignal` currently carries a free-text `reason`
string with no enforced vocabulary; existing callers write ad-hoc messages ("no rows passed audit",
"missing in stream or synthesized files").

**Decision**: Introduce a small closed vocabulary of reason *prefixes* (`era_unavailable:`,
`no_source_data:`, `not_yet_extracted:`) so tooling (the coverage auditor, FR-002/FR-003) can
programmatically distinguish "this signal cannot exist for this build" from "this signal exists but
this tile lacks it" without parsing free text. Existing free-text reasons remain valid (no prefix
required); only the new corrective code paths adopt the vocabulary.

**Alternatives considered**: A new enum type was considered and rejected as unnecessarily invasive
to the existing `UnavailableSignal` contract (would touch every existing caller); a string
convention is additive and backward-compatible.

## Decision 5 — relative-height target: per-tile min-max normalization to `[0, 1]`, with an explicit degenerate-tile floor

**Finding**: The rejected `WdlPriorNet` target (`build_wdl_target`/`decode_wdl_target` in
`harvester/spec103/wdl_prior_model.py`) normalizes against a *global* constant range
(`HEIGHT_GLOBAL_MIN`/`HEIGHT_GLOBAL_MAX`), which is exactly what makes the target absolute — two
tiles with identical relief at different altitudes get different normalized values. A per-tile
min-max mapping (`(h - tile_min) / max(tile_max - tile_min, floor)`) makes the target strictly a
function of relief shape, satisfying FR-007 (constant per-tile offset invariance) by construction:
adding a constant to every height in a tile shifts `tile_min`/`tile_max` by the same constant and
leaves the normalized values unchanged.

**Decision**: Adopt per-tile min-max to `[0, 1]`, with a small floor (e.g. 1.0 world unit) on the
denominator so a genuinely flat tile normalizes to a well-defined constant (0.5) rather than
dividing by ~0. The target contract records `tile_min`/`tile_max` alongside the normalized field so
reconstruction (decode back to world-unit height) is exact, not approximate — this pair is the
"Relative-Height Target Contract" entity from spec.md.

**Alternatives considered**: Z-score (mean/std) normalization was considered; rejected because std
on a near-flat tile is a noisier denominator than range, and min-max keeps the target boundable to
`[0, 1]`, simplifying the decode contract and matching the existing WDL lattice's `[0, 1]`
convention (`decode_wdl_target` already clips to `[0, 1]` before rescaling).

## Decision 6 — model architecture: small CNN encoder-decoder, single residual output, no pretrained backbone commitment yet

**Finding**: The standing "time-to-signal over rigor" memory and constitution IV (residual chain,
no multi-task) both point toward the smallest model that produces a real signal first. The rejected
`WdlPriorNet` was already a from-scratch spatial CNN (not a pretrained backbone) at moderate
parameter count; its failure was the *target*, not obviously the architecture family.

**Decision**: Reuse the from-scratch small-CNN-encoder / spatial-decoder shape (minimap RGB in,
dense per-pixel or per-chunk height-field out) as the Phase 2 starting point, retargeted to the
relative-height contract from Decision 5, sized to train in minutes on the available 16 GB GPU on a
~600-tile corpus. No DepthAnything-family or other large pretrained backbone (standing memory).
Exact layer counts/widths are an implementation-time tuning detail, not a planning decision — the
contract (Decision 5) and the one-signal constraint (constitution IV) are what this plan fixes.

**Alternatives considered**: A pretrained ImageNet backbone (as `WdlPriorNet`'s RGB normalization
constants hint it may have partially assumed) was considered; deferred rather than rejected outright
— if the from-scratch model underperforms, swapping the encoder is a contained follow-up that
doesn't reopen the target contract.
