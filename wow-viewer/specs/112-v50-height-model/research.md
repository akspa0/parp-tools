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

## Decision 2 — RESOLVED: the 256/1024 gap was authored-minimap provenance leakage, not concurrency

**Confirmed mechanism (2026-07-18, after the T012 rebuild reproduced the gap)**: the v22 stream
profile serializes the AUTHORED client minimap from the MPQ under the key `minimap_rgb`
(`RawArraySerializer.cs` line 359). `_cmd_build` filled every declared signal found in the stream,
so each tile first received the authored minimap labeled `freshly_extracted`, and the synthesized
256px PNG overwrote it only where synthesis succeeded. On the 220 Kalimdor tiles (and 41 Azeroth
tiles) where synthesis legitimately skipped ("no referenced BLP texture could be decoded" —
reproduced bit-identically at both resolutions on a single-tile probe of Kalimdor (19,12)), the
authored imagery silently remained, while `minimap_rgb_1024` (which has no authored counterpart in
any client) stayed honestly empty. The lost-row pattern proved it before the code did: contiguous
map-edge runs (Kalimdor rows 0–28 etc.), which no race or memory-pressure mechanism produces, and
Kalidar showed zero gap because it has no authored minimaps at all.

**Fix**: `signal_takes_stream_data()` in `v50_build_dataset.py` — a signal whose
`authoritative_source` is `synthetic-minimap` never accepts stream data; a tile without a
synthesized PNG is unavailable at BOTH resolutions (the spec's US1 edge case verbatim), restoring
parity by construction and ending the authored/synthesized provenance mix inside the store.

**Open design question — RESOLVED by the user 2026-07-18 (see Decision 7).** Authored client
minimaps are the model's real deployment input; they are now a first-class signal
(`minimap_rgb_authored`) and the model trains on both sources.

### Superseded intermediate finding (kept for the record)

## Decision 2 (superseded) — initial race hypothesis DISPROVEN by code audit; diagnosis moved to an instrumented A/B run

**Finding (updated during implementation, 2026-07-18)**: The originally suspected mechanism — a
plain `Dictionary` in `NativeMpqService` mutated across `Parallel.ForEach` worker threads — does
not survive a full read of the class. Every mutation of `_scannedArchives`/`_archives`/
`_knownFileHashes`/`_hashToName` happens in load/scan methods (`LoadArchives`,
`ScanMapMpqArchives`, `LoadListfile*`) that run single-threaded at startup, before the tile loop;
the read paths (`FileExists`, `ReadFile`, `ReadFromScannedArchive`, `ReadFileFromArchive`) only
read those collections and open a fresh per-call `FileStream` with `FileShare.Read` — stateless
and safe under concurrent readers, in-process and cross-process alike. There is no dictionary race
to fix, and fixing one anyway would have been a phantom repair.

What remains true: 1024px coverage trails 256px on three of four maps (Kalimdor 0.76, Azeroth
0.92, PVPZone02 0.40 vs ~1.00) while Kalidar shows *no* gap (0.64 = 0.64, both limited by the same
20 genuinely texture-less tiles) — so the loss is real, resolution-correlated, and map-dependent.
Surviving candidate mechanisms, in rough order of plausibility: memory/allocation pressure in the
1024 process (16× the pixel buffers per tile across up to core-count concurrent workers, with
broad `catch` blocks in the decode path converting any failure into a "texture could not be
decoded" skip); cross-process I/O pressure from the two resolutions running simultaneously against
the same archives; or a genuinely resolution-dependent code path not yet identified.

**Decision**: Do not fix blind. `synthetic-minimap` gained `--synthesis-workers N` (default -1 =
unbounded, matching current behavior; 1 = fully sequential) so a bounded, user-run A/B matrix can
isolate the mechanism: (a) 1024 alone, default workers; (b) 1024 alone, workers=1; (c) 1024
concurrent with 256, default workers. Coverage equal in (a)/(b) but degraded in (c) implicates
cross-process pressure; degraded in (a) but clean in (b) implicates in-process parallelism; clean
everywhere implicates the pipeline-runner context specifically. The fix follows the measurement.
FR-004's acceptance (1024 row-set equals 256 row-set on rebuilt stores) governs regardless of
which mechanism is confirmed.

**Alternatives considered**: Synchronizing `NativeMpqService`'s collections (the original plan) —
rejected as demonstrably unnecessary after the audit. Serializing the two resolution passes in
`_cmd_build` unconditionally — deferred; it is the likely mitigation if (c) is the confirmed
mechanism, but adopting it before measurement would mask the cause instead of naming it.

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

## Decision 7 — authored + synthetic minimaps are BOTH used, as separate training rows (user-directed 2026-07-18)

**Finding**: Fixing the provenance leak (Decision 2) removed the authored client minimap from the
store entirely, leaving `minimap_rgb`/`minimap_rgb_1024` synthesized-only. The user flagged this as
a correctness problem, not a cleanup: a model that will ultimately decompile *real* minimaps
(screenshots, private-server exports, authored client tiles) must be trained on that real imagery,
or it only learns to invert our compositor's specific lighting/texture-averaging quirks — a domain
gap that makes the model useless on real input. "We need to use the originals if we are going to
train a model on fucking anything useful."

**Decision**:
- **Store**: the frozen catalog gains `minimap_rgb_authored` (harvest-stream sourced, uint8
  256×256×3, partial coverage, honestly unavailable where the client shipped no minimap BLP — never
  zero-substituted). `minimap_rgb`/`minimap_rgb_1024` are now documented synthesized-only. The v22
  stream already decodes the authored image (`TryLoadMinimapFromMpq`); `_cmd_build` captures it
  under the new key before the synthesis override, at zero re-harvest cost.
- **Curriculum**: each kept tile emits up to two rows — one per available minimap source — paired
  with the SAME height target and all the same auxiliary terrain signals. The curriculum's
  `minimap_rgb` column is the per-row model input (synthetic or authored); an `minimap_source`
  index column records which. Both rows share one `source_group_id` and the split is assigned per
  group, so a tile's rows can never straddle train/val (the leak-safety invariant the trainer
  rechecks). The 1024px synthetic upscaler target is excluded from this height curriculum (separate
  lane, no authored counterpart).
- **Model**: unchanged — it reads `minimap_rgb` per row and predicts relative height. The
  authored/synthetic distinction is data, not architecture; the model never sees the label. The
  training summary records the authored/synthetic row split for analysis.

**Alternatives considered**: authored-only (rejected — discards synthetic augmentation the user
wanted kept); two input channels in one row (rejected — the user explicitly chose separate rows,
and it also forces every tile to have both sources, which many don't); synthetic-only (the rejected
status quo that prompted this decision).
