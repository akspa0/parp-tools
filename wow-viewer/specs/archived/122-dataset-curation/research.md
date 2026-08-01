# Research: Canonical Dataset Curation and Signal-Mismatch Bucketing

## Part A — Design Decisions

### D-01: C# project placement and CLI surface

**Decision**: New library `wow-viewer/src/core/WowViewer.Core.Curation`, exposed via a new
`curate` subcommand on the existing `WowViewer.Tool.Harvest` (`tools/harvest/WowViewer.Tool.Harvest`).

**Rationale**: Constitution II (Library-First) requires one canonical owner per capability, with
CLI tools as thin wrappers. Curation is neither a format reader/writer (owned by
`WowViewer.Core.IO`) nor a general primitive (`WowViewer.Core`) — it is a derived-analysis layer
that reads already-decoded `TerrainTileTensorPack` output and classifies it. Giving it its own
library keeps `WowViewer.Core.IO`'s scope (format decode) from blurring into a second concern
(quality classification), and keeps the classification logic reusable by anything else that holds
a tensor pack (viewer tooling, future audits) without depending on the harvest tool. Adding
`curate` to the existing harvest tool, rather than a fourth tool project
(`WowViewer.Tool.Curate`), avoids the sprawl the constitution's "CLI tools are thin wrappers"
language warns against — `WowViewer.Tool.Harvest/Program.cs` already hosts a wide range of
harvest-adjacent analysis subcommands (`discover-maps`, `dump-texture-names`, `relief-to-map`,
`extract-holes`, `capture-objects`), so one more subcommand for "another pass over harvest output"
is consistent with its existing role, not scope creep.

**Alternatives considered**: (a) Fold curation directly into `WowViewer.Core.IO` — rejected,
blurs format-reader ownership with a distinct analysis concern and makes `Core.IO` responsible for
a second kind of correctness (decode correctness vs. data-quality correctness). (b) A new
`WowViewer.Tool.Curate` project — rejected, adds a fourth tool for a capability that is naturally
"one more harvest-adjacent pass," and would duplicate CLI plumbing (argument parsing, client-root
resolution) the harvest tool already has.

### D-02: On-disk shape of the Curation Manifest

**Decision**: Two companion Parquet tables written next to a v50 store's own `index.parquet`:
`curation_manifest.parquet` (one row per tile — identity, bucket assignments, synthetic-fidelity
summary) and `curation_findings.parquet` (one row per finding — tile identity, category, severity,
reason, evaluability state). Plus one small `v50-curation-run-v1` JSON provenance record per
invocation (store identity, build fingerprint, checks run, per-bucket tile counts, timestamp — not
row data).

**Rationale**: This repo already has a consistent, working pattern for exactly this shape of data:
`index.parquet` (row identity), `decoded_metadata.parquet` (per-tile sidecar, read by
`spec111/lighting_buckets.py`), `embeddings.parquet`/`curated_embeddings.parquet` (Spec
119/120), and `MismatchReport.to_parquet()` in the very script this feature retires. Two tables
(wide-per-tile + long-per-finding) rather than one avoids forcing a variable-length "list of
findings" into fixed columns, and both remain queryable with plain pandas/pyarrow filtering — no
scan of the Zarr array data is needed to answer "give me the mismatched bucket" (FR-009). Keeping
this fully separate from the Zarr store means the store's writer contract (`harvester/v50/store.py`)
needs zero changes, directly satisfying FR-014 (curation is read-only with respect to the dataset
it classifies) and the constitution's Streaming-First principle (the Zarr store remains the sole
store-of-record for signal arrays).

**Alternatives considered**: (a) Embed curation columns directly into `index.parquet` — rejected,
couples curation's evolving logic to the store-writer's schema and would require the v50 store
writer to be re-run or patched every time a bucket rule changes; a companion table can be
regenerated independently. (b) A single JSON blob — rejected, does not scale to per-finding
querying without a full parse, and every other row-level dataset artifact in this repo is Parquet,
not JSON (JSON is reserved for small provenance/run records, an already-established convention
this plan keeps).

### D-03: Invocation point relative to the harvest pipeline

**Decision**: `curate` is a separate, independently re-runnable command
(`WowViewer.Tool.Harvest curate --clients-root ... --build ... --store <v50 store path>`), run
after a store's `build`/`finalize` step, not folded into every `harvest-map`/`harvest-stream`
invocation as a mandatory inline stage.

**Rationale**: Some checks need cross-tile, map-level context (the lighting-bucket reconciliation
invariant in `spec111/lighting_buckets.py`'s `MapAccumulator` sums per-map, not per-tile), which is
awkward to compute mid-stream inside a single-tile harvest loop. More importantly, curation logic
is expected to keep evolving — spec111's own lighting-bucket logic changed after the harvest step
that produced its inputs, and this feature's whole premise is that quality logic should be
re-runnable against already-harvested data without re-harvesting. Keeping `curate` separate but
still inside the same `WowViewer.Tool.Harvest` binary (not a new tool) balances both concerns: one
binary, two independently-invokable passes.

**Alternatives considered**: (a) Inline curation into every harvest call — rejected, forces a
re-harvest (expensive, sometimes requiring the full client library) every time a curation rule is
tuned. (b) A fully separate tool — rejected per D-01's reasoning against tool sprawl.

**CLI convention**: `curate` follows the dry-run-first pattern used by every CLI in this repo
(Python and, where present, C#) — printing the planned tile count, which checks will run, and
output paths, requiring an explicit `--write` to persist anything. This is a UX convention carried
over from the Python side (e.g. `v50_cleanup_artifacts.py plan`, every `--confirm-run`-gated
trainer) rather than an existing C#-side precedent, since no prior C# harvest subcommand has needed
a stateful write-vs-plan distinction at this granularity.

### D-04: Legacy script migration path (per script, not a blanket rule)

**Decision**: Evaluate each of the six named scripts individually rather than applying one
uniform "make it a shim" rule, following the Spec 109 Phase 6 precedent (which found a real
caller-search was necessary before disposing of anything):

| Script | Disposition |
|---|---|
| `v16_curation.py` | Becomes a thin reader of `curation_manifest.parquet`/`curation_findings.parquet` via a new `harvester/curation_store.py`; existing function names (`is_blank_what_plate`, `DIFFICULTY_BUCKETS`, etc.) kept as backward-compatible wrappers if a real-caller search (task-time, not assumed) finds other modules still importing them directly. |
| `mismatch_detector.py` | Same treatment — becomes a thin reader; its `MismatchTile`/`MismatchReport` dataclasses may be kept as a compatibility view over the new manifest if callers exist. |
| `spec111/lighting_buckets.py` | Its `MapAccumulator` reconciliation invariant is ported as a canonical C# bucket dimension (FR-006); the Python module becomes a thin reader or, if a caller search finds nothing left calling it directly, a documented-historical header — determined at implementation time, not pre-decided here. |
| `build_v16_curation_manifest.py` | Closest to purely historical: a full CLI tied to the legacy V16 store shape, not v50. Gets a documented-retired header pointing to `curate` as the v50-era replacement; kept, not deleted, matching this repo's never-delete-working-legacy-code convention. |
| `v50_audit_signal_coverage.py` | **Not a migration target.** Different concern — this is Spec 109's per-signal coverage/defect audit (artifact-lifecycle and inventory), not per-tile quality classification. Stays as-is unless a real overlap is found at implementation time. |
| `v50_audit_artifacts.py` | **Not a migration target**, same reasoning as above — this owns Spec 109's inventory/cleanup-plan machinery, a distinct concern from tile-level curation. |

**Rationale**: The spec's FR-015 requires that no script "silently compute a second, divergent
definition" of a bucket or mismatch this feature covers — it does not require every named script to
become a shim regardless of what it actually does. Two of the six scripts turned out, on rereading,
to own a genuinely different concern (artifact/store lifecycle vs. per-tile quality), and forcing
them into the migration would misrepresent their purpose and risks the exact kind of undocumented
scope creep this repo's memory-bank repeatedly flags as a past failure mode (e.g. the Spec 116
"tests pass ≠ documented CLI works" lesson, generalized: don't claim a migration is complete
without checking what the target script actually does).

**Alternatives considered**: A blanket "all six become shims" rule — rejected as the initial framing
in the spec's own problem statement, corrected here after rereading `v50_audit_signal_coverage.py`
and `v50_audit_artifacts.py`'s actual docstrings (Spec 109 T019 read-only audit commands: inventory
and verify-v18, not tile-level bucket/mismatch classification).

### D-05: Test / validation strategy

**Decision**: (1) C# unit tests under a new `wow-viewer/tests/WowViewer.Core.Curation.Tests/`
project against synthetic in-memory tensor-pack fixtures (mirrors the existing
`WowViewer.Core.PM4.Tests` convention) — one fixture per known-clean, known-blank,
known-height-normal-mismatched, and known-low-fidelity-synthetic case. (2) A real (non-fixture)
smoke run of `curate` against an existing on-disk v50 store, gating on full tile coverage
(SC-006/FR-008). (3) Before any legacy script is marked retired, an explicit SC-003 comparison:
run `mismatch_detector.py`'s existing logic and the new C# `HeightNormalMismatchDetector` against
the identical real tiles and diff the flagged sets, written up as a comparison report, not just
asserted. (4) `data-harvester/tests/test_curation_store.py` validating the new Python-side reader
round-trips the C#-written Parquet schema correctly (column names/dtypes agree) — this project's
established "NPZ shard format is the contract, both sides must agree" principle, generalized here
to the new Parquet curation contract.

**Rationale**: Matches this repo's repeatedly-stated lesson (recorded in memory-bank under both
Spec 116 and the "verify CLI docs against argparse" feedback note) that unit tests passing in
isolation is not sufficient proof a documented pipeline actually works end-to-end on real data —
every prior spec's credible "done" claim has come with a real-store smoke run, not fixtures alone.

---

## Part B — External Research: Comparable Image-to-Terrain Projects

Two independent research passes were run against HuggingFace and GitHub: one scoped to
game/simulation-oriented image-to-terrain projects and generative terrain synthesis research, one
scoped to the remote-sensing/geospatial DEM-from-imagery literature (the closest real scientific
analogue). Full agent reports are preserved as session evidence; this section extracts what is
load-bearing for curation design specifically. Neither pass found a project that changes this
feature's scope — both are corroborating context, per the spec's Assumptions section ("informed by,
not blocked on").

### B-1: Nothing targets our exact regime — a genuine, not concerning, gap

Neither search found any project predicting height from a **compressed, stylized, ~2 yards/px game
minimap** — every comparable RGB→height result in the literature (Vaihingen 9cm, Potsdam 5cm,
DFC2019 1.3m, GBH 3m being the coarsest confirmed benchmark) operates at resolutions 1.5–40x finer
than ours, on calibrated real photography rather than painted/JPEG-compressed game art. This is
consistent with, not contradictory to, our own recorded finding that RGB-alone plateaus at
tile-mean on this project's data — the field's implicit resolution floor for bare RGB regression
sits at or finer than ~3m GSD, and nobody has published a result at our coarser, lower-fidelity
operating point in either direction. This is background context for future model-design specs, not
something curation needs to encode.

### B-2: Coarse-prior + residual-detailer is an independently validated shape

This is the most load-bearing external finding for the project overall (though it is a model-design
concern, not a curation one — recorded here for continuity into whichever future spec picks model
architecture back up). "Guided DEM/DSM super-resolution" — fusing a coarse elevation prior with
RGB guidance to predict a residual/refined field — is a named, active subfield (Real-GDSR,
Prompt2DEM, GDEMSR/GSRMTL, MFSR, and others), independently converging on the same two-stage shape
already validated in this project's own Spec 114 detailer work (9–11% MAE gain over coarse-only).
Reported improvement magnitudes in that literature (3–50% depending on setup) bracket our own
result. Model sizes in this literature are modest (ResNet-18/50, EfficientNetV2-lite, Swin-Tiny,
shallow CNNs) — consistent with, though never explicitly framed around, a consumer-VRAM budget.

One factual, non-actionable note: the most directly analogous 2025/2026 paper found (MFSR) uses the
Depth Anything model family as an auxiliary guidance channel. This project's Depth Anything
blacklist ([[feedback_no_depth_anything]]) stands regardless — recorded here only because it is a
literature fact a future reader might otherwise stumble on and misread as a recommendation.

### B-3: No precedent for discrete/categorical texture-layer prediction

Both research passes independently confirm this project's MCLY/MCAL brush-library framing
(discrete per-chunk texture-layer indices into a reused brush alphabet) has no confirmed external
precedent. Adjacent work either treats terrain texture as continuous image synthesis (diffusion/GAN
pixel output — MESA, Geodiffussr, TerraFusion) or classifies real photographed terrain surface
material for robotics navigation (a different task: perception of real photography, not authored
brush selection). One candidate lead (arXiv:1707.03383, allegedly a GAN-heightmap→segmentation→
splatmap pipeline) could not be verified against primary text and is flagged, not claimed. This
supports treating texture-layer curation as this project's own problem to solve without an external
recipe to borrow.

### B-4: Data curation precedent — directly relevant to this spec's design

This is the section most directly load-bearing for **this** feature (as opposed to future
model-design work):

- **Spatial leakage is a named, well-studied failure mode outside this project too.** A GeoAI
  handbook and companion tooling ([spatialCV](https://github.com/geoai-lab/spatialCV)) report that
  most reviewed CNN geoscience studies used non-spatial cross-validation, and that this can inflate
  reported performance by up to 28% versus spatially-blocked holdouts — directly corroborating this
  project's own measured 99.6% train/val spatial-adjacency leak (recorded in
  [[project_adt_corpus_structure]]) as a known, serious, externally-precedented problem, not an
  idiosyncratic one. Standard mitigation is spatial blocking **with a buffer zone**, not just literal
  overlap exclusion — worth a future note on the Spec 116 held-out-split machinery, though out of
  this feature's scope.
- **Per-tile quality bitmasks are established practice at production scale.** The PGC
  ArcticDEM/REMA/EarthDEM pipeline applies an adaptive outlier filter plus an explicit per-strip
  bitmask flagging cloud/water contamination — architecturally the same shape as this feature's
  bucket + finding design (a durable, queryable, non-destructive quality label), not a filter that
  discards data. This is external validation of the spec's central "partition, don't discard"
  requirement (User Story 2) — a real production geospatial pipeline uses the same pattern for the
  same reason.
- **Two-stage "train broad, refine on a curated subset" is documented practice**, not just this
  project's intuition: one generative-DEM paper trains on its full corpus, then fine-tunes on an
  SSIM-filtered high-quality subset. This requires keeping the broad (including lower-quality) set
  available through both stages — independent corroboration of the user's mid-session correction
  that curation must expose bad/mismatched data for deliberate use, not only a clean-only view.
- **Every confirmed curation practice in this literature is a single-signal, per-tile heuristic**
  (land-cover class, cloud-cover %, missing-data %, elevation range) — none does cross-signal
  relational consistency checking (e.g., checking whether an RGB-implied cue corroborates the height
  ground truth, this feature's height-vs-normal mismatch check). This project's relational curation
  approach — consistent with its own "ADT is a relational database"
  ([[project_adt_is_a_relational_database]]) framing — appears to be ahead of, not behind, published
  practice in this specific respect. No external recipe exists to borrow for the mismatch-detection
  half of this feature; the existing in-repo logic (`mismatch_detector.py`,
  `MinimapShadingMatch.cs`) is the right and only starting point.

### B-5: What this means for scope (no changes required)

Neither research pass surfaced a reason to add, remove, or redefine any bucket or mismatch category
already specified. The external precedent instead **confirms two design choices already made**:
(1) durable, non-destructive per-tile quality labeling over silent filtering (B-4, ArcticDEM
pattern) and (2) that this project's relational/cross-signal curation is genuinely ahead of
documented external practice, meaning the six scattered scripts being consolidated here are not
behind the field — they were simply never given one durable home.
