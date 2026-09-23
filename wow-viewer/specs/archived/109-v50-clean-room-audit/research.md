# Research Decisions: V50 Clean-Room Dataset and Repository Reset

## Decision 1 — V50 is a new data authority, not a metadata rename

**Decision**: Create new per-build v50 stores with complete manifests and independently derived
identities. Never promote a store because its path or attributes say v50.

**Rationale**: The current candidate builder can stamp arbitrary old inputs as v50 without proving
their contents. Trust must follow evidence, not labels.

**Alternatives considered**: Rename V18 directories; add only `release=v50.1`; keep the compact mixed
store as the canonical dataset. All were rejected because they preserve unknown provenance or create
training-specific copies.

## Decision 2 — V18 migration is per signal, per row, and copy-on-proof

**Decision**: Audit each V18 signal independently. Copy passing payloads bit-for-bit with hashes and
lineage; freshly extract failing or missing signals; never port known-defective `holes_16`.
`liquid_mask` and `liquid_height` are additionally `fresh-only`: historic stores cannot establish
that a WL fallback used the corrected continuous surface-quads rasterizer. A fresh WL source must
declare `wl_liquid_surface_quads_v1`; non-WL sources retain their reader identity in row lineage.

**Rationale**: Prior audit evidence showed sound core arrays alongside a systemic hole-mask defect,
per-tile coverage gaps, and a historic WL sparse-stamp projection defect. Whole-store or whole-row
certification would overclaim.

**Alternatives considered**: Trust the prior six-row audit; rebuild everything from clients; port all
V18 arrays and fix later. The chosen hybrid preserves proven work without carrying known defects and
reduces unnecessary client reads.

## Decision 3 — Canonical stores are complete; curricula are manifests

**Decision**: Maintain complete per-build v50 stores. Real/synthetic or other training curricula are
immutable row-selection manifests over canonical stores, not copied Zarr subsets.

**Rationale**: This makes provenance explicit and avoids repeated full-payload copies, directly
supporting disk-space recovery.

**Alternatives considered**: One merged store; one copied store per experiment; keep the current
240-row mixed-store builder. These complicate build lineage or waste space.

## Decision 4 — Client location is configured and content is fingerprinted

**Decision**: Accept `H:\CLIENTS`, the user-approved faster-SSD library, as a runtime client-root
argument. Manifests bind logical build identity and content fingerprints; source code does not embed
the machine-local path.

**Rationale**: The user has a larger, faster build library, and the constitution forbids hardcoded
client paths. Build truth should survive a local path move.

**Alternatives considered**: Keep project-local client copies; hardcode the SSD root; store only the
absolute path. Each either wastes space or makes the workflow machine-specific.

## Decision 5 — Cleanup is a manifest transaction

**Decision**: Separate cleanup into read-only inventory, reviewed manifest, user-run apply, and
post-cleanup verification. Bind every target by resolved path and observed identity.

**Rationale**: Generated roots are ignored by Git and contain mixed-value user artifacts. Direct
recursive deletion is unsafe and unauditable.

**Alternatives considered**: Delete by age; delete entire version directories; keep everything.
These can destroy dependencies or fail the disk-recovery goal.

## Decision 6 — Rename by moving ownership, not adding wrappers forever

**Decision**: `harvester.v50` becomes the implementation owner. Spec-named modules may temporarily
delegate for compatibility but may not remain the v50 authority.

**Rationale**: The current v50 commands are wrappers over Spec 103/108 scripts, while those scripts
were modified to carry v50 behavior. This leaves two contradictory owners and makes cleanup harder.

**Alternatives considered**: Keep wrappers permanently; rename files in place without compatibility
proof; duplicate implementations. The chosen route preserves callers while converging ownership.

## Decision 7 — Approved and protected roots policy for cleanup safety

**Decision**: Explicitly separate directory hierarchies into write-eligible/deletion candidate "Approved Roots" and strictly read-only "Protected Roots".

**Rationale**: To prevent accidental deletion of source code, specifications, client libraries, or active research assets during dry-runs and cleanup apply commands.

- **Approved Roots**:
  - `wow-viewer/output/`
  - `wow-viewer/data-harvester/tmp/`
  - `wow-viewer/data-harvester/checkpoints/`
  - `wow-viewer/data-harvester/models/`
- **Protected Roots**:
  - `wow-viewer/specs/`
  - `wow-viewer/docs/`
  - `wow-viewer/src/`
  - `wow-viewer/tests/`
  - `H:\CLIENTS`

## Decision 8 — `write_v50_store` must never destroy a good store before its replacement is proven

**Decision**: `write_v50_store` writes to a staging directory beside the target path and only
replaces the target once every array has been written without error, with retry-with-backoff around
the final directory swap.

**Rationale**: A real user-run `build --confirm-run` against `H:\CLIENTS` Kalimdor produced a
genuinely complete, valid 951-tile store, but the pipeline's `finalize` step was fed the wrong
manifest (the blank release template, not the manifest `build` actually produced) and so always
reported `finalization_state=incomplete`. Because `write_v50_store` previously opened its target
with `zarr.open_group(..., mode="w")` unconditionally, any retry -- prompted by that false-negative
report, or any process/host interruption mid-write -- silently erased the good store and forced a
full restart from tile 0. See `docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`'s
Phase 8 incident write-up for the full root-cause trace against the actual output on disk.

**Alternatives considered**: A fully incremental/checkpointed per-tile writer (write each tile to the
store as it streams in, so a crash loses at most one tile) would remove data loss risk entirely, but
is a substantially larger rewrite of `_cmd_build`'s in-memory accumulation loop. Deferred as a
documented follow-up rather than an unrequested rewrite, since the confirmed incident's actual cause
was the wrong `finalize` manifest plus the destructive unconditional `mode="w"` retry, not a mid-run
crash -- both of which the smaller staging-directory fix fully closes.

## Decision 9 — a non-complete `finalize` must not abort the multi-map pipeline, and object tiles get their own curation manifest

**Decision**: (1) `finalize_store_report()` returns every concrete reason a store isn't complete, and
`v50_pipeline_runner.py` treats a non-complete `finalize` as expected/non-fatal for a map (it still
runs curation for that map and moves on) rather than aborting the whole run. (2) The pipeline writes
two curation manifests per map: the existing strict, object-free one (`--max-object-coverage 0.0`)
and a new object-inclusive one (`--max-object-coverage 1.0`) that keeps every tile touched by an
MDDF/MODF footprint.

**Rationale**: The first real post-Phase-8 full-corpus run hit both gaps on the first try. Azeroth's
`finalize` legitimately reported `incomplete` (2 real tiles lack `minimap_rgb` because minimap
synthesis correctly skipped them for having no texture data), but `v50_pipeline_runner.py`'s
unconditional `check=True` treated that as fatal and aborted before PVPZone02 or Kalidar ever ran --
over exactly the kind of dirty row that curation exists to drop. Separately, `spec103_curate_dataset.py`'s
`--max-object-coverage 0.0` default is correct for minimap-to-height reconstruction specifically (an
object occludes the ground, making "true height under it" an impossible target for that task from the
minimap alone -- its own docstring says so), but baking that as the *only* curation profile silently
dropped 518/951 (54.5%) of Kalimdor, 355/685 (51.8%) of Azeroth, 3/64 (4.7%) of PVPZone02, and 12/56
(21.4%) of Kalidar as `object_contaminated` from the only curated view that existed, discarding real
data for anything object-aware even though v50's frozen signal catalog deliberately keeps
`object_precise_mask`/`object_instance_mask` as first-class signals.

**Alternatives considered**: Making `finalize` itself always exit 0 was rejected -- it must keep
failing closed on a real trust question (is this store complete); the fix is the *caller* no longer
conflating "not complete" with "stop everything." Replacing the strict curation profile outright
(rather than adding a second manifest) was rejected per explicit user direction: the strict
object-free manifest is still the correct input for height-reconstruction work, so both need to exist
side by side rather than picking one policy for the whole corpus. Neither curation manifest
duplicates array data -- both are Parquet row-reference lists over the same raw, untouched store.
