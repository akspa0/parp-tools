# V50 Clean-Room Dataset and Repository Audit — 2026-07-15

**Owner**: Spec 109

**Status**: Clean-slate bootstrap complete 2026-07-16; no dataset promoted

## Phase 2/3 foundational + trust-boundary implementation — 2026-07-17

- Implemented `harvester/v50/` (contracts, identity, client_evidence, path_policy, inventory,
  verify_v18, verify_store) and the read-only `scripts/v50_audit_artifacts.py inventory`/
  `verify-v18` commands, all fixture-tested (114 passed, 2 skipped for symlink privilege, 0
  failed). `v50_contract.py`'s release-identity gates now live in `harvester.v50.contracts`; the
  old module is a thin re-export shim so existing Spec 103/108 callers are unaffected.
- Ran the real (non-fixture) read-only inventory command against everything currently on disk:
  `wow-viewer/output`, `wow-viewer/models`, `data-harvester/checkpoints`, `data-harvester/tmp`,
  `data-harvester/models`. Report: `output/reports/v50/v50.1/inventory.json`.

  | Owner | Kind | Path | Bytes |
  |---|---|---|---:|
  | output | unknown | `output/synthesized-minimaps` | 954,670,730 |
  | models | unknown | `models/v18` | 10,848,359,743 |
  | models | unknown | `models/v21` | 25,156,952 |
  | models | unknown | `models/v23` | 1,871,311,176 |
  | checkpoints | checkpoint | `data-harvester/checkpoints/d1_best.pt` | 44,737,027 |
  | checkpoints | checkpoint | `data-harvester/checkpoints/d1_final.pt` | 44,737,799 |
  | checkpoints | checkpoint | `data-harvester/checkpoints/v15_best.pt` | 328,818,359 |
  | checkpoints | checkpoint | `data-harvester/checkpoints/v15_final.pt` | 328,819,211 |
  | tmp | unknown | `data-harvester/tmp/v18_smoke` | 205,050 |
  | tmp | unknown | `data-harvester/tmp/v22_smoke` | 12,696,389 |
  | harvester_models | unknown | `data-harvester/models/spec077` | 1,095,721,766 |

  Every record above is `trust_state=unverified`, `disposition=quarantine`, `proof_level=inventory`
  regardless of path or filename — none of this is promoted or claimed compatible with v50 by this
  pass. Directory-kind entries show `unknown` rather than a guessed kind: the classifier only infers
  kind from file suffixes, and a directory (a nested tree of possibly-mixed content) is not
  guessed at.
- `verify-v18`'s per-signal audit (T018) does not yet cross-validate against a fresh client
  extraction (plan.md Phase 2 step 2) — it audits an already-decoded V18 store's content for known
  defects (blacklisted signals, non-finite values, `has_*` truthfulness) using a caller-supplied
  signal catalog, since Spec 109's frozen v50 signal table (T002) is not finalized yet. Smoke-tested
  against a synthetic fixture store: correctly rejected a NaN-poisoned row and blacklisted
  `holes_16` in every row regardless of its content.

## Phase 4 reviewable cleanup planning — 2026-07-17

- Implemented `harvester/v50/dependencies.py` (scans manifest/report JSON for references by path
  or content hash -- no not-yet-frozen manifest schema hardcoded) and `cleanup.py` (matches
  `v50-cleanup-plan.schema.json` exactly: a candidate only reaches `targets` once it already has
  `dependency_check=pass` and `approved=true`; failing candidates are absent, not marked-rejected).
  `scripts/v50_cleanup_artifacts.py plan` is the only subcommand -- no `apply` exists yet (Phase 7,
  user-run-only, requires the reviewed plan's hash and explicit confirmation).
- Real local dry run against the Phase 3 real inventory: with zero dispositions reviewed, correctly
  0 targets. With the two genuinely-disposable `data-harvester/tmp/v18_smoke` and `.../v22_smoke`
  scratch artifacts explicitly reviewed and dispositioned `remove-candidate` with a stated
  replacement proof, the plan correctly proposed exactly those 2 targets (12,901,439 bytes),
  `dependency_check=pass`, `approved=true`, `dry_run_complete=true`. Nothing was deleted; there is
  still no apply path.
- Fixture proof: the exact scenario in tasks.md's Independent Test (protected, depended-on,
  out-of-root, and safe-obsolete targets in one fixture) included only the safe-obsolete target.

## Phase 5 complete v50 dataset builder — 2026-07-17

- Implemented `harvester/v50/store.py` (`write_v50_store`, `read_v50_manifest`, `finalize_store`),
  `migrate.py` (`plan_signal_migration`, `MigrationLedger`/`MigrationLedgerEntry`, `copy_signal_row`),
  `build.py` (`build_harvest_stream_command`, `run_fresh_extraction` gated behind
  `confirm_run`, `read_harvest_stream` reusing `harvester.raw_reader.read_tile_blob` for the C#
  harvest-stream inner-blob format rather than reimplementing it), and `curriculum.py`
  (`build_curriculum`/`CurriculumManifest`, row-reference-only, reuses
  `harvester.spec103.prefab_curation.validate_source_group_split` for partition-leakage checks).
- Rewrote `scripts/v50_build_dataset.py` from the Phase 1 fail-closed placeholder into 5 real
  subcommands: `migrate-v18`, `build`, `verify`, `finalize`, `curriculum`.
- Fixture tests: `uv run python -m pytest tests/v50/ tests/test_v50_contract.py tests/spec103/
  tests/spec111/ -q` → 164 passed, 2 skipped (symlink privilege), 0 failed.
- Smoke-tested all 5 subcommands end-to-end against a synthetic 3-row V18 Zarr fixture (real
  Zarr/Parquet I/O, nothing mocked): `migrate-v18 --write-store` produced a partial store
  (`finalization_state=incomplete`); `finalize` against a stale manifest template correctly refused
  (`incomplete`, exit 1 — the check catching a real mismatch, not a bug), then succeeded
  (`complete`, exit 0) against the manifest actually read back from the written store; `verify`
  passed with `proof_level=full` against the correct observed hash and failed closed
  (`proof_level=contract`, exit 1, naming the exact signal and both hash values) against a
  deliberately wrong one; `curriculum` produced a row-reference-only manifest across a train/val
  split.
- Explicitly not proven yet: the real client-backed paths. `build`'s C# harvester launch and
  `migrate-v18`/`verify` against a real V18 build under `H:\CLIENTS` are implemented but require the
  user to select a build and authorize the run (`--confirm-run` for `build`; no equivalent gate is
  needed for `migrate-v18`/`verify` since they only read already-decoded local stores, but they have
  not been pointed at real data yet). See `quickstart.md` sections 3-5.

## Phase 6 canonical command-ownership convergence — 2026-07-17

- Moved the real implementations of the WDL-prior (train/infer/evaluate/visualize) and
  terrain-refiner (train/infer) commands out of top-level `scripts/*_spec103_*.py` files and into
  `harvester.v50.wdl_prior_train`/`wdl_prior_infer`/`wdl_prior_evaluate`/`wdl_prior_visualize`/
  `terrain_refiner_train`/`terrain_refiner_infer`. All 6 `scripts/v50_*.py` entry points now import
  `main` only from their `harvester.v50` owner.
- Disposition of the 6 historical `scripts/{train,infer}_spec103_{wdl_prior,v7}.py` /
  `evaluate_spec103_wdl_prior.py` / `visualize_spec103_wdl_prior.py` files: **kept as thin
  re-export shims, not deleted.** A repo-wide caller search (`grep` across `wow-viewer/` for each
  of the 6 module names) found every one is load-bearing: `tests/spec103/test_wdl_prior_sanity.py`
  imports `filter_deployable_rows` and `V7TileDataset` from those exact module names and
  subprocess-invokes `infer_spec103_wdl_prior.py` by file path, and
  `runpod/spec103/{train,smoke,verify_bundle}.sh` invoke `train_spec103_v7.py`/`infer_spec103_v7.py`
  by file path. None is a deletion candidate; each now contains only re-exports plus
  `if __name__ == "__main__": raise SystemExit(main())`, never a second implementation that could
  drift from its `harvester.v50` owner.
- The same caller search caught a real regression this move would otherwise have shipped
  silently: `scripts/package_spec103_runpod.py`'s RunPod bundle packager listed
  `scripts/train_spec103_v7.py`/`infer_spec103_v7.py` as files to ship but never packaged
  `src/harvester/v50` -- the bundled shims would have imported a module the bundle didn't
  contain. Fixed by adding `src/harvester/v50` to `_SOURCE_DIRS` (that package has no dependency
  beyond what the bundle already ships: numpy, pyarrow, torch, zarr, Pillow, all stdlib otherwise).
- Added `tests/v50/test_command_compatibility.py` (14 tests, T038): every canonical `v50_*.py`
  entry imports `main` only from its `harvester.v50` owner and never from a historical
  spec103-named module; every historical shim re-exports that owner and defines no second
  `main()`; a WDL checkpoint from the wrong release is rejected by `load_model`; all 4 moved
  command-owner modules import the identical `harvester.v50.contracts` gate objects (`is`
  comparison), not a locally reimplemented copy that could silently drift.
- A full (not v50-scoped) `uv run python -m pytest tests/ -q` run surfaced one more real
  regression: `tests/test_v50_build_command.py` still asserted Phase 1's retired fail-closed
  placeholder refusal message. That message was intentionally retired by Phase 5's real
  subcommands; the test was rewritten against the current CLI contract (subcommand required,
  unrecognized subcommand rejected with the matching argparse error and exit code 2).
- Proof: `tests/spec103/ tests/v50/ tests/test_v50_contract.py tests/spec111/` -> 178 passed, 2
  skipped, 0 failed. Full `tests/` -> 568 passed, 43 skipped, 3 failed; the 3 failures
  (`tests/v24/test_export_map.py`, `tests/v25/test_h1_coarse.py` x2) were confirmed via `git
  stash` to reproduce identically against committed `HEAD` (`b84d30aa`, before any Phase 6
  change) -- pre-existing and unrelated. All 12 touched scripts individually smoke-tested via
  `--help`; all 12 clean.

## Phase 7 reviewed cleanup apply (code + dry-run refresh) — 2026-07-17

- Implemented `harvester/v50/cleanup.py`'s `apply_cleanup_plan()`, `CleanupApplyResult`, and
  `CleanupApplyError`: refuses without explicit `confirm=True`; refuses unless the caller's
  `expected_plan_id` matches the plan's own `plan_id` exactly (a stale or hand-edited plan cannot
  be applied); re-resolves every target against `PathPolicy` at execution time rather than
  trusting the plan's own `approved_roots` snapshot; rehashes each target's real on-disk content
  immediately before deleting it, skipping (not deleting) anything that drifted since the plan
  was built. A target already absent is treated as already-removed (a prior interrupted apply),
  not an error, and its bytes are not recounted -- reapplying the same plan is idempotent.
- Added the `apply` subcommand to `scripts/v50_cleanup_artifacts.py` (`--plan`, `--plan-id`,
  `--approved-root`/`--protected-root`, `--confirm`, `--output`). Smoke-tested end-to-end against
  a real synthetic fixture file: a wrong `--plan-id` was refused (exit 1, file untouched); the
  correct `--plan-id --confirm` run actually deleted the fixture and reported `1 removed, 0
  skipped, 22 bytes recovered`; re-running the identical command afterward reported `1 removed, 0
  skipped, 0 bytes recovered` (idempotent).
- Added `tests/v50/test_cleanup_apply.py` (9 tests, T045): identity gate (wrong plan_id, missing
  confirm), successful removal, content-drift skip, re-tampered protected-root rejection at
  apply time, interrupted-run resumability, and `to_dict()` round-trip.
- Refreshed the real (non-fixture) inventory and cleanup dry-run plan against everything
  currently on disk: 13 artifacts now (2 new since Phase 4 are this report directory's own prior
  output and `models/.gitignore`, neither a disposal candidate). The same 2 genuinely-disposable
  targets as Phase 4 (`data-harvester/tmp/v18_smoke`, `.../v22_smoke`) reproduce byte-for-byte:
  2 targets, 12,901,439 bytes expected recovered,
  `plan_id=sha256:fc2c657b42c33fd852a57f4873e657cd8ccbcef021487057a2eeddb826a4e346`. **Nothing has
  been deleted.** The exact user-run apply command is documented in `quickstart.md` section 8.
- Proof: `tests/v50/ tests/test_v50_contract.py tests/spec103/ tests/spec111/
  tests/test_v50_build_command.py` -> 189 passed, 2 skipped, 0 failed. Full `tests/` -> 577
  passed, 43 skipped, 3 failed (same 3 pre-existing, unrelated failures as Phase 6).
- Not done, and explicitly gated on the user: actually running `apply` against the real
  `tmp/v18_smoke`/`tmp/v22_smoke` targets (T050), the post-apply verification that follows it,
  and the memory-bank compression (T051) that was deliberately deferred until after that real run
  so it only needs to happen once.

## Clean-slate completion — 2026-07-16

- The user successfully emptied workspace `output/` and `wow-viewer/output/` with the guarded
  cleanup workflow after repairing sandbox-owned ACLs.
- More than 200 GB of legacy datasets, models, temporary client copies, validation artifacts, caches,
  and scratch outputs were removed.
- `H:\CLIENTS` remains the approved known-good client library and was outside cleanup scope.
- The small 1.0.0 native-client decompile bundle remains preserved under ignored local
  `wow-viewer/test_data/native-research/1.0.0-decomp` for M2 research.
- The output roots are now intentionally empty. All future datasets and models must be new v50
  artifacts with provenance; nothing deleted by this cleanup is trusted or eligible for reuse.
- This is the reboot handoff checkpoint. The next work starts at Spec 109 Phase 0/1 contract and
  inventory implementation, then builds fresh v50 stores and models without legacy-output fallback.

## Audit rule

Every pre-v50 dataset, checkpoint, generated prior, model output, manifest, and derived report is
unverified until its provenance and content contract are independently proven. A v50 label is not
proof. Missing proof fails closed.

## Proof levels

1. **Inventory proof**: the artifact or workflow exists at the recorded location.
2. **Contract proof**: metadata, schema, shapes, dtypes, lineage fields, and compatibility gates agree.
3. **Sampled content proof**: bounded rows reproduce against an independent authoritative source.
4. **Full content proof**: all rows and hashes pass the approved verification procedure.
5. **Quality proof**: a model meets its stated held-out and deployment-facing acceptance gates.

Passing an earlier level does not imply a later one.

## Initial evidence

- Clean-slate decision: the user declared legacy artifacts under workspace `output/` and
  `wow-viewer/output/` disposable so v50 datasets/models start from empty output roots. A guarded
  user-run cleanup command now refuses tracked files, reparse points, and out-of-root targets.
- Preflight found one tracked generated PM4 identity JSON and one obsolete untracked Python scratch
  script in `wow-viewer/output`; neither had callers. The tracked generated result was removed from
  the output tree before cleanup could be enabled.
- Preserved the small `output/ghidra_1.0.0` native-client research bundle by relocating it to ignored
  local evidence at `wow-viewer/test_data/native-research/1.0.0-decomp`; it is outside both cleanup
  roots and remains available for M2 research.
- Cleanup measurement exposed sandbox-owned pytest output whose ACL grants only the sandbox owner,
  SYSTEM, and Administrators. The tool now records partial measurement and per-target deletion
  failures; the quickstart contains a bounded elevated ACL repair for the two output roots only.
- User-confirmed client authority: `H:\CLIENTS` is a safe, known-good, faster SSD library and is the
  preferred configured source for v50 client-backed verification and fresh extraction. Every build
  still requires an evidence fingerprint.
- Branch at audit start: `v0.5.1`.
- The worktree already contained modified Spec 103/108 implementation files and untracked v50
  commands, contract code, and tests. They are preserved as user work.
- `wow-viewer/output/v50` did not exist, and no immediate child of
  `wow-viewer/output/datasets` contained `v50` in its name.
- Local historical dataset roots included `albedo`, `spec102`, `spec103`, `spec108`,
  `teacher-prior`, `v18`, `v22`, `v24`, and `v25`, plus analysis products. These names are
  inventory evidence only.
- Local model/output roots also existed under `wow-viewer/output/models`, `wow-viewer/models`,
  `wow-viewer/data-harvester/models`, and `wow-viewer/from_cloud`.
- No tracked payload with a common ML artifact extension (`pt`, `pth`, `onnx`, `npz`, `parquet`,
  `zarr`, `zip`, or `safetensors`) was found under `wow-viewer`; local output roots are ignored.
- The repository root contained untracked `.python-version`, `pyproject.toml`, and `uv.lock` for a
  separate `parp-tools-mcp` environment, while project policy requires Python work to live under
  `wow-viewer/data-harvester`.
- The active Spec Kit pointer still named Spec 108 when the audit began.

## Initial findings and dispositions

| Surface | Evidence | Initial state | Disposition | Required next proof |
|---|---|---:|---|---|
| Historical dataset roots | Many local roots under ignored `wow-viewer/output/datasets` | Unverified | Quarantine | Per-artifact provenance and schema inventory; independent sampled comparison before any promotion |
| Historical model/checkpoint roots | Local output/model/cloud-download roots exist | Unverified | Quarantine | Bind each artifact to a verified dataset, split, code identity, and metrics report |
| `v50_build_dataset.py` | Refuses production until the Spec 109 canonical owner exists | Fail-closed | Keep blocked | Replace only with fixture-proven migration and client-backed fresh extraction |
| Other `v50_*.py` commands | Thin wrappers over Spec 103/108 scripts | Inventory only | Migrate | Focused parity tests, then move ownership out of historical spec command modules |
| Current mixed-store builder | Accepts arbitrary real/synthetic stores and writes v50 metadata | Failed trust gate | Quarantine | Reject unverified sources; bind source schema, manifest, hashes, and row lineage before writing output |
| `v50_contract.py` | Checks release identity plus WL surface provenance and fresh-only liquid policy | Partial contract proof | Keep/extend | Validate complete store schema, index identity, source identities, row lineage, and downstream hashes |
| Modified Spec 103/108 modules | Carry v50 behavior while retaining historical owner paths and aliases | Unreviewed user work | Verify/migrate | Focused tests plus an ownership plan that preserves historical compatibility intentionally |
| Root Python project files | Separate MCP environment at repository root | Boundary violation candidate | Remove-candidate | Confirm no required workspace tool depends on them, then remove without touching the harvester environment |
| Ignored output roots | Git ignores `wow-viewer/output` and other generated/model roots | Local user data | Keep quarantined | Produce an artifact manifest before any pruning |
| Oversized continuity files | `activeContext.md` was 834 lines; `progress.md` was 663 lines | Hygiene debt | Migrate/compress | Preserve current truth in Spec 109 and archive stale chronology in a reviewed pass |

## Blocking defect: metadata laundering

The current mixed-store builder writes `model_family=v50`, a `v50.N` release, and a v50 schema onto
its output after accepting caller-provided real and synthetic stores. It does not first prove the
input schema, producer identity, immutable content hash, authoritative client source, index
truthfulness, or row-level lineage. An arbitrary historical store can therefore be copied into a
v50-labeled store.

Until that defect is closed, no output from this builder is trusted regardless of its v50 metadata.

## Frozen liquid policy — 2026-07-16

- V50 will include `liquid_mask` and `liquid_height` as valuable terrain-supervision signals, but
  both are **fresh-only** in the first V50 release. No historic V16/V18 payload can prove that its
  WL fallback avoided the former sparse-stamp/checkerboard projection.
- A newly extracted WL source is accepted only when its signal metadata contains
  `wl_liquid_surface_quads_v1`, proving the continuous nine-quad surface rasterizer. MH2O/MCLQ
  sources are not required to claim that WL-specific marker, but their authoritative reader must
  be retained in row lineage.
- `scripts/v50_build_dataset.py` now stops with a non-zero result rather than delegating to the
  legacy Spec 108 mixed-copy builder. It does not create or relabel a V50 store. The future Spec
  109 per-build writer must enforce this policy before writing any liquid payload.

## Phase 1 audit slices

1. Define the artifact-record and provenance-manifest fields, including proof-level semantics.
2. Implement a read-only inventory that emits `unverified` records and never promotes by name.
3. Add source verification gates to the dataset builder before it can write a v50 store.
4. Add full schema/index/lineage/content identities to the v50 store contract.
5. Prove fail-closed behavior with fixtures; do not read a broad dataset or launch training.
6. Review the inventory and approve individual sampled/full verification commands for user execution.
7. Only after verification, plan migration or removal of duplicate command owners and local artifacts.

## Explicitly not proven

- No historical dataset has been validated.
- No historical checkpoint or generated prior is compatible with v50.
- No v50 dataset exists yet.
- No model-quality claim carries forward.
- No cleanup candidate is safe to delete yet.

The final bullet above described the initial audit state and is superseded by the user-approved
clean-slate completion: the two output roots were explicitly designated disposable and successfully
emptied. It does not authorize deletion outside those roots.

## Frozen Signal Catalog (T002) — Updated 2026-07-17

The v50.1 release signal catalog defines the exact, verified data elements allowed in the clean-room store:

| Signal | dtype | Shape | V50 Policy | Required | Notes |
|---|---|---|---|---|---|
| `height_257` | float32 | (257,257) | copy-if-verified | **yes** | Core terrain heightmap. |
| `normal_xyz` | float32 | (257,257,3) | copy-if-verified | no | MCNR normals. |
| `normal_mask` | bool | (257,257) | copy-if-verified | no | MCNR availability mask. |
| `alpha_256` | float32 | (256,256,4) | copy-if-verified | no | MCAL texture blend weights. |
| `holes_16` | bool | (16,16) | **blacklisted** | no | Uncorrected hole masks (FR-017). |
| `liquid_mask` | float32 | (256,256) | **fresh-only** | no | Historic WL liquid presence (fresh-only). |
| `liquid_height` | float32 | (256,256) | **fresh-only** | no | Historic WL liquid surface height (fresh-only). |
| `liquid_type_256` | uint8 | (256,256) | copy-if-verified | no | Liquid classification. |
| `mcnk_flags_16` | int32 | (16,16) | copy-if-verified | no | Chunk flags. |
| `minimap_rgb` | uint8 | (256,256,3) | copy-if-verified | no | **Synthesized** terrain minimap (compositor output). Partial coverage is honest where a tile lacks usable texture data; curriculum/pair-set selection requires real row lineage. NEVER the authored client image — see `minimap_rgb_authored` (Spec 112). |
| `minimap_rgb_1024` | uint8 | (1024,1024,3) | copy-if-verified | no | **4x Resolution** synthesized minimap for Real-ESRGAN upscaler. Same honest partial coverage as `minimap_rgb`; row coverage must equal it. |
| `minimap_rgb_authored` | uint8 | (256,256,3) | copy-if-verified | no | **Authored client minimap** — the real in-game render decoded from the MPQ. Harvest-stream sourced, NEVER synthesized. Partial coverage (only tiles the client shipped a minimap BLP for); honestly unavailable elsewhere, never zero-substituted. This is the real deployment input a decompilation model must ultimately consume (Spec 112, user-directed 2026-07-18). |
| `mccv_rgb` | float32 | (257,257,3) | copy-if-verified | no | **MCCV Vertex Colors** (vertex lighting/shading). |
| `shadow_mask` | float32 | (256,256) | copy-if-verified | no | MCSH shadow. |
| `mcly_texture_ids` | int32 | (16,16,4) | copy-if-verified | no | Per-chunk texture IDs. |
| `mcly_layer_mask` | float32 | (16,16,4) | copy-if-verified | no | Layer presence. |
| `mcnr_mask_257` | bool | (257,257) | copy-if-verified | no | Normal coverage mask. |
| `ground_intent_height_257` | float32 | (257,257) | copy-if-verified | no | WDL-derived ground intent. |
| `mcly_tileset_ids` | int32 | (16,16,4) | copy-if-verified | no | Per-chunk tileset IDs. |
| `wdl_outer_17` | float32 | (17,17) | copy-if-verified | no | WDL-scale coarse height lattice, outer samples (`height_257[::16,::16]`, Spec 108 FR-001 / `TerrainWdlLattice`). Already streamed by the harvester; only newly cataloged (Spec 117). |
| `wdl_inner_16` | float32 | (16,16) | copy-if-verified | no | WDL-scale coarse height lattice, inner samples (`height_257[8::16,8::16]`), offset half-step from the outer grid. |
| `wdl_outer_present` | bool | (17,17) | copy-if-verified | no | Per-sample validity for `wdl_outer_17`: a gap is a real MCVT vertex absence, never fabricated or interpolated to fill the point. |
| `wdl_inner_present` | bool | (16,16) | copy-if-verified | no | Per-sample validity for `wdl_inner_16`, same convention as `wdl_outer_present`. |
| `object_geometry_visible_mask_257` | float32 | (257,257) | copy-if-verified | no | Strict visible-object mask (Spec 118 FR-001): 1.0 only where a transformed M2/WMO triangle is visible above the raw MCVT surface (+0.25 clearance, liquid-aware) — never the full placement footprint. Streamed by the harvester (Full/V16 profiles only, not V22); only newly cataloged. Tiles whose strict target is ineligible carry no array and are excluded-and-counted, never fabricated. |
| `object_geometry_visible_source_257` | uint8 | (257,257) | copy-if-verified | no | Per-pixel class of the front-most visible object fragment (Spec 118 FR-003): 0 = none, 1 = doodad (M2Triangle), 2 = building (WmoTriangle). 0 exactly where `object_geometry_visible_mask_257` is 0. |
| `object_geometry_visible_instance_257` | int32 | (257,257) | copy-if-verified | no | Per-object instance id of the front-most visible fragment (Spec 118 FR-002): 0 = none, 1..K = per-tile compact ids (MDDF placements first, then MODF) resolved via the per-tile `object_geometry_visible_instances` metadata table. New dense array painted by the strict rasterizer under the same front-most rule as the source tag. |
| `object_mask` | float32 | (257,257) | copy-if-verified | no | Placement-footprint object mask, painted by `AlphaTensorPackBuilder.BuildObjectMasks` from MDDF/MODF placements alone (doodads = scale-sized circles, WMOs = MODF bounding rects) — no MDX geometry load required, so it populates on the 0.5.3 **alpha** harvest path where the strict `object_geometry_visible_*` signals (ADT-builder only) zero-fill. This is the v18-proven object mask. Serialized under this exact key by the V22 profile. |
| `object_precise_mask` | float32 | (257,257) | copy-if-verified | no | Soft-edged variant of `object_mask` (v18 "precise object mask"): `PaintSoftCircle`/`PaintSoftRect` per placement, same alpha-builder source. Over-masks doodads (footprint, not visible-portion) but is populated on alpha where the strict signals are empty. |
| `object_instance_mask` | int32 | (257,257) | copy-if-verified | no | Per-placement instance id (1..K) for `object_mask`, alpha-builder-painted. Lets loss/segmentation address individual objects even though the mask is footprint-based. |

### Catalog amendment 2026-07-22 (Spec 118) — strict visible-object signals added

**Rationale**: Spec 118 reintroduces the object signal dropped below — correctly, as a per-object,
occlusion-aware (visible-portion-only) mask with a class label. The strict geometry target
(`TerrainVisibleObjectMaskRasterizer` + `AdtTensorPackBuilder.BuildStrictTerrainVisibleObjectMask`)
already computed the visibility-correct mask and class tag and already streamed them under these
exact names in the Full/V16 profiles; the instance-id array is the one new dense array, painted in
the same raster pass under the same front-most rule. This amendment only adds the three names to
the frozen catalog so the existing v50 store builder's 1:1 name-matched extraction selects them —
the same gap shape as the Spec 117 amendment.

**Correction 2026-07-22 (same day)**: the strict `object_geometry_visible_*` signals are produced
ONLY by `AdtTensorPackBuilder` (Full/V16 profiles). The 0.5.3 corpus harvests through
`AlphaTensorPackBuilder` (the alpha WDT path), which does NOT produce them — so on the real v50
build they zero-fill (0/951 tiles). Deferring the footprint masks therefore left the alpha corpus
with NO usable object signal at all. Re-cataloged the legacy `object_mask`/`object_precise_mask`/
`object_instance_mask` (rows above): they are painted by `AlphaTensorPackBuilder.BuildObjectMasks`
from placements alone (no MDX load), populate on alpha, and are the v18-proven masks. They over-mask
(footprint, not visible-portion) — a known limitation, but a populated approximate mask beats an
empty strict one. The strict signals stay cataloged for when an ADT-builder/model-loading path
materializes them; on alpha they remain zero-filled and must not be used as the object signal.

**Approved by**: this session's Spec 118 implementation pass, 2026-07-22 (data plumbing only, no
model decision implied).

### Catalog amendment 2026-07-21 (Spec 117) — WDL lattice signals added

**Rationale**: Spec 117 needs the exact 545-point WDL-scale lattice (17×17 outer + 16×16 inner,
Spec 108 FR-001) as a first-class per-tile signal so a standalone minimap-only predictor can be
trained and scored before any chain-integration is attempted. `TerrainWdlLattice` was already
computed from real MCVT vertex data in `AdtTensorPackBuilder` and already streamed by
`RawArraySerializer.WriteTerrainVertexArrays` under these exact names in every stream profile
(Full/V16/V22) — this amendment only adds them to the frozen catalog so the existing v50 store
builder's 1:1 name-matched extraction (`scripts/v50_build_dataset.py::_cmd_build`) selects them.
No new harvester code, no new store writer code: the store builder, row-lineage tracking, and
"excluded and counted, never fabricated" behavior for a signal missing on some tiles were already
generic before this amendment.

**Approved by**: this session's Spec 117 implementation pass, 2026-07-21 (data plumbing only, no
model decision implied).

### Catalog amendment 2026-07-18 (Spec 112) — `minimap_rgb_authored` added

**Rationale**: A model that decompiles minimaps into terrain must ultimately consume the *real*
authored client minimap, not only our compositor's synthesized render — training on synthetic-only
imagery creates a domain gap against the actual deployment input. The v22 harvest stream already
decodes the authored client minimap (`TryLoadMinimapFromMpq`); before this amendment it was written
under the same `minimap_rgb` key as synthesis output and silently discarded/mixed. The amendment
gives the authored image its own honestly-labeled signal and clarifies that `minimap_rgb`/
`minimap_rgb_1024` are synthesized-only.

**Approved by**: the user, 2026-07-18, in session ("we need to use the originals if we are going to
train a model on fucking anything useful"). Training will pair BOTH sources with the same height
target as separate curriculum rows (user choice).

### Dropped Signals (Deferred or Removed)
- `object_mask`, `object_precise_mask`, `object_instance_mask`: Deferred — footprint-based and replaced by the Spec 118 strict visible-object signals (`object_geometry_visible_mask_257`/`_source_257`/`_instance_257`, cataloged 2026-07-22).
- `object_roof_mask`, `object_roof_confidence`: Removed (broken/dead signals).
- `object_filtered_mask`, `model_focus_mask`: Removed (derivative of broken masks).
- `mddf_mask`, `modf_mask`: Removed (synthesized/interpolated projections).
- `model_above_terrain_mask`: Removed (requires volumetric redesign).

## Canonical Curation Entrypoint (Spec 122, added 2026-07-30)

Dataset quality curation (difficulty/coverage/lighting buckets, height-normal-mismatch/non-finite/
has-flag-mismatch/synthetic-fidelity-gap findings) for the v50 lane now has one canonical, durable
entrypoint: `WowViewer.Tool.Harvest curate --client-root <path> --store <v50 store>`
(`wow-viewer/src/core/WowViewer.Core.Curation`), read from Python via
`harvester.curation_store.load_curation_manifest`/`load_curation_findings`. It classifies every
tile in a store's `index.parquet` — including bad/mismatched ones — into a durable, equally-
queryable Parquet manifest under `<store>/curation/<curation_run_id>/`; it never silently drops a
tile. Five pre-existing scattered Python curation implementations were found and are documented
in-place (not converted to shims, since real callers on V16/V18/V23-era store shapes depend on
their current behavior): `v16_curation.py`, `mismatch_detector.py`, `spec111/lighting_buckets.py`,
`build_v16_curation_manifest.py`, and `spec103_curate_dataset.py` (the last is a fifth,
previously-undocumented script — the one that actually produced the real curation output already
on disk under `curation-0_5_3_3368-<Map>*/`, a drop-filter that discards ~80% of tiles with only
aggregate reasons, not durable per-tile records). New v50-lane curation work should read the
canonical manifest, not add another scattered implementation. See
`specs/122-dataset-curation/` for the full design.

## Approved and Protected Roots (T003)

To guarantee safety during clean-room dataset builds and cleanup:
- **Approved Roots** (write-eligible/deletion candidates under dry-run):
  - `wow-viewer/output/`
  - `wow-viewer/data-harvester/tmp/`
  - `wow-viewer/data-harvester/checkpoints/`
  - `wow-viewer/data-harvester/models/`
- **Protected Roots** (strictly read-only, never deleted or written):
  - `wow-viewer/specs/`
  - `wow-viewer/docs/`
  - `wow-viewer/src/`
  - `wow-viewer/tests/`
  - `H:\CLIENTS` (or any other external client root)

## Phase 8 incident — real build silently wiped and restarted (2026-07-17)

**Symptom (user-reported)**: a real `build --confirm-run` run against `H:\CLIENTS` Kalimdor would
run for the full ~8-11 minutes, then "randomly" delete everything it had generated and restart the
whole extraction from tile 0, with no visible reason.

**Root cause, confirmed against real output already on disk, not reproduced from a guess**: at the
time of investigation, `output/datasets/v50/v50.1/0_5_3_3368-Kalimdor.zarr` was a completely valid,
complete 491 MB / 951-tile store -- its own `zarr.json` attrs carried `row_count: 951` and real,
non-placeholder content hashes for every signal. But the sibling
`0_5_3_3368-Kalimdor.manifest.json`, written by `v50_pipeline_runner.py`'s `finalize` step, showed
`row_count: 0` and all-zero placeholder hashes with `finalization_state: "incomplete"`. Two bugs
combined to turn that false-negative into real data loss:

1. `v50_pipeline_runner.py`'s `finalize_cmd` passed `--manifest
   ./v50_configs/v50-manifest-template-0_5_3_3368.json` -- the blank release template with
   `row_count: 0` -- instead of the manifest `build` had actually produced. `finalize_store()`
   therefore always found `row_lineage count 951 != manifest row_count 0` and every hash
   mismatched, so it reported `finalization_state=incomplete`/exit 1 for every build, however good.
   Quickstart's own Phase 5 smoke-test record already said the correct input is "the manifest read
   back from the real written store" -- but no CLI path ever wrote that manifest to a file; it only
   ever lived in the Zarr store's own `attrs`.
2. `write_v50_store()` (`harvester/v50/store.py`) opened its target with `zarr.open_group(...,
   mode="w")` unconditionally, which erases any pre-existing store at that path with no resume, no
   backup, and no confirmation. So the retry that followed the false "incomplete" report -- or any
   process/host interruption mid-write -- destroyed the good store and forced a full restart.

**Fix**:

- `build` now also writes its real, just-computed manifest to disk via a new `--write-manifest`
  path (`harvester/v50/build.py`/`scripts/v50_build_dataset.py`); `v50_pipeline_runner.py`'s
  `finalize_cmd` now points at that file instead of the blank template.
- `write_v50_store()` now writes to a staging directory beside the target path and only replaces
  the target once every array has been written without error, with a short retry-with-backoff
  around the final directory swap (Windows can transiently deny a rename/rmtree on a
  just-finished-writing directory, e.g. an antivirus or indexer scan; retrying clears it rather
  than failing outright). A build that fails or is interrupted partway now leaves a prior good
  store at the target path untouched.
- Regression coverage: `tests/v50/test_store.py` proves a failed second write leaves the prior good
  store's manifest byte-identical, a successful second write does replace it, and no staging
  directory is left behind on success.

**Not yet fixed (known follow-up, out of scope for this pass)**: `_cmd_build` still accumulates the
entire map's tile stream in Python memory and only calls `write_v50_store` once at the very end, and
the harvest-stream/minimap-synthesis pass runs inside one `tempfile.TemporaryDirectory()` whose
`__exit__` deletes all synthesized minimap PNGs on any unhandled exception mid-run. Neither bug was
implicated in the confirmed incident above (the store on disk proved `build` itself completed
cleanly), but a genuine mid-run crash still loses that run's synthesized-minimap work and reports no
partial progress. Left as a documented gap rather than a larger unrequested rewrite.

## Phase 9 incident — the first real full-corpus run hit two more gaps immediately (2026-07-18)

**Symptom (user-reported)**: running the fixed `v50_pipeline_runner.py --confirm` against the full
`0_5_3_3368` corpus for the first time, Kalimdor completed, but the run then crashed with a Python
traceback partway through Azeroth's `finalize` step and stopped entirely -- PVPZone02 and Kalidar
never ran.

**Root cause 1 -- a legitimate `finalize` failure was still treated as fatal for the whole run**:
Azeroth's `finalize` genuinely reported `finalization_state=incomplete` (not the Phase 8 false
negative -- the manifest fed to it this time was the correct one). Diagnosing why required a
hand-rolled script reproducing `finalize_store`'s internal mismatch computation, because the CLI only
ever printed the bare state, never the reason. That diagnosis found two real tiles (row 8 = tile
(2,2), row 9) that legitimately lack `minimap_rgb`: their MCLY texture data was absent, so minimap
synthesis correctly skipped them (consistent with `alpha_256`/`mcnk_flags_16`/`shadow_mask` also
being `unavailable` for the same rows). This is exactly the kind of dirty tile Spec 103 curation
exists to drop -- but `v50_pipeline_runner.py`'s `finalize_cmd` still ran through `run_command(...,
check=True)`, so `finalize`'s exit 1 raised `CalledProcessError` and killed the whole script before
curation, PVPZone02, or Kalidar ever ran.

**Root cause 2 -- the one curation pass silently discarded all object-touched tiles from the corpus**:
the pipeline's only curation step passes `spec103_curate_dataset.py --max-object-coverage 0.0`. That
script's own docstring explains the policy is scoped to one task: "Height under an object is occluded
in the minimap, so an object tile is an impossible target and must be DROPPED, not learned" -- a real
constraint for minimap-to-height reconstruction, not a general judgment that object tiles are bad
data. Baking it in as the *only* curated view meant every real, wanted object tile (518/951 = 54.5%
of Kalimdor, 355/685 = 51.8% of Azeroth, 3/64 = 4.7% of PVPZone02, 12/56 = 21.4% of Kalidar) was
absent from the one manifest that existed, even though v50's frozen signal catalog deliberately keeps
`object_precise_mask`/`object_instance_mask` as first-class signals for object-aware work.

**Fix**:

- `harvester/v50/store.py` gained `FinalizeReport`/`finalize_store_report()`: same completeness
  check as `finalize_store()`, but returns every concrete mismatch reason (missing/mismatched
  signal, row-count disagreement, or up to 5 named rows + a count per required signal missing
  lineage). `_cmd_finalize` now prints every reason, not just the bare state.
- `v50_pipeline_runner.py` is now resilient per map: a `build` failure skips the rest of that map
  only (other maps still run); a non-complete `finalize` no longer aborts anything -- it prints the
  reasons and curation still runs for that map, since curation is what actually drops those rows. A
  final per-map summary table (`build`/`finalize`/`curate`/`curate_object_inclusive`) prints at the
  end of every run so a long run's outcome isn't buried in scrollback.
- The pipeline now writes a second curation manifest per map,
  `curation-<build>-<Map>-object-inclusive/` (`--max-object-coverage 1.0`, object filter effectively
  off), alongside the unchanged strict one. Both apply the same missing_signal/blank_minimap/
  height_normal_mismatch checks; only the object policy differs. Neither manifest ever duplicated
  array data -- both are Parquet row-reference lists over the same raw, untouched store, so no data
  was ever actually lost, only absent from one curated *view*.
- Completed the interrupted real run by hand with the fixed tools: all four maps now have both
  manifests. Strict kept Kalimdor 421/951, Azeroth 328/685, PVPZone02 60/64, Kalidar 24/56.
  Object-inclusive kept Kalimdor 939/951, Azeroth 683/685, PVPZone02 63/64, Kalidar 36/56 (Kalidar's
  ceiling here is the 20 missing-minimap rows, not objects).
- Regression coverage: `tests/v50/test_store.py` proves `finalize_store_report` names the specific
  signal and row(s) responsible, and reports zero mismatches for a genuinely complete store.
  `uv run python -m pytest tests/v50/ tests/test_v50_contract.py tests/test_v50_build_command.py -q`
  -> 120 passed, 2 skipped, 0 failed.
