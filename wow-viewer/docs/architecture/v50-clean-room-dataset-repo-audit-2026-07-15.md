# V50 Clean-Room Dataset and Repository Audit — 2026-07-15

**Owner**: Spec 109

**Status**: Clean-slate bootstrap complete 2026-07-16; no dataset promoted

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
| `v50_build_dataset.py` | Thin wrapper over `spec108_build_mixed_curriculum.py` | Inventory only | Migrate | Establish a canonical v50 owner after source-verification gates exist |
| Other `v50_*.py` commands | Thin wrappers over Spec 103/108 scripts | Inventory only | Migrate | Focused parity tests, then move ownership out of historical spec command modules |
| Current mixed-store builder | Accepts arbitrary real/synthetic stores and writes v50 metadata | Failed trust gate | Quarantine | Reject unverified sources; bind source schema, manifest, hashes, and row lineage before writing output |
| `v50_contract.py` | Checks release syntax and three top-level attributes | Partial contract proof | Keep/extend | Validate complete store schema, index identity, source identities, row lineage, and downstream hashes |
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
