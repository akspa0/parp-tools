# Implementation Plan: V18 Dataset Canonical Contract

**Branch**: `v0.5.0-dev` | **Date**: 2026-05-27 | **Spec**: [`wow-viewer/specs/001-v18-dataset-spec/spec.md`](wow-viewer/specs/001-v18-dataset-spec/spec.md)

**Input**: Feature specification from [`wow-viewer/specs/001-v18-dataset-spec/spec.md`](wow-viewer/specs/001-v18-dataset-spec/spec.md)

**Note**: This plan is authored manually on the current branch because the local Speckit setup script hard-fails outside numbered feature branches. The implementation direction still follows the approved spec: V18 is the direct versioned successor to the current V16 dataset build flow.

## Summary

Build V18 by copying forward the current V16 dataset-build workflow, keeping the
same streaming harvester-driven shape, and promoting the recently patched-on
signals plus decoded metadata into the canonical build itself. The key
implementation move is a versioned V18 workflow fork centered on a new
[`build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1)-style script, not a net-new architecture. The V18 build must emit a publishable store in one run under [`wow-viewer/output/datasets/v18/`](wow-viewer/README.md:728) with no required post-build patch phase.

The broader parser → decoded → dataset direct-pipeline redesign is intentionally
deferred to a future V20 effort so V18 can stay focused on contract closure and
proof instead of architecture replacement.

**Proof owner reminder** ⚠️: for full ADT MCLY-layer terrain plus visible object
evidence, bounded `gillijimproject_refactor/src/MdxViewer` validation capture is
still the reference proof lane until wow-viewer captures are visually proven,
not merely command-complete.

## Technical Context

**Language/Version**: Python 3.11+ via `uv`, with existing C# / .NET 10 harvest and validation tools reused as-is

**Primary Dependencies**: `numpy`, `zarr`, `pyarrow`, `Pillow`, existing harvester tooling under [`wow-viewer/tools/harvest/`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:54), existing validation capture tooling under [`wow-viewer/tools/validation-capture/`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:55)

**Storage**: Per-build Zarr v3 stores plus Parquet metadata tables and JSON validation reports under a new V18 dataset root

**Testing**: Real-data `uv run` dataset builds and validations, plus [`dotnet build wow-viewer/WowViewer.slnx -c Debug`](wow-viewer/README.md:89) for dependent C# tools when required

**Target Platform**: Windows 11 development workstation using staged client roots under [`output/tmp/wowarchive-clients/`](AGENTS.md:116)

**Project Type**: Python data-harvester CLI workflow with thin orchestration over shared C# tools

**Performance Goals**:

- preserve the current single-pass streaming build shape from [`build_v16_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1)
- avoid intermediate loose shard regeneration beyond the current contract
- eliminate required follow-up patch passes for signals promoted into V18

**Constraints**:

- remain on the current branch; do not require branch renaming or Speckit setup scripts
- no writes outside [`wow-viewer/`](wow-viewer/README.md:1)
- no new reader rewrites; reuse the existing harvest/validation tooling
- no `H:\CLIENTS`; use only staged client roots
- keep plans bite-sized and independently validatable
- do not expand V18 into the future V20 direct-pipeline redesign

**Scale/Scope**:

- one Zarr store per staged client build
- current corpus expectation spans the six active builds documented in [`wow-viewer/README.md`](wow-viewer/README.md:35)
- scope includes build, resume, merge, validation, and consumer contract alignment for V18 dataset stores

## Constitution Check

*GATE: Must pass before implementation. Re-check after design and again after code lands.*

- **Repo independence**: Pass. All planned work stays under [`wow-viewer/`](wow-viewer/README.md:1).
- **Library-first**: Pass with constraint. Reuse the existing harvest and validation tools; do not duplicate format readers. The version bump is a workflow fork, not a parser fork.
- **Real-data validation**: Required for signoff. Every phase ends with staged-client proof under [`output/tmp/wowarchive-clients/`](AGENTS.md:116).
- **Residual model chain**: Pass. This feature changes dataset build ownership, not model topology.
- **Streaming-first dataset pipeline**: Pass. V18 must preserve the current harvester → Python writer streaming pattern from [`build_v16_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1).
- **No untrusted client paths**: Pass only if all commands and docs stay on staged-client roots.
- **Phase discipline**: Required. Do not start consumer cutover until the V18 build and validation contract is proven.

## Project Structure

### Documentation (this feature)

```text
wow-viewer/specs/001-v18-dataset-spec/
├── spec.md
├── checklists/
│   └── requirements.md
└── plan.md
```

### Source Code (repository root)

```text
wow-viewer/
├── data-harvester/
│   ├── scripts/
│   │   ├── build_v16_dataset.py
│   │   ├── build_v18_dataset.py              # new versioned canonical builder
│   │   ├── patch_v16_renderer_truth.py       # logic to absorb or retire
│   │   ├── patch_v18_object_roof_masks.py    # logic to absorb or retire
│   │   └── validate_v18_training_ready.py    # optional follow-on if needed
│   └── src/harvester/
│       ├── v18_dataset.py
│       ├── v16_dataset.py
│       └── v16_2_dataset.py
├── output/
│   └── datasets/
│       ├── v16/
│       └── v18/
└── tools/
    ├── harvest/
    └── validation-capture/
```

**Structure Decision**: Implement V18 as a versioned workflow fork in [`wow-viewer/data-harvester/scripts/`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1), backed by the same shared harvest and validation tools already used by V16. V18 output stores live under a dedicated [`wow-viewer/output/datasets/v18/`](wow-viewer/README.md:728)-style root.

## Phase 1 — Freeze the V18 Promotion Surface

Goal: lock exactly what V18 is promoting from the current V16 + patch workflow so implementation stays narrow.

1. Audit the current V16 canonical builder surface in [`build_v16_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:63).
2. Enumerate the signals already emitted directly by V16 versus the signals currently added by patch flows.
3. Freeze the initial promoted V18 signal set, with special attention to the renderer-truth arrays from [`patch_v16_renderer_truth.py`](wow-viewer/data-harvester/scripts/patch_v16_renderer_truth.py:1) and the current object-roof patch lane.
4. Freeze the required V18 metadata artifacts: [`index.parquet`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1363), [`decoded_metadata.parquet`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1912), [`signal_validation.json`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1346), and [`decoded_metadata_validation.json`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1424).
5. Record which still-derived guidance channels remain loader-time derivations rather than build-time stored arrays.

Validation:

- One written promotion matrix exists in the implementation notes or commit message.
- No ambiguity remains about which former patch-on signals are mandatory in V18 Phase 1.

## Phase 2 — Copy Forward the V16 Builder into a V18 Builder

Goal: create the new V18 build entrypoint as a version bump of the working V16 builder rather than a redesign.

1. Copy forward [`build_v16_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1) into a new V18 builder script.
2. Change dataset root ownership from the V16 output root to a dedicated V18 output root.
3. Update operator-facing command names, banners, usage text, and output paths to speak in V18 terms only.
4. Preserve the existing streaming harvest path, resume semantics, merge entrypoints, and validation command shapes unless a V18 contract requirement forces a bounded change.
5. Keep decoded metadata writing and validation enabled by default in the V18 workflow.
6. Keep compatibility with the same staged-client discovery expectations already used by V16.

Validation:

- The V18 script boots, prints help, and resolves the same staged build inputs as V16.
- A bounded dry run or limited real run creates a V18 store root without touching V16 outputs.

## Phase 3 — Absorb Current Patch Steps into the Canonical V18 Build

Goal: move required post-build patch behavior into the main V18 build so the store is publishable immediately.

1. Inline or shared-call the renderer-truth patch logic from [`patch_v16_renderer_truth.py`](wow-viewer/data-harvester/scripts/patch_v16_renderer_truth.py:151) into the V18 build finalization path.
2. Inline or shared-call the current object-roof mask patch lane if it is part of the approved Phase 1 promoted signal set.
3. Ensure promoted signals are written during the normal V18 build flow, not as a second mandatory operator command.
4. Update index rows, coverage flags, and build metrics so promoted signals are represented as first-class V18 outputs.
5. Preserve explicit optional/additive handling for any signal that is not yet part of the mandatory V18 launch contract.
6. Keep raw-blob preservation out of the critical path unless explicitly enabled, per the spec.

Validation:

- One bounded V18 build writes the approved promoted signals without running a separate patch script.
- Coverage metadata for promoted signals is visible directly in the V18 output artifacts.
- **Reminder** 📌: image-derived promoted signals are contract-visible but still
  require separate visual proof from the bounded MdxViewer lane before being
  treated as semantically trustworthy.

## Phase 4 — Tighten Resume, Merge, and Finalization Rules

Goal: make V18 publishable and resumable without losing parity guarantees.

1. Promote finalized-status checks into the V18 workflow so incomplete stores remain clearly in-progress.
2. Require V18 resume flows to refresh signal and decoded-metadata validation before final publication.
3. Carry decoded metadata parity guarantees through V18 merge paths.
4. Carry promoted-signal coverage through V18 merge paths, including explicit fallback behavior where older source stores lack newer metadata or signal fields.
5. Ensure missing or placeholder provenance is represented explicitly instead of silently dropping rows.
6. Keep the no-patch publication rule enforceable through strict mode.

Validation:

- Resume after interruption does not produce a finalized V18 store until validations rerun.
- A bounded merge produces complete decoded metadata coverage and preserved promoted-signal accounting.

## Phase 5 — Consumer Alignment and Training-Surface Compatibility

Goal: let V18 consumers open the new stores without inventing a second incompatible dataset family.

1. Audit [`v18_dataset.py`](wow-viewer/specs/024-v18-canvas-paste-refinement-layer/spec.md:24) and related loaders for assumptions tied to the V16 root or patched-signal availability.
2. Update V18 consumer surfaces to prefer canonical V18 stores when present.
3. Keep V16/V16.2 compatibility paths alive only where needed for transition or comparison.
4. Ensure the promoted V18 signals line up with the current loader expectations for object guidance, renderer-truth guidance, and decoded metadata-backed provenance.
5. Avoid widening scope into model redesign; only make the minimum consumer changes required to read the V18 store contract.

Validation:

- At least one existing V18 consumer path can open a bounded V18 store and resolve its required arrays and metadata.
- No consumer requires a mandatory legacy patch script to read the new store.

## Phase 6 — Real-Data Proof and Operator Documentation

Goal: prove the V18 workflow on staged builds and document the new canonical path.

1. Run bounded real-data V18 builds on at least one early-era build and one later-era build from [`wow-viewer/README.md`](wow-viewer/README.md:37).
2. Capture reproducible commands, output roots, validation reports, and any coverage limitations.
3. Document the V18 operator path as the new canonical dataset build workflow.
4. Clearly mark which legacy V16 patch scripts are now compatibility/remediation-only after V18 adoption.
5. Update continuity docs if proof changes the active workflow boundary.

**Proof note** 🧭: if wow-viewer capture emits flat or all-black artifacts, do
not treat that as proof. Re-route to bounded MdxViewer compatibility evidence and
record the failure honestly.

Validation:

- Bounded staged-client proof exists for at least two builds with V18 outputs.
- Validation reports show parity success for decoded metadata and approved promoted signals.
- Operator docs describe one canonical V18 build path with no required post-build patch phase.

## Complexity Tracking

No constitution violations are currently required.

The only deliberate tradeoff is a versioned workflow fork at the Python script
surface. This is acceptable because it preserves the existing shared C# reader
ownership, keeps the implementation narrow, and matches the requested “copy
forward and bump the version” direction without duplicating format readers.

Tiny reminder marker 🧪: when progress feels “too smooth,” restate the current
proof owner, current proof artifact, and current unproven gap before continuing.
