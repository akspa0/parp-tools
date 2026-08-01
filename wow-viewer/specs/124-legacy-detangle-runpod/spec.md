# Feature Specification: Legacy Python Lane Detangling + New C# RunPod Tooling for v50

**Feature Branch**: `124-legacy-detangle-runpod`

**Created**: 2026-07-30

**Status**: Draft

**Input**: User description: "we need to then port all runpod stuff to c# wherever possible, and
detangle the mess here. leave only stitching in python." Preceded by a same-session finding that a
naive "archive the old curation scripts" attempt would have broken the active V23→V24 RunPod
pipeline, and followed by a full dependency/lane audit of `data-harvester`'s pre-v50 Python tree.

## Problem Statement

`data-harvester` contains roughly a dozen historical dataset/model "lanes" (V14/D1, R1, V15, V16,
V16.1, V16.2, V17, V18, V19, V20, V21, V22, V23, V24, Spec102/V25, Spec103, Spec108, Spec111, plus
an unversioned "Spec 077" research island) alongside the current V50 lane. This is not a clean
"old vs. new" split. A full audit this session (grep-verified import graph, cross-checked against
each lane's own spec status and docstrings) found:

- Several files that look V16-era by name are genuinely load-bearing for still-relevant work: the
  curation helper `v16_curation.py` is imported by V23 (which feeds the still-deployable V24
  pipeline), and `v16_1_dataset.py` is imported by V18/19/20.
- V18 itself is not dead — it is still user-invokable per its own documentation and its curation
  manifest is packaged directly into the V23 RunPod bundle.
- V22 is not dead — it is imported by V23 and documented as feeding a still-open spec (V23-based
  height predictor).
- **Spec103 is not legacy at all** — the package is the current model/loss backend for several
  live V50 trainers (WDL-prior train/infer/evaluate/visualize, terrain-refiner train/infer). Only
  one file in that package, `spec103_curate_dataset.py`, was actually superseded by this session's
  Spec 122 work, and that has already been repointed.
- By contrast, V14/D1, R1, V15, most of V16 (excluding the curation helper), V16.2, V17, V19, V20,
  V21, Spec102/V25 (whose own spec document's status is literally "BLOCKED... no training is
  authorized"), and an unversioned research island (object-library/fractal-brush/teacher-prior
  code) have **zero import edges into any still-relevant lane**, confirmed by exhaustive grep, and
  are safe to physically move out of the active tree.
- Separately: there is **no RunPod deployment infrastructure for the current V50 lane at all**.
  The three existing RunPod lanes (spec103, V23, V24) each have their own packaging script,
  provisioning script, and shell-script bundle, all still functioning and documented as
  deploy-blocked-on-transfer-infrastructure rather than broken. Porting those specific, working,
  frozen scripts to C# would be pure risk for no gain. The actual gap — and the place a new C#
  RunPod tool adds real, safe value — is that V50 (including this session's new Spec 123 model
  work) has nothing at all yet.
- The RunPod packagers that exist are not pure file-copying: two of them (`package_v23_runpod.py`,
  `package_v24_runpod.py`) perform real Zarr-store row/field subsetting via the Python `zarr`
  library, which this project has no C# equivalent for. The pod-provisioning logic
  (`setup_spec077_runpod.py`, shared as a base by the V23/V24 setup scripts) is, by contrast,
  ordinary HTTP-API and process orchestration with no Python-specific or ML-specific dependency —
  a genuinely portable candidate.

## Governing Principle

Only code with zero confirmed import edges from any still-relevant lane is archived, and only after
the same real-caller-search discipline this session already applied is repeated and the full test
suite still passes. Nothing is moved on the assumption that "it looks old." New C# work targets the
actual gap (RunPod tooling for the v50 lane, which has none) rather than risking a port of frozen,
working, already-deploy-blocked infrastructure for lanes that are not this session's concern.

## Relationship To Existing Specs

- **Follows**: Spec 122 (dataset curation), which already repointed `v50_pipeline_runner.py` off
  `spec103_curate_dataset.py` and left documentation-only pointers on the five curation-adjacent
  scripts it touched, without moving any of them (their real-caller search is the model this spec's
  US1 repeats at wider scope).
- **Supports**: Spec 123 (the new ground-up terrain height model), which currently has no RunPod
  packaging path of its own; this spec's new V50 RunPod tooling is where that would eventually
  live, though Spec 123 itself is scoped to run locally on a single consumer GPU and does not
  require this spec to be done first.
- **Does not touch**: the V23, V24, or Spec103 RunPod bundles, packagers, provisioners, or shell
  scripts. All three are confirmed still-referenced, working, and (per standing project memory)
  deploy-blocked on transfer infrastructure, not on code. This spec does not reopen that.

## Out Of Scope (Explicit)

- Modifying, porting, or "improving" the existing V23, V24, or Spec103 RunPod packaging/
  provisioning scripts or their shell-script bundles. They are frozen-but-working; touching them is
  a distinct, separately-authorized decision, not implied by "detangle the mess."
- Building a C# Zarr reader/writer. The two existing packagers that need real Zarr row/field
  subsetting (`package_v23_runpod.py`, `package_v24_runpod.py`) keep doing that in Python; this
  spec does not attempt to replicate Zarr I/O in C#. If a future V50 RunPod packager needs to
  subset a v50 Zarr store, that specific need is a separate, later decision — not assumed here.
- Archiving, moving, or otherwise touching any lane this session's audit could not confirm dead
  with high confidence: `v16_curation.py`, `v16_1_dataset.py`, V18, V22, V23, V24, Spec103,
  Spec108, Spec111, and `pm4_asset_matching` (flagged uncertain, not exhaustively traced) are all
  explicitly left alone by this spec.
- Deleting anything outright. Every archived file is moved (git-tracked, so fully recoverable via
  history), never deleted, matching this project's established convention.
- Any change to model training code, dataset schemas, or the v50 signal catalog.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Confirmed-Dead Lanes Physically Leave the Active Tree (Priority: P1)

A developer working in `data-harvester` no longer has to mentally filter out a dozen historical,
fully-superseded dataset/model generations while looking for currently-relevant code. Every file
this session's audit confirmed has zero import edges from any still-relevant lane is moved to a
clearly-labeled archive location, preserving its git history.

**Why this priority**: This is the actual "detangle" the user asked for, scoped to what is
provably safe rather than what merely looks old.

**Independent Test**: Before the move, run the full test suite and record the pass count. After the
move, run it again from the new locations (or confirm the moved lanes' own tests move with them and
are excluded from the default suite) and confirm zero new failures in the code that remained.

**Acceptance Scenarios**:

1. **Given** the audit's confirmed-dead list (V14/D1, R1, V15, V16 dataset/model files excluding
   `v16_curation.py`, V16.2, V17, V19, V20, V21, Spec102/V25, and the unversioned research island
   excluding `setup_spec077_runpod.py`), **When** the move runs, **Then** every listed file is
   relocated to an archive location with its git history intact, and no file outside this list is
   touched.
2. **Given** the move is complete, **When** the full test suite runs against the remaining active
   tree, **Then** it passes with the same results as before the move (accounting for the moved
   lanes' own tests moving with them), proving no live code depended on anything moved.
3. **Given** a file the audit flagged as uncertain (not confirmed dead), **When** the move runs,
   **Then** that file is explicitly left in place, not included by default, and separately listed
   as a follow-up needing its own dedicated verification before any future move.
4. **Given** the archived location, **When** a future developer looks there, **Then** it is clearly
   labeled as historical/superseded, distinct from the active tree, so nothing new gets built on it
   by accident.

---

### User Story 2 - New C# RunPod Bundle Packaging for the v50 Lane (Priority: P2)

An operator preparing a v50 (or Spec 123) training run for cloud GPU execution can produce a
deployable bundle (selected source files, a content-hash manifest, and an archive) using a C# tool,
without needing the Python-specific Zarr-subsetting machinery the older lanes' packagers use
(v50/Spec 123 packaging does not need to subset a Zarr store the way V23/V24 do, since it packages
source code and small config/checkpoint artifacts, not a store).

**Why this priority**: This is where "port RunPod stuff to C#" produces real, safe value — filling
a genuine gap (v50 has no packaging tool at all) rather than risking a working system.

**Independent Test**: Run the new C# packaging tool against the current v50 source tree and confirm
it produces a bundle whose file list and content hashes are independently verifiable, with a
dry-run mode that reports the plan without writing anything.

**Acceptance Scenarios**:

1. **Given** a configured set of source files/directories for the v50 lane, **When** the packaging
   tool runs, **Then** it produces an archive plus a manifest recording every included file's path
   and content hash.
2. **Given** the same source tree, **When** the tool runs twice, **Then** the manifest's hashes are
   identical both times (deterministic output).
3. **Given** the tool is run without an explicit confirmation flag, **When** it executes, **Then**
   it prints the planned file list and exits without writing an archive (dry-run-first, matching
   every other CLI in this project).
4. **Given** a bundle the tool already produced, **When** its manifest is checked against the
   archive contents, **Then** every recorded hash matches, proving the bundle has not been silently
   altered after packaging.

---

### User Story 3 - New C# RunPod Pod Provisioning (Priority: P2)

An operator can create, monitor, and tear down a RunPod GPU pod, and transfer a packaged bundle to
and from it, using a C# tool — reusing the same general-purpose orchestration logic
(`setup_spec077_runpod.py`'s shared base class today) that already works for the existing V23/V24
lanes, since that logic is HTTP-API and process orchestration with no Python-specific or ML-
specific dependency.

**Why this priority**: This is the second genuinely portable piece identified by the audit — real
infrastructure logic, not a Python-ecosystem-specific dependency — and it is shared, reusable
infrastructure any future lane (including v50) benefits from having in a form that does not require
the Python environment to be present just to provision a pod.

**Independent Test**: Run the new C# provisioning tool against RunPod's API in a dry-run/plan mode
and confirm it reports the same pod configuration decisions (GPU type, datacenter, image) the
existing Python provisioner would make for equivalent inputs, without actually creating a pod
unless explicitly confirmed.

**Acceptance Scenarios**:

1. **Given** the same GPU/datacenter/budget inputs the existing Python provisioner accepts, **When**
   the C# tool plans a pod, **Then** its reported configuration matches what the Python version
   would choose, so the two are provably equivalent for the cases exercised.
2. **Given** a packaged bundle from User Story 2, **When** the C# tool transfers it to a pod,
   **Then** the transfer completes and is verified (e.g. by re-checking the manifest hashes on the
   remote side).
3. **Given** the tool is invoked without explicit confirmation, **When** it runs, **Then** it
   reports the plan (what pod would be created, at what cost) and takes no billed action — matching
   this project's standing rule that only the user launches billed/heavy operations.
4. **Given** a pod the tool provisioned, **When** the operator asks it to tear the pod down,
   **Then** it does so and confirms the pod no longer exists.

---

### User Story 4 - Ambiguous Lanes Get a Documented "Why This Stays" Record, Not Silence (Priority: P3)

A future developer or agent who might otherwise repeat this session's near-miss (almost archiving
`v16_curation.py` and breaking the V23→V24 pipeline) finds a clear, dated record of exactly why
each ambiguous or load-bearing older-looking file was deliberately left in place, and what would
need to be true before it could be safely reconsidered.

**Why this priority**: This is the process fix that makes User Story 1's careful judgment call
durable — without it, the next session has to redo this entire audit from scratch or risk the same
near-miss.

**Independent Test**: A reader can find, for each file the audit flagged as load-bearing or
uncertain, a short note stating which still-relevant lane depends on it and why archiving was
rejected.

**Acceptance Scenarios**:

1. **Given** the audit's list of load-bearing older-looking files (`v16_curation.py`,
   `v16_1_dataset.py`, V18, V22, Spec103, Spec108, Spec111), **When** a developer looks for why each
   is still in the active tree, **Then** a dated note names the specific dependent (e.g. "imported
   by V23, which feeds V24") rather than requiring them to re-derive it.
2. **Given** the one file flagged genuinely uncertain (`pm4_asset_matching`, not exhaustively
   traced), **When** a developer considers archiving it, **Then** the record states plainly that it
   was not fully verified and names the specific follow-up check needed first.

---

### Edge Cases

- A file the audit's grep-based method could not confidently classify (e.g. dynamic imports,
  string-based module loading): it is treated as load-bearing by default (User Story 4's "leave
  alone" bucket), never archived on an unconfirmed guess.
- A test file whose corresponding source file is archived: the test moves with it, and the default
  test-collection scope is updated so the active suite does not silently start failing to collect
  (or silently stop running) tests for code that no longer exists at the old path.
- A future lane wants to reuse something from an archived file: it remains fully readable and
  git-history-intact at its new location; nothing is lost, only relocated.
- The new C# packaging tool is pointed at a source set that does not yet exist for v50 (e.g. before
  Spec 123 lands): the tool must fail closed with a clear "nothing to package" message, not produce
  an empty or misleading bundle.
- The new C# provisioning tool loses network connectivity mid-provision: it must report the failure
  clearly and never claim a pod exists when it does not (or vice versa) — no silent inconsistency
  between what the tool believes and RunPod's actual state.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST relocate every file confirmed, by explicit real-caller-search
  evidence, to have zero import edges from any still-relevant lane (V50, Spec103, V22, V23, V24,
  V18, or the curation-adjacent files Spec 122 already accounted for) to a clearly-labeled archive
  location, preserving version-control history.
- **FR-002**: The system MUST NOT relocate, modify, or delete any file this session's audit flagged
  as load-bearing or uncertain (`v16_curation.py`, `v16_1_dataset.py`, V18, V22, Spec103, Spec108,
  Spec111, `pm4_asset_matching`) without a fresh, dedicated verification pass specific to that file.
- **FR-003**: The full existing test suite MUST pass, with no new failures in the remaining active
  tree, after the archival move.
- **FR-004**: A dated record MUST exist naming, for each load-bearing or uncertain file left in
  place, which still-relevant lane depends on it (or why its status is unconfirmed) and what would
  need to be verified before it could be reconsidered.
- **FR-005**: The system MUST provide a new C# tool that packages a configured set of source
  files/directories into a bundle plus a content-hash manifest, deterministically, with a dry-run
  mode that reports the plan without writing anything.
- **FR-006**: The system MUST provide a new C# tool that provisions, monitors, and tears down a
  RunPod GPU pod and transfers a packaged bundle to/from it, reusing the same general provisioning
  decisions (GPU/datacenter/image selection) the existing Python provisioner makes, with a
  dry-run/plan mode that takes no billed action.
- **FR-007**: Neither new C# tool MUST require or embed any Zarr-store-specific subsetting logic;
  if a future need for that arises, it is explicitly out of this spec's scope.
- **FR-008**: Neither new C# tool MUST modify, wrap, or replace the existing V23/V24/Spec103
  packaging or provisioning scripts; both are new, additive, v50-scoped tools.
- **FR-009**: All new C# tool invocations MUST be dry-run-first and MUST NOT launch any billed pod
  or transfer without an explicit confirmation flag (project-wide rule: only the user launches
  billed/heavy operations).
- **FR-010**: The archival move MUST NOT delete anything; every relocated file remains fully
  readable and recoverable at its new location.

### Key Entities

- **Archive Location**: the clearly-labeled, non-active location confirmed-dead lanes move to;
  distinct from the active `data-harvester` tree, git-history-intact.
- **Load-Bearing/Uncertain Record**: the dated documentation naming why each ambiguous file was
  left in place and what a future re-verification would need to check.
- **RunPod Bundle Manifest**: the content-hash manifest the new C# packaging tool produces,
  independently verifiable against the archive it describes.
- **Pod Provisioning Plan**: the new C# provisioning tool's dry-run output — the pod configuration
  it would create, without actually creating it.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every file on the confirmed-dead list is relocated with git history intact, and the
  active `data-harvester` tree's total file count for pre-v50 lanes drops accordingly.
- **SC-002**: The full test suite's pass count for the remaining active tree is unchanged (modulo
  the moved lanes' own tests moving with them) before and after the archival move.
- **SC-003**: Zero files flagged load-bearing or uncertain by this session's audit are touched by
  the archival move.
- **SC-004**: The new C# packaging tool produces a bundle and manifest for the current v50 source
  tree, and re-running it produces byte-identical manifest hashes.
- **SC-005**: The new C# provisioning tool's dry-run plan for at least one real GPU/datacenter
  configuration matches what the existing Python provisioner would choose for the same inputs.
- **SC-006**: A reader can locate, for every load-bearing/uncertain file, a dated explanation of why
  it was not archived, without re-deriving the dependency chain themselves.

## Assumptions

- The audit performed this session (grep-verified import graph, cross-checked against each lane's
  own spec status/docstrings) is trusted as the basis for the confirmed-dead list; this spec does
  not re-run that audit from scratch, but FR-003's full-test-suite pass is the safety net that
  catches anything the audit missed.
- "Wherever possible" for RunPod porting is read as: port the genuinely language-agnostic
  infrastructure logic (file bundling/hashing, HTTP-API pod orchestration) — not model training,
  not Zarr I/O, and not the working V23/V24/Spec103 bundles themselves. Re-implementing PyTorch
  training or Zarr I/O in C# is a different, much larger undertaking this spec does not attempt.
- `pm4_asset_matching` remains explicitly unverified (User Story 4) rather than guessed at; a future
  dedicated pass is expected before it is either archived or confirmed load-bearing.
- The new C# RunPod tools are additive infrastructure for the v50/Spec 123 lane; nothing in this
  spec requires Spec 123's model work to be complete first, and nothing in Spec 123 requires this
  spec's tooling to exist first — they are parallel, not sequential.
