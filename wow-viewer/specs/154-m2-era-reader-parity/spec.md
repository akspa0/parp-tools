# Feature Specification: M2 Reader Era Parity (1.x – 3.0.1)

**Feature Branch**: `v0.5.3-dev` (this repository keeps specs on the active dev branch)

**Created**: 2026-08-15

**Status**: Draft

**Input**: User description: "spec something that fixes our damn m2 reader for 1.x-3.0.1, since it works fine in 3.3.5-4.0.0 data to render the m2's, so whatever we do in those builds, make work for the other m2 data"

## Context

Model reading is correct for the Alpha `MDLX` route and is reported correct for the late-3.x / 4.x
`MD20` route that the viewer renders from. Everything between those two ends is broken or refuses to
load, which blocks any cross-era model work — including the question that surfaced it: whether the
2.x Blood Elf rig is the 0.5.3 High Elf rig re-skinned.

### Measured current state (2026-08-15, real staged clients)

Every row below was read from a staged client on that date. Nothing here is inferred from another
build.

| Declared | Build measured | Model | Result |
|---|---|---|---|
| `MDLX` | 0.5.3.3368 | `Creature\HighElf\HighElfMale_Warrior.mdx` | **Works.** 54 bones, 106 sequences, 32 attachments, 128 pivot points |
| `MD20 0x100` | 2.0.0.5610 (pre-release) | `CHARACTER\BloodElf\Male\BloodElfMale.m2` | **Broken.** `bones=0`; geometry dies at bone index 10 |
| `MD20 0x100` | 2.0.0.5610 (pre-release) | `Character\NightElf\Male\NightElfMale.m2` | **Broken.** Identical failure, identical index |
| `MD20 0x107` | 3.0.1.8303 (pre-release) | `CHARACTER\BloodElf\Male\BloodElfMale.M2` | **Refused.** Unhandled "2.x TBC era, not yet supported" |
| `MD20 0x108` | 3.3.0.10958 (retail) | `CHARACTER\BloodElf\Male\BloodElfMale.M2` | **Works.** 151 bones, 155 sequences, 14 aliases, geometry available, 375 bone lookups |
| `MD20 0x109`+ | 4.0.0.11927 (beta) | `CHARACTER\BloodElf\Male\BloodElfMale.M2` | **Broken.** Unhandled failure reading camera records |

**The broken range is exactly `0x100` through `0x107`.** 3.0.1 declares `0x107` and 3.3.0 declares
`0x108`, so the boundary is crisp at the declared version and falls precisely where the refusal range
ends. The range named in the request — 1.x through 3.0.1 — is exact, not approximate.

**The reference is specifically `0x108`, not "3.3.5 through 4.0.0".** 3.3.0 reads cleanly; the 4.0.0
beta does not. Whatever `0x109`+ needs, it is not established by the `0x108` route working.

**A usable Blood Elf skeleton already exists today.** 3.3.0 yields 151 bones with geometry. The
driving comparison — 0.5.3 High Elf against a Blood Elf rig — is therefore reachable *before* any
reader repair, using the `MDLX` and `0x108` routes that already work. Whether a 3.3.0 rig is close
enough to a 2.x rig to answer the question is itself a question, but the comparison is not blocked.
Note the counts are not expected to match (54 against 151): later rigs added bones, so the comparison
must be structural — does the earlier bone set appear within the later one, with corresponding parents
and pivots — not a count equality.

Three measured defects:

- **D1 — Bone data is discarded for the `0x100` era route.** The reader completes and reports zero
  bones. Nothing downstream can pose, compare, or export a skeleton.
- **D2 — The fallback path reads bones with the wrong layout.** The geometry route falls back to the
  late-3.x reader, which locates the bone array at a different header position and steps through it
  at 88 bytes per bone. The `0x100` era's own recorded layout puts the array elsewhere and steps at
  108 bytes — a 20-byte drift per bone. `CHARACTER\BloodElf\Male\BloodElfMale.m2` and
  `Character\NightElf\Male\NightElfMale.m2` from the 2.0.0.5610 pre-release fail **identically at
  bone index 10**, which is where the accumulated drift first lands on non-float bytes. The correct
  layout for this era is already recorded in the codebase, with native-client references; it was
  never connected to a bone parser.
- **D3 — The "working" late route is not verified working.** `CHARACTER\BloodElf\Male\BloodElfMale.M2`
  from the 4.0.0.11927 beta raises an **unhandled** error while reading camera records — it does not
  degrade, it terminates. This directly contradicts the premise that 4.x is sound. Two readings are
  possible and this specification does not choose between them: 11927 is an outlier pre-release, or
  the route the viewer renders through differs from the route model inspection uses. **US1 decides
  this by measurement.**

A fourth observation, load-bearing for scope: **the version word does not identify the layout.**
`0x100` already means two mutually incompatible layouts, disambiguated today by probing the header.
The 2.0.0.5610 pre-release reports `0x100` — which is why a Burning Crusade client never reached the
"2.x unsupported" refusal at all. Any era mapping asserted without per-build measurement is a guess.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Know which builds are actually broken (Priority: P1)

Someone working across client eras can produce, from the staged client library, a build-by-build
table stating for each client: its build identity, what the model files declare, which route handled
them, and — per model — whether identity, skeleton, sequences, geometry, and cameras were read, or
how reading failed.

**Why this priority**: Every other story depends on knowing the true mapping, and the premise of this
work is already known to be partly wrong (D3). Fixing readers against an assumed mapping repeats a
failure this project has paid for before. This story also has standalone value: it converts "the M2
reader is buggy as hell" into a defect list with owners.

**Independent Test**: Run the survey across the staged clients and read the table. It is complete
when every client in the library has a row per surveyed model, no row says "unknown", and D3 is
resolved to a named cause.

**Acceptance Scenarios**:

1. **Given** the staged client library, **When** the survey runs over a fixed set of character models
   present across eras, **Then** it emits one row per build/model with build identity, declared
   version, route taken, and per-section outcome.
2. **Given** a model that fails to read, **When** the survey records it, **Then** the row names the
   section and the element index at which reading failed, rather than only that it failed.
3. **Given** the 4.0.0.11927 result, **When** the survey completes, **Then** the record states whether
   the failure is specific to that build and whether the rendering route and the inspection route
   reach the same outcome for the same file.
4. **Given** any two builds that declare the same version word, **When** they resolve to different
   layouts, **Then** the table records the evidence that distinguished them.
5. **Given** builds separated by only a patch increment — the staged 3.0.1 builds 8303, 8334 and 8391
   are the worked example — **When** the survey runs, **Then** each is surveyed and recorded
   separately, and any difference between them is reported rather than collapsed.

---

### User Story 2 - Skeletons load for the `0x100` era (Priority: P2)

Someone loading a 1.x-era or early-2.x-era character model receives its full skeleton: every bone,
each bone's parent, and each bone's pivot — with no bone silently missing and no value that is not a
real number.

**Why this priority**: This is the concrete blocker (D1 + D2) for the driving use case, and the
correct layout is already recorded — the work is connecting it, not discovering it.

**Independent Test**: Load the Blood Elf and Night Elf character models from the 2.0.0.5610
pre-release and confirm a plausible, complete, finite skeleton rather than zero bones or a failure at
bone index 10.

**Acceptance Scenarios**:

1. **Given** a `0x100`-era character model, **When** it is read, **Then** the reported bone count is
   non-zero and matches the count the file declares.
2. **Given** that model, **When** its bones are read, **Then** every pivot is a finite value and every
   parent index is either "no parent" or a valid in-range bone.
3. **Given** that model, **When** its bone parents are walked from any bone, **Then** the walk reaches
   a root without revisiting a bone.
4. **Given** a model previously failing at bone index 10, **When** it is read, **Then** it reads to
   completion.

---

### User Story 3 - Mid-era builds load instead of refusing (Priority: P3)

Someone opening a model from a build that currently refuses outright gets model data, or a failure
that names what was not understood and at which position — never a blanket "this era is unsupported".

**Why this priority**: Real value, but narrower than it appears: the pre-release Burning Crusade
client measured here declares `0x100` and never hits the refusal. US1 establishes how many staged
builds actually land in the refusing range before effort is spent here.

**Independent Test**: For each staged build that US1 shows lands in the refusing range, open its
character models and confirm data is returned or a positioned, specific failure is reported.

**Acceptance Scenarios**:

1. **Given** a build in the currently-refusing range, **When** its character model is opened, **Then**
   identity, skeleton, and sequences are returned.
2. **Given** a genuinely unsupported layout, **When** it is opened, **Then** the failure names the
   section and position that was not understood.
3. **Given** any model in any staged build, **When** reading fails for any reason, **Then** it fails as
   a reported error and never as an unhandled termination.

---

### User Story 4 - Compare a rig across two eras (Priority: P4)

Someone can take a model from the Alpha route and a model from any repaired M2 route and get their
skeletons and sequence tables in one shape that can be compared directly — bone names or identifiers,
parent structure, pivots, and per-sequence identity and duration.

**Why this priority**: This is what the driving question actually needs, and it is far smaller than a
motion-export pipeline.

**This story is not blocked by US2/US3.** The `MDLX` and `0x108` routes both work today, so a 0.5.3
High Elf rig can be compared against a 3.3.0 Blood Elf rig immediately. US2/US3 widen which builds can
participate — letting a 2.x Blood Elf rig into the comparison, which is the era the question is really
about — but they are not a precondition for a first answer.

**Independent Test**: Emit the comparable shape for a 0.5.3 High Elf model and a Blood Elf model, plus
Night Elf and Human as controls, and confirm the outputs can be diffed without hand-massaging.

**Acceptance Scenarios**:

1. **Given** an Alpha-route model and an M2-route model, **When** both are emitted, **Then** the two
   outputs carry the same fields with the same meanings.
2. **Given** two models with structurally identical rigs, **When** they are compared, **Then** the
   comparison reports correspondence; **Given** two unrelated rigs, **Then** it reports difference.
3. **Given** a comparison result, **When** it is recorded, **Then** it carries the build identity of
   both sides.

### Edge Cases

- A model declaring a version word that no staged build exhibits.
- Two builds declaring the same version word with incompatible layouts — already real at `0x100`.
- A model whose declared section count disagrees with the bytes actually present.
- A model with zero bones legitimately (props, doodads) versus one whose bones were silently dropped —
  these must not look alike.
- A bone parent index forming a cycle, or pointing outside the array.
- A skeleton that reads without error but is subtly wrong: correct count, finite values, implausible
  pivots. US1's survey is what catches this class, not per-model success alone.
- Companion/external animation data present in some eras and absent in others.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST record, per staged build, the build identity, the declared model version,
  the layout actually used, and the evidence that selected it.
- **FR-002**: The system MUST NOT infer a build's layout from another build's version word. Layout
  selection MUST rest on evidence observed in the file being read.
- **FR-003**: The system MUST read the complete bone set for every era it claims to support: count,
  parent linkage, and pivot per bone.
- **FR-004**: The system MUST reject a skeleton it cannot read completely, rather than returning a
  partial or empty one that reads as valid.
- **FR-005**: The system MUST distinguish "this model has no bones" from "this model's bones were not
  read", and MUST surface that distinction to callers.
- **FR-006**: Reading failure MUST identify the section and the element position at which it occurred.
- **FR-007**: No model in the staged library may cause an unhandled termination. Every failure MUST be
  a reported, catchable error.
- **FR-008**: The system MUST continue to read Alpha-route models exactly as it does today, with no
  change to their observable output.
- **FR-009**: The system MUST continue to read the late-3.x/4.x route at least as well as today, with
  any behaviour change recorded as a deliberate correction and evidenced.
- **FR-010**: The system MUST expose skeleton and sequence data in one shape shared across the Alpha
  route and the M2 routes, sufficient to compare two rigs without per-era special handling.
- **FR-011**: Every support claim MUST name the exact build it was verified against and a model read
  from it. Support claimed without such a record is not support, and support for one build MUST NOT be
  recorded as support for any other build — including the adjacent patch.
- **FR-012**: Where the codebase already records a layout for an era, the reader MUST consume that
  record rather than restating the layout separately.

### Key Entities

- **Build identity**: The specific client a file came from — version and build number — carried
  alongside every measurement and every support claim.
- **Layout profile**: The description of where a model's sections sit and how large each element is,
  for one era; the authority a reader consults.
- **Skeleton**: The bone set for one model — per bone, an identity, a parent, and a pivot.
- **Sequence table**: The animations a model declares — per entry, an identity, a duration, and
  whether it stands alone or refers elsewhere.
- **Survey record**: One build/model row from US1 — identity, declared version, layout used, per-section
  outcome, and failure position where applicable.
- **Rig comparison**: Two skeletons plus their sequence tables in the shared shape, with both build
  identities attached.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every client in the staged library has a complete survey record, with no "unknown"
  layout and no unexplained failure.
- **SC-002**: The 4.0.0.11927 camera failure is resolved to a named cause, and the record states
  whether the rendering and inspection routes agree for the same file.
- **SC-003**: 100% of character models surveyed from builds declaring `0x100` return a non-zero,
  complete, finite skeleton whose bone count matches the file's own declaration.
- **SC-004**: Zero models in the staged library cause an unhandled termination.
- **SC-005**: Every model that reads today still reads, with identical output for the Alpha route.
- **SC-006**: A 0.5.3 High Elf rig and a Blood Elf rig can be compared in one operation, with Night
  Elf and Human available as controls, and the result names both build identities.
- **SC-007**: Every build the system claims to support names at least one model it was verified
  against, and no build is claimed on the strength of a different build's result.
- **SC-008**: No client at or beyond 4.0.1 is read, surveyed, or referenced by this work.

## Assumptions

- The staged client library is the source of truth; no client files are added to the repository, and
  client roots stay runtime configuration.
- Readers remain library-owned; command-line surfaces stay thin and gain no parsing logic.
- **The reference is `0x108`, measured at 3.3.0.10958 — not "3.3.5 through 4.0.0".** The 4.0.0 beta
  fails (D3), so `0x109`+ is treated as unverified and is not the thing being matched. If `0x109`+
  proves defective, repairing it is in scope, but it is not the yardstick.
- **The range "1.x through 3.0.1" is exact.** Measured: 3.0.1 declares `0x107`, 3.3.0 declares `0x108`.
  The broken range is precisely `0x100`–`0x107` and the boundary is crisp at the declared version.
  *(An earlier draft assumed 3.0.1 and 3.3.5 shared a version word and called the range approximate.
  That assumption was wrong and measurement replaced it.)*
- **Hard scope ceiling: 4.0.0.** Nothing beyond it is read, planned for, surveyed, or referenced.
  Later formats are not a future phase of this work and must not shape any decision in it.
- **The unit of support is the build, not the version, and not the expansion.** These are rolling
  releases: structurally significant changes land in `0.0.1` patch releases, unannounced and without a
  version-word bump. Two builds one patch apart can differ in ways that break a reader written against
  the other. The staged library already shows this — three separate 3.0.1 builds (8303, 8334, 8391)
  are three distinct artifacts, and `0x100` covers two incompatible layouts while a Burning Crusade
  pre-release declares `0x100` and a later Burning Crusade build declares `0x107`.
- Consequently "era" is only ever a label for a **measured group of builds**, never a category a build
  can be placed in by reasoning about its release name or its neighbours. A reader may be claimed to
  support a build only if that build was read. Support for one build never implies support for the
  build one patch away. This is what FR-002 and FR-011 exist to enforce, and it is why US1 surveys
  every staged build rather than one representative per expansion.
- Motion export — BVH, FBX, pose clips, and the pose farm of archived Spec 053 — is **out of scope**.
  That tooling does not exist (verified: no project, no commits) and this work does not create it.
  US4 delivers only the comparable skeleton and sequence shape.
- Rendering, materials, textures, particles, and skin profile selection are out of scope except where
  a defect blocks skeleton or sequence reading.
- Correct-looking output is not proof of correct parsing; plausibility checks over a surveyed corpus
  are the evidence, not single-model success.
