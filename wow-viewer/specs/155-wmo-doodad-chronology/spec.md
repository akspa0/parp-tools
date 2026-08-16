# Feature Specification: Asset Reference Inventory — Expected vs Catalogued vs Present

**Feature Branch**: `v0.5.3-dev` (this repository keeps specs on the active dev branch)

**Created**: 2026-08-16

**Status**: Draft

**Input**: User description: "take inventory of every doodad in every wmo… a way to log the doodads loaded in wmo's, and then a way to analyze all wmo's, and then also a way to analyze all mdx or m2 files as well, for texture data that may or may not exist… We need an inventory of what the game data expects exists, versus what the listfiles provide, as we believe there are a lot of missing assets that no one knows about, which may have related data in the existing data corpus, which could be used to patch things back into full function."

## Context

Game data is full of references: a world object names the doodads it places, a world object names the
textures its materials use, a model names the textures it draws with. Each reference is a claim that
some asset exists. **Nobody has ever compared the full set of those claims against what the game
actually ships.**

This feature builds that comparison, delivered through the existing inspection tooling, which already
reads every format involved and needs analysis surfaces rather than new readers.

### The phenomenon, and why it is worth sweeping for

The real engine draws an untextured object **bright neon green**. This holds in at least every
pre-alpha and beta Vanilla build. A missing texture is therefore not a silent defect — it is a
recognisable artifact, and the corresponding reference is recoverable from the data.

Two effect objects on the side of Mt. Hyjal are the illustration. They appear as green smoke because
their water-spray texture is missing. They became well known when Classic launched in 2018 and
explorers found them exactly where they had been claimed to be, and inspecting them in this project's
own viewer showed the cause directly: the texture reference resolves to nothing.

**Those two objects are known. The point of this feature is everything that is not.** One anecdote
became famous because someone happened to walk past it; there is no reason to think it is rare, and no
way to find the rest by walking. Only a full-corpus sweep can say how large the missing-asset
population actually is. Known instances are useful afterwards as a sanity read on a report that already
exists — nothing here is scoped around finding a particular object.

### Three sets, not two

The interesting output is not a list — it is the disagreement between three sets:

- **Expected** — every asset the game data references.
- **Catalogued** — every asset the listfiles name.
- **Present** — every asset actually readable from the build.

| Catalogued | Present | Meaning |
|---|---|---|
| yes | yes | Working reference; nothing to report |
| yes | no | The catalogue claims an asset the build does not contain |
| no | yes | Catalogue gap — the asset is there but unnamed by any listfile |
| no | no | **Missing asset.** The Mt. Hyjal case |

And separately: **present but never referenced** — assets the game ships and nothing uses. These are
orphans, and they are the donor pool for repair, because a missing reference and an unreferenced asset
that resembles it are very often the same asset under a changed name.

### Measured grounding (2026-08-16, staged clients)

| Build | Files catalogued | World objects | Models |
|---|---|---|---|
| 0.5.3.3368 | 42,765 | 492, as per-asset containers under the loose `World` tree | 5,545 |
| 3.0.1.8303 | 131,106 | 9,711, as packaged archive entries | 17,296 |

Packaging differs by build and the existing data-access layer already handles both. One caution
carries forward: **archive internal listfiles are the "catalogued" set, not the "present" set.**
Per-asset containers carry no internal listfile, so an index built from them names one world object for
the earliest build rather than 492. That is precisely the catalogue-versus-present distinction this
feature exists to measure — but it must not be mistaken for the corpus itself when sweeping.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Sweep every world object and every model in a build (Priority: P1)

Someone points the tooling at a build and gets the complete reference ledger for it — every world
object and every model, every reference each makes, and whether each resolves. Reporting the references
of a single asset falls out of the same extraction and is available as a debugging view.

**Why this priority**: The whole premise is that there are many unknown missing assets. One-at-a-time
inspection cannot find what nobody knows to look for; only a full sweep can, and the size of the
missing population is itself the first thing worth knowing.

**Independent Test**: Sweep the earliest staged build end to end and confirm the examined counts match
its actual contents — 492 world objects and 5,545 models — with every reference carrying an outcome.

**Acceptance Scenarios**:

1. **Given** a build, **When** it is swept, **Then** every world object and every model in it is
   examined.
2. **Given** a sweep, **When** it completes, **Then** it reports how many assets of each kind it
   examined, so an under-counted sweep is visible rather than silent.
3. **Given** a build packaging world objects as per-asset containers, **When** it is swept, **Then** all
   492 are examined, not the one an internal-listfile index would name.
4. **Given** an asset that cannot be read at all, **When** the sweep encounters it, **Then** it is
   recorded as unreadable and the sweep continues.
5. **Given** a swept build, **When** results are reported, **Then** the count of unresolved references
   is stated, so the scale of the missing-asset population is visible immediately.
6. **Given** a single asset, **When** its references are dumped, **Then** the same extraction reports
   its doodad and texture references with their outcomes.

---

### User Story 2 - Compare what the data expects against what the catalogue names (Priority: P2)

Someone gets, for one build, the disagreement between what the game data references, what the listfiles
name, and what the build actually contains — including assets that ship but nothing references.

**Why this priority**: This is the headline deliverable and the thing that has never been produced. It
sits behind the sweep only because it is a function of the sweep being complete.

**Independent Test**: Produce the comparison for one build and confirm every referenced asset lands in
exactly one category, with orphans listed separately.

**Acceptance Scenarios**:

1. **Given** a swept build, **When** the comparison runs, **Then** every referenced asset is classified
   as working, catalogue-claims-but-absent, catalogue-gap, or missing.
2. **Given** an asset present in the build but named by no listfile, **When** it is classified, **Then**
   it is reported as a catalogue gap and **not** as missing.
3. **Given** an asset the listfiles name but the build does not contain, **When** it is classified,
   **Then** it is reported as such and not conflated with an asset nothing references.
4. **Given** assets present but referenced by nothing, **When** the comparison runs, **Then** they are
   listed as orphans.
5. **Given** the comparison for a build, **When** it is reported, **Then** counts per category are
   stated, so the scale of the missing-asset population is visible.

---

### User Story 3 - Find the asset that was probably meant (Priority: P3)

For each missing reference, someone learns whether the build contains an asset that is plausibly the
intended one — an orphan with a near-identical name, a differing extension, a moved path, a spelling or
casing drift.

**Why this priority**: This converts the inventory into something repairable, and it is where the
"related data already in the corpus" idea is tested. It carries the feature's main false-positive risk,
so it is specified before any repair happens.

**Independent Test**: Run candidate matching over one build's missing references and confirm each
candidate is an asset verified present in that same build, with the nature of the difference stated.

**Acceptance Scenarios**:

1. **Given** a missing reference with no plausible candidate, **When** it is classified, **Then** it is
   reported as having none.
2. **Given** a missing reference where a present asset differs only in spelling, punctuation, casing,
   extension, or path, **When** it is classified, **Then** that candidate is reported with the nature of
   the difference.
3. **Given** any candidate, **When** it is reported, **Then** it is verified present **in that same
   build** — never drawn from another build and never invented.
4. **Given** several plausible candidates, **When** they are reported, **Then** all are listed and none
   is silently chosen.
5. **Given** the full missing-reference population of a build, **When** candidate matching runs,
   **Then** it reports how many have no candidate, one, or several — coverage across the population,
   not a per-case verdict.

---

### User Story 4 - Date assets across builds (Priority: P4)

Someone gets, across the staged builds, the window in which each asset first appears, and where it
happens, when it disappears.

**Why this priority**: Valuable and originally the driving goal, but it depends on the per-build ledger
being complete, and the missing-asset finding delivers value without it.

**Independent Test**: Produce the timeline across at least three builds and confirm every asset carries
an introduction window bounded by named builds.

**Acceptance Scenarios**:

1. **Given** an asset present in one build and absent from an earlier one, **When** the timeline is
   built, **Then** its introduction is recorded as a window bounded by those two named builds.
2. **Given** an asset present earlier and absent later, **When** the timeline is built, **Then** the
   disappearance is recorded as its own fact.
3. **Given** any timeline entry, **When** it is reported, **Then** it names the builds it rests on and
   states that its granularity is between-build.
4. **Given** two builds separated only by a patch increment, **When** they are compared, **Then** they
   are treated as distinct artifacts and neither stands in for the other.
5. **Given** a reference that is missing in one build and resolves in another, **When** the timeline is
   built, **Then** that is recorded — an asset that arrived late is distinguishable from one that never
   shipped.

---

### User Story 5 - Repair broken references, on purpose and reversibly (Priority: P5)

Someone can have missing references repointed at assets that genuinely exist in that build, with a full
record of what changed and the ability to undo it.

**Why this priority**: The payoff, but it modifies data, so it comes last and only once candidate
matching is trustworthy.

**Independent Test**: Repair a set of references, confirm each change is recorded with its evidence, and
confirm the original state can be restored exactly.

**Acceptance Scenarios**:

1. **Given** repair is not requested, **When** any analysis runs, **Then** nothing is modified.
2. **Given** a repair is applied, **When** it completes, **Then** the record holds the original
   reference, the replacement, and the evidence for the match.
3. **Given** a repair is applied, **When** it is undone, **Then** the data returns to its exact prior
   state.
4. **Given** a missing reference with no candidate or several, **When** repair runs, **Then** it is left
   untouched and reported.

---

### User Story 6 - Know what the conversion tools can actually do (Priority: P6)

Someone gets a straight statement of which conversion operations currently work, which are broken, and
how each compares to the maturity of terrain reading — before any parity is promised.

**Why this priority**: "Make them work properly again" cannot be scoped before the current state is
known. This project has twice been caught by a capability that was documented or assumed but not real.

**Independent Test**: Run each conversion operation against real staged data and record the outcome,
then compare that record against what the tools claim to do.

**Acceptance Scenarios**:

1. **Given** each conversion operation, **When** it is exercised against real data, **Then** its outcome
   is recorded with the build it ran against.
2. **Given** an operation that fails, **When** it is recorded, **Then** the record states what failed,
   not only that it failed.
3. **Given** the survey is complete, **When** parity work is scoped, **Then** it targets recorded
   defects rather than assumed ones.
4. **Given** any documented capability, **When** it is compared against the survey, **Then**
   documentation that overstates what exists is corrected.

### Edge Cases

- An asset intentionally missing to produce an in-game effect. The inventory reports the fact; it does
  not decide intent, and repair must never assume a missing asset is a defect.
- A reference that resolves only because a listfile names it, with nothing actually readable behind it.
- The same asset present under two paths differing only in case.
- A reference using a different extension than the asset actually shipped with.
- An asset referenced by hundreds of objects — references and assets must never be conflated in counts.
- A build where the catalogue is far smaller than what is present, which is already the measured case.
- A model or world object that cannot be read at all; it must not abort a sweep.
- Renames across builds, which appear as one disappearance plus one introduction unless recognised. The
  timeline must say it cannot tell rather than manufacture an introduction.
- Assets referenced by data outside world objects and models, which this feature does not sweep and must
  not implicitly claim are unreferenced.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST report, for a single world object, every doodad and every texture it
  references.
- **FR-002**: The system MUST report, for a single model, every texture it references.
- **FR-003**: The system MUST sweep every world object and every model in a build.
- **FR-004**: The system MUST obtain the corpus through the data-access layer that already resolves both
  per-asset containers and packaged archive entries, and MUST NOT derive it from archive internal
  listfiles.
- **FR-005**: The system MUST report how many assets of each kind it examined, so an under-counted sweep
  is visible rather than silent.
- **FR-006**: The system MUST classify each referenced asset as working, catalogue-claims-but-absent,
  catalogue-gap, or missing — treating "named by a listfile" and "readable from the build" as separate
  facts that are never merged.
- **FR-007**: The system MUST report assets that are present but referenced by nothing.
- **FR-008**: The system MUST propose candidate matches for missing references only from assets verified
  present in the same build, MUST state the nature of the difference, and MUST list all candidates
  rather than choosing.
- **FR-009**: The system MUST continue a sweep when an individual asset cannot be read, recording it as
  unreadable.
- **FR-010**: The system MUST express asset introduction as a window bounded by two named builds and
  MUST state the granularity of any chronology claim.
- **FR-011**: The system MUST treat each build as a distinct artifact; a finding for one build MUST NOT
  be recorded as a finding for another, including the adjacent patch.
- **FR-012**: The system MUST NOT modify any data unless repair is explicitly requested.
- **FR-013**: Every repair MUST record the original reference, the replacement, and the evidence, and
  MUST be reversible to the exact prior state.
- **FR-014**: The system MUST leave untouched, and report, any missing reference it cannot repair
  unambiguously.
- **FR-015**: The system MUST record the current working state of each conversion operation against real
  data before parity work is scoped.
- **FR-016**: Every record MUST carry the build identity it came from, and MUST contain paths, outcomes,
  and provenance only — never client file content.

### Key Entities

- **Build identity**: The specific client an observation came from, carried by every record.
- **Referencing asset**: One world object or model in one build.
- **Reference**: One claim by a referencing asset that some asset exists, with its kind — placed doodad,
  world-object texture, or model texture.
- **Referenced asset**: The target, tracked across builds by path.
- **Catalogue entry**: An asset named by a listfile. Naming is not existence.
- **Presence**: Whether an asset is actually readable from the build, determined independently of the
  catalogue.
- **Reference classification**: Working, catalogue-claims-but-absent, catalogue-gap, or missing.
- **Orphan**: An asset present in a build and referenced by nothing swept.
- **Candidate match**: A present asset in the same build proposed for a missing reference, with the
  nature of the difference.
- **Introduction window**: The interval, named by two builds, in which an asset first appears.
- **Repair record**: Original, replacement, evidence, and what is needed to reverse it.
- **Conversion capability record**: One operation, one build, what happened.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A sweep of a build examines its entire corpus — every world object and every model on a
  readable route — and reports the total count of unresolved references, so the size of the
  missing-asset population is known rather than estimated.
- **SC-002**: A sweep reports examining 492 world objects and 5,545 models for the earliest staged
  build; a run reporting 1 world object indicates the internal-listfile index was used instead of the
  data-access layer and is a failure.
- **SC-003**: 100% of references found by a sweep carry a resolution outcome; none is unclassified.
- **SC-004**: 100% of referenced assets are classified into exactly one of the four categories, with
  per-category counts reported.
- **SC-005**: Zero assets are reported missing solely because a listfile omitted them.
- **SC-006**: Every proposed candidate is verified present in the same build; zero candidates reference a
  non-existent or cross-build asset.
- **SC-007**: A sweep completes over a full build without a single unreadable asset aborting it.
- **SC-008**: Every asset in the timeline carries an introduction window bounded by two named builds.
- **SC-009**: Zero data is modified when repair is not requested.
- **SC-010**: 100% of applied repairs are reversible to the exact prior state, demonstrated by restoring
  and comparing.
- **SC-011**: Every conversion operation has a recorded outcome against real data before any parity
  claim is made.
- **SC-012**: Every record names the build it came from.

## Assumptions

- Delivery is through the existing inspection tooling. The readers for world objects, their doodad and
  texture tables, and model texture tables already exist and are reused, not reimplemented; what is
  added is per-asset reporting, corpus-wide sweeping, and the comparison.
- The staged client library is the source of truth. No client data enters the repository; records carry
  paths, outcomes, and provenance only. Client roots stay runtime configuration.
- **A missing asset is a finding, not a defect.** Some are deliberate, producing in-game effects. The
  Mt. Hyjal objects are the reason this distinction is written down rather than assumed away.
- The neon-green rendering of untextured objects is treated as established for pre-alpha and beta
  Vanilla builds and is used as the in-world corroboration of a missing texture, not as a detection
  method — detection is from the data.
- **`uniqueId` is out of scope as a chronology source.** It dates placements, not assets. Correlating
  the two is separate work.
- Asset introduction is dated between builds. Finer granularity is not assumed and is not claimed.
- A rename cannot in general be distinguished from a removal plus an introduction; where the system
  cannot tell, it says so.
- Sound and other asset classes are out of scope. The same method may extend later; nothing here
  assumes it.
- Orphan detection is bounded by what is swept. An asset referenced only from data this feature does not
  read would appear as an orphan, and the reports state that limit.
- Repair operates on data the operator owns and has chosen to modify. It is never part of analysis.
