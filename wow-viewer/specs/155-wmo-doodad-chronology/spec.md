# Feature Specification: WMO Doodad Inventory and Asset Chronology

**Feature Branch**: `v0.5.3-dev` (this repository keeps specs on the active dev branch)

**Created**: 2026-08-16

**Status**: Draft

**Input**: User description: "take inventory of every doodad in every wmo, to determine the order in which each object was introduced into the game… per-build… and across many builds… to keep track of objects through the years or missing objects or objects with incorrect names that just don't load because they don't exist… We should offer the ability to fixup old data or broken data in that way, as part of our conversion tools, which need to be made to work again, properly."

## Context

There is no accurate record of which art assets entered the game when. `uniqueId` is the chronology of
record for **world layout**, but it is a doodad-placement clock and does not date the assets
themselves. What every WMO references, tracked across builds, is the other available signal — and it
additionally exposes references that resolve to nothing, which are silently invisible in a running
client.

### Measured grounding (2026-08-16, staged clients)

| Build | Files known | WMO packaging | Corpus size |
|---|---|---|---|
| 0.5.3.3368 | 42,765 | One per-asset container per WMO under the loose `World` tree | 532 |
| 3.0.1.8303 | 131,106 | Ordinary entries inside packaged archives | 9,711 |

**The corpus is already readable, and the existing data-access layer already handles both shapes.**
The archive catalogue scans the loose tree for per-asset containers, the data source maps a container
back to the logical asset path it holds, the native archive service knows these are listfile-less
single-file archives and already de-duplicates their double registration so enumeration does not emit
each WMO twice, and the V14 converter documents that it handles per-asset containers automatically.
The viewer reads this data today. **Nothing about corpus access needs to be invented.**

**One surface does not see them, and that is a usage trap, not a defect.** Building an index cache from
archive *internal listfiles* returns a single WMO for the earliest build, because per-asset containers
carry no internal listfile by design. That surface answers "what does this archive's listfile declare",
which is a different question from "what world objects does this build contain". Choosing it for corpus
enumeration would under-count 532 as 1 and produce a timeline dating every asset later than it arrived,
while looking authoritative.

**Consequences that shape this feature:**

- The inventory is built on the existing data-access layer, which already resolves both packaging
  shapes. It is not built on the listfile index.
- Where a build keeps its world objects, and how they are packaged, is **build-dependent**; the two
  staged shapes above already differ. The feature reports what it found rather than assuming a layout.
- "This asset does not exist" is the feature's most dangerous claim. It must rest on a failed lookup
  through the data-access layer against what the build actually contains — never on absence from an
  index, which is exactly the trap above in its most damaging form.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Inventory every doodad reference and say which resolve (Priority: P1)

Someone gets, for one build, every doodad reference made by every world object in that build, each
marked as resolving to a real asset or not — with the corpus taken from the data-access layer that
already reads this data, and its size reported so the reader can see it is complete.

**Why this priority**: This is the inventory itself and the direct source of the missing-asset finding.
It stands alone: even with no chronology, a per-build list of references that resolve to nothing is
immediately useful. Corpus access is not a precondition to build — it exists — but the reported corpus
size is the reader's check that the right surface was used, so it is part of this story rather than a
separate one.

**Independent Test**: Produce the inventory for one build, confirm the reported corpus size matches
what the build holds (532 for the earliest staged build, not 1), and confirm every reference carries a
resolution outcome with none left unclassified.

**Acceptance Scenarios**:

1. **Given** a build's world objects, **When** the inventory runs, **Then** every doodad reference from
   every object is recorded with the object that made it.
2. **Given** any build, **When** the inventory runs, **Then** it reports how many world objects it
   examined and how they were packaged, so an under-counted corpus is visible rather than silent.
3. **Given** a build packaging world objects as per-asset containers, **When** the inventory runs,
   **Then** all of them are examined — the earliest staged build yields 532, not 1.
4. **Given** a reference, **When** it is resolved, **Then** the outcome distinguishes "found",
   "not found", and "could not be checked" — the third is never silently merged into the second.
5. **Given** a reference absent from an index but present in the build's actual contents, **When** it
   is resolved, **Then** it is reported as found.
6. **Given** the same object appearing in more than one build, **When** inventories are compared,
   **Then** its references can be compared across those builds.

---

### User Story 2 - Separate what never shipped from what was misnamed (Priority: P2)

For each reference that resolves to nothing, someone learns whether the build contains a near-match
that is plausibly the intended asset, or whether nothing resembling it exists.

**Why this priority**: This is what converts a list of broken references into something actionable, and
it is the precondition for any repair. It also carries the feature's main false-positive risk, so it is
specified before repair rather than alongside it.

**Independent Test**: Run classification over one build's unresolved references and confirm each is
labelled, with the evidence for any near-match stated and inspectable.

**Acceptance Scenarios**:

1. **Given** an unresolved reference with no plausible match, **When** it is classified, **Then** it is
   labelled as having no candidate.
2. **Given** an unresolved reference where a real asset in the same build differs only in spelling,
   punctuation, casing, extension, or path, **When** it is classified, **Then** the candidate is
   reported with the nature of the difference.
3. **Given** any candidate, **When** it is reported, **Then** it is an asset that exists **in that same
   build** — never one drawn from a different build or invented.
4. **Given** several plausible candidates, **When** they are reported, **Then** all are listed and none
   is silently chosen.

---

### User Story 3 - Date each asset across builds (Priority: P3)

Someone gets, across the staged builds, the window in which each asset first appears and — where it
happens — when it disappears.

**Why this priority**: This is the driving goal. It sits behind the inventory only because it is a
function of the inventory being complete: a timeline built over an under-counted corpus is worse than
no timeline, because it looks authoritative while dating every asset later than it arrived.

**Independent Test**: Produce the timeline across at least three builds and confirm every asset carries
an introduction window bounded by named builds.

**Acceptance Scenarios**:

1. **Given** an asset present in one build and absent from an earlier one, **When** the timeline is
   built, **Then** its introduction is recorded as a window bounded by those two named builds.
2. **Given** an asset present in an earlier build and absent later, **When** the timeline is built,
   **Then** the disappearance is recorded as its own fact, not as an error.
3. **Given** any timeline entry, **When** it is reported, **Then** it names the builds the claim rests
   on and states that its granularity is between-build.
4. **Given** two builds separated by only a patch increment, **When** they are compared, **Then** they
   are treated as two distinct artifacts and neither stands in for the other.

---

### User Story 4 - Test whether ordering within a file dates assets more finely (Priority: P4)

Someone learns whether the order in which doodads appear inside a world object's own tables carries
introduction chronology — and gets a straight answer, including "it does not".

**Why this priority**: A finer-grained clock than between-build would be valuable, but it is a
hypothesis, and US3 supplies the ground truth to test it against. Reporting a within-file chronology
before testing it would be asserting a clock nobody has checked.

**Independent Test**: Take assets whose introduction window is already known from US3, check whether
within-file ordering predicts that known order at better than chance, and report the result either way.

**Acceptance Scenarios**:

1. **Given** assets with known introduction windows, **When** within-file ordering is tested against
   them, **Then** the test reports agreement, disagreement, or no relationship.
2. **Given** the test finds no relationship, **When** it is reported, **Then** the null result is
   recorded as a finding and within-file ordering is not used for chronology.
3. **Given** the test is run, **When** it is reported, **Then** it states whether it could have
   detected the relationship had one existed — a test that could not have seen the effect is not
   evidence of its absence.
4. **Given** the test finds a relationship, **When** any chronology uses it, **Then** that chronology
   states it rests on the finer-grained signal and cites the validation.

---

### User Story 5 - Repair broken references, on purpose and reversibly (Priority: P5)

Someone can have broken references repointed at assets that genuinely exist in that build, with a full
record of what changed and the ability to undo it.

**Why this priority**: The payoff, but it modifies data, so it comes last and only after classification
is trustworthy.

**Independent Test**: Repair a set of references, confirm each change is recorded with its evidence,
and confirm the original state can be restored exactly.

**Acceptance Scenarios**:

1. **Given** repair is not requested, **When** any analysis runs, **Then** nothing is modified.
2. **Given** a repair is applied, **When** it completes, **Then** the record holds the original
   reference, the replacement, and the evidence for the match.
3. **Given** a repair is applied, **When** it is undone, **Then** the data returns to its exact prior
   state.
4. **Given** an unresolved reference with no candidate, **When** repair runs, **Then** it is left
   untouched and reported.
5. **Given** several candidates, **When** repair runs, **Then** it does not choose silently.

---

### User Story 6 - Know what the conversion tools can actually do (Priority: P6)

Someone gets a straight statement of which conversion operations currently work, which are broken, and
how each compares to the maturity of terrain reading — before any parity is promised.

**Why this priority**: "Make them work again properly" cannot be scoped until the current state is
known. This project has already been burned by a documented tool that did not exist and by a "working"
route that measurement falsified; establishing the baseline first is the cheap way not to repeat it.

**Independent Test**: Run each conversion operation against real staged data and record the outcome,
then compare that record against what the tools claim to do.

**Acceptance Scenarios**:

1. **Given** each conversion operation, **When** it is exercised against real data, **Then** its
   outcome is recorded with the build it was run against.
2. **Given** an operation that fails, **When** it is recorded, **Then** the record states what failed
   rather than only that it failed.
3. **Given** the survey is complete, **When** parity work is scoped, **Then** it targets recorded
   defects rather than assumed ones.
4. **Given** any documented capability, **When** it is compared against the survey, **Then**
   documentation that overstates what exists is corrected.

### Edge Cases

- A build whose world objects are packaged in a way no other build uses.
- A build where the index and the actual contents disagree — which is already the measured case.
- A reference that resolves in one build and not in another; this is data, not an error.
- An asset present under two different paths in the same build.
- A reference that resolves only because a supplemental index claims it, with nothing actually present.
- An object referenced by nothing, and an object referencing nothing.
- Assets renamed between builds, which appear as one disappearance and one introduction unless
  recognised; the timeline must not present a rename as a new asset without saying it cannot tell.
- Case-insensitive path collisions, where two references differ only in case and both resolve.
- A build in which the same asset is referenced by hundreds of objects — the inventory must scale
  without conflating references with assets.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST obtain each build's world-object corpus through the existing data-access
  layer, which already resolves both per-asset containers and packaged archive entries. It MUST NOT
  derive the corpus from archive internal listfiles, which do not describe per-asset containers.
- **FR-002**: The system MUST report how many world objects it examined and how they were packaged, so
  that an under-counted corpus is visible in the output rather than silent.
- **FR-003**: The system MUST NOT assume any build's storage layout from another build's layout.
- **FR-003**: The system MUST record every doodad reference made by every world object in a build,
  attributed to the object that made it.
- **FR-004**: The system MUST resolve each reference against what the build actually contains, and MUST
  NOT conclude absence from an index alone.
- **FR-005**: The system MUST distinguish "found", "not found", and "could not be checked", and MUST
  never merge the third into the second.
- **FR-006**: The system MUST classify each unresolved reference as having no candidate, one candidate,
  or several, and MUST state the evidence for any candidate.
- **FR-007**: A candidate MUST be an asset present in the same build. The system MUST NOT propose an
  asset from another build or one that does not exist.
- **FR-008**: The system MUST express asset introduction as a window bounded by two named builds, and
  MUST state the granularity of any chronology claim.
- **FR-009**: The system MUST validate the within-file ordering hypothesis against between-build
  ground truth before any chronology relies on it, MUST report a null result as a finding, and MUST
  state whether the test could have detected the effect.
- **FR-010**: The system MUST treat each build as a distinct artifact. A finding for one build MUST NOT
  be recorded as a finding for any other, including the adjacent patch.
- **FR-011**: The system MUST NOT modify any data unless repair is explicitly requested.
- **FR-012**: Every repair MUST record the original reference, the replacement, and the evidence, and
  MUST be reversible to the exact prior state.
- **FR-013**: The system MUST leave untouched, and report, any unresolved reference it cannot repair
  unambiguously.
- **FR-014**: The system MUST record the current working state of each conversion operation against
  real data before parity work is scoped.
- **FR-015**: Every record the system produces MUST carry the build identity it came from.
- **FR-016**: Records MUST contain paths, outcomes, and provenance only — never client file content.

### Key Entities

- **Build identity**: The specific client an observation came from, carried by every record.
- **Corpus discovery**: For one build, where its world objects live, how they are packaged, how many
  were found, and any disagreement between the build's index and its actual contents.
- **World object**: One WMO in one build, with the doodad references it makes.
- **Doodad reference**: One reference from one object to one asset, with its resolution outcome.
- **Asset**: A referenced art file, tracked across builds by path.
- **Resolution outcome**: Found, not found, or not checkable — plus how it was determined.
- **Candidate match**: A real asset in the same build proposed for an unresolved reference, with the
  nature of the difference.
- **Introduction window**: The bounded interval, named by two builds, in which an asset first appears.
- **Repair record**: Original, replacement, evidence, and what is needed to reverse it.
- **Conversion capability record**: One operation, one build, what happened.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The inventory reports examining **532** world objects for the earliest staged build, and
  a count matching actual contents for every other staged build. A run reporting **1** for the earliest
  build indicates the listfile index was used instead of the data-access layer and is a failure.
- **SC-002**: 100% of doodad references in a surveyed build carry a resolution outcome; none is
  unclassified.
- **SC-003**: Zero references are reported missing solely because an index omitted them.
- **SC-004**: 100% of unresolved references are classified as having no candidate, one, or several.
- **SC-005**: Every proposed candidate is verified present in the same build; zero candidates reference
  a non-existent or cross-build asset.
- **SC-006**: Every asset in the timeline carries an introduction window bounded by two named builds,
  and every chronology claim states its granularity.
- **SC-007**: The within-file ordering hypothesis has a recorded result — including, if that is the
  answer, that no relationship was found and that the test had the power to find one.
- **SC-008**: Zero data is modified when repair is not requested.
- **SC-009**: 100% of applied repairs are reversible to the exact prior state, demonstrated by
  restoring and comparing.
- **SC-010**: Every conversion operation has a recorded outcome against real data before any parity
  claim is made.
- **SC-011**: Every record names the build it came from.

## Assumptions

- The staged client library is the source of truth. No client data enters the repository; records carry
  paths, outcomes, and provenance only.
- Client roots are runtime configuration and are never baked into source.
- Existing readers are reused, not reimplemented. The world-object and doodad readers, the world
  name-table reader, and the format converters already exist and are the canonical owners of their
  formats.
- **`uniqueId` is out of scope as a chronology source here.** It is the world-layout clock and dates
  placements, not assets. This feature is the independent signal; correlating the two is separate work.
- Asset introduction is dated **between builds**. Finer granularity is available only if US5 validates
  it, and is not assumed.
- A rename cannot in general be distinguished from a removal plus an introduction. Where the system
  cannot tell, it must say so rather than pick.
- Texture, sound, and other asset classes are out of scope; this feature is doodad references from
  world objects. The same method may extend later, but nothing here assumes it.
- Repair operates on data the operator owns and has chosen to modify. It is never part of analysis.
- The number of staged builds available bounds timeline resolution; the timeline states which builds it
  rests on and does not interpolate between them.
