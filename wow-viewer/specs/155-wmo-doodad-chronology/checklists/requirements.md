# Specification Quality Checklist: Asset Reference Inventory

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-16
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Validation Notes

**Iteration 1 drew the wrong conclusion and it has been corrected.** The first draft claimed the WMO
corpus "cannot currently be enumerated", promoted corpus discovery to P1, and set SC-001 as closing a
500× enumeration defect. **That was wrong.** The capability exists and is deliberate: the archive
catalogue scans the loose tree for per-asset containers, the data source maps a container back to the
logical asset path it holds, the native archive service explicitly identifies these as listfile-less
single-file archives and already de-duplicates their double registration so enumeration does not emit
each WMO twice, and the V14 converter documents that it handles per-asset containers automatically.
The viewer reads this data today.

The real finding is much smaller: **building an index cache from archive internal listfiles is the
wrong surface for corpus enumeration**, because per-asset containers carry no internal listfile by
design. That surface answers "what does this archive's listfile declare", not "what world objects does
this build contain". Choosing it returns 1 instead of 532.

Corrections applied: corpus discovery is no longer a user story — the inventory (was P2) is now P1 and
absorbs corpus reporting as a self-check; every story moved up one priority; FR-001 now requires the
data-access layer and forbids deriving the corpus from listfiles; FR-002 requires the examined count to
be reported so an under-count is visible rather than silent; SC-001 reframed from "close a defect" to
"a run reporting 1 means the wrong surface was used".

**The lesson worth keeping**: the trap is real even though the defect was not. A plan that reaches for
the index cache because it is the convenient enumeration surface would produce a timeline dating every
asset later than it arrived, while looking authoritative. That is why the corpus-source requirement
survives in corrected form rather than being deleted.

**Iteration 2 — the spec was solving an adjacent problem and has been rewritten.**

The first two drafts scoped this as "inventory doodads in world objects, then date them". Two things
were wrong:

- *Scope was too narrow.* The ask covers three reference kinds, not one: doodads placed by world
  objects, textures used by world objects, and **textures used by models**. The driving example is a
  model texture, so a doodad-only inventory would have missed the exact case that motivated the work.
- *The listfile was treated as a hazard to route around.* It is not — it is **one of the two sides of
  the comparison**. The deliverable is the disagreement between what data expects, what listfiles
  name, and what the build contains. Iteration 1 over-corrected into "never trust the listfile", which
  would have discarded half the signal. The corrected framing keeps "named by a catalogue" and
  "readable from the build" as separate facts and reports the four ways they can disagree, plus
  orphans as the repair donor pool.

The chronology, originally the headline, is now P5. It depends on the ledger being complete, and the
missing-asset inventory delivers value without it.

**A positive control now anchors the whole feature.** The Mt. Hyjal effect objects are a known, in-world
verified instance of a missing texture — visible because the engine draws untextured geometry neon
green. SC-001 requires an untargeted sweep to flag them, and a sweep that does not is failed regardless
of anything else it reports. This project has been caught reporting null results from detectors that
could not have seen the thing; a known-true instance closes that hole and is far stronger than any
synthetic test.

**One distinction the spec now protects deliberately**: a missing asset is a *finding*, not a defect.
Some are intentional and produce in-game effects — the Mt. Hyjal objects being the reason anyone knows
that. Repair must never assume absence is breakage.

**Zero [NEEDS CLARIFICATION] markers.** Every candidate ambiguity had either a defensible default or a
measurable answer:

- *How finely can assets be dated?* Between builds, by set difference — defensible now. Finer
  granularity is US5's hypothesis, explicitly allowed to return null.
- *Which asset classes?* Doodad references from world objects, as asked. Textures and sounds are out,
  recorded as an assumption rather than a question.
- *What about `uniqueId`?* Recorded as out of scope with the reason: it dates placements, not assets.
  The user's framing already treats this feature as the signal *beyond* `uniqueId`.

**Standing risks to carry into planning:**

- **"This asset does not exist" is the feature's most dangerous output.** The measured index gap means
  absence-from-index and absence-from-build are different facts. FR-004, FR-005 and SC-003 exist for
  this; a plan that resolves references against an index alone reintroduces the defect the spec was
  written to avoid.
- **US5 must be allowed to fail.** The within-file ordering hypothesis is the kind of claim this
  project has already seen test null once, on a related chronology question, because it was the wrong
  clock. FR-009 requires both the result and a statement of whether the test could have detected the
  effect — a null from an underpowered test is not a finding.
- **Renames are indistinguishable from removal-plus-introduction** without further evidence. The
  timeline must say so rather than choose; a plan that silently treats one as the other will
  manufacture introductions that never happened.
- **Repair modifies data.** FR-011 through FR-013 and SC-008/SC-009 are the guardrails. Planning must
  keep analysis and repair separable, so that no analysis path can mutate anything.
- **US7 is a survey before a promise.** "Make the converters work properly" cannot be scoped before the
  current state is recorded. This project has twice been caught by a capability that was documented or
  assumed but not real; the survey is the cheap defence.
