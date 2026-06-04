# Research: PM4 Asset Matching

## Decision 1: Treat The Freeze-Prone `Export PM4 Obj Set` Path As A Replacement Target, Not The Design Owner

**Decision**: Build a new library-first PM4 export workflow and let any surviving viewer menu action become a thin host or review trigger.

**Rationale**:

- The user’s primary complaint is that the current export path can freeze the whole program.
- A blocking viewer interaction is not a credible owner for corpus export and automated matching.
- The matching pipeline needs deterministic, testable export behavior that works outside the shell.

**Alternatives considered**:

- Patch the existing viewer export path only: rejected because it keeps the core workflow UI-bound.
- Keep manual export as the primary path and add background threads later: rejected because it does not fix ownership or reproducibility.

## Decision 2: Use PM4 Object Segments As The Automation Primitive

**Decision**: Standardize the pipeline around exported PM4 object segments that carry one coherent grouping identity, cross-tile span, and derived signal record.

**Rationale**:

- Matching and placement need a stable unit of comparison.
- The user explicitly called out `ck24ObjectId` as the practical segmentation owner for the current lane.
- Segment records can carry ambiguity metadata without forcing a premature “final” PM4 hierarchy closure.

**Alternatives considered**:

- Match directly on raw surface rows: rejected because it is too fine-grained and unstable for replacement placement.
- Match on whole tiles or `Field04` buckets: rejected because recent corpus evidence shows those do not map cleanly to one object.

## Decision 3: Use Zarr-Backed Signal Corpora For Both PM4 Segments And Asset References

**Decision**: Store PM4 segment signals and staged WMO/M2 reference signals in Zarr-backed corpora with aligned index metadata and reproducible manifests.

**Rationale**:

- The repo already treats Zarr as the durable dataset/storage layer for large signal corpora.
- Matching needs to compare many PM4 segments against many asset references without regenerating signals every run.
- Zarr lets the pipeline stream slices instead of materializing the whole corpus in memory.

**Alternatives considered**:

- Thousands of per-object JSON/NPY files: rejected as fragile and hard to version.
- SQLite-only storage: rejected because the repo’s existing signal ecosystem is already Zarr-first.

## Decision 4: Separate Candidate Generation From Placement Synthesis

**Decision**: Split the automation into two stages: (1) ranked asset matching and (2) placement synthesis from accepted or top-ranked candidates.

**Rationale**:

- Matching accuracy and placement accuracy are different proof surfaces.
- Researchers need to inspect unresolved or ambiguous match reports before trusting replacement placements.
- The separation keeps the first signed-off placement output proposal-grade rather than silently mutating maps.

**Alternatives considered**:

- One monolithic “match and place” step: rejected because it hides failure modes and makes debugging harder.

## Decision 5: Start With Deterministic Scoring Before Any Learned Matcher

**Decision**: The first slice should use deterministic geometric/topological/signal scoring and report its reasoning explicitly.

**Rationale**:

- The user wants to avoid the current broken manual tools, not replace them with another opaque black box immediately.
- Deterministic scoring is easier to validate against known placements and easier to debug when the wrong asset ranks first.
- A deterministic baseline can later seed or supervise stronger learned matchers if needed.

**Alternatives considered**:

- Train a learned matcher immediately: rejected because the current pipeline first needs trustworthy segment/reference corpora and interpretable validation.

## Decision 6: Make Replacement Placement Output Proposal-Grade First

**Decision**: Emit machine-readable placement proposal manifests first, and defer direct ADT/WDT mutation or world-writeback to a later slice.

**Rationale**:

- The user’s immediate need is automation that can reconstruct likely missing placements from PM4 data.
- Proposal-grade output is safer and easier to validate against known-good tiles before any writeback step exists.
- This keeps the first slice bounded and avoids reopening map-writing surfaces unnecessarily.

**Alternatives considered**:

- Directly write replacement placements into map files: rejected because it couples two proof surfaces too early.

## Decision 7: Keep Viewer Review Bounded And Secondary

**Decision**: Treat CLI/report outputs as the primary automation surface and keep any viewer review panel or report loader strictly secondary.

**Rationale**:

- The current manual UI tools are explicitly not trusted as the workflow owner.
- Review still matters, but it should consume the automation outputs rather than recreate the matching process interactively.

**Alternatives considered**:

- Rebuild the manual matcher UI first: rejected because it delays the automation lane the user actually wants.
