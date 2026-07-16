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

**Rationale**: Prior audit evidence showed sound core arrays alongside a systemic hole-mask defect
and per-tile coverage gaps. Whole-store or whole-row certification would overclaim.

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
