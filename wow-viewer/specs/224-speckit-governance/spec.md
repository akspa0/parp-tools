# Feature Specification: Agent Governance — Scope Fidelity, Receipts & Spec Hygiene (Spec 224)

**Feature Branch**: `224-speckit-governance`
**Created**: 2026-09-06
**Status**: Implementing Phase 1
**Input**: Operator directive 2026-09-06 — "None of what I asked for was implemented — instead I end up with random shit I never asked for… if it is in the spec, that means _someone_ fabricated things I never wanted, and made it gospel fact without my consent. We must now do a set of speckit plans to: 1. write rules and refactor AGENTS.md so we never run into this nonsense again while using speckit; 2. require receipts of implemented code, with proof of function and precisely what we spec'd as the expected result with real data; 3. implement some form of speckit plan cleanup, perhaps via a custom skill, run once a month, instilled with a timestamp in AGENTS.md."

## Context & Motivation

Repeated project failures share one root: agents implement scope the operator never specified,
record it in specs as if it were approved, and the specs then rot — clogging every future chat's
context with stale plans while real code drifts from its documents. Recent concrete instances:

- Spec 212 Phase 2 described decorative reticle/compass/bezel HUD meshes the operator never asked
  for; an agent implemented them and marked the tasks complete.
- Spec 223 Phase 6 was documented as "source complete; acceptance pending" after the operator had
  already reported four acceptance failures.
- Output data has appeared in unmanaged locations on the operator's drives.
- Implemented specs are never archived, so "the code talks for itself" is violated at session start.

This spec makes those behaviors rule violations with enforcement mechanisms, not style preferences.

---

## User Stories & Testing

### US1 — No unrequested scope (P0)

An agent working any task implements exactly what the named spec/task states. If the agent believes
additional work is needed, it stops and proposes a spec amendment; the operator approves or rejects
before any code is written. Nothing enters a spec document without operator-originated wording or
explicit operator approval.

**Why**: fabricated scope is the single most expensive recurring defect in this project.

**Acceptance criteria**:
- AGENTS.md contains a scope-fidelity rule that names this spec as its authority.
- A task checklist item may only be marked complete if its description matches a spec criterion —
  a reviewer diffing task text against spec criteria finds no orphan features.
- Any spec document found to contain scope the operator did not state is corrected in the same
  session it is discovered, with a dated operator-correction note.

### US2 — Receipts for every completed task (P0)

A task marked complete in any `tasks.md` carries a receipt: files changed with paths, the exact
verification commands run and their exit status, and a criterion→evidence table mapping each spec
acceptance criterion to real output (real data, real client, or real file — never synthetic-only
proof for a data claim). Source/build/test output alone never satisfies a criterion that names
runtime, visual, or data behavior.

**Why**: "build green" has repeatedly been recorded as if it were feature acceptance.

**Acceptance criteria**:
- AGENTS.md defines the receipt format and forbids marking a task complete without one.
- The monthly audit (US4) treats any checked task lacking a receipt as invalid and un-checks it.
- Evidence files live in the owning spec's `evidence/` directory, never in memory-bank.

### US3 — No scattered artifacts (P0)

Every generated file — captures, exports, builds, datasets, temporaries — is written inside the
repository or a project-managed output root. No agent writes to desktops, temp dirs, drive roots,
or other ad-hoc locations.

**Why**: the operator has found data "in weird random places on the system's drives."

**Acceptance criteria**:
- AGENTS.md states the write-location rule.
- Code review for any change that creates files confirms the path is project-managed or
  operator-supplied.

### US4 — Monthly speckit cleanup with timestamp ledger (P1)

A `speckit-cleanup` skill runs on the first session of each month (or when the operator invokes
it). It: (a) audits every spec's `tasks.md` against the real code — checked tasks must have
receipts and their named symbols must exist; (b) archives fully implemented and operator-closed
specs out of the active registry; (c) compresses memory-bank files, moving landed narrative to
`archive/`; (d) updates the timestamp ledger in AGENTS.md. `activeContext.md` stays short enough
that a new session reads only STATUS.md, activeContext.md, and the one owning spec.

**Why**: stale specs and memory banks waste tokens, time, and cause agents to "continue" dead lanes.

**Acceptance criteria**:
- The skill exists, is installed in the operator's skill directory, and is documented in AGENTS.md.
- AGENTS.md carries a `Last cleanup` timestamp the skill updates on every run.
- After a run, no implemented spec remains listed as active in `specs/STATUS.md`, and
  `activeContext.md` contains no narrative older than the most recent archived handoff.

### US5 — Spec-sync discipline (P1)

When an implementation diverges from its spec — including when the implementation is "better" —
the agent updates the spec in the same change or opens a dated amendment note. The spec and the
code never disagree at session end.

**Why**: the operator identifies silent divergence as the reason specs rot.

**Acceptance criteria**:
- AGENTS.md requires spec-sync in the same pass as non-trivial code changes.
- The monthly audit flags any active spec whose last code change postdates its last spec edit
  without an amendment note.

---

## Functional Requirements

- **FR-1 — Scope freeze**: The named spec/task text is the complete implementation contract.
  Omissions and improvements are proposed, never self-authorized.
- **FR-2 — Receipt gate**: `tasks.md` checkboxes are evidence-gated. No receipt, no check.
- **FR-3 — Write containment**: All outputs live in the repo or operator-named paths.
- **FR-4 — Cleanup mechanism**: The `speckit-cleanup` skill plus the AGENTS.md timestamp ledger.
- **FR-5 — Spec-sync**: Code and spec agree at session end, or a dated amendment explains the delta.
- **FR-6 — Context discipline**: Implemented specs are archived; hot context is STATUS.md +
  activeContext.md + the one owning spec.

## Success Criteria

- **SC-1**: A fresh session reading only STATUS.md + activeContext.md + the owning spec can begin
  correct work with no stale-lane confusion.
- **SC-2**: The next operator acceptance review finds zero implemented-but-unspec'd features.
- **SC-3**: Every checked task in every active spec has a receipt or gets un-checked by the audit.

## Dependencies

- Applies to all specs; owner of the AGENTS.md governance section and the cleanup skill.
- Spec 212's Phase 2 scope correction (2026-09-06) is the triggering instance and is recorded there.

## Stakeholders

- Operator: sole approver of scope; runs the monthly cleanup or delegates it explicitly.
- All agents/harnesses: bound by the AGENTS.md rules this spec authors.
