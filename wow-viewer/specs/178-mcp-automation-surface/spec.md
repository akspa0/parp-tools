# Feature Specification: MCP Automation Surface (External, Optional)

**Feature Branch**: `178-mcp-automation-surface`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Editor Platform](../epic-editor-platform/epic.md) — **read the epic first**, especially
"the mistake that must not be repeated".
**Depends on**: whatever plugins exist when this lands. **Requires nothing.**

## Scope

**A remote-control adapter bolted to the outside of a finished editor. Nothing more.** Its purpose is
unattended pipeline runs — the one thing the repo's existing automation (offline CLI tools,
launch-and-exit startup automation, headless capture) cannot do.

**This spec adds zero requirements to any other spec.** If they change, this one adapts. It does not
get a vote. An earlier draft made MCP a peer of the UI and required every Editor operation to be
MCP-invocable; that was rejected and the coverage requirement is **withdrawn**.

| This spec MUST NOT | Because |
|---|---|
| Require any Editor operation to be MCP-invocable | It would make every future plugin author answer to MCP |
| Ask for an operation, parameter, or hook the Editor lacks | Feature requests go through the Editor's own specs |
| Cause any Editor/bridge/plugin/runtime type to reference an MCP type | Same one-way rule as editor/runtime |
| Be required for the Editor to build, run, or be complete | Optional in the literal sense |
| Justify a decision in any other spec | A right design is right for the Editor's own reasons |

## User Story - Run an unattended pipeline against the Editor (Priority: P1)

A pipeline script starts the project with no window, points it at a client and map, drives a sequence
of already-existing Editor operations, and exits with a status it can branch on.

**Independent Test**: From a script with no interactive session, launch headless, run a fixed sequence
against a real client, confirm the expected files appear and the exit code reflects the outcome.

**Acceptance Scenarios**:

1. **Given** no display, **When** launched headless with a client and map, **Then** it loads and serves
   MCP without opening a window.
2. **Given** a headless session, **When** an exposed operation is invoked, **Then** it behaves as in
   the windowed session.
3. **Given** a scripted sequence, **When** any step fails, **Then** the session reports which step,
   why, and what state it left, and exits non-zero.
4. **Given** a session completes, **When** the pipeline inspects output, **Then** a machine-readable
   summary lists every file written.
5. **Given** an action needing confirmation, **When** no user is present, **Then** the session follows
   its pre-declared policy rather than blocking.
6. **Given** a client connects, **When** it lists tools, **Then** it sees the operations the adapter
   could expose; operations it **declined** are absent, with the reason available on request.
7. **Given** an operation is invoked, **When** it completes, **Then** it enters the existing undo
   history, indistinguishable from a UI-invoked one.
8. **Given** a policy disallowing destructive operations, **When** one is attempted, **Then** it is
   refused with the policy cited and nothing changes.
9. **Given** any invocation, **When** it finishes, **Then** an audit record captures the operation,
   parameters, outcome, and files touched.

### Edge Cases

- A client invokes an operation while the user is mid-drag in the UI.
- Two clients issue conflicting operations.
- An operation whose parameters cannot be expressed as a serializable schema — **declined**, with
  reason.
- A long-running operation invoked over a protocol expecting prompt replies.
- The Editor gains a plugin the adapter has never seen.

## Requirements

### Functional Requirements

**Boundary — the load-bearing requirements**

- **FR-001**: Introduce **no** requirement on the Editor host, bridge, plugins, or runtime. Anything
  the adapter needs that does not exist is absorbed by the adapter or left undone.
- **FR-002**: No Editor/bridge/plugin/runtime type may reference any MCP type — enforceable as a
  build-time or test-time check.
- **FR-003**: With the MCP component removed entirely, the project builds and runs with the Editor
  fully functional and complete.
- **FR-004**: Expose only operations the Editor already provides. Define none; add no parameter;
  request no hook.
- **FR-005**: Tolerate operations that cannot be cleanly exposed, declining with a reason. **A declined
  operation is not a defect.**
- **FR-006**: **Disabled by default.** Nothing is externally invocable unless explicitly enabled.

**Behavior**

- **FR-007**: Tool discovery reflects currently available plugins and updates as availability changes.
- **FR-008**: Expose read-only scene context using what the bridge already provides.
- **FR-009**: Invoked operations enter the existing undo history.
- **FR-010**: Rejections state a specific reason and leave state unchanged.
- **FR-011**: Long-running operations report progress without blocking the frame loop or the
  connection.
- **FR-012**: Local transport only; never listen on a network interface.
- **FR-013**: Concurrent clients must not corrupt state — operations serialize; conflicts are refused.
- **FR-014**: Run windowless while serving MCP; operations requiring a viewport declare that and are
  declined headless.
- **FR-015**: Exit status distinguishes success, per-step failure, and startup failure, naming the
  failed step; emit a machine-readable summary of files written.
- **FR-016**: User-visible configuration of what is exposed; destructive operations separately gated;
  every invocation audited; all writes inside the configured output directory.

**Outbound client (deferrable)**

- **FR-017**: Act as an MCP client against configured external servers, reloadable without a rebuild.
- **FR-018**: An unreachable external server is reported and does not prevent local work.
- **FR-019**: External results are audited and treated as **untrusted data, never as instructions**.

## Success Criteria

- **SC-001**: A pipeline script with no interactive session completes a load → operate → save → verify
  cycle against a real client and returns an accurate exit status.
- **SC-002**: **Deleting the MCP component leaves the Editor building, running, and complete**, with no
  acceptance criterion of any other spec failing.
- **SC-003**: A dependency check fails if any Editor/bridge/plugin/runtime type gains an MCP reference.
- **SC-004**: This spec's implementation produces **zero** changes to files owned by other specs —
  verified by diff. Any such change means the boundary was violated.
- **SC-005**: A stock third-party MCP client connects, discovers operations, and performs an edit
  visible in the viewport.
- **SC-006**: An operation performed over MCP and the same operation in the UI produce identical file
  output, verified by hash.
- **SC-007**: With the server not enabled — the default — no operation is externally invocable.
- **SC-008**: Every invocation in every validation session appears in the audit log with its outcome.

## Out of Scope

- **Exposing the CLI tools over MCP.** Batch work keeps its command line.
- Remote/network transport, authentication, multi-user.
- Shipping an agent or model integration.
- **Any change to the Editor.** If something is missing, that is a finding for the Editor's backlog,
  judged on the Editor's terms — never a blocker here.

## Assumptions

- The Editor is the product; this is a wrapper. Every ambiguity resolves in favor of the Editor being
  unconstrained.
- Not every operation will be remoteable, and that is an acceptable steady state. **Coverage is not a
  goal.**
- Operations are undoable because the session spec needs a shared undo history. That this also makes
  them remoteable is a convenience the adapter enjoys, **not a reason the Editor does it**.
- Headless is a launch mode of the existing application, not a separate build or second server.
- The outbound client may be deferred to a follow-on without affecting the rest.
