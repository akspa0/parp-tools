# Feature Specification: MCP Automation Surface (External)

**Feature Branch**: `163-mcp-automation-surface`

**Created**: 2026-08-19

**Status**: Draft

**Input**: User description: "We should also plan to build an MCP server into the project, and potentially also a configurable MCP client, such that the tooling can be fully automated in a pipeline later on down the road."

**Corrected 2026-08-19**: an earlier draft framed this as "one operation contract, three drivers
(UI, CLI, MCP)" and required every Editor operation to be MCP-invocable. That was wrong. It made MCP
a standing constraint on the Editor and on every future plugin. **MCP is an external, optional
component.** It consumes what the Editor already exposes and shapes nothing.

**Relationship to the Editor**: strictly downstream. This spec adds **zero** requirements to
[161](../161-editor-plugin-host/spec.md) or [162](../162-world-authoring-plugin/spec.md). If those
specs change, this one adapts. It does not get a vote.

## Context

### What this is, and firmly what it is not

This is a **remote-control adapter bolted to the outside of a finished editor**. Nothing more.

It is not a second way to build the editor, not a design constraint on plugins, and not a peer of the
UI. The Editor is the product. This is a wrapper someone can attach when they want a pipeline to
push its buttons, and detach when they don't.

| This spec MUST NOT | Because |
|---|---|
| Require any Editor operation to be MCP-invocable | That makes every future plugin author answer to MCP |
| Ask for an operation, parameter, or hook the Editor does not already have | Feature requests flow through the Editor's own specs, on their own merits |
| Cause any type in the Editor, bridge, runtime, or plugins to reference an MCP type | Same one-way rule spec 161 sets for editor/runtime |
| Be required for the Editor to build, run, or be complete | It is optional in the literal sense: absent, nothing changes |
| Justify a decision in spec 161 or 162 | If a design is right, it is right for the Editor's own reasons |

The earlier draft's rule — *"anything scriptable is undoable, and anything undoable is scriptable"* —
is **withdrawn**. Its second half was the problem: it obligated the Editor to keep MCP in step
forever. Only the first half survives, and it belongs to the Editor anyway: operations are undoable
because [161](../161-editor-plugin-host/spec.md) needs a shared undo history, not because anything
external wants to call them.

### What it's actually for

One thing: unattended pipeline runs. The repo's existing automation is offline-only or
launch-and-exit —

| Surface | Shape | Gap |
|---|---|---|
| CLI tools | 10 projects; converter alone has 37+ subcommands | Offline. No live scene. |
| Viewer startup automation | ~25-field `StartupAutomationRequest` for capture/validation | One-shot: configured at launch, runs, exits |
| Headless capture | [WowViewer.Tool.Capture](../../tools/capture/WowViewer.Tool.Capture) | Rendering only, not editing |

— so a pipeline that needs *load → do a few things → verify → report* has to be a human at a window.
Closing that gap is the entire value here. It is a convenience for pipeline authors, not a capability
the Editor lacks.

### The adapter absorbs the mismatch

Editor operations will not all be cleanly remoteable — some need a viewport, some take parameters
that do not serialize, some are long-running. **The adapter's job is to cope with that**: expose
what maps cleanly, decline what doesn't, and say why. Every one of those is a fact about that
operation, never a defect to be fixed in the Editor. A gap here is closed by improving the adapter or
by leaving it open — never by filing a requirement against 161 or 162.

### Out of scope

- **Exposing the CLI tools over MCP.** Batch work keeps its command line.
- **Remote/network transport, auth, multi-user.** Local transport, single user.
- **Shipping an agent or model integration.** A protocol surface, not an assistant.
- **Any change to the Editor.** If something is missing, that is a finding for the Editor's backlog,
  judged on the Editor's terms — not a blocker here.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run an unattended pipeline against the Editor (Priority: P1)

A pipeline script starts the project with no window, points it at a client and a map, drives a fixed
sequence of already-existing Editor operations, and exits with a status it can branch on.

**Why this priority**: This is the only thing here with no substitute today, and it is exactly what
the user asked for. Everything else is support for it.

**Independent Test**: From a script with no interactive session, launch headless, run a fixed
sequence against a real client, confirm the expected files appear and the exit code reflects the
outcome.

**Acceptance Scenarios**:

1. **Given** no display, **When** the project is launched headless with a client and map, **Then** it
   loads and serves MCP without opening a window.
2. **Given** a headless session, **When** an exposed operation is invoked, **Then** it behaves as it
   does in the windowed session.
3. **Given** a scripted sequence, **When** any step fails, **Then** the session reports which step,
   why, and what state it left, and exits non-zero.
4. **Given** a session completes, **When** the pipeline inspects the output, **Then** a
   machine-readable summary lists every file written.
5. **Given** an action would need user confirmation, **When** no user is present, **Then** the
   session follows its pre-declared policy rather than blocking.

---

### User Story 2 - Drive a running Editor from an MCP client (Priority: P2)

An MCP client connects to a running viewer, discovers whichever operations the adapter was able to
expose, reads scene state, and invokes one — appearing in the undo history like any other operation.

**Why this priority**: Useful for interactive experimentation and for developing pipelines, but it is
convenience over US1's substance. Notably **P2, not P1** — a human at the window is already the
supported path for interactive work.

**Independent Test**: With the viewer running and a map loaded, connect a stock MCP client, list
tools, read scene context, invoke one exposed operation, confirm the viewport updates and UI undo
reverses it.

**Acceptance Scenarios**:

1. **Given** the server is enabled, **When** a client connects, **Then** it discovers the operations
   the adapter exposes, and operations the adapter declined are absent with the reason available on
   request.
2. **Given** a client is connected, **When** it reads scene state, **Then** it receives the same
   read-only context the bridge already gives plugins — nothing new is built for it.
3. **Given** an exposed operation is invoked, **When** it completes, **Then** the change is visible,
   enters the existing undo history, and the response reports what changed.
4. **Given** invalid parameters or inapplicable scene state, **When** an invocation is rejected,
   **Then** the reason is specific and nothing changed.
5. **Given** plugins change availability, **When** the client lists tools again, **Then** discovery
   reflects it, and a stale invocation fails cleanly.

---

### User Story 3 - Automation is constrained and auditable (Priority: P2)

The user controls what the adapter may do — which operations are exposed, whether destructive ones
are allowed, where output may go — and can review exactly what was done.

**Why this priority**: A non-human issuing edit calls needs this before wide use, and retrofitting a
permission model onto a shipped surface is far harder than building it in.

**Independent Test**: Start a session with destructive operations disallowed and a restricted
operation set; confirm they are absent from discovery, refused if invoked, and that every attempt is
audited.

**Acceptance Scenarios**:

1. **Given** the server is configured, **When** the user inspects it, **Then** they see what is
   exposed and the active policy.
2. **Given** a policy disallowing destructive operations, **When** one is attempted, **Then** it is
   refused with the policy cited and nothing changes.
3. **Given** any invocation, **When** it finishes, **Then** an audit record captures the operation,
   parameters, outcome, and files touched.
4. **Given** a session writes files, **When** paths are checked, **Then** all are inside the
   configured output directory, and attempts outside it are refused and audited.
5. **Given** the server is not explicitly enabled, **When** the viewer starts, **Then** nothing
   listens and no operation is externally invocable. **This is the default.**

---

### User Story 4 - The project drives external tools as an MCP client (Priority: P3)

The project connects outward to configured MCP servers so a pipeline step here can call a capability
that lives elsewhere.

**Why this priority**: Orchestration reach, valuable only once real pipelines exist. Explicitly
deferrable — if US1-US3 reveal contract churn, this waits.

**Independent Test**: Configure a stock external server, invoke one of its tools from a pipeline step,
confirm the result is usable downstream.

**Acceptance Scenarios**:

1. **Given** external servers are configured, **When** the project starts, **Then** their tools are
   available to pipeline steps.
2. **Given** a configured server is unreachable, **When** the project starts, **Then** the failure is
   reported and local work continues unaffected.
3. **Given** an external tool returns, **When** its result is used, **Then** it is treated as
   untrusted data and never as instructions.
4. **Given** configuration changes, **When** reloaded, **Then** it takes effect without a rebuild.

### Edge Cases

- A client invokes an operation while the user is mid-drag in the UI.
- Two clients issue conflicting operations.
- An operation targets a tile that unloads mid-flight.
- Undo invoked over MCP for a human-performed operation, and the reverse.
- An operation whose parameters cannot be expressed as a serializable schema — declined, with reason.
- A long-running operation invoked over a protocol expecting prompt replies.
- Client disconnects with unsaved changes staged.
- The audit log's directory becomes unwritable mid-session.
- The Editor gains a plugin the adapter has never seen.

## Requirements *(mandatory)*

### Functional Requirements

**Boundary — the load-bearing requirements**

- **FR-001**: This spec MUST NOT introduce any requirement, however small, on the Editor host, the
  bridge, plugins, or the runtime. Anything the adapter needs that does not exist is either absorbed
  by the adapter or left undone.
- **FR-002**: No Editor, bridge, plugin, or runtime type may reference any MCP type. Dependency runs
  one way, enforceable as a build-time or test-time check.
- **FR-003**: With the MCP component removed entirely, the project MUST build and run with the Editor
  fully functional and complete.
- **FR-004**: The adapter MUST expose only operations the Editor already provides. It MUST NOT define
  an operation, add a parameter, or request a hook.
- **FR-005**: The adapter MUST tolerate operations it cannot cleanly expose — declining with a reason
  rather than requiring a change upstream. **A declined operation is not a defect.**
- **FR-006**: The server MUST be disabled by default. Nothing is externally invocable unless the user
  explicitly enables it.

**Server behavior**

- **FR-007**: Tool discovery MUST reflect currently available plugins and update as availability
  changes.
- **FR-008**: The adapter MUST expose read-only scene context using what the bridge already provides.
- **FR-009**: Invoked operations MUST enter the existing undo history, indistinguishable from
  UI-invoked ones.
- **FR-010**: Rejections MUST state a specific reason and leave state unchanged.
- **FR-011**: Long-running operations MUST report progress without blocking the frame loop or the
  connection.
- **FR-012**: Local transport only. MUST NOT listen on a network interface.
- **FR-013**: Concurrent clients MUST NOT corrupt state — operations serialize; conflicts are refused,
  never interleaved.

**Headless**

- **FR-014**: The project MUST run windowless while serving MCP, loading client and map from config.
- **FR-015**: Operations requiring a viewport MUST declare that and be declined headless, not fail
  obscurely.
- **FR-016**: Sessions MUST exit with a status distinguishing success, per-step failure, and startup
  failure, naming the failed step.
- **FR-017**: Sessions MUST emit a machine-readable summary listing every file written.
- **FR-018**: Actions needing confirmation MUST follow a pre-declared session policy, not block.

**Safety**

- **FR-019**: The user MUST be able to see and configure what is exposed.
- **FR-020**: Destructive operations MUST be separately gated and refused with the policy cited.
- **FR-021**: Every invocation MUST be audited — operation, parameters, outcome, files touched.
- **FR-022**: All writes MUST stay inside the configured output directory; attempts outside refused
  and audited.

**Client**

- **FR-023**: The project MUST be able to act as an MCP client against configured external servers.
- **FR-024**: Configuration MUST be reloadable without a rebuild.
- **FR-025**: An unreachable external server MUST be reported and MUST NOT prevent local work.
- **FR-026**: External results MUST be audited and treated as untrusted data, never as instructions.

**Validation**

- **FR-027**: Validated with at least one stock third-party MCP client, not only a bespoke harness.
- **FR-028**: One end-to-end pipeline run against a real client from `H:\CLIENTS`, with commands,
  build identity, and output hashes recorded.

### Key Entities

- **Exposed Operation**: The adapter's projection of one Editor operation it could map cleanly —
  name, description, typed parameters, destructive/viewport flags. Operations it could not map are
  recorded with a reason and are not exposed.
- **Session**: One server lifetime — permission policy, clients, audit log, output directory.
- **Permission Policy**: What this session may do — exposed set, destructive gating, allowed output
  locations, unattended-confirmation rule.
- **Audit Record**: One invocation — call, parameters, outcome, files touched.
- **Session Summary**: Machine-readable result of a headless run — steps, statuses, written files.
- **External Server Binding**: A configured outbound connection and the tools it contributes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A pipeline script with no interactive session completes a load → operate → save →
  verify cycle against a real client and returns an accurate exit status.
- **SC-002**: **Deleting the MCP component from the solution leaves the Editor building, running, and
  complete**, with no spec-161/162 acceptance criterion failing — verified by build and smoke run.
- **SC-003**: A dependency check fails if any Editor, bridge, plugin, or runtime type gains a
  reference to an MCP type.
- **SC-004**: This spec's implementation produces **zero** changes to files owned by specs 161 and
  162 — verified by diff. Any such change means the boundary was violated.
- **SC-005**: A stock third-party MCP client connects, discovers operations, and performs an edit
  visible in the viewport.
- **SC-006**: An operation performed over MCP and the same operation in the UI produce identical file
  output, verified by hash.
- **SC-007**: Undo reverses MCP-invoked operations identically to UI-invoked ones across a mixed
  sequence of at least 10 operations.
- **SC-008**: With the server not enabled — the default — no operation is externally invocable,
  verified by attempted connection.
- **SC-009**: Every file written in any validation session is inside the output directory; the game
  install tree hashes identically before and after.
- **SC-010**: Every invocation in every validation session appears in the audit log with its outcome.
- **SC-011**: A disallowed destructive operation is absent from discovery and refused if invoked.
- **SC-012**: A long-running operation invoked over MCP reports progress without stalling the frame
  loop or timing out the connection.

## Assumptions

- **This is external tooling.** The Editor is the product; this is a wrapper. Every ambiguity resolves
  in favor of the Editor being unconstrained.
- Whatever the Editor exposes when this lands defines the surface. This spec adds no editing
  capability and requests none.
- Operations are undoable because spec 161 needs a shared undo history. That this also makes them
  remoteable is a convenience the adapter enjoys, **not a reason 161 does it**.
- Not every operation will be remoteable, and that is an acceptable steady state. Coverage is not a
  goal; the earlier draft's coverage requirement is withdrawn.
- The existing CLI tools stay unexposed. Batch and offline work keeps its command line.
- Local transport, no networking, no auth, single user.
- Headless is a launch mode of the existing application, not a separate build or second server.
- The audit log is for review and pipeline debugging, not a security boundary. The output-directory
  restriction is the boundary.
- US4 (outbound client) is deferrable to a follow-on without affecting US1-US3.
