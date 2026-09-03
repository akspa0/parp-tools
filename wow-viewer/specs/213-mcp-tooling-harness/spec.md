# Feature Specification: MCP Tooling Harness

**Feature Branch**: `213-mcp-tooling-harness` (authored on `v0.5.3`; specs 193–211 follow the same convention)

**Created**: 2026-09-02

**Status**: Draft

**Input**: User description: "add an MCP server or client to the tooling, so all the tooling can be harnessed through such methods, using the latest modelcontextprotocol.io specification, to interoperate with a separate private orchestration and job layer with VLM/LLM inference that consumes the TensorStore/Zarr datastores this program builds"

## Context & Motivation

This repository's capability is spread across ten independent command-line tools —
`inspect`, `harvest`, `capture`, `converter`, `enrich`, `mask-validate`, `validation-capture`,
`wdl-read`, `wmo-minimap` — plus the viewer itself. Each is reachable only by constructing an exact
argument line, and the argument line is the *only* contract. That has already cost real time: the
standing lesson recorded in memory is that "tests pass" does not mean "the documented command runs",
because the documentation and the argument parser drift apart with nothing holding them together.

Separately, an external orchestration and job layer with VLM/LLM inference — a private project — is
intended to drive this repository's tooling and consume the TensorStore/Zarr datastores it produces.
That project speaks the Model Context Protocol. Today there is no way for it to reach anything here
except by shelling out to argument lines it would have to hard-code and keep in sync.

An MCP server over the tooling solves both at once. It gives every tool a **declared, typed,
introspectable contract** that a machine can enumerate and validate against — which is exactly the
thing whose absence caused the CLI documentation drift — and it gives the external orchestrator a
standard way in. The tools keep working as CLIs; the server is an additional front door onto the same
capability, not a replacement for it.

The scope decision for this spec is **server first**. This repository is the side that *has*
capability to expose. A client — the viewer reaching outward to remote inference or job servers — is
a later phase, specified once the server's contract has been proven against a real external caller.

Spec 212 covers the spatial UI shell, the other half of the same long-range direction. The two are
independent: neither blocks the other.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - An external harness discovers and runs a tool (Priority: P1)

An operator points the external orchestration project at this repository's MCP server. Without
reading any documentation or hard-coding an argument line, that harness lists the available tools,
reads each one's declared inputs and what they mean, calls one with typed arguments, and receives a
structured result it can act on.

**Why this priority**: This is the entire feature in one slice. A server that exposes even one tool
this way already proves the contract, the transport, and the round trip, and already gives the
external project a way in. Everything else is breadth on top of a shape that must be right first.

**Independent Test**: Connect a standard MCP client to the server, enumerate tools, call one
read-only tool with valid arguments, and confirm the result is structured and correct. Then call it
with invalid arguments and confirm the error is actionable.

**Acceptance Scenarios**:

1. **Given** a running server, **When** a client requests the tool list, **Then** it receives every
   exposed tool with a name, a human-readable description, and a complete input schema.
2. **Given** an exposed tool, **When** a client calls it with valid arguments, **Then** it receives a
   structured result rather than scraped console text.
3. **Given** an exposed tool, **When** a client calls it with arguments that violate the declared
   schema, **Then** the call is rejected before any work begins, with an error naming the offending
   argument.
4. **Given** a tool that fails during execution, **When** the failure occurs, **Then** the client
   receives an error result that distinguishes the failure from a protocol-level fault and carries
   enough detail to diagnose it.
5. **Given** a client that speaks the current Model Context Protocol specification, **When** it
   connects, **Then** version negotiation succeeds without special-casing this server.

---

### User Story 2 - The declared contract is the same contract the CLI enforces (Priority: P1)

A tool's MCP schema and its command-line parser cannot disagree. When someone adds, renames, or
retypes an argument, either both surfaces change or the build fails.

**Why this priority**: This is the defect this feature exists to prevent, and it is P1 because
retrofitting it later means writing every schema twice and re-introducing exactly the drift the
memory bank records. If the schema is a hand-maintained copy of the parser, the feature has made the
problem worse rather than better.

**Independent Test**: Change an argument in one tool's parser and confirm the mismatch is caught
automatically, without a human noticing it.

**Acceptance Scenarios**:

1. **Given** a tool exposed over MCP, **When** its declared input schema is compared against its
   command-line argument parser, **Then** the two agree on every argument's name, type, and
   requiredness.
2. **Given** a change to a tool's arguments on either surface, **When** the test suite runs, **Then**
   any divergence between the two surfaces fails the build.
3. **Given** a newly added tool, **When** it is not registered on both surfaces, **Then** this is
   reported rather than silently omitted.

---

### User Story 3 - Long-running work reports progress and can be cancelled (Priority: P2)

Harvests, captures and corpus scans run for minutes or hours. A harness that starts one can watch it
advance, and can stop it without killing the server.

**Why this priority**: Without this, the long-running tools — which are the ones an orchestration
layer most wants to drive — are unusable over the protocol: the caller cannot tell a slow run from a
hung one. It is P2 only because the short read-only tools in US1 prove the contract first.

**Independent Test**: Start a long-running tool through the server, observe progress advance, cancel
it mid-run, and confirm both that it stopped and that the server remained healthy and responsive.

**Acceptance Scenarios**:

1. **Given** a long-running tool invoked through the server, **When** it executes, **Then** the caller
   receives progress updates that advance, rather than silence until completion.
2. **Given** a running invocation, **When** the caller cancels it, **Then** the work stops, partial
   output is not presented as complete, and the server continues serving other requests.
3. **Given** a client that disconnects mid-invocation, **When** the disconnect occurs, **Then** the
   invocation is cancelled and its resources released.
4. **Given** two invocations requested at once, **When** both run, **Then** neither corrupts the
   other's output.

---

### User Story 4 - Datastores and artifacts are addressable as resources (Priority: P2)

The harness can enumerate and read what this repository has produced — datastore entries, curation
manifests, harvested artifacts, evidence files — by stable address, without knowing where they sit on
disk.

**Why this priority**: This is what makes the interoperation with the external orchestration project
substantive rather than nominal: that project's stated purpose is to consume these datastores. It
follows US1 because it needs the same transport and contract to already work.

**Independent Test**: Enumerate resources over the protocol, read one, and confirm its content and
declared type match the artifact on disk.

**Acceptance Scenarios**:

1. **Given** produced artifacts, **When** a client lists resources, **Then** it receives stable
   addresses with declared content types.
2. **Given** a resource address, **When** a client reads it, **Then** it receives that artifact's
   content, correctly typed.
3. **Given** an artifact that does not exist or is not exposed, **When** a client reads its address,
   **Then** it receives a clear not-found error and no partial content.
4. **Given** a large artifact, **When** a client reads it, **Then** the read does not exhaust server
   memory or block other requests.

---

### User Story 5 - The exposed surface is bounded and the operator controls it (Priority: P1)

The server exposes only what it was configured to expose. It does not become an arbitrary
command-execution or filesystem-read endpoint on the operator's machine, and the operator can see and
change what is reachable.

**Why this priority**: A server that fronts ten tools which read client archives and write files is a
capable thing. Getting its boundary wrong is the one failure in this feature that is not merely
inconvenient. It is P1 because the boundary has to be designed in from the first exposed tool, not
added once the surface is wide.

**Independent Test**: Attempt to reach a path and a tool outside the configured surface and confirm
both are refused; then change the configuration and confirm the reachable surface changes with it.

**Acceptance Scenarios**:

1. **Given** a configured root, **When** a caller requests a path outside it, **Then** the request is
   refused, including via traversal or symlink.
2. **Given** a tool not in the configured exposure set, **When** a caller invokes it, **Then** the
   call is refused and the tool is absent from the tool list.
3. **Given** any tool argument, **When** it reaches the underlying tool, **Then** it cannot cause
   execution of anything other than that tool.
4. **Given** a tool that writes, **When** it is invoked, **Then** it writes only within configured
   output roots.
5. **Given** a running server, **When** the operator inspects it, **Then** they can see which tools
   and roots are exposed and what has been invoked.
6. **Given** no explicit configuration, **When** the server starts, **Then** it exposes a read-only
   surface by default; write-capable and long-running tools require deliberate enablement.

---

### User Story 6 - The viewer consumes remote MCP servers (Priority: P4 — deferred)

The viewer acts as an MCP *client*, reaching outward to the external orchestration layer's servers
for job submission and VLM/LLM inference.

**Why this priority**: Deferred by explicit scope decision. It is recorded here so the server's
contract is designed with a known second phase in mind, but it is not built in this spec and its
acceptance criteria are deliberately left to be written when it is.

**Independent Test**: Not applicable in this phase.

---

### Edge Cases

- **A tool writes to standard output as its real result.** Several tools today report by printing.
  Turning that into a structured result, without silently losing detail the operator relies on, has to
  be specified per tool rather than assumed.
- **A tool needs a client data root that is machine-local.** Client paths are configuration and must
  never enter source or portable configuration; the server must resolve them without embedding them.
- **A tool prompts or blocks on input.** An invocation that waits for a human over a protocol
  connection hangs forever.
- **Argument values that name paths.** Every path argument is a potential escape from the configured
  root.
- **Two invocations writing the same output path.** Concurrency at the tool level, not the protocol
  level.
- **A tool that exhausts memory or runs unbounded.** The server must survive a tool that does not.
- **Protocol version mismatch.** A client older or newer than the server's supported specification
  version.
- **Very large results.** A corpus scan result that does not fit comfortably in one response.
- **Transport disconnection mid-write.** A tool that has already produced side effects when the caller
  vanishes.
- **The viewer holds the build output.** The standing constraint that the viewer must be closed before
  tests run has an analogue here: a server process holding assemblies open.

## Requirements *(mandatory)*

### Functional Requirements

**Protocol and contract**

- **FR-001**: The system MUST implement a Model Context Protocol server conforming to the current
  published specification, including capability negotiation and version handling.
- **FR-002**: The system MUST expose each configured tool with a name, a human-readable description,
  and a complete, machine-readable input schema.
- **FR-003**: The system MUST validate every invocation against the declared schema and reject
  non-conforming calls before any work is performed.
- **FR-004**: The system MUST return structured results rather than requiring callers to parse console
  output.
- **FR-005**: The system MUST distinguish tool execution failures from protocol faults, and carry
  diagnostic detail on both.
- **FR-006**: The system MUST expose produced artifacts as addressable, typed resources with stable
  addresses.

**Contract integrity**

- **FR-007**: A tool's declared MCP input schema and its command-line argument parser MUST be derived
  from one shared definition, so that the two cannot describe different contracts.
- **FR-008**: The system MUST fail the build when any exposed tool's two surfaces diverge, or when a
  tool is registered on one surface and not the other.
- **FR-009**: The existing command-line entry points MUST continue to work unchanged.

**Execution**

- **FR-010**: The system MUST report progress for long-running invocations.
- **FR-011**: The system MUST support cancelling an in-flight invocation, and MUST cancel invocations
  whose caller has disconnected.
- **FR-012**: The system MUST remain responsive to other requests while an invocation is running.
- **FR-013**: The system MUST isolate concurrent invocations so that neither corrupts the other's
  output.
- **FR-014**: The system MUST survive a tool that fails, hangs, or exhausts its resources, without
  taking the server down.

**Boundary and operator control**

- **FR-015**: The system MUST expose only explicitly configured tools, and MUST omit unexposed tools
  from discovery entirely.
- **FR-016**: The system MUST confine all reads and writes to configured roots, and MUST reject
  traversal and symlink escapes.
- **FR-017**: The system MUST NOT provide arbitrary command execution or unrestricted filesystem
  access under any argument value.
- **FR-018**: The system MUST default to a read-only surface, requiring deliberate enablement for
  write-capable and long-running tools.
- **FR-019**: The system MUST let the operator inspect the exposed surface and the record of what has
  been invoked.
- **FR-020**: Machine-local client data roots MUST be supplied as configuration and MUST NOT appear in
  source or in portable configuration.

### Key Entities

- **Tool Contract**: The single shared definition of one tool's name, description, arguments, types,
  requiredness, and result shape — the source both the CLI parser and the MCP schema are derived from.
- **Tool Registry**: The set of contracts the server knows about, and which of them the current
  configuration exposes.
- **Invocation**: One in-flight execution — its arguments, progress, cancellation state, result or
  error, and the caller it belongs to.
- **Resource Address**: A stable identifier for a produced artifact, independent of its location on
  disk.
- **Exposure Policy**: The operator's configuration of which tools are reachable, which roots may be
  read and written, and whether write-capable tools are enabled.
- **Invocation Record**: What was called, by whom, with what arguments, and what happened.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A standard MCP client, with no code specific to this repository, can discover and
  successfully invoke every exposed tool.
- **SC-002**: 100% of exposed tools have a declared schema that provably matches their command-line
  parser, enforced by an automated check that fails the build on divergence.
- **SC-003**: An external harness can run a complete task — discover, invoke, read the produced
  artifact — with zero hard-coded argument lines.
- **SC-004**: Invalid arguments are rejected before any work begins, in 100% of cases, with an error
  that names the offending argument.
- **SC-005**: A long-running invocation reports progress at least every 5 seconds, and cancellation
  takes effect within 5 seconds.
- **SC-006**: No sequence of tool arguments can read or write outside the configured roots, or cause
  execution of anything other than an exposed tool — verified by an explicit negative test suite
  covering traversal, absolute paths, and symlinks.
- **SC-007**: Every existing command-line invocation documented in the repository continues to work
  unchanged after the feature lands.
- **SC-008**: The server survives a deliberately failing, hanging, and memory-exhausting tool without
  becoming unresponsive.

## Assumptions

- **Server first; client deferred.** This spec builds the server only. The client (US6) is recorded
  for design context and specified later, once the server contract has been exercised by a real
  external caller.
- **The current published MCP specification is the target**, per the external project's requirement
  that both sides be current. The exact version is pinned during planning, not here, because the
  specification revises on its own schedule.
- **The CLI is not replaced.** The tools remain independently runnable. The server is a second front
  door onto the same capability.
- **One shared contract is non-negotiable.** Deriving the schema and the parser from one definition is
  the mechanism by which this feature prevents the documented CLI-drift defect rather than
  reproducing it. A hand-maintained parallel schema does not satisfy FR-007.
- **Python owns the datastore.** Resource exposure of Zarr/TensorStore data reads what Python has
  produced. This feature does not implement any datastore access in C#; that boundary is a standing
  architectural constraint, not a preference.
- **Testability follows the standing constraint.** No test project references the viewer, so contract
  definitions, schema derivation, argument validation, path confinement and registry logic belong in
  `WowViewer.Core*`.
- **The external orchestration project is out of scope.** Its job layer, inference, and internals are
  not specified or depended upon here. This feature's obligation is to present a standard-conforming
  surface that any conforming harness can drive.
- **Transport choice is a planning decision.** Which transports to support is settled in `plan.md`
  against the pinned specification version.

## Out of Scope

- Building the MCP client (deferred to a later spec).
- Any part of the external orchestration/inference project.
- Implementing Zarr or TensorStore access in C#.
- Replacing or deprecating the existing command-line tools.
- Authentication schemes beyond what the pinned specification defines, and any remote/multi-tenant
  hosting model.
