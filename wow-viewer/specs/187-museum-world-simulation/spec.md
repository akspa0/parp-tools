# Feature Specification: Single-player museum world simulation

**Feature Branch**: `187-museum-world-simulation`

**Created**: 2026-08-23

**Status**: Draft

**Depends on**: [186 Server data transformer](../186-server-data-transformer/spec.md) — this spec
consumes the store 186 produces and adds nothing to its requirements.

**Input**: User description: "there's a server that I'd like to reinvent with some sort of llm and
steady state style logic replacement, for 1.12.1+ versions of WoW... a museum that the user can play
with, with any version of the game, and see how things changed over time... We can do MCP based calls
in and out of the server component, which is meant to act as a single player solution for old
versions of WoW, to make them more interactive in the viewer."

## Context

Spec 186 makes the viewer able to *know* who lived in a place. This spec makes something *happen*
there.

Every existing option for that is a 20-year-old realm server: a C++ daemon fronting a MySQL schema,
implementing a network protocol this project has no use for, carrying an authentication and realm
layer for a single local player, and shaped throughout by the constraint of serving thousands of
concurrent clients over 2004 bandwidth. `tortoise-wow` is a good example of the form and also a good
example of why the form is wrong here.

This spec builds it as what it actually is: **a deterministic single-player world simulation, embedded
beside the viewer, reading from the store spec 186 produces.** No network protocol, no realm layer, no
database at runtime, no client interception.

### Why determinism is the load-bearing requirement

Two goals pull against each other: a reproducible automated test loop that proves the renderer
survives complex creature and player interactions, and model-driven actors. If model output sits
inside the simulation, no run is reproducible — a passing scenario proves nothing and a failing one
cannot be investigated.

The resolution is to make the core deterministic — same content version, same seed, same input log,
same result — and treat every intelligent actor as an **input source** feeding that core, exactly like
a human player's input. A model then becomes replayable, because its decisions are recorded as inputs
and can be replayed without the model.

### Where a model may act, and where it may never

The point of the model is not flavour text. It is to **replace the per-entity scripting layer** — the
thousands of lines of hand-written quest, spell and creature scripts a traditional server carries, one
authored fragment at a time. A small local model reading well-structured content can stand in for that
layer, which is what makes "a very simple set of core server needs" achievable.

That only works if the model is kept out of the path where integrity lives:

| tier | what it covers | model allowed? |
|---|---|---|
| **Stored fact** | Returning what the store holds — a creature's health, a quest's reward, a spawn's position | **Never.** Returned verbatim or not at all. |
| **Invariant** | Tick order, movement, collision, damage arithmetic, state transitions | **Never.** Deterministic rules over stored facts. |
| **Open decision** | What an unscripted creature does next; what a quest giver says; how a crowd reacts | **Yes**, and the decision is recorded as an input. |
| **Retrieval** | Finding *which* records are relevant to a situation | **Yes**, as ranking only — it selects records, never authors them. |

A model that paraphrases a stored fact has corrupted the museum, silently and unfalsifiably. A model
that chooses what a creature does has done its job. The difference is whether its output can be
*returned as data* or only *consumed as a decision*, and that is mechanically checkable.

Retrieval is the subtle one: an embedding index makes lookup fuzzy but stays deterministic given a
fixed index, because the same query against the same index returns the same ranking. So the index
version is part of what a run records.

### What exists today

`SqlWorldPopulationService` places static spawns in the viewer. There is no tick, no behaviour, no
quest state, and no combat. This spec is entirely new construction on top of 186's store.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run a deterministic world simulation with no network layer (Priority: P1)

A researcher starts a simulation over a chosen content version and map, advances it, and observes
entity state change over time — spawning, movement, aggression, death, respawn — with no protocol, no
realm, and no login.

**Why this priority**: This is the component that does not exist. It is the difference between a
viewer that knows things and a museum you can play with.

**Independent Test**: Run the same content version, seed, and input log twice and compare the
resulting state streams; they must be identical. Then confirm no network listener exists.

**Acceptance Scenarios**:

1. **Given** a content version, a map, and a seed, **When** the simulation runs a fixed number of
   ticks against a fixed input log, **Then** repeating it produces an identical state stream.
2. **Given** a running simulation, **When** the process is inspected, **Then** it has opened no
   game-protocol listener and performs no client packet handling.
3. **Given** a simulation, **When** it is asked for its state at a tick, **Then** it answers without
   requiring a rendering surface.
4. **Given** a saved run, **When** it is replayed from its seed and input log, **Then** it reaches the
   same state without the original actors being present.

---

### User Story 2 - Play the museum inside the viewer (Priority: P2)

A user loads a world in the viewer, starts the simulation, and interacts with what is there —
creatures react, objects respond, and the user can move through the world as a participant rather than
an observer.

**Why this priority**: The stated goal, and the first story a person can look at and judge.

**Acceptance Scenarios**:

1. **Given** a loaded map with simulation running, **When** the user approaches a hostile creature,
   **Then** the creature reacts and the reaction is visible.
2. **Given** simulation running, **When** the user stops it, **Then** the viewer returns to plain
   observation with no residual state.
3. **Given** a simulation error, **When** it occurs, **Then** the viewer reports it and continues
   rendering rather than terminating.

---

### User Story 3 - Drive and observe the simulation from outside (Priority: P2)

An external tool starts a scenario, issues actions, reads state back, and receives notification when
something of interest happens — and the simulation can also call outward to a configured service when
it needs a decision.

**Why this priority**: The interoperability requirement and the substrate for the automated test loop.

**Independent Test**: Drive a scripted scenario end to end from outside the process, then delete the
adapter and confirm the simulation still builds, runs and is complete.

**Acceptance Scenarios**:

1. **Given** a running simulation, **When** an external tool issues an operation, **Then** it behaves
   identically to the same operation issued in-process.
2. **Given** the adapter is removed, **When** the project is built and run, **Then** the simulation is
   complete and no core component references the adapter.
3. **Given** the simulation needs an outward decision, **When** no service is configured, **Then** it
   falls back to deterministic built-in behaviour rather than stalling.
4. **Given** an outward call is made, **When** the run is later replayed, **Then** the recorded
   response is reused and the external service is not called again.

---

### User Story 4 - Let a local model stand in for the scripting layer (Priority: P2)

A researcher configures a locally served small model, and unscripted entities start behaving — a
creature with no authored script decides what to do, a quest giver answers in its own voice — while
every stored fact the player sees still comes verbatim from the store.

**Why this priority**: This is the "steady state logic replacement" that makes the approach worth
doing instead of hand-authoring a scripting layer per entity. P2 rather than P1 because the
deterministic core and its recording mechanism must exist first.

**Independent Test**: Run a scenario with the model configured, then replay the recorded run with the
model unreachable and confirm an identical state stream. Separately, verify no value a player can read
was produced by the model.

**Acceptance Scenarios**:

1. **Given** an entity with no authored behaviour, **When** the simulation needs its next action and a
   model is configured, **Then** the model supplies a decision and that decision is recorded as an
   input.
2. **Given** a recorded run that consulted a model, **When** it is replayed with the model unreachable,
   **Then** the state stream is identical.
3. **Given** a request for a stored fact, **When** it is served, **Then** it comes from the store
   verbatim and no model participates in producing it.
4. **Given** a model returns a decision the rules do not permit, **When** it is applied, **Then** it is
   rejected by the deterministic core and a permitted fallback is used, with the rejection recorded.
5. **Given** no model is configured, **When** the simulation runs, **Then** every entity still behaves
   by built-in deterministic rules and nothing blocks.
6. **Given** retrieval is model-assisted, **When** the same query runs against the same index version,
   **Then** it returns the same ranking, and the index version is part of the run record.

---

### User Story 5 - Prove the renderer survives complex interaction, automatically (Priority: P3)

A scripted scenario stages a demanding interaction — many creatures, combat, deaths, spawns — and
produces a pass/fail signal about whether the renderer and interface handled it, without a human
watching.

**Why this priority**: A real gap: today the only renderer proof is a person looking at it. P3 because
it consumes Stories 1 and 3.

**Acceptance Scenarios**:

1. **Given** a scenario, **When** it runs unattended, **Then** it produces a named verdict with the
   evidence behind it.
2. **Given** a build with a deliberately introduced rendering fault, **When** the scenario runs,
   **Then** the verdict is a failure naming the affected stage.
3. **Given** a failing scenario, **When** the verdict is produced, **Then** it includes what is needed
   to reproduce the failure.

---

### Edge Cases

- An outward decision service that is slow, unavailable, or returns something unusable — the
  simulation must stay deterministic and must not hang.
- A replay whose recorded outward responses no longer match the current content version.
- A scenario that stresses the renderer hard enough to fail for reasons unrelated to the simulation.
- Content for a version whose client assets are not present.
- A simulation running faster than real time for testing, where behaviour must not depend on wall
  clock.
- A source that supplies contradictory behaviour for one entity, where the simulation must pick one
  and say which.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The simulation core MUST be deterministic: identical content version, seed, and input log
  MUST produce an identical state stream.
- **FR-002**: The simulation core MUST NOT implement any game network protocol, and MUST NOT open a
  game-protocol listener or interpret client packets.
- **FR-003**: The simulation core MUST run without a rendering surface.
- **FR-004**: The simulation core MUST read content only from the store spec 186 produces, and MUST NOT
  require a database engine at any point.
- **FR-005**: The simulation core MUST expose its operations as a library surface that the viewer
  consumes in-process.
- **FR-006**: Every external adapter MUST be optional; removing all of them MUST leave the simulation
  complete and buildable, and no core component may reference an adapter type.
- **FR-007**: Non-deterministic decision sources MUST be modelled as inputs to the core, never as logic
  inside it, and their responses MUST be recorded so a run can be replayed without them.
- **FR-008**: A stored fact MUST be served verbatim from the store. No model may produce, paraphrase,
  complete, or substitute any value presented to a user as content.
- **FR-009**: A model MAY supply open decisions for entities with no authored behaviour, and every such
  decision MUST be validated against the deterministic rules before taking effect; a decision the rules
  do not permit MUST be rejected, replaced by a permitted fallback, and recorded as rejected.
- **FR-010**: Model-assisted retrieval MUST rank existing records only and MUST NOT author them; the
  retrieval index version MUST be part of the run record.
- **FR-011**: When a configured outward decision source is unavailable, the simulation MUST fall back
  to deterministic built-in behaviour and MUST NOT block.
- **FR-012**: The simulation MUST record runs such that any run can be replayed from its recorded
  inputs.
- **FR-013**: Scenario runs MUST produce a named verdict with supporting evidence, and MUST be able to
  fail.
- **FR-014**: Simulation behaviour MUST NOT depend on wall-clock time.
- **FR-015**: The system MUST NOT read from, write to, or interoperate with a live game client's
  network traffic.
- **FR-016**: Simulation failures MUST be surfaced to the viewer without terminating rendering.
- **FR-017**: Where ingested sources supply contradictory behaviour for one entity, the simulation MUST
  select one and report which source it used, rather than blending them.

### Key Entities

- **Simulation run**: a content version, a seed, an input log, and the resulting state stream.
- **Input**: anything influencing the simulation from outside — user action, scripted action, or a
  recorded decision from an external source.
- **Operation surface**: the set of actions and queries the simulation offers, independent of caller.
- **Adapter**: an optional binding of the operation surface to an external transport.
- **Scenario**: a scripted run with a verdict and its evidence.
- **Decision source**: an external supplier of open decisions, with a deterministic fallback.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The same content version, seed, and input log produce byte-identical state streams across
  100% of repeated runs.
- **SC-002**: The simulation runs headless with no rendering surface and no game-protocol listener,
  verified by inspection of the running process.
- **SC-003**: No database engine is installed, running, or required, and no runtime component issues a
  database query.
- **SC-004**: Removing every external adapter leaves the project building and the simulation complete,
  with zero references from core components to adapter types.
- **SC-005**: A run that consulted an external decision source replays to the identical state stream
  without that source being reachable.
- **SC-006**: A scenario run against a deliberately broken build produces a failing verdict naming the
  affected stage; the same scenario against a good build passes.
- **SC-007**: A user can enter a populated area in the viewer, provoke a creature, and see it react.
- **SC-008**: With a model configured, an entity with no authored behaviour acts; with the model
  removed, it still acts by built-in rules and nothing blocks.
- **SC-009**: Zero values presented to a user as content originate from a model, verified by inspection
  of the serving path.
- **SC-010**: Rejected model decisions appear in the run record.

## Assumptions

- Target clients are 1.12.1 and later, with custom-modified 1.12.1 builds as the primary case; other
  eras are reachable through the same store but are not the first target.
- This is single-player. Concurrency, authentication, realms, and anti-cheat are not requirements and
  their absence is a design choice, not a gap.
- Fidelity to any particular server project's behaviour is not a goal. Where sources disagree, the
  museum picks a documented behaviour and says so rather than reproducing a specific server's quirk.
- Combat, quest, and reward logic are deterministic rules over stored content, not a reimplementation
  of any specific server's code.
- The intended model is small and locally served — the user already runs a Gemma-class model through
  `llama.cpp` on this machine and reaches it over MCP. The spec is deliberately **model-agnostic**: it
  requires a decision source with a fallback, not a named model.
- A small model is viable because 186's ingest is lossless: it is handed exact, complete, structured
  records and asked for a decision, rather than asked to have memorised anything.
- Real-client and real-session proof, long runs, and any model-serving runs are user-run.

## Out of Scope

- Any game network protocol, packet handling, realm or login layer.
- Interoperating with, intercepting, or proxying a live game client's traffic.
- Multiplayer of any kind.
- Reimplementing or porting an existing server project's source.
- Ingesting server databases — that is spec 186, which this spec consumes.
- Training or fine-tuning a model; a domain-specialised adapter is a configuration change under FR-009.
- Writing into Blizzard container formats.
