# Feature Specification: 5.0.1 Physics — Decode the Data, Drive a Licensed Solver

**Feature Branch**: `214-mop-physics-domino` (authored on `v0.5.3`; specs 193–213 follow the same convention)

**Created**: 2026-09-02

**Status**: Draft

**Input**: User description: "spec out and use ghidra to figure out the physics system that the 5.0.1 client added to the engine, since the animations of some objects in 5.0.1 are entirely dependent on that system existing, especially for flags that whip around in the snow storm weather effects... very important that we plan for that stuff now, with real data from the real client binary."

## Context & Motivation

Some 5.0.1 content does not animate at all in the viewer, and cannot, because its motion is not
authored. Flags, banners, cloaks and tabards have no keyframes for the part that moves — the client
*simulates* them. Without the simulation those models are static geometry, and no amount of work on
the animation path will fix it, because there is no animation to play.

The system responsible was added to the engine after the alpha era. Reconnaissance against the real
binary (recorded in
[`memory-bank/workstream-atmosphere-501-ghidra.md`](../../memory-bank/workstream-atmosphere-501-ghidra.md))
establishes what it is:

- The solver is **Domino**, a distinct engine living at
  `Engine\Source\Domino/{Common,Dynamics,Collision}` in Blizzard's tree. Its headers are named
  directly in surviving assert strings: `dmIsland.h`, `dmContact.h`, `dmDynamicTree.h`,
  `dmHullBuilder.h`, `dmPolytope.h`, `dmMeshData.h`, `dmTreeMesh.h`, `dmDistance.h`.
- WoW's own adapter sits *beside* it, not inside it: `Engine\Source\Physics/PhysData.h`, with
  `Physics.cpp` (`0x00d7592c`), `PhysicsInt.cpp` (`0x00d75a40`) and `PhysData.cpp` (`0x00d75dac`).
- `"Domino Assertion: %s, in file %s, line %d"` (`0x00e0ac8c`) reaches its handler `FUN_00c29680`,
  whose **~90 calling functions** span a contiguous **`0x00c26e50`–`0x00c4e012`** region. Each call
  site carries its own file and line, so the handler's callers are a self-labelling map of the
  solver.
- The client does **not** simulate everything: `"Physics culling dist set to %f."` (`0x00d759e8`)
  and paired enable/disable strings for both *processing* and *culling* prove a runtime budget.

## The solver is licensed in, not reimplemented

**Domino is Blizzard's proprietary engine and is not reproduced by this feature.** The operator's
direction is to use an existing, permissively licensed C# physics library and to keep all
copyrighted engine code out of the tooling.

This changes what the reverse engineering is *for*, and the distinction matters both legally and
technically:

- **In scope**: the *data* — what a physicalised asset declares (bodies, shapes, joints, limits,
  masses) and how it binds to a model — and the *observable behaviour* the client produces from it,
  including its budget rules. Formats and behaviour are what a compatible reader needs.
- **Out of scope**: transcribing Domino's algorithms, structures or source-derived logic into this
  repository. Decoding the solver's internals to reproduce them would make this a derivative of
  Blizzard's engine, which is exactly what the operator has ruled out.

So the deliverable is: **read Blizzard's physics data, drive an independent solver with it, and match
the client's observable motion.** Where the two disagree, the finding is recorded rather than
"fixed" by importing Blizzard's method.

This also resolves the scope concern that an earlier draft of this spec raised. Writing a rigid-body
solver with broadphase, hull generation and an island solver from scratch is a multi-month effort
whose correctness is invisible until nearly complete. Adopting a maintained library removes that
entirely and leaves the genuinely novel work: the asset format, the binding to models, the budget,
and the era gating.

**Candidate libraries** (final selection and license verification belong in `plan.md`, not here):
`BepuPhysics v2` and `Jitter2` are pure C# and avoid native interop, which suits this repository's
cross-platform targets; NVIDIA's `PhysX` and `Bullet` are mature natives with .NET bindings. Each
must have its license confirmed as permissive at the version actually used, and recorded, before it
is adopted — the spec requires the check rather than assuming any particular answer.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The solver's shape is recovered from the binary (Priority: P1)

An engineer can read a written contract for the physics **data** — what a body is, what a shape is,
what a joint is, how they are stored, and what the client's observable behaviour and budget rules
are — with every claim carrying the address it came from. The contract describes what must be *read*
and *matched*, not Domino's internal algorithms.

**Why this priority**: Nothing can be implemented before this exists, and this is the step that the
project's own history says gets skipped. Every field named in the contract must be one that was
*read*, not one that was recognised by name. It is also where the legal boundary is drawn: the
contract records formats and behaviour, and explicitly does not transcribe engine internals.

**Independent Test**: Pick any claim in the contract at random; it names an address, and decompiling
that address supports the claim.

**Acceptance Scenarios**:

1. **Given** the assertion handler `FUN_00c29680`, **When** its callers are enumerated, **Then** the
   Domino region is bounded and each cluster is attributed to a source header and line from its own
   call site — used to locate the data and behaviour boundary, not to transcribe the algorithms.
2. **Given** the recovered map, **When** the contract is written, **Then** every structure layout,
   field offset and stride in it cites the address it was measured at.
3. **Given** a field whose purpose is inferred rather than measured, **When** the contract is
   written, **Then** it is recorded as unverified and named for what it demonstrably is, not for
   what it is assumed to be.
4. **Given** the WoW-side `PhysData` layer, **When** it is decoded, **Then** the boundary between
   Blizzard's adapter and Domino proper is stated explicitly, and the two are not described as one
   system.
5. **Given** the contract, **When** it is reviewed, **Then** it contains no transcribed engine
   algorithm or source-derived logic — only data layouts, observable behaviour and budget rules.

---

### User Story 2 - Physicalised assets are located and parsed (Priority: P1)

The viewer can find the physics description attached to a model, read it, and report what bodies,
shapes and joints it declares — without simulating anything yet.

**Why this priority**: The data must be readable before the solver has anything to solve, and this
is independently valuable: it makes it possible to *see* which 5.0.1 assets are physicalised and
what they declare, which is currently unknown.

**Independent Test**: Point the reader at 5.0.1 assets and enumerate every physics body, shape and
joint, with counts that can be checked against the file bytes.

**Acceptance Scenarios**:

1. **Given** a physicalised model, **When** its physics description is loaded, **Then** bodies,
   shapes, joints and their parameters are reported.
2. **Given** a model with no physics description, **When** it is loaded, **Then** it renders exactly
   as it does today, with no error and no penalty.
3. **Given** a malformed or truncated physics description, **When** it is loaded, **Then** it is
   rejected with a diagnostic and the model still renders.
4. **Given** the shape types the format declares, **When** any is encountered, **Then** it is either
   parsed or explicitly reported as unsupported — never silently skipped.

---

### User Story 3 - A third-party solver is integrated and bodies come to rest (Priority: P2)

A permissively licensed physics library is selected, its license recorded, and it is driven by the
decoded data. A rigid body released in the scene falls under gravity, collides with world geometry
and with other bodies, and settles. It does not sink, jitter, or gain energy.

**Why this priority**: This is the first point at which simulation exists rather than being
described. "It comes to rest and stays there" is a demanding acceptance test — it fails first when
the data is mapped onto the solver incorrectly, which is the actual risk now that the solver itself
is not being written here.

**Independent Test**: Drop bodies of each supported shape onto known geometry and measure resting
penetration, settling time, and energy over time.

**Acceptance Scenarios**:

0. **Given** a candidate physics library, **When** it is adopted, **Then** its license is verified as
   permissive at the version used and is recorded, and no proprietary engine code enters the
   repository.
1. **Given** a body released above a surface, **When** it is simulated, **Then** it comes to rest in
   contact with that surface and stays there.
2. **Given** a body at rest, **When** simulation continues, **Then** its total energy does not
   increase and it does not drift.
3. **Given** a stack of bodies, **When** simulated, **Then** the stack does not sink into itself or
   explode.
4. **Given** a body moving quickly at a thin surface, **When** simulated, **Then** it does not pass
   through it.
5. **Given** the same scene and the same starting state simulated twice, **When** compared, **Then**
   the results match — the simulation is deterministic.

---

### User Story 4 - Flags, banners and cloth move (Priority: P2)

The motion that prompted this work. Flags and banners on 5.0.1 models move as attached soft bodies,
respond to wind, and are constrained at their attachment points.

**Why this priority**: This is the visible payoff and the reason the request was made. It sits after
US3 because cloth is solved by the same constraint machinery, but it is called out separately so
that it is reached — and can be judged — well before the full-fidelity phases complete.

**Independent Test**: Load a 5.0.1 model with a physicalised flag and observe motion under a
controlled wind input; compare to reference footage of the client.

**Acceptance Scenarios**:

1. **Given** a model with a physicalised flag, **When** it is rendered with a non-zero wind input,
   **Then** the flag moves and remains attached at its constrained points.
2. **Given** zero wind, **When** rendered, **Then** the cloth hangs under gravity and settles rather
   than remaining rigid or oscillating forever.
3. **Given** cloth and the model it is attached to, **When** the model moves or animates, **Then**
   the cloth follows the attachment and reacts to the movement.
4. **Given** cloth and collision geometry, **When** they meet, **Then** the cloth does not pass
   through the body it belongs to.

---

### User Story 5 - The simulation is budgeted the way the client budgets it (Priority: P2)

Physics has a distance cull and can be switched off, matching the runtime controls the client
exposes. A crowded scene does not lose its frame rate to simulation.

**Why this priority**: The binary proves the client culls by distance; a reimplementation that
simulates everything is not a faithful one and will not perform. This project has a standing problem
with object-count frame cost, and adding an unbudgeted per-object solver to it would compound a known
defect.

**Independent Test**: Measure frame-time distribution with a physicalised crowd at varying cull
distances, and with physics disabled.

**Acceptance Scenarios**:

1. **Given** a physics culling distance, **When** objects are beyond it, **Then** they are not
   simulated.
2. **Given** an object crossing the cull boundary, **When** it re-enters, **Then** it resumes without
   a visible pop or a burst of accumulated motion.
3. **Given** physics disabled, **When** the scene renders, **Then** it costs no more than it did
   before this feature existed.
4. **Given** a fixed budget, **When** more objects are eligible than the budget allows, **Then** the
   overflow degrades predictably rather than by dropping frames.

---

### User Story 6 - Joints and articulated assemblies hold together (Priority: P3)

The joint types the format declares — the weld/spherical/shoulder/prismatic/revolute family — behave
according to their constraints.

**Why this priority**: Required for full-solver fidelity and for anything articulated, but not for
the flag motion that motivated the work. Depends on US3's contact and island machinery.

**Independent Test**: Construct an assembly for each joint type and verify its constrained and free
degrees of freedom.

**Acceptance Scenarios**:

1. **Given** each supported joint type, **When** simulated, **Then** it constrains exactly the
   degrees of freedom it is meant to and permits the others.
2. **Given** a jointed assembly at rest, **When** simulated, **Then** it does not separate or drift.
3. **Given** a joint with declared limits, **When** driven past them, **Then** the limits hold.

---

### User Story 7 - Era behaviour is gated and provenance is carried (Priority: P1)

Physics activates only for eras that actually had it. Alpha content is unaffected. Every physics
decision records which era profile produced it.

**Why this priority**: P1 despite being last in the list, because it is a constraint on *all* the
other work rather than a stage of it, and because this project has paid for era-blind decoding
before — MCNR component order is era-split and the shared decoder applied one era's order
everywhere. Domino does not exist in 0.5.3; a system that assumes it does is wrong for the
project's primary target.

**Independent Test**: Load alpha content and confirm the physics path does not engage; load 5.0.1
content and confirm it does; confirm both record which profile decided.

**Acceptance Scenarios**:

1. **Given** 0.5.3 content, **When** it loads, **Then** no physics path engages and rendering is
   unchanged from before this feature.
2. **Given** 5.0.1 content, **When** it loads, **Then** the physics path engages.
3. **Given** a build the era profile does not recognise, **When** it loads, **Then** it is flagged
   rather than silently assigned a default.
4. **Given** any physics result, **When** it is inspected, **Then** it carries the era profile and
   evidence source that produced it.

---

### Edge Cases

- **A decoded field that is never written.** A structure member the solver only reads is not
  evidence of its meaning; it may be padding, a cache, or dead.
- **Inlined and merged functions.** An optimised build may fold several Domino functions into one,
  so a one-function-to-one-header mapping will not always hold.
- **The ~90 asserting functions are a floor.** Functions that never assert are invisible to the
  pivot and must be found another way.
- **Determinism versus floating point.** Frame-rate-dependent integration, or reordering, will make
  results irreproducible; the acceptance criteria demand reproducibility.
- **Bodies that never sleep.** Without a sleep/deactivation rule, a crowd accumulates permanent cost.
- **Degenerate input geometry.** Zero-area shapes, coincident points, and hulls that cannot be built.
- **Attachment to an animated skeleton.** Cloth constrained to a bone that is itself moving each
  frame is a moving boundary condition, not a static one.
- **Very large time steps.** A stalled frame must not be integrated as one enormous step.
- **The alpha client has none of this.** Absence in 0.5.3 is the expected finding, not a decode
  failure.
- **Transports are a separate path.** `TransportPhysics.dbc` is data-driven and may not use the same
  route as model cloth.

## Requirements *(mandatory)*

### Functional Requirements

**Evidence**

- **FR-001**: The system MUST produce a written Domino contract in which every structure layout,
  field offset, stride and update-order claim cites the binary address it was measured at.
- **FR-002**: The system MUST distinguish measured facts from inferences, and MUST name fields for
  what they demonstrably are rather than for what an inherited name suggests.
- **FR-003**: The system MUST state the boundary between Blizzard's `Physics`/`PhysData` adapter and
  Domino proper.
- **FR-004**: Reverse-engineering sessions MUST be read-only with respect to the Ghidra program
  unless the operator authorises otherwise, and any exception MUST be recorded.
- **FR-005**: The recovered contract MUST be limited to data layouts, observable behaviour and budget
  rules, and MUST NOT transcribe engine algorithms or source-derived logic.

**Licensing**

- **FR-006**: The system MUST use an existing third-party physics library rather than a
  reimplementation of Domino.
- **FR-007**: Any adopted library's license MUST be verified as permissive at the version actually
  used, and MUST be recorded in the repository.
- **FR-008**: No proprietary or copyrighted engine code MUST enter the repository, in any form,
  including transcribed or machine-translated algorithms.

**Data**

- **FR-009**: The system MUST locate and parse the physics description associated with a model.
- **FR-010**: The system MUST report every declared body, shape and joint, and MUST explicitly report
  any construct it does not support rather than skipping it.
- **FR-011**: The system MUST render a model with no physics description exactly as it does today.
- **FR-012**: The system MUST reject malformed physics data with a diagnostic while still rendering
  the model.

**Simulation** *(delivered by the adopted library, driven by the decoded data)*

- **FR-013**: The system MUST simulate rigid bodies under gravity with collision against world
  geometry and other bodies.
- **FR-014**: The system MUST resolve contacts such that bodies come to rest without sinking,
  jittering, or gaining energy.
- **FR-015**: The system MUST prevent fast-moving bodies from passing through thin geometry.
- **FR-016**: The system MUST support the declared joint family and honour declared joint limits.
- **FR-017**: The system MUST simulate attached cloth/soft bodies with wind and gravity input,
  constrained at their attachment points, and MUST follow an animated attachment.
- **FR-018**: The system MUST produce identical results for identical inputs and starting state.
- **FR-019**: The system MUST decouple simulation stepping from frame rate and MUST bound the effect
  of an unusually long frame.
- **FR-020**: The system MUST deactivate bodies that have come to rest and reactivate them when
  disturbed.

**Budget**

- **FR-021**: The system MUST support a physics culling distance beyond which objects are not
  simulated.
- **FR-022**: The system MUST allow physics to be disabled entirely, returning pre-feature cost.
- **FR-023**: The system MUST resume a re-entering object without a visible discontinuity.
- **FR-024**: The system MUST degrade predictably when eligible objects exceed the budget.

**Era**

- **FR-025**: The system MUST gate physics by era profile and MUST NOT engage it for eras that did
  not have it.
- **FR-026**: The system MUST flag builds the era profile does not recognise rather than defaulting
  them.
- **FR-027**: Every physics result MUST carry the era profile and evidence source that produced it.

### Key Entities

- **Physics Contract**: The written, address-cited description of Domino's object model and update
  order — the deliverable of US1 and the input to everything else.
- **Rigid Body**: A simulated object with mass, inertia, position, orientation and velocity, in an
  active or sleeping state.
- **Collision Shape**: The geometry a body collides with — the primitive, hull and mesh forms the
  format declares — which is not the geometry it renders.
- **Joint**: A constraint between two bodies, of a declared type, with declared limits.
- **Contact**: A resolved touch between two shapes, carrying the information needed to separate them.
- **Simulation Island**: A connected group of bodies solved together, and the unit at which sleep is
  decided.
- **Cloth Body**: A deformable body attached to a model, driven by wind, gravity and its attachment.
- **Simulation Budget**: The cull distance, active-object limit and enable state governing cost.
- **Era Profile**: The per-build statement of which physics capabilities exist, carried as provenance.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of structural claims in the physics contract cite a binary address; a reviewer
  can select any claim at random and confirm it at that address.
- **SC-002**: Every function reached from the Domino assertion handler is attributed to a source
  header and line, and the count of attributed functions is reported against the ~90 measured floor.
- **SC-003**: Every physics construct declared by 5.0.1 assets is either parsed or explicitly
  reported unsupported — zero silently skipped.
- **SC-004**: A body released above a surface rests within 0.01 units of penetration and its energy
  does not increase over 60 seconds of simulation.
- **SC-005**: Identical inputs reproduce identical results across runs — zero divergence.
- **SC-006**: Flags and banners on 5.0.1 models move under wind and remain attached, and a side-by-side
  against client reference footage is judged a match by the operator.
- **SC-007**: With physics disabled, frame cost is indistinguishable from the pre-feature baseline,
  measured against the frame-time distribution rather than a static-camera average.
- **SC-008**: With a physicalised crowd at the client's cull distance, simulation stays within its
  declared frame budget.
- **SC-009**: 0.5.3 content renders identically to the pre-feature baseline — zero observable change.
- **SC-010**: Every physics result carries an era profile and evidence source; unrecognised builds
  are flagged, never defaulted.
- **SC-011**: The adopted physics library's license is permissive at the version used and is recorded
  in the repository; a review of the diff finds zero transcribed proprietary engine code.

## Assumptions

- **The binary and Ghidra project are available.** `Wow.exe` 5.0.1.15464 is loaded in the project at
  `C:\WoW4-data\MoPBeta\ghidra\Mists of Pandaria 5.0.1.15464`. If that becomes unavailable, this
  feature is blocked, not merely slowed.
- **Decode from 5.0.1, gate by era.** 5.0.1 is used because it is the *complete* implementation. It
  is not evidence about 0.5.3, and this project has already paid twice for treating one era's
  decode as universal.
- **The solver is licensed in, not written.** Operator direction: use an existing permissively
  licensed C# physics library; no copyrighted engine code in the tooling. Full *capability* is still
  in scope — rigid bodies, collision, contacts, joints and cloth — but it is delivered by the library
  and driven by decoded data. The novel work is the asset format, the model binding, the budget and
  the era gating.
- **Domino is decoded as a contract, never reproduced.** Formats and observable behaviour are read;
  algorithms are not transcribed. This is both the operator's instruction and the safer footing.
- **The assertion pivot is the entry point, not the whole map.** ~90 functions is a floor.
- **Simulation must be visible to be judged.** Reference footage comparison is an operator task; this
  spec does not claim automated visual proof.
- **Testability follows the standing constraint.** No test project references the viewer, so the
  solver, the physics data reader, and era gating belong in `WowViewer.Core*`.
- **Operator-owned work stays operator-owned.** Real-client capture, reference footage, and
  frame-rate proof are the operator's to run.
- **Spec 215 owns weather.** Wind arrives at this feature as an input; where it comes from is spec
  215's concern. The two connect at that input and nowhere else.

## Out of Scope

- Weather simulation, precipitation and wind *generation* (spec 215).
- Sky, cloud, fog and lighting (specs 160, 147, 143 — see the shared evidence note).
- Reimplementing Domino, or any part of a proprietary engine.
- Server-side or gameplay physics, collision-driven movement, and pathing.
- Establishing what 0.5.3 did instead; that is separate work against the 0.5.3 binary.
- Authoring or editing physics data.
