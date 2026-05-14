# wow-viewer Engine Modernization Plan

## Status

- status: active
- date: 2026-05-14
- owner: `wow-viewer`
- intent: evolve `wow-viewer` from a viewer-led migration into a real engine program with WoW-data support, modern rendering backends, and future ML-native content seams

## Thesis

`wow-viewer` is no longer just a viewer port.

The viewer/editor host is one product surface. The actual target is a modern engine that:

1. uses old WoW client data as a first-class world/content source,
2. preserves strong format and runtime truth against the original games,
3. can grow beyond WoW-specific constraints into a general world runtime,
4. can eventually consume tiny trained models as data generators, synthesis helpers, and runtime content systems.

The right mental model is:

- `WowViewer.Core` and `WowViewer.Core.IO` are becoming content and world-foundation layers.
- `WowViewer.Core.Runtime` is becoming the engine runtime.
- `WowViewer.App` is becoming the current `game-viewer` host product of that engine.
- the current viewer cutover is a sub-problem inside the larger engine plan.

## Product Direction

Near term:
- build a real engine that can faithfully open, inspect, convert, and render WoW-era data
- replace the legacy viewer/runtime ownership with `wow-viewer` ownership
- make import/export and data-root management the first practical editor shell, not a late afterthought

Medium term:
- support original-world playback, inspection, and editing with a modern engine architecture
- support custom data domains that are not constrained to Blizzard file formats
- support metadata-driven feature gating and multi-root workflows across different compatible game data sets

Long term:
- support prompt-driven or model-driven world generation
- support generated imagery, structures, props, terrain, and characters as engine content inputs
- support a future external dataset-management system without baking that private system into this repo

## Hard Direction Changes

This plan replaces the old framing where the main goal was "port the viewer."

New framing:

1. `wow-viewer` is an engine program first.
2. `game-viewer` is only one engine host.
3. WoW data is the first major content compatibility target, not the final boundary.
4. ML work is not a sidecar; it is a planned future content source, but it must enter through explicit engine seams.
5. The engine core must stay game-neutral; WoW-style formats are compatibility profiles, not the shape of the engine itself.

## Non-Negotiable Constraints

1. `gillijimproject_refactor` remains read-only reference input.
2. file-format ownership stays in `wow-viewer` shared libraries.
3. runtime ownership stays in `WowViewer.Core.Runtime`.
4. rendering backend ownership must not leak format or world-logic concerns back into app glue.
5. ML/runtime integration must consume explicit engine contracts, not reach directly into archive readers or ad hoc tools.
6. one phase at a time; every phase ends with proof and an honest remaining boundary.

## Backend Strategy

### Primary backend

- Vulkan

### Required fallback backend

- OpenGL

### Why this is the right split

- Vulkan is the correct long-range engine backend for the user's intended architecture: explicit resource control, modern frame graphs, compute-friendly workflows, and future coexistence with small ML-driven generation or inference systems.
- OpenGL remains the practical compatibility fallback and proof surface while the engine backend is maturing.
- The runtime and engine contracts must stay backend-agnostic. Vulkan should be the lead implementation, not the only shape the architecture can express.

## Engine Pillars

### Pillar A — Content Truth

The engine must continue to be grounded in real source data:

- ADT/WDT/WDL/WMO/M2/MDX/BLP/DBC/DB2/PM4 read truth
- staged-client validation truth
- converter and harvest truth
- exact provenance on derived content
- artifact-preservation truth across raw, decoded, normalized, and converted representations

### Pillar A1 — Artifact Preservation

The engine must preserve witness to the genuine artifact:

- exact raw bytes where lawful and practical
- version/build identity
- buggy or accidental shipped content
- provenance from source artifact to normalized engine representation
- explicit difference between faithful artifact playback and cleaned-up forward-native content

### Pillar B — Runtime Truth

The engine runtime must own:

- world/session bootstrap
- visibility and LOD policy
- streaming and residency policy
- pass routing
- terrain/liquid/sky/object submission contracts
- diagnostics and frame summaries

### Pillar C — Backend Separation

Backend layers must consume runtime packets rather than invent policy:

- Vulkan backend
- OpenGL fallback backend
- optional later headless/test backend

### Pillar D — Data/Model Interop

The ML side must remain modular:

- V14 stays the current training architecture
- model outputs become engine content inputs only through explicit contracts
- no direct "LLM magic" shortcuts inside core runtime paths

### Pillar E — Beyond-WoW Extensibility

The engine must be able to host:

- WoW-native content
- Warcraft 3-compatible asset/archive content where the profile contract supports it
- forward-native content profiles such as GLB + textures + sidecar metadata
- converted/custom assets
- future generated terrain/props/buildings/characters
- non-WoW world semantics where practical

## Architecture Stack

```text
Applications / Hosts
  game-viewer (current host in `WowViewer.App`)
  future editor host
  future headless/runtime tools

Engine Runtime
  WowViewer.Core.Runtime
  world session
  visibility/LOD
  streaming
  render packet generation
  simulation-facing contracts

Engine Rendering
  backend-agnostic render graph/contracts
  Vulkan backend
  OpenGL fallback backend

Content Foundation
  WowViewer.Core
  WowViewer.Core.IO
  universal content contracts
  provenance and artifact layers
  maps/models/textures/db/archive readers and writers
  converters
  harvest contracts

Data + ML Pipeline
  WowViewer.Tool.Harvest
  data-harvester/
  V14 training/inference tools
  future engine-consumable generated content outputs
```

## Core Reframe Of Existing Projects

### `WowViewer.Core`

Evolve from shared domain library into engine-facing content contracts:

- universal content contracts
- provenance contracts
- world tiles and terrain contracts
- object placement contracts
- material/texture metadata contracts
- future generated-content contracts

### `WowViewer.Core.IO`

Remain the canonical source of file/data truth:

- archive access
- file readers/writers
- converters
- import/export bridges

### `WowViewer.Core.Runtime`

This becomes the real engine center:

- world runtime
- render-facing frame composition
- streaming/residency
- asset lifecycle
- simulation-ready world services

### `WowViewer.App`

This is one engine host:

- `game-viewer` shell
- diagnostic app
- future editor shell
- runtime proof harness

It is not the engine itself.

## Editor And Interop Direction

The engine's first editor shell should be import/export-first.

That means the initial UX should prioritize:

- game/data root management
- asset inventories
- import/export and conversion flows
- compatibility/profile diagnostics
- render preview workspaces

The detailed host plan for this lives in:

- `wow-viewer/docs/architecture/wow-engine-editor-and-interop-plan-2026-05-14.md`
- `wow-viewer/docs/architecture/game-viewer-host-plan-2026-05-13.md`
- `wow-viewer/docs/architecture/game-viewer-plan-pack-2026-05-14/README.md`

## ML And Generated-Content Direction

The repo should explicitly plan for three ML integration lanes:

### Lane 1 — Offline supervision and training

- current V14 terrain-model work
- future asset or content models
- dataset generation and validation

### Lane 2 — Offline content generation

- generated terrain
- generated props/buildings
- generated characters
- image-to-world or image-to-asset transforms

### Lane 3 — Runtime assist systems

- small local models for data expansion, tagging, layout hints, or procedural guidance
- small models as bounded runtime generators where latency and determinism are acceptable

Hard rule:
- runtime ML systems must enter through asset/content services or world-generation services, not by bypassing engine ownership seams

## Future Dataset-Management Boundary

The user has a separate non-public dataset-management project.

This repo should assume:

- that project is external
- this repo should expose clean import/export and manifest seams
- no private-coupled architecture should be baked into `wow-viewer`

Therefore:

- prefer manifests, artifact contracts, and content-package boundaries
- avoid hardwiring repo logic to one external orchestration system
- keep the engine core compatible with forward-native content profiles that never touch WoW-style storage

## Execution Phases

### Phase E0 — Engine Plan Reset

Intent:
- align docs and routing so future work treats `wow-viewer` as an engine program, not just a viewer port

In scope:
- this plan
- reclassify the viewer migration plan as a sub-plan
- clarify backend direction: Vulkan first, OpenGL fallback

Proof:
- repo-local architecture docs updated and internally consistent

### Phase E1 — Engine Runtime Contracts

Intent:
- define the runtime contracts that make backend and host separation durable

Must include:
- backend-agnostic frame graph contracts
- render packet families for terrain, liquid, sky, WMO, MDX/M2, overlays
- world/session lifecycle contracts
- asset residency and streaming contracts

Primary owner:
- `wow-viewer/src/core/WowViewer.Core.Runtime`

Proof:
- focused contract tests
- CLI/runtime diagnostics that enumerate packet families and backend readiness

### Phase E2 — Vulkan Backend Baseline

Intent:
- land the first real Vulkan engine backend

Must include:
- device/swapchain/frame lifecycle
- resource allocation strategy
- command recording flow
- basic offscreen/world-frame proof

Proof:
- deterministic clear-frame or bounded terrain frame proof from `WowViewer.App`

### Phase E3 — OpenGL Fallback Baseline

Intent:
- keep a functional fallback backend without making it the architecture owner

Must include:
- same runtime packet consumption shape as Vulkan where practical
- comparable diagnostics and proof commands

Proof:
- same bounded frame exercised through OpenGL fallback path

### Phase E4 — World Rendering Closure

Intent:
- move from bounded previews into a real world renderer baseline

Must include:
- terrain submission
- liquid submission
- skybox/lighting baseline
- WMO and M2/MDX world placement consumption
- runtime-driven visibility and LOD

Proof:
- real-data captures on staged clients with pass counters and clear remaining gaps

### Phase E5 — Engine Host Expansion

Intent:
- raise the app from diagnostic host to practical engine host

Must include:
- game manager
- world navigation and inspection
- asset workspaces
- import/export entry workflows
- runtime diagnostics
- capture and validation workflows

Proof:
- routine engine validation no longer requires legacy `MdxViewer`

### Phase E6 — Authoring And Conversion Surfaces

Intent:
- make the engine useful for real workflows, not just viewing

Must include:
- converter UX integration
- import/export flows
- bounded world-editing or authoring seams where grounded

Proof:
- end-to-end operator workflow from source data to engine-visible output

### Phase E7 — ML Content Ingest Contracts

Intent:
- define how future models hand content into the engine

Must include:
- manifests for generated terrain/assets
- versioned content contracts
- provenance fields
- validation tools for generated content packages

Proof:
- one synthetic/generated content path loaded by the engine without special-case hacks

### Phase E8 — Beyond-WoW World Mode

Intent:
- prove the engine can host non-WoW-native content semantics

Must include:
- engine-owned world/content packages that are not archive-dependent
- runtime path that does not assume WoW client roots

Proof:
- one bounded custom world loaded through the same engine/runtime stack

### Phase E9 — Multi-Profile Modding Tool Closure

Intent:
- make the engine/editor a real multi-profile modding tool rather than only a WoW runtime host

Must include:
- compatibility-profile routing
- metadata-driven feature gating
- DBC/DB2 editor ownership baseline
- cross-root interop flows

Proof:
- one shell can switch between at least multiple supported profile families and expose different tool surfaces honestly

## What This Plan Explicitly Does Not Do

- It does not collapse V14 training work into renderer work.
- It does not let "future AI game" aspirations justify vague architecture today.
- It does not replace proof with speculation.
- It does not turn `wow-viewer` into a generic engine overnight.
- It does not discard WoW-data truth; WoW compatibility remains the proving ground.

## Immediate Next Planning Follow-Ups

1. Retarget the current `game-viewer-host-plan-2026-05-13.md` as the app-host/viewer sub-plan under this engine plan.
2. Write a dedicated Vulkan/OpenGL engine-backend plan that defines runtime packet families, renderer modules, and proof commands.
3. Write the editor/interop plan for import/export-first UX, game-manager ownership, compatibility profiles, and DBC editing.
4. Write an ML-to-engine content contract plan describing how V14-era outputs and future generated assets enter the engine.

## Validation Language Rule

- engine-contract proof is primary
- backend proof is separate from host-shell proof
- legacy `MdxViewer` evidence is compatibility evidence only
- do not claim "modern replacement engine" until the engine can load and render real worlds through its own runtime/backend stack without legacy ownership seams
