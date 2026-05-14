# wow-viewer Editor And Interop Plan

## Status

- status: active
- date: 2026-05-14
- owner: `wow-viewer`
- parent: `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md`
- intent: define the editor/tool identity of the engine, with import/export-first UX, multi-game compatibility layers, and metadata-driven feature gating

## Thesis

The engine should not start life as "a renderer with some tools around it."

It should start life as:

1. an import/export tool for assets and world data,
2. a renderer that can visualize every supported asset family through explicit render layers,
3. a `game-viewer`-hosted editor that can switch between multiple data roots and collaborate across them,
4. a metadata-driven modding tool that enables and disables features based on the active game/data profile.
5. an artifact-preserving tool that can distinguish raw historical artifacts from normalized or forward-native content.

The initial UI should reflect that.

## Product Framing

Near-term host identity:
- one tool for MPQ-era WoW modding and asset/world interoperability
- shipped through the current `game-viewer` app host while `wow-viewer` remains the library/runtime foundation
- developed now inside `wow-viewer`, but aimed at a future extracted engine repo where `game-viewer` is a host and `wow-viewer` is one profile/personality library

Early compatibility targets:
- WoW `0.5.x`
- WoW MPQ-era clients through the existing old-school conversion lane
- Warcraft 3 asset/archive compatibility where the formats and runtime semantics overlap meaningfully

Early forward-native target:
- custom content roots built from GLB + textures + per-object metadata sidecars
- `Museums` roots as the named family for the user's own evolving forward-native data model

Longer-term identity:
- generalized engine/editor that can host both Blizzard-era content and future custom/generated content

## Non-Negotiable Product Rules

1. The UI starts with import/export and data-source management, not only with "open a viewer window."
2. Asset rendering must be layered by asset family and backend/render-pass needs.
3. Compatibility must be profile-driven, not hardcoded as one permanent WoW version.
4. Feature availability must be metadata-driven wherever possible.
5. DBC/DB2 read and edit support is part of the tool vision, not an optional sidecar.
6. Cross-root workflows must be designed intentionally; copying data between clients/maps is a first-class use case.
7. The shell must not assume that every profile is archive-backed or WoW-shaped.
8. The shell must treat profile/personality libraries as first-class citizens rather than one-off hardcoded exceptions.
9. Audio inspection and playback diagnostics belong in the same managed-shell story as rendering diagnostics.
10. Browser/embed output surfaces should be supported through explicit delivery contracts, not by making the editor shell browser-shaped by default.

## Core User Stories

### User story 1 — import/export-first

The user launches the tool and the first meaningful workflows are:

- register game/data roots
- inspect content inventories
- import or export assets
- convert assets/world data between supported formats
- open editor/render workspaces from those managed roots

### User story 2 — full asset renderer

The tool can render supported asset types through explicit layers:

- terrain
- liquids
- sky
- WMO/world objects
- M2/MDX models
- overlays and diagnostics
- future generated/custom assets
- future browser/embed previews through a WebGL-facing delivery component when the active workspace supports it

### User story 3 — multi-root game manager

The user can register multiple game roots and switch quickly between them:

- multiple WoW clients
- Warcraft 3 installs/data roots
- future custom content roots
- forward-native GLB + metadata roots

The editor understands which data profile is active and adjusts tools accordingly.

### User story 4 — cross-root collaboration

The user can move/copy content between roots or maps:

- asset-to-asset conversion
- map-to-map copy/paste
- source-to-target placement transfer
- world data migration workflows

### User story 5 — metadata-driven editor behavior

The editor should not pretend every root supports every feature.

Instead:
- detect game/build/profile metadata
- load the right schema/definition surfaces
- enable/disable editor features accordingly

## Compatibility Model

### Compatibility layer concept

The engine/editor should support compatibility by profile:

- `Custom.Forward.GlbMetadata`
- `Museums.ForwardNative.v0`
- `WoW.Alpha.0.5.x`
- `WoW.PreRelease.0.6-0.7`
- `WoW.LK.3.x`
- `WoW.Cata.4.0.0`
- `Warcraft3.Classic`
- future custom/generated profile families

Each profile controls:

- archive rules
- file families
- supported import/export paths
- renderer features
- audio resolution features
- editor tools
- metadata schema bindings

### Why Warcraft 3 belongs early

The user direction is correct:

- Warcraft 3 shares meaningful MPQ/archive and MDX-era commonality with early WoW data
- this makes it a plausible early second compatibility target
- it is useful as a forcing function so the tool does not calcify around one exact WoW-only assumption set

Hard rule:
- add Warcraft 3 support through explicit compatibility-profile seams, not by muddying WoW-specific logic

## Game Manager

## Purpose

The engine needs a `Game Manager` or `Data Root Manager` as a first-class subsystem.

It should own:

- registered game/data roots
- display names and labels
- detected game family
- detected version/build metadata
- active profile binding
- schema/definition routing
- optional loose overlay roots
- import/export targets

## Minimum capabilities

1. Register one or more roots.
2. Detect or confirm profile family/version.
3. Store per-root metadata and capabilities.
4. Switch active roots quickly in the UI.
5. Open multiple roots in one session.
6. Drive feature gating and conversion choices from the active root metadata.
7. Support multiple Museums roots the same way it supports multiple game roots.
8. Surface which profile/personality library is bound to each root.

## Stretch capabilities

- cross-root clipboard/package operations
- side-by-side world/session comparison
- donor-source selection for conversions
- future collaboration packages or content bundles

## Metadata-Driven Feature Gating

### Principle

The tool should mimic the Blizzard-editor pattern the user described:

- not one rigid version lock
- one editor shell
- feature availability controlled by data/profile metadata

### Data sources for gating

- game binary version metadata
- archive layout signals
- DBCD + WoWDBDefs definitions
- known file-family support per profile
- discovered DBC/DB2 presence and schema coverage

### Examples

Feature gating should decide things like:

- which file families are legal to edit
- which converters are offered
- which DBC/DB2 tables are editable
- which render layers are active
- which terrain/liquid/material controls exist
- whether world/session tools expose Alpha-specific, LK-specific, or Warcraft-3-specific behavior
- whether a profile is raw-artifact-first, forward-native, or mixed

## DBC/DB2 Editing Direction

DBC/DB2 support should expand from read/lookup toward editor ownership.

The tool vision includes:

- read all supported DBC/DB2 tables
- inspect schema and rows
- edit supported tables safely
- export/import table changes
- surface profile/version schema differences honestly

Hard rule:
- schema-driven editing only
- no blind binary poking UX
- every edit flow must know the active game/data profile

## Import/Export Surface

The tool should eventually expose import/export across these categories:

- terrain/world data
- models
- textures
- world objects
- metadata tables
- generated/custom content packages
- raw artifact captures and normalized derivative packages

Initial UI emphasis should be:

1. data-root selection
2. asset/world inventory
3. import/export/conversion actions
4. preview/render workspace entrypoints

## Renderer Layer Model

The renderer should be organized by explicit asset/render layers, not one giant scene blob.

Core layers:

- terrain layer
- liquid layer
- sky layer
- WMO/world-object layer
- M2/MDX model layer
- overlay/selection/diagnostic layer
- audio diagnostics layer in the shell, fed by engine-neutral audio runtime state

Compatibility layers:

- WoW Alpha/LK/Cata differences
- Warcraft 3 asset/runtime compatibility shims
- forward-native GLB + metadata adapters
- future custom/generated content adapters

This has two goals:

1. make rendering ownership clean
2. make feature gating possible without giant conditionals everywhere

## Editor Shell Direction

The initial editor shell should prioritize these workspaces:

### Workspace 1 — Game Manager

- register/switch roots
- profile detection
- capability summary
- feature gating preview

### Workspace 2 — Asset Browser

- browse supported assets by family
- preview metadata
- launch import/export flows

### Workspace 3 — Conversion And Interop

- convert assets/worlds between supported profiles
- choose source and target roots
- run copy/paste/migration workflows

### Workspace 4 — Render Preview

- preview selected assets and worlds using the runtime/backend layers
- leave room for a WebGL-facing preview/export or embedded web panel without making it the primary renderer path

### Workspace 5 — Database Editor

- inspect and edit DBC/DB2 tables for the active root/profile

### Workspace 6 — World Editor

- longer-range world/map editing surface
- only after the above foundation exists

## Relationship To The User's Future Game

The user's separate game effort is still exploratory:

- NPC AI is still early
- no formal dataset contract yet
- no final map/canvas/world structure yet

That means this repo should not overfit to speculative game semantics now.

Instead it should provide:

- strong import/export
- strong compatibility layers
- clean content/runtime contracts
- a path for future generated content packages to enter safely

That is the correct bridge between today's WoW modding tool and tomorrow's prompt-driven/custom game workflows.

## Ordered Phases

Execution detail for those phases now lives in:

- `wow-viewer/docs/architecture/game-viewer-plan-pack-2026-05-14/README.md`

### Phase I0 — Editor/Interop Plan Reset

Intent:
- establish the editor identity as import/export-first plus game-manager-led

Proof:
- this plan plus linked updates in the engine and viewer sub-plans

### Phase I1 — Game Manager Contracts

Intent:
- define root registration, profile detection, and feature-gating contracts

Must include:
- registered root records
- profile identity
- capability summary
- active-root switching contract

Proof:
- CLI or app diagnostics can enumerate registered roots and computed capabilities

### Phase I2 — Asset Inventory And Import/Export Shell

Intent:
- make import/export the first real UI workflow

Must include:
- managed root selection
- asset family inventory
- import/export action surface

Proof:
- operator can select a root and run at least one supported import/export flow from the shell

### Phase I3 — Compatibility Profiles

Intent:
- formalize compatibility layers for WoW families and Warcraft 3

Must include:
- profile registry
- profile capability flags
- Warcraft 3 early support boundary

Proof:
- tool can distinguish at least one WoW root and one Warcraft 3 root with different feature surfaces

### Phase I4 — Render Layer Closure

Intent:
- expose explicit renderer layers for all core asset families

Must include:
- terrain/liquid/sky/object/model/overlay layer contracts
- layer-specific diagnostics

Proof:
- app/runtime diagnostics report active layer set for a chosen root/profile

### Phase I5 — DBC/DB2 Editor Baseline

Intent:
- move from lookup-only to safe schema-driven table editing

Proof:
- one bounded DBC/DB2 edit workflow on a supported profile with import/export proof

### Phase I6 — Cross-Root Interop

Intent:
- support copying or migrating data between roots/maps

Proof:
- one bounded source-to-target interop workflow proven through the app/tool shell

## Immediate Follow-Ups

1. update the engine plan so import/export-first UI and game-manager ownership are top-level goals
2. update the `game-viewer` host sub-plan so its first shell slices reflect game-manager and asset-interoperability entrypoints
3. later write a dedicated compatibility-profile technical plan for WoW + Warcraft 3 schema and feature routing
4. later write a dedicated forward-native content-profile plan for GLB + textures + metadata-sidecar roots
5. treat `Museums` as the named supported forward-native family while its exact storage spec evolves
