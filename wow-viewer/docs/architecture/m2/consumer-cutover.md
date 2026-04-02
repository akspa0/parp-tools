# Consumer Cutover

## Purpose

This document explains how the current M2 knowledge should turn into implementation work across:

- `wow-viewer` as the canonical library/runtime owner
- `MdxViewer` as a compatibility consumer and proof harness

## Canonical Library Targets

### `WowViewer.Core`

Use for:

- typed M2 model metadata contracts
- version or build-aware profile descriptors
- active section and effect metadata contracts

### `WowViewer.Core.IO`

Use for:

- root-model parsing
- strict span and layout validation
- skin file reading
- external animation file reading

### `WowViewer.Core.Runtime`

Use for:

- choose/load/init skin ownership
- active section materialization
- effect recipe selection
- runtime flags
- future lighting, animation state, and scene-submission seams

## Immediate Implementation Priorities

### Priority 1

Strengthen the library-owned M2 foundation around:

- build-aware parser dispatch
- numbered-skin ownership for later clients
- explicit early-beta and pre-release exceptions

### Priority 2

Land typed active-section and effect metadata that preserve:

- section counts and remap outputs
- unresolved `0x20` and `0x40` semantics
- effect-family selections

### Priority 3

Land external animation ownership and model-local runtime state:

- `%04d-%02d.anim`
- alias and ready-state metadata
- later lighting or animated material state

### Priority 4

Only after the above, expand into:

- scene submission
- family-aware batching
- consumer cutover and parity harnesses

## MdxViewer Use Rules

`MdxViewer` should be used for:

- real-data smoke checks
- parity probes on fixed assets
- temporary compatibility wiring when a `wow-viewer` seam already exists

`MdxViewer` should not be used for:

- inventing the long-term M2 runtime contract
- adding new isolated parser or effect-routing logic that `wow-viewer` does not own

## Legacy Reference Map

Use these older docs as targeted source material, not as the primary implementation entrypoints.

### Native evidence and session logs

- `../m2-native-client-research-2026-03-31.md`
- `../ab-session-a-2026-04-01/*`

### Historical migration and workflow docs

- `../../../gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md`
- `.github/prompts/wow-viewer-m2-runtime-plan-set.prompt.md`
- `.github/prompts/m2-cross-build-native-investigation.prompt.md`

### Early-beta and pre-release implementation references

- `../../../gillijimproject_refactor/documentation/wow-200-beta-m2-light-particle-terrain-guide.md`
- `../../../gillijimproject_refactor/documentation/pre-release-3.0.1-m2-wow-exe-guide.md`
- `../../../gillijimproject_refactor/specifications/3.0.1.8303/Contracts/M2_MDX_Implementation_Contract_3.0.1.8303.md`

### Raw structure references

- `../../../gillijimproject_refactor/reference_data/wowdev.wiki/M2.md`
- `../../../gillijimproject_refactor/reference_data/wowdev.wiki/M2.skin.md`
- `../../../wow-viewer/libs/ModernWoWTools/Warcraft.NET/Docs/M2.md`

## Non-Claims

- this cutover document does not claim full runtime parity in either `wow-viewer` or `MdxViewer`
- it does not claim all native M2 semantics are closed
- it does not replace the raw evidence logs; it only consolidates how to use them
