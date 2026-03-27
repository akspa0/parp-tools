# Prompt Registry

Codex-oriented prompt docs live here. They are converted from `.github/prompts/*` plus the active memory-bank workflow guidance.

## Core wow-viewer prompts

- `wow-viewer-tool-suite-plan-set.md`
- `wow-viewer-pm4-library-implementation.md`
- `wow-viewer-shared-io-implementation.md`
- `wow-viewer-shared-io-library-plan.md`
- `wow-viewer-tool-migration-sequence-plan.md`
- `wow-viewer-tool-inventory-cutover-plan.md`
- `wow-viewer-cli-gui-surface-plan.md`
- `wow-viewer-bootstrap-layout-plan.md`

## Viewer performance and parity prompts

- `scene-culling-batching-performance-plan.md`
- `archive-io-performance-plan.md`
- `cache-residency-performance-plan.md`
- `bls-shader-parity-performance-plan.md`
- `lighting-dbc-expansion-implementation-plan.md`
- `sky-environment-parity-implementation-plan.md`
- `m2-material-parity-implementation-plan.md`
- `wow-400-terrain-blend-recovery.md`

## PM4 and repair prompts

- `pm4-ck24-world-mapping-implementation.md`
- `development-repair-implementation-plan.md`

## WDL chooser prompts

- `wdl-spawn-chooser-validation-checklist.md`
- `wdl-spawn-chooser-runtime-triage.md`
- `wdl-spawn-chooser-implementation-plan.md`

## Pre-release M2 prompts

- `pre-release-m2-rendering-recovery.md`
- `pre-release-3-0-1-m2-runtime-triage.md`
- `pre-release-3-0-1-m2-implementation-plan.md`
- `pre-release-3-0-1-m2-ghidra-followup.md`

## Terrain recovery prompts

- `brokenasfuck-3x-support.md`
- `alpha-regression-audit.md`
- `add-terrain-regression-tests.md`

## Notes

- `AGENTS.md` is the always-on Codex instruction surface.
- These prompt docs are reusable planning or execution playbooks; they are documentation assets unless the host explicitly wires them into prompt tooling.
- When workflow boundaries materially change, keep these files aligned with `AGENTS.md`, `.codex/README.md`, `wow-viewer/README.md`, and the relevant memory-bank or plan files.
