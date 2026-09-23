# Implementation Plan: Lighting Quick Controls and Confident Hover Inspection

**Branch**: `107-lighting-quick-inspection` | **Date**: 2026-07-15 | **Spec**: [spec.md](spec.md)

## Summary

Expose the existing LIT/DBC lighting state in Tools > Quick, use FogEnd to bound detailed terrain while keeping the real WDL horizon visible for an additional 2,500 world units, and render an exact-path hover card only for a single ray-confirmed scene match.

## Technical Context

**Language/Version**: C#/.NET 10. **Testing**: focused xUnit plus viewer build. **Scope**: `ViewerApp`, `WorldScene`, existing LIT panel. **Constraint**: no lighting reader rewrite and no capture/training run.

## Constitution Check

Pass: viewer-only UI and visibility consumption changes, no parser duplication, no client-path assumption, and no user-owned heavy work.

## Phase 0 — Findings

1. Preserve LIT/DBC as the FogEnd authority; the defect is `GetSceneFarPlane`'s 6000-unit minimum, not fog parsing. The existing `TerrainManager` AOI keeps detailed ADTs fog-driven and restores corresponding 17x17/16x16 WDL tiles as ADTs unload, so the 2,500-unit extension is WDL horizon space rather than expanded full-detail terrain.
2. Preserve the detailed Lighting utility as evidence owner; Quick becomes its discoverable scene-control surface.
3. Treat ray hits as precise and brush/overlap hits as ambiguous for hover-card purposes.

## Phase 1 — Implement and validate

1. Keep the pure far-plane helper's 1-unit minimum, change its padding to 2,500 world units, and use it for terrain and VLM paths while leaving the fog-driven detailed-ADT AOI and WDL replacement mechanism intact.
2. Add concise Quick lighting controls: time, fog start/end, LIT override state, active range, and a one-action link to Utilities > Lighting.
3. Mark `HoveredAssetInfo` with whether it came from a precise ray hit; suppress the popup for brush-only candidates while preserving click selection.
4. Add focused tests for far-plane behavior and hover precision; build the viewer.
5. Update memory-bank continuity and commit code/spec evidence together.

## Out of Scope

Native direction-model implementation (Spec 106), sky objects, LIT/DBC decoding, and capture/training execution.
