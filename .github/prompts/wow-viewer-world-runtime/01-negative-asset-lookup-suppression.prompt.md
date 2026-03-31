---
description: "Implement or plan the first world-runtime stabilization slice for wow-viewer and MdxViewer. Use when repeated .skin lookups, failed MDX retry spam, negative asset misses, or noisy asset logs are causing hidden performance loss before the deeper WorldScene split."
name: "wow-viewer World Runtime 01 Negative Asset Lookup Suppression"
argument-hint: "Optional asset family, hotspot file, or missing-lookup symptom to prioritize"
agent: "agent"
---

Implement or plan the first stabilization slice for the `WorldScene` split: stop repeated missing asset work from distorting runtime behavior and logs.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md`
4. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldAssetManager.cs`
5. `gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs`
6. `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
7. `gillijimproject_refactor/src/MdxViewer/ViewerApp_CaptureAutomation.cs`
8. `wow-viewer/README.md`
9. `.github/copilot-instructions.md`

## Goal

Remove obvious repeated asset-miss work, especially `.skin` companion lookups and failed MDX retries, before larger runtime-service extraction starts.

## Current Concrete Problem

- `WorldAssetManager` currently retries cached failed MDX loads and deferred failed MDX loads.
- M2-family loads still walk candidate `.skin` paths even when the practical answer is already known to be "no usable companion skin exists for this model path".
- similar companion-skin loops exist in standalone load paths and WMO doodad M2 loading.
- this creates log spam, repeated file probes, and likely frame-time noise that makes broader renderer work harder to measure accurately.

## Non-Negotiable Constraints

- Do not turn this slice into the full `WorldScene` service split.
- Do not over-abstract asset loading just to match a future runtime design.
- If a reusable negative-cache contract belongs in `WowViewer.Core.Runtime`, move only that narrow seam.
- If the fastest correct fix needs to stay in `MdxViewer` first, do that and describe it honestly as compatibility prep.
- Do not claim renderer-performance closure from reducing one miss path.

## What The Work Must Produce

1. the exact repeated asset-miss paths to stop
2. the exact files that should own negative-cache or failure-residency behavior
3. the exact logging changes needed so the app reports the problem once without flooding
4. a narrow proof plan using build, tests where practical, and optional fixed-shot capture smoke
5. explicit follow-on hooks for slice 02 instead of another broad cleanup bucket

## Deliverables

Return all items:

1. the exact negative-lookup or retry seam to change
2. why it is the first world-runtime slice
3. exact files to change
4. exact validation to run
5. what should stay out of scope
6. what counters or telemetry should be surfaced afterward

## First Output

Start with:

1. the concrete retry or miss loop you think is most expensive right now
2. the narrowest fix that stops repeated work without hiding real errors
3. the proof that would show the log spam and retry path are actually reduced
4. what you are explicitly not solving yet