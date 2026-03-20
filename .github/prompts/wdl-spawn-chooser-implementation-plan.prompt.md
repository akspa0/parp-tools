---
description: "Implement a focused fix for WDL spawn chooser regressions in MdxViewer without broad terrain/model refactors."
name: "WDL Spawn Chooser Implementation Plan"
argument-hint: "Optional failing map or known failing state transition"
agent: "agent"
---

Implement the next code slice to restore WDL spawn chooser behavior.

## Read First

1. gillijimproject_refactor/memory-bank/activeContext.md
2. gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md
3. .github/prompts/wdl-spawn-chooser-runtime-triage.prompt.md

## Scope Guardrails

- Keep this as a surgical viewer-flow fix.
- Do not mix in unrelated PM4, terrain alpha, or model rendering changes.
- Do not claim closure without real runtime validation.

## Required Files

1. gillijimproject_refactor/src/MdxViewer/ViewerApp.cs
2. gillijimproject_refactor/src/MdxViewer/ViewerApp_WdlPreview.cs
3. gillijimproject_refactor/src/MdxViewer/Terrain/WdlPreviewCacheService.cs
4. gillijimproject_refactor/src/MdxViewer/Terrain/WdlPreviewRenderer.cs

## Required Implementation Order

1. Reproduce and pin the exact failing branch in chooser flow.
2. Fix warm-state/enablement or open/fallback logic first.
3. Fix chooser selection-to-spawn commit path second.
4. Keep fallback-to-default-load behavior intact for true preview failures.
5. Build MdxViewer and report build status.
6. Run real-data runtime validation on both Alpha-era and 3.x map sets.

## Deliverables

Return all items:

1. exact code changes made and why
2. any behavior intentionally unchanged
3. build status
4. automated test status
5. runtime validation status split by map/version
6. residual risk

## Required Language

- If no tests were added, say so explicitly.
- If only build was run, say so explicitly.
- If runtime validation is pending, say so explicitly.
