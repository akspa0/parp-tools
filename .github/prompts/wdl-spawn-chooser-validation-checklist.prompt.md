---
description: "Run a strict runtime validation checklist for WDL spawn chooser fixes so regressions are not marked complete on build-only evidence."
name: "WDL Spawn Chooser Validation Checklist"
argument-hint: "Optional map list, client version, or known failing scenario"
agent: "agent"
---

Use this prompt after a candidate chooser fix to validate behavior end-to-end.

## Read First

1. gillijimproject_refactor/memory-bank/data-paths.md
2. gillijimproject_refactor/memory-bank/activeContext.md
3. gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md

## Validation Matrix

Run at minimum:

1. one Alpha-era map with WDL available
2. one 3.x map with WDL available

For each map, verify:

1. map row shows correct spawn readiness state
2. `Spawn` can be clicked when preview is ready
3. preview dialog opens without silent fallback
4. clicking preview selects a spawn tile marker
5. `Load Map` from dialog applies the selected spawn
6. true preview failure still falls back to default spawn load

## Required Evidence

Collect and report:

1. map/version tested
2. observed warm state transitions
3. whether selection marker appears and updates
4. resulting spawn coordinates/camera position behavior
5. any failure logs/messages

## Deliverables

Return all items:

1. pass/fail per matrix item
2. blockers and probable file/function owner
3. whether issue is fully fixed, partially fixed, or still broken
4. explicit statement of tests run (build/tests/runtime)

## Non-Negotiable Rule

Do not mark the chooser fixed from compile success or static inspection alone.
