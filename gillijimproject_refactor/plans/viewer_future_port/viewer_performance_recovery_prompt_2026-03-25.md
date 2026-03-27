# Viewer Performance Recovery Prompt

Use this prompt in a fresh planning chat when the goal is to make the active viewer materially faster before layering on more renderer fidelity and scene liveness.

## Prompt

Design a concrete implementation plan to improve real-world performance in `gillijimproject_refactor/src/MdxViewer`, with emphasis on large world scenes that currently feel far too slow for a 2003-era game dataset on modern hardware.

The plan must assume:

- current world-scene performance can still fall below `30 FPS` on real content
- reverse-engineering notes already show that historical clients used more aggressive render-list precompute, batching, specialization, and distance-based layer collapse than the active viewer currently does
- the viewer is already doing more than terrain: world WMOs, MDX/M2 placements, minimap overlays, PM4 overlays, taxi actors, and optional SQL-driven spawns all compete for frame time
- future work wants even more scene complexity, including richer SQL actor fidelity and enhanced terrain shaders, so performance recovery is a prerequisite rather than cleanup

## What The Plan Must Produce

1. A prioritized performance recovery stack.
2. A distinction between measurement/profiling work and optimization work.
3. A first practical `v0.4.6` slice that can land without a renderer rewrite.
4. A follow-up stack for `v0.5.0` where deeper renderer architecture work becomes justified.
5. A validation plan based on frame-time measurement and real datasets.
6. A risk register covering regressions, visibility bugs, and fidelity tradeoffs.

## Required Constraints

- do not treat more aggressive hiding/culling as a free performance win if it recreates the object pop-in problems the user already complained about
- keep measurement visible: if the viewer lacks the counters needed to prove an optimization, say what must be instrumented first
- separate terrain cost, object cost, transparent-sort cost, PM4 cost, and SQL actor cost instead of calling the whole problem “OpenGL slow”
- leverage the known `0.5.3` findings around render-list precompute, terrain layer-count collapse, and stronger specialization, but do not claim parity from those notes alone
- identify the smallest slice that reduces frame cost before the enhanced terrain path lands

## High-Priority Topics To Include

- world-scene submission/batching churn
- terrain precompute / render-list strategy
- transparent item recollection/sorting pressure
- scene residency and prefetch policy
- PM4 overlay cost controls that remain truthful for debugging
- SQL actor budgets for future liveness work
- how enhanced terrain shaders can avoid becoming a performance regression multiplier

## Suggested Deliverable Structure

1. Current bottleneck inventory
2. Measurement plan
3. `v0.4.6` recovery slice
4. `v0.5.0` deeper recovery slice
5. Validation and benchmarking plan
6. Risks and non-goals

## Validation Rules

- require real scene measurements on the fixed development data paths before calling the recovery successful
- do not describe a plan as validated from builds alone
- if the plan depends on profiler instrumentation not yet present, say that explicitly