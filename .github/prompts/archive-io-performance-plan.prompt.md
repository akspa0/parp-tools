---
description: "Tighten archive I/O, file-set lookup, and background prefetch in MdxViewer using the 4.0.0 engine's archive/cache evidence as the guide."
name: "Archive IO Performance Plan"
argument-hint: "Optional hotspot, asset family, map path, or lookup symptom to prioritize"
agent: "agent"
---

Implement the archive I/O performance slice in `gillijimproject_refactor/src/MdxViewer` with repeated asset-read cost and scene-streaming latency as the runtime goal.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/memory-bank/data-paths.md`
4. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
5. `gillijimproject_refactor/documentation/wow-400-engine-performance-recovery-guide.md`
6. `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDataSource.cs`
7. `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Services/NativeMpqService.cs`
8. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldAssetManager.cs`

## Goal

Reduce repeated archive/path-resolution work and make background asset warming measurably more useful without breaking correctness on the active MPQ-era viewer path.

## Why This Slice Exists

- Ghidra evidence from `4.0.0.11927` shows explicit archive objects, MPQ stack cache code, streamed MPQs, file mapping imports, and async I/O primitives.
- The active viewer already has a good foundation in `MpqDataSource` and `WorldAssetManager`, but it still spends real work on repeated probing, fallback search, and coarse prefetch policy.
- This is a root-cause performance slice. It should improve asset-read hot paths directly instead of only hiding them behind bigger caches.

## Scope

- In scope:
  - `MpqDataSource` lookup and read hot paths
  - raw-byte cache hit behavior and instrumentation
  - prefetch queue policy and worker handoff
  - `WorldAssetManager` calls that repeatedly trigger archive reads or fallback probing
- Out of scope unless direct evidence forces it:
  - renderer-side shader changes
  - terrain decode rewrites
  - speculative replacement of the MPQ reader architecture

## Non-Negotiable Constraints

- Do not share one mutable primary MPQ reader across worker threads.
- Do not replace explicit path or case-correct lookup with broad fuzzy heuristics.
- Do not treat larger caches as a substitute for cheaper hot-path logic.
- Keep fixes aligned with the current viewer/data-source architecture.
- Update memory-bank files if runtime behavior, instrumentation, or known limits materially change.

## Required Implementation Order

1. Identify the hottest `ReadFile` and path-probe call sites on the active viewer path.
2. Confirm which repeated checks are actually redundant versus correctness-preserving.
3. Add or tighten instrumentation first if the current code does not expose enough signal.
4. Land the smallest lookup/read-path reduction that removes confirmed waste.
5. Only then adjust prefetch policy so it warms data the scene is likely to need next.
6. Update memory-bank notes with the exact behavior changed and what is still unverified.

## Investigation Checklist

- Verify how often `MpqDataSource.ReadFile(...)` falls through direct lookup into canonical or fallback probes.
- Verify whether high-frequency callers already normalize paths before calling into the data source.
- Check whether the raw-byte cache is being bypassed by path spelling differences.
- Check whether prefetch currently warms the same asset families the scene actually requests next.
- Prefer exact counters and timings over generic “feels faster” claims.

## Validation Rules

- Build the changed viewer solution.
- If runtime validation on fixed real data is not run, say so explicitly.
- If automated tests are not added or run, say so explicitly.
- Do not claim generalized streaming improvement unless you have runtime evidence or explicit instrumentation to support it.

## Deliverables

Return all items:

1. exact archive-I/O behavior changed
2. files changed and why
3. what redundant work was removed versus what still remains
4. build status
5. automated-test status
6. runtime-validation status
7. memory-bank updates made

## First Output

Start with:

1. the current hottest archive/path-resolution seam in the active viewer
2. what evidence from the 4.0.0 engine supports this as a real performance target
3. what files will be changed first
4. what you are explicitly not changing in this pass