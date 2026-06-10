# v0.5.0 New Repo Library Migration Prompt

Use this prompt in a fresh planning chat when the goal is to define `v0.5.0` as the break from the `parp-tools` R&D repo into the new production repo at `https://github.com/akspa0/wow-viewer`.

## Prompt

Design a concrete migration plan for `v0.5.0` where the shipping viewer/tool stack moves into `akspa0/wow-viewer` and stops treating `parp-tools` as the long-term home for production code.

This plan must assume:

- `parp-tools` is now the R&D / archaeology / experiment repo
- `wow-viewer` is the intended production repo for the next generation of the viewer and related tools
- the new repo should be centered on one canonical shared library for WoW data/runtime/tooling logic, with the viewer and tools split into separate consumers of that library
- code that currently lives across `MdxViewer`, `WoWMapConverter.Core`, `gillijimproject-csharp`, rollback/reference utilities, and other first-party/internal project fragments must be inventoried and re-authored into that new canonical library/tool stack instead of remaining a permanent pile of cross-repo drift
- commodity dependencies like UI frameworks, OpenGL bindings, image libraries, and other non-domain plumbing should be evaluated separately from domain-specific logic; do not assume every third-party dependency needs to be rewritten
- the current repo still matters as source material and validation reference, but it should not be treated as the future release repo
- practical layout guidance from external maintainers is worth following here: the main renderer application should have an obvious top-level home in the new repo instead of being buried under historical project clutter
- external upstream projects should remain external and live under a clear `libs/` root in the new repo; examples explicitly in scope are `Warcraft.NET`, `DBCD`, `WoWDBDefs`, `Alpha-Core`, `WoWTools.Minimaps`, and `SereniaBLPLib`
- the intent is to rebuild every first-party parsing / reading / writing / conversion subsystem in our own library rather than permanently depending on ad-hoc internal legacy libraries spread across the current repo
- repository bootstrap should automatically pull critical upstream data/dependency repos where appropriate, including `wow-listfile` from `https://github.com/wowdev/wow-listfile`

## What The Plan Must Produce

1. A target repo layout for `wow-viewer`.
2. A canonical library boundary map showing what belongs in:
   - shared format/runtime library
   - viewer app
   - CLI/tools
   - optional research/reference packages
3. A migration inventory for domain logic currently spread across:
   - `src/MdxViewer`
   - `src/WoWMapConverter/WoWMapConverter.Core`
   - `src/gillijimproject-csharp`
   - still-relevant rollback/reference code
   - imported project code that should become first-party code in the new stack
4. A phased extraction/reimplementation plan that preserves the active viewer as a reference during the transition.
5. A dependency-retirement map that distinguishes:
   - logic to absorb into the new library
   - logic to keep as external dependency
   - logic to discard as dead or R&D-only
6. A first shipping vertical slice for the new repo.
7. A validation strategy based on real data and real scenes, not mock confidence.

## Upstream Dependency Baseline

The plan should treat the following as the current high-value upstream inputs to preserve or evaluate explicitly:

- `https://github.com/wowdev/WoWDBDefs`
- `https://github.com/wowdev/wow-listfile`
- `https://github.com/ModernWoWTools/Warcraft.NET`
- `https://github.com/ModernWoWTools/ADTMeta`
- `https://github.com/Marlamin/WoWTools.Minimaps`
- `https://github.com/WoW-Tools/SereniaBLPLib`
- `https://github.com/Marlamin/wow.tools.local`
- `https://github.com/Kruithne/wow.export`

The plan may also evaluate targeted use of:

- `Warcraft.NET` adapters as a continuing compatibility seam rather than a permanent excuse to keep first-party parsing logic fragmented
- `ModernWoWTools` `MapUpconverter` as a possible downstream stage for `3.3.5 -> later-client` outputs once Alpha data can be brought forward to a valid `3.3.5` representation
- `ADTMeta` extension or augmentation if a cleaner metadata pipeline for this use case is worth building on top of that foundation

## Repo Layout Guidance

The plan should treat the following as the default repo-shape bias unless there is a strong reason to do otherwise:

- the main renderer application should live in one obvious top-level app folder such as `src/mapviewer`, `src/viewer`, or a similarly direct name
- the renderer/runtime app can have its own dedicated folder set separate from the shared library, instead of remaining buried inside the same tree as parsing/conversion code
- first-party shared domain libraries should live under a clear library root such as `libs/` or `src/lib/`
- external/vendor-style dependencies should not be mixed into the same folder as first-party domain code and should remain under a dedicated `libs/` root when they come from upstream repos
- upstream-cloned datasets and support repos such as `wow-listfile` should be bootstrapped automatically instead of requiring manual ad-hoc setup every time
- standalone tools and converters should live under a dedicated `tools/` root instead of being scattered alongside the main viewer app
- repo readers should be able to understand where the shipping app lives within a few seconds of opening the repo

The exact folder names are less important than the constraint: the future production repo must have a sane, readable top-level layout for the viewer, libraries, dependencies, and tools.

## Required Constraints

- Do not propose a vague “rewrite everything in one pass” plan.
- Do not assume the current repo structure should be mirrored unchanged into the new repo.
- Do not bury the main viewer application under multiple historical or ambiguous project roots in the new repo.
- Do not carry first-party parser/writer/conversion logic forward as a patchwork of old internal libraries when the stated goal is one canonical owned library.
- Preserve readable FourCC handling internally and only reverse at I/O boundaries.
- Keep Alpha terrain handling and standard terrain handling explicitly separated where the format/risk boundary is real.
- Treat terrain alpha, placement writing, model/WMO version conversion, SQL spawn ingestion, and PM4 structure as separate risk seams with separate acceptance criteria.
- Be explicit about what stays experimental in `parp-tools` versus what must become canonical in `wow-viewer`.
- Be explicit that upstream external libraries remain under `libs/` and should track their original repos where practical, while first-party domain logic is re-owned in our own library.
- Be explicit about which upstream repos are cloned automatically during bootstrap and which are optional/manual research dependencies.
- Distinguish raw PM4 data relationships from viewer-derived grouping/anchor conventions.
- Require real-data validation before claiming that a re-authored subsystem is production-ready.

## High-Priority Questions The Plan Must Answer

1. What should `wow-viewer`'s top-level package/project layout be?
2. Should the main renderer app live under a clear root like `src/mapviewer` or an equivalent direct path, and what naming keeps that obvious?
3. What is the smallest first-party shared library slice that proves the split is working?
4. Which current first-party libraries or fragments should be absorbed first, especially `gillijimproject-csharp` and the parsing/writing logic now split across viewer/converter code?
5. Which upstream libraries should stay under `libs/` and continue tracking their original repos instead of being forked into first-party ownership?
6. Which upstream repos should be cloned automatically during repo bootstrap, including support data like `wow-listfile`?
7. How should the viewer be thinned so it becomes a consumer of the shared library instead of the place where format/runtime truth accretes?
8. How should CLI/tools be structured so they stop duplicating logic and instead ride the same contracts as the viewer?
9. What parts of the current stack should remain R&D-only in `parp-tools` even after `wow-viewer` exists?
10. Where should the performance overhaul sit in the migration order so the new repo does not inherit the same structural bottlenecks?

## Suggested Deliverable Structure

1. Milestone intent
2. New repo target structure
3. Canonical library boundary map
4. Migration phases
5. First shipping vertical slice
6. Dependency-retirement map
7. Real-data validation plan
8. Upstream dependency policy
9. Explicit non-goals

## Stretch Direction

The plan may include stretch work around extending existing external tools, including evaluating whether alpha-era asset support could be contributed upstream to `Noggit` / `noggit-red`, but this must stay explicitly separate from the core `v0.5.0` migration goals unless a small concrete contribution slice is identified.

## Validation Rules

- Be explicit about what is still inferred versus proven.
- If a subsystem only has build validation, say that is not enough.
- Do not treat archived code, imported projects, or synthetic fixtures as proof for the new canonical library.
- For terrain, lighting, placement, SQL scene fidelity, and performance work, require real-data runtime validation before describing the migration slice as complete.
