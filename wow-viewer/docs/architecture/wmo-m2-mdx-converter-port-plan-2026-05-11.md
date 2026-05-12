# WMO / M2 / MDX Converter Port Plan

Date: 2026-05-11

## Status Update (May 12, 2026)

This plan is now partially landed.

Landed in `wow-viewer`:

- `WmoV17ToV14Converter` and `WmoV14ToV17Converter` are owned in shared I/O and wired through converter CLI entry points.
- `M2ToMdxConverter` and `MdxToM2Converter` are also owned in shared I/O and wired through converter CLI entry points.
- `convert-lk-to-alpha --bundle-wmos` now emits Alpha-compatible monolithic `WMO v14` roots using `MOMO` plus embedded `MOGP` groups, rewrites bundled WMO texture tables inside `MOMO` roots, and rewrites Alpha placement paths to the local bundle outputs.
- `convert-lk-to-alpha --bundle-m2s` now emits local bundled `MDX` outputs and rewrites bundled texture references beside those outputs.
- the WMO downgrade path now has two extra compatibility guards beyond the earlier structural downgrade rules:
   - spatial merge when a source root would exceed the practical Alpha-era `384` group ceiling
   - batch-boundary group splitting when any `v17` `MOBA.firstIndex` exceeds the legacy `ushort` ceiling used by `v14`
- real staged Kalimdor proof for `4.0.0.11927 -> 0.5.3.3368` now reports `311` converted WMOs, `1` missing root, and `0` conversion failures under the temp `wmo-full-debug-after-split` validation output.

Still open:

- `M2 -> MDX` remains a structural downgrade lane, not Alpha-client runtime parity proof.
- sequence and animation compatibility for Alpha `MDX` remains open, especially around `SEQS` fidelity and broader native animation expectations.
- `MDX -> M2` remains structural rather than renderer- or runtime-parity complete.
- do not treat successful converter builds or inspect re-reads as active Alpha-client signoff for all bundled MDX assets.

## Purpose

Plan the next converter slice needed to make Cataclysm-era object content viable in the Alpha 0.5.3 target lane without routing new ownership back into `gillijimproject_refactor`.

The immediate product goal is narrow:

- make Cataclysm `WMO v17` content downgrade to Alpha-compatible `WMO v14`
- make Cataclysm `M2` content downgrade to Alpha-compatible `MDX`
- do that inside `wow-viewer` shared I/O and converter surfaces, with `MdxViewer` used only as an optional compatibility proof host

The reverse directions (`WmoV14ToV17`, `MdxToM2`) matter for completeness, but they are not the first critical path for `4.0.0 -> 0.5.3` map conversion.

## Current State

What already exists:

- `wow-viewer` already owns modern read-side surfaces for `WMO`, `M2`, and `MDX`
- `wow-viewer` already has active `M2` runtime ownership work and `MDX` summary/runtime surfaces
- `wow-viewer` now owns active converter code for:
   - `WmoV14ToV17Converter`
   - `WmoV17ToV14Converter`
   - `M2ToMdxConverter`
   - `MdxToM2Converter`
   - `convert-wmo-v17-to-v14`
   - `convert-wmo-v14-to-v17`
   - `convert-m2-to-mdx`
   - `convert-mdx-to-m2`

What is still not complete in active `wow-viewer`:

- no broad runtime-parity proof for the landed converter directions
- no Alpha-runtime-safe `MDX` animation/signature guarantee for the full Cataclysm doodad corpus
- no complete policy for every unsupported `M2` feature family such as particles, ribbons, attachments, or effect-heavy materials
- no claim yet that all generated `MDX` or `M2` outputs are suitable for active viewer or native-client runtime use

What the repo evidence says:

- `WMO` is the easier port. The workspace already has real upgrade and downgrade implementations plus tests and writer helpers in legacy/reference code.
- `WMO` is now the more mature converter lane. The hard object-side blockers found so far were Alpha-specific constraints, not missing basic format ownership:
   - Alpha `WMO v14` expects monolithic `MOMO`-wrapped roots with embedded `MOGP` groups rather than later split root plus `_NNN.wmo` companions.
   - the downgrade path must handle both the practical Alpha-era `384` group ceiling and legacy `MOBA` `ushort` batch-index limits.
- `M2 <-> MDX` is still harder. `wow-viewer` now has minimal converter ownership, but broad semantic mapping for animation tracks, sequence fidelity, materials, particles, ribbons, and effect routing is still incomplete.

## Ownership Rule

All new implementation belongs in `wow-viewer`.

Target ownership:

- shared model contracts: `wow-viewer/src/core/WowViewer.Core/Wmo`, `.../M2`, `.../Mdx`
- format read and write logic: `wow-viewer/src/core/WowViewer.Core.IO/Wmo`, `.../M2`, `.../Mdx`
- converter orchestration and shared downgrade helpers: same `Core.IO` format families unless a later shared conversion namespace becomes clearly necessary
- CLI entry points: `wow-viewer/tools/converter/WowViewer.Tool.Converter`
- tests: `wow-viewer/tests/WowViewer.Core.Tests`

Reference code in `gillijimproject_refactor`, `parpToolbox`, `PM4Tool`, and `archived_projects` is extraction input only.

## Scope Decision

Do not start with full bidirectional parity.

The implementation order should be:

1. `WmoV17ToV14`
2. `WmoV14ToV17`
3. `M2ToMdx`
4. `MdxToM2`

Reason:

- `WmoV17ToV14` is directly on the Cataclysm-to-Alpha critical path and has the strongest existing reference implementation.
- `WmoV14ToV17` should follow while the shared WMO write path is still fresh.
- `M2ToMdx` is the model-side critical downgrade path for Cataclysm doodads.
- `MdxToM2` is useful for symmetry and future workflows, but it is not required to unblock `4.0.0 -> 0.5.3` conversion.

## Non-Goals

This plan does not attempt to finish:

- full active-viewer parity for all object rendering
- full native-material parity for every M2 effect family in the first slice
- a new long-range shell surface in `WowViewer.App`
- broad archaeology cleanup in `gillijimproject_refactor`
- speculative renaming or redesign of existing format-family contracts without proof

## Workstreams

## Workstream 0 - Shared Proof Inputs

Before implementation, collect and pin the reference inputs used by later slices.

Deliverables:

- inventory the exact legacy/reference converter files used as extraction input
- identify which fixtures can become `wow-viewer` tests and which remain real-data manual proofs
- record proof boundaries for each family: synthetic test, file-structure inspect proof, runtime compatibility proof

Minimum reference set:

- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV17ToV14Converter.cs`
- `parpToolbox/src/WoWToolbox/WoWToolbox.WmoV14Converter/WmoV14ToV17Converter.cs`
- `parpToolbox/src/WoWToolbox/WoWToolbox.Core/WMO/WmoV17Writer.cs`
- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/M2ToMdxConverter.cs`
- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/MdxToM2Converter.cs`
- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/*`
- `wow-viewer/src/core/WowViewer.Core.IO/M2/*`
- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/*`

Exit condition:

- the fresh chat starts from a pinned list of source files and expected proof commands instead of rediscovering them

## Workstream 1 - WmoV17ToV14

This is the first implementation slice.

Objective:

- convert modern split `WMO v17` root plus group files into Alpha-compatible monolithic `WMO v14`

Implementation target:

- add `wow-viewer` shared writer support for `WMO v14`
- port downgrade mapping into `wow-viewer`
- expose a converter command in `WowViewer.Tool.Converter`

Expected subtasks:

1. Lift a shared intermediate model from the legacy converter instead of writing chunks directly from ad hoc byte reads.
2. Reuse or extend `wow-viewer` read-side `WmoSummaryReader` and deeper readers where they already cover needed fields.
3. Implement a `WmoV14Writer` in `wow-viewer`.
4. Port root and group downgrade mapping rules.
5. Preserve group ordering, bounds, portal data, doodad sets, and material references.
6. Treat Alpha `v14` roots as monolithic `MOMO` containers with embedded `MOGP` groups, not as later split root-plus-companion-group files.
7. Preserve both known Alpha-era hard limits in the downgrade path:
   - practical `384` group ceiling on the emitted legacy root
   - legacy `MOBA` batch `firstIndex` `ushort` ceiling, handled by batch-boundary group splitting before legacy write
6. Add focused tests for:
   - root version change `17 -> 14`
   - group count and required chunk presence
   - legacy `MOMO` packaging for the Alpha lane if required by the target format path
   - oversized `MOBA` batch ranges splitting into multiple legacy embedded groups while staying inspect-readable
7. Add inspect-level proof for converted output.

Validation:

- focused unit tests in `wow-viewer/tests/WowViewer.Core.Tests`
- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- `wmo inspect` proof on produced files
- optional `MdxViewer` compatibility proof against staged 0.5.3 client, described explicitly as compatibility evidence only

Exit condition:

- a Cataclysm `v17` WMO can be downgraded by `wow-viewer` without depending on legacy converter code at runtime

## Workstream 2 - WmoV14ToV17

This should follow immediately after the downgrade path while the writer and shared model are still active.

Objective:

- convert Alpha monolithic `WMO v14` into split `WMO v17`

Implementation target:

- add `WmoV17Writer` ownership to `wow-viewer`
- port upgrade mapping using the same shared intermediate contracts from Workstream 1

Expected subtasks:

1. Implement `WmoV17Writer` in `wow-viewer`.
2. Port material and group-info expansion rules.
3. Write split root and group outputs deterministically.
4. Mirror the existing real-data header-version and structural tests.

Validation:

- synthetic/focused tests matching the existing toolbox proof shape
- `wmo inspect` proof for root and group outputs

Exit condition:

- `wow-viewer` owns both WMO directions and no longer depends on legacy writer surfaces for this family

## Workstream 3 - M2ToMdx Minimal Downgrade Lane

This is the first model-family slice that matters for Cataclysm doodads in Alpha.

Objective:

- convert `M2` plus required skin data into Alpha-compatible `MDX`

Important boundary:

- do not try to reach full native visual parity in the first slice
- first land a strict, testable downgrade lane for static geometry, material references, bounds, basic animation metadata, and placement-safe output

Implementation target:

- add `MDX` writer ownership to `wow-viewer`
- build on existing `M2ModelReader`, `M2GeometryReader`, `M2SkinReader`, and `MDX` domain contracts

Expected subtasks:

1. Define the minimum intermediate representation needed to map `M2` into `MDX` without flattening away later-needed semantics.
2. Treat `%02d.skin` selection as an explicit input seam, consistent with native findings already documented in `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`.
3. Implement `MdxWriter` in `wow-viewer`.
4. Port geometry, texture-name, material, sequence, bone, and pivot mapping at a minimal strict subset.
5. Make unsupported features explicit instead of silently inventing data.
6. Keep the current proof boundary explicit: bundled `MDX` texture rewrites and inspect re-read validity are landed, but Alpha runtime-safe animation fidelity is not yet closed.
6. Add focused tests for:
   - signature/version correctness
   - non-empty geometry round-trip through `mdx inspect`
   - texture table and bounds presence
   - skin-fed index data actually reaching MDX geosets

Validation:

- focused tests in `WowViewer.Core.Tests`
- `mdx inspect` proof on produced output
- optional old-viewer load proof only as compatibility evidence

Exit condition:

- `wow-viewer` can produce structurally valid `MDX` from at least a bounded subset of real `M2` assets used by the Cataclysm-to-Alpha lane

## Workstream 4 - M2ToMdx Extended Feature Coverage

Only start after the minimal downgrade lane is stable.

Objective:

- improve downgrade fidelity for the model features that actually block Alpha-side usefulness

Candidate scope:

- sequence mapping improvements
- transparency and render-state mapping
- lights
- attachments and helpers
- particle/ribbon downgrade policy
- explicit unsupported-feature reporting instead of silent lossy conversion

Validation:

- targeted regression fixtures per feature family
- compatibility renders where a staged Alpha client or compatibility host can exercise the converted asset

Exit condition:

- the downgrade lane is good enough for the target Cataclysm object set actually used in converted maps, not merely for synthetic smoke tests

## Workstream 5 - MdxToM2

This is explicitly deferred behind the downgrade-critical work.

Objective:

- convert Alpha `MDX` into later `M2` plus `.skin`

Why deferred:

- not required for the immediate `4.0.0 -> 0.5.3` goal
- depends on the same writer/model seams as the harder `M2ToMdx` path
- easier to do once `wow-viewer` already owns both `MDX` writing and `M2` write-side contracts

Validation:

- focused `m2 inspect` proof
- `.skin` selection proof
- optional runtime proof in `wow-viewer` or legacy compatibility host

## Fresh-Chat Execution Order

The first fresh implementation chat should not try to cover all four converters.

Use this execution order:

1. land `WmoV17ToV14`
2. validate it with focused tests and inspect proof
3. land `WmoV14ToV17`
4. validate it with focused tests and inspect proof
5. land `M2ToMdx` minimal downgrade lane
6. validate it with focused tests and inspect proof
7. decide whether `MdxToM2` is still needed for the immediate milestone

## Validation Language

Use strict proof language during implementation:

- unit tests and inspect output prove file-structure behavior
- a `wow-viewer` build or test pass is implementation proof, not runtime signoff
- `MdxViewer` or staged-client loading is compatibility evidence only
- do not describe any slice as complete until one executable proof exists in `wow-viewer`

## First Slice Recommendation

The fresh chat should start with `WmoV17ToV14`.

Reason:

- strongest reference implementation base
- direct Cataclysm-to-Alpha need
- lower semantic risk than `M2ToMdx`
- lets `wow-viewer` establish writer-side converter ownership for object formats before the harder model-family work

## Expected Outcome

If this plan is followed, `wow-viewer` ends up with:

- canonical shared ownership of WMO version conversion
- a first real object-model downgrade lane for Cataclysm doodads
- a clean path to move Cataclysm-era object conversion out of legacy code and into the repo that already owns the terrain and alphaWDT conversion seams
