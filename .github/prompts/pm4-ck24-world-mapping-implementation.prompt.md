---
description: "Implement PM4 support in gillijimproject_refactor using a full cross-tile CK24 decode and a validated PM4-to-world coordinate contract instead of tile-local mesh dumps."
name: "PM4 CK24 World Mapping Implementation"
argument-hint: "Optional PM4 tile, CK24 key, failing coordinate example, or output layer to prioritize"
agent: "agent"
---

Implement the next PM4 support slice in `gillijimproject_refactor` using CK24 object grouping and validated world-coordinate mapping as the primary architecture.

## Read First

1. `gillijimproject_refactor/src/MdxViewer/memory-bank/converter_plan.md`
2. `gillijimproject_refactor/.agent/workflows/mscn-first-pipeline.md`
3. `gillijimproject_refactor/WoWRollback/WoWRollback.PM4Module/PipelineCoordinateService.cs`
4. `gillijimproject_refactor/WoWRollback/WoWRollback.PM4Module/Decoding/Pm4Decoder.cs`
5. `gillijimproject_refactor/WoWRollback/WoWRollback.PM4Module/Decoding/Pm4ChunkTypes.cs`
6. `gillijimproject_refactor/.windsurf/rules/data-paths.md`

## Goal

Build PM4 support around the complete PM4 dataset, not around one-tile exports.

The intended end state is:

1. all PM4 tiles decode through one reusable PM4 model
2. CK24 groups can be aggregated across multiple ADT tiles
3. PM4-derived object layers can be converted into correct world coordinates
4. the viewer can place PM4 geometry and PM4-derived object anchors in the same world space as ADT terrain and placements

## Non-Negotiable Architecture

- Do not treat each PM4 tile as an isolated object catalog.
- Do not throw away `CK24`, `MdosIndex`, `MPRL`, or tile provenance during decoding.
- Do not assume all PM4 coordinate-bearing chunks share the same coordinate space.
- Do not claim the transform is correct until it is checked against the fixed development dataset.
- Prefer the richer rollback PM4 decoder contract over the thinner core parser when there is a mismatch.

## Required Data Truths

- `MSUR.PackedParams` contains the effective `CK24` key.
- `MSUR.MdosIndex` links surfaces to `MSCN` scene nodes.
- CK24 grouping can span multiple PM4 files / ADT tiles.
- The existing pipeline already exports layers of the same object via CK24; the remaining hard problem is correct world-coordinate placement.

## Required Starting Files

1. `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Formats/PM4/Pm4File.cs`
2. `gillijimproject_refactor/WoWRollback/WoWRollback.PM4Module/Decoding/Pm4Decoder.cs`
3. `gillijimproject_refactor/WoWRollback/WoWRollback.PM4Module/Decoding/Pm4ChunkTypes.cs`
4. `gillijimproject_refactor/WoWRollback/WoWRollback.PM4Module/Decoding/Pm4MapReader.cs`
5. `gillijimproject_refactor/WoWRollback/WoWRollback.PM4Module/PipelineCoordinateService.cs`
6. `gillijimproject_refactor/WoWRollback/WoWRollback.PM4Module/Decoding/MscnObjectDiscovery.cs`

## Required Implementation Order

1. Confirm which PM4 decode contract should become canonical in core.
2. Preserve and expose CK24, MSCN, MPRL, and tile provenance in reusable APIs.
3. Build a cross-tile CK24 registry before adding viewer rendering.
4. Define one authoritative PM4-to-world transform API.
5. Validate transformed positions against known ADT placements from the fixed development data.
6. Only then add viewer/debug layers for PM4 geometry, object anchors, or matched assets.

## Required Layer Model

At minimum, keep these logical outputs separate:

- terrain or nav/background surfaces
- CK24 object groups
- matched WMO candidates
- matched M2 candidates
- unmatched geometry candidates
- placement/reference markers

## Validation Rules

- Use the fixed development dataset from `test_data/development/World/Maps/development`.
- If you did not validate coordinates against real PM4 + ADT data, say so explicitly.
- If you only built a decoder or viewer path without cross-tile aggregation, say so explicitly.
- If automated tests were not added or run, say so explicitly.
- Build success is not proof that the world-coordinate mapping is correct.

## Deliverables

Return all items:

1. Canonical PM4 decode model implemented or proposed
2. Exact CK24 aggregation model
3. Exact coordinate contract used for PM4-derived placements
4. Files changed and why
5. Validation status split into build, tests, and real-data coordinate checks
6. Any memory-bank or plan updates still required

## First Output

Start with:

1. current PM4 decode state summary
2. whether CK24 aggregation or coordinate mapping is the first blocker
3. the exact file you will use as the coordinate source of truth
4. the smallest code slice you will change first