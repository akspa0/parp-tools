# Unified Format I/O Overhaul Prompt

Use this prompt in a fresh planning chat when working on the shared read/write/conversion overhaul for WoW terrain, map, model, and WMO data.

## Prompt

Design a concrete implementation plan for a unified shared format I/O library in `gillijimproject_refactor` that becomes the single source of truth for reading, writing, and converting:

- Alpha `WDT` / monolithic terrain data
- LK 3.3.5 split `ADT` / `WDT`
- relevant 4.x split `ADT` / `WDT` variants already partially supported in the viewer
- `MDX` / `M2`
- `WMO v16` / `WMO v17`
- placement data including `MDDF` / `MODF`

The plan must assume the current repo has drifted knowledge:

- `MdxViewer` now contains newer practical runtime-read behavior and version handling in several areas.
- `WoWMapConverter.Core` still contains older assumptions and is not a complete source of truth.
- The existing map converter should not be treated as closed for Alpha placement downconversion. Writing correct Alpha `MODF` / `MDDF` placement data is still an explicit open seam.
- Do not assume broad roundtrip safety just because some readers or isolated converters already exist.
- PM4 viewer forensics now have one pragmatic structural contract that should be preserved during planning:
	- use `CK24` family as the root viewer bucket
	- split next by `MSLK`-linked subgroup
	- then by optional `MDOS` subgroup
	- then by connectivity/component part
	- treat centroids as derived display anchors, not as proven raw PM4 node records
	- treat `MSUR.AttributeMask` colors as value-identification buckets first, not as closed semantics

## What The Plan Must Produce

1. A target architecture for the shared library.
2. A migration map showing what logic currently lives in:
	- `src/MdxViewer`
	- `src/WoWMapConverter/WoWMapConverter.Core`
	- any still-relevant rollback/reference sources
3. A phased extraction strategy that avoids breaking the active viewer.
4. A list of format contracts that should become canonical first.
5. A clear statement of which conversions are currently real, partial, or speculative.
6. A validation strategy that uses real data and does not over-claim.

## Required Constraints

- Do not propose a vague “rewrite everything” plan.
- Favor incremental extraction with a first vertical slice that can actually land.
- Preserve readable FourCC handling internally and only reverse at I/O boundaries where needed.
- Keep Alpha and standard terrain handling explicitly separated where that distinction is structurally meaningful.
- Treat terrain alpha, placement writing, and model/WMO version translation as separate risk areas with separate acceptance criteria.
- Assume real-data validation is mandatory before claiming any cross-version conversion works.
- If the plan carries PM4-derived structure into the shared library, distinguish raw PM4 relationships from viewer-derived grouping/anchor data explicitly.

## High-Priority Open Seams To Address

- Alpha placement writing for `MODF` / `MDDF` during downconversion.
- Shared terrain read/write contracts across Alpha, LK 3.3.5, and supported 4.x split-ADT variants.
- Consolidating `MdxViewer` runtime knowledge for `M2` / `MDX` and `WMO` parsing/writing into reusable non-renderer code.
- Defining a canonical placement/model/WMO conversion pipeline rather than separate ad-hoc tools.
- Ensuring tool UIs and CLIs all call the same library instead of maintaining divergent code paths.
- Capturing PM4 structural findings without over-closing semantics:
	- the current useful hierarchy is `CK24 -> MSLK-linked group -> MDOS bucket -> connectivity part`
	- `MSUR.AttributeMask` / `GroupKey` should stay available as inspectable subgroup labels even when their final semantics remain open

## Suggested Deliverable Structure

1. Current-state inventory
2. Canonical-library proposal
3. Phase plan
4. First vertical slice
5. Risk register
6. Real-data validation plan
7. Cutover plan for existing tools

## Validation Rules

- Be explicit about what is not yet proven.
- If no automated tests are proposed for a seam, say why and give the real-data validation step instead.
- Do not count archived code, library tests, or synthetic fixtures as proof for the active pipeline.

## Fixed Data Reminder

Use the fixed paths already documented in the repo memory bank, especially:

- `test_data/development/World/Maps/development`
- `test_data/WoWMuseum/335-dev/World/Maps/development`

Do not ask for alternate paths unless those fixed paths are genuinely missing.