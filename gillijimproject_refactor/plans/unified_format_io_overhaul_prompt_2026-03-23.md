# Unified Format I/O Overhaul Prompt

Use this prompt in a fresh planning chat when working on the shared read/write/conversion overhaul for WoW terrain, map, model, and WMO data, especially when that overhaul is being moved into `https://github.com/akspa0/wow-viewer` for `v0.5.0`.

## Prompt

Design a concrete implementation plan for a unified shared format I/O library that becomes the single source of truth in `wow-viewer` for reading, writing, and converting:

- Alpha `WDT` / monolithic terrain data
- LK 3.3.5 split `ADT` / `WDT`
- relevant 4.x split `ADT` / `WDT` variants already partially supported in the viewer
- `MDX` / `M2`
- `WMO v16` / `WMO v17`
- placement data including `MDDF` / `MODF`

The plan must assume the current repo has drifted knowledge:

- `MdxViewer` now contains newer practical runtime-read behavior and version handling in several areas.
- `WoWMapConverter.Core` still contains older assumptions and is not a complete source of truth.
- `gillijimproject-csharp` and other first-party/internal base libraries should be treated as migration input, not as permanent owned architecture to keep scattering through the next repo.
- The existing map converter should not be treated as closed for Alpha placement downconversion. Writing correct Alpha `MODF` / `MDDF` placement data is still an explicit open seam.
- Do not assume broad roundtrip safety just because some readers or isolated converters already exist.
- `parp-tools` should now be treated as the R&D/archeology source repo, not the long-term production home for the canonical library.
- new-repo layout sanity matters: the main renderer app in `wow-viewer` should have a direct, readable home, while shared libraries and standalone tools should live under clearly separate roots.
- upstream libraries such as `Warcraft.NET`, `DBCD`, `WoWDBDefs`, `Alpha-Core`, `WoWTools.Minimaps`, and `SereniaBLPLib` should stay under a `libs/` policy and continue tracking their original repos where practical; the goal is to rebuild our own first-party parsing/writing/conversion logic, not to fork everything indiscriminately.
- upstream support data such as `wow-listfile` should be treated as automatic bootstrap material rather than manual tribal-knowledge setup.
- PM4 viewer forensics now have one pragmatic structural contract that should be preserved during planning:
	- use `CK24` family as the root viewer bucket
	- split next by `MSLK`-linked subgroup
	- then by optional `MDOS` subgroup
	- then by connectivity/component part
	- treat centroids as derived display anchors, not as proven raw PM4 node records
	- treat `MSUR.AttributeMask` colors as value-identification buckets first, not as closed semantics

## What The Plan Must Produce

1. A target architecture for the shared library.
2. A statement of how that library lives in `wow-viewer` instead of continuing as another half-owned subsystem inside `parp-tools`.
3. A migration map showing what logic currently lives in:
	- `src/MdxViewer`
	- `src/WoWMapConverter/WoWMapConverter.Core`
	- `src/gillijimproject-csharp`
	- any still-relevant rollback/reference sources
	- imported project code that should become first-party code in the new library/tool stack
4. A phased extraction strategy that avoids breaking the active viewer while `wow-viewer` becomes viable.
5. A list of format contracts that should become canonical first.
6. A clear statement of which conversions are currently real, partial, or speculative.
7. A validation strategy that uses real data and does not over-claim.

## Required Constraints

- Do not propose a vague “rewrite everything” plan.
- Favor incremental extraction with a first vertical slice that can actually land.
- Do not recreate the current repo sprawl inside `wow-viewer`; the new repo should have an obvious viewer-app root plus separate library and tool roots.
- Preserve readable FourCC handling internally and only reverse at I/O boundaries where needed.
- Keep Alpha and standard terrain handling explicitly separated where that distinction is structurally meaningful.
- Distinguish domain-specific code that should move into first-party ownership from commodity dependencies that should stay external.
- Treat the current first-party/internal parser/writer libraries as things to rationalize and re-own in one canonical library, not as permanent layers to keep stacking.
- Treat terrain alpha, placement writing, and model/WMO version translation as separate risk areas with separate acceptance criteria.
- Assume real-data validation is mandatory before claiming any cross-version conversion works.
- If the plan carries PM4-derived structure into the shared library, distinguish raw PM4 relationships from viewer-derived grouping/anchor data explicitly.

## High-Priority Open Seams To Address

- Alpha placement writing for `MODF` / `MDDF` during downconversion.
- Shared terrain read/write contracts across Alpha, LK 3.3.5, and supported 4.x split-ADT variants.
- Consolidating `MdxViewer` runtime knowledge for `M2` / `MDX` and `WMO` parsing/writing into reusable non-renderer code.
- Defining a canonical placement/model/WMO conversion pipeline rather than separate ad-hoc tools.
- Ensuring tool UIs and CLIs all call the same library instead of maintaining divergent code paths.
- Deciding what can stay as upstream external dependency under `libs/` versus what must be fully re-authored into our own first-party library.
- Evaluating whether existing upstream tools or libraries should be adapted rather than rewritten, including `MapUpconverter`, `ADTMeta`, `wow.export`, and `wow.tools.local` as reference/integration seams.
- Capturing PM4 structural findings without over-closing semantics:
	- the current useful hierarchy is `CK24 -> MSLK-linked group -> MDOS bucket -> connectivity part`
	- `MSUR.AttributeMask` / `GroupKey` should stay available as inspectable subgroup labels even when their final semantics remain open

## Suggested Deliverable Structure

1. Current-state inventory
2. New repo layout proposal
3. Canonical-library proposal
4. Phase plan
5. First vertical slice
6. Risk register
7. Real-data validation plan
8. Cutover plan for existing tools
9. New-repo adoption and retirement plan for the old stack

## Stretch Direction

The plan may include optional upstream collaboration work, such as identifying a small, concrete alpha-era asset-support contribution for `Noggit` / `noggit-red`, but that should remain explicitly separate from the core canonical-library migration.

## Validation Rules

- Be explicit about what is not yet proven.
- If no automated tests are proposed for a seam, say why and give the real-data validation step instead.
- Do not count archived code, library tests, or synthetic fixtures as proof for the active pipeline.

## Fixed Data Reminder

Use the fixed paths already documented in the repo memory bank, especially:

- `test_data/development/World/Maps/development`
- `test_data/WoWMuseum/335-dev/World/Maps/development`

Do not ask for alternate paths unless those fixed paths are genuinely missing.