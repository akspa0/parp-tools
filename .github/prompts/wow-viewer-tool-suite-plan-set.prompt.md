---
description: "Route wow-viewer migration work to the right detailed Copilot prompt. Use when the task is choosing between PM4 implementation for inspect or audit or linkage or MSCN or unknowns work, shared-I/O implementation for ADT or WDT or WMO or BLP or DBC families, bootstrap, shared-library ownership planning, tool inventory, CLI or GUI parity, or migration sequencing."
name: "wow-viewer Tool Suite Plan Set"
argument-hint: "Describe the tool family, file type, migration problem, or planning slice you want to attack"
agent: "agent"
---

Choose the right detailed planning or implementation prompt for the `wow-viewer` tool-suite refactor.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `.github/copilot-instructions.md`

## Goal

Route the current request to the correct focused prompt in `.github/prompts/` so the planning work stays specific, detailed, and tied to the actual tool/library mess in `parp-tools`.

## Companion Prompts

- `wow-viewer-pm4-library-implementation.prompt.md`
- `wow-viewer-shared-io-implementation.prompt.md`
- `wow-viewer-world-runtime-plan-set.prompt.md`
- `wow-viewer-bootstrap-layout-plan.prompt.md`
- `wow-viewer-shared-io-library-plan.prompt.md`
- `wow-viewer-tool-inventory-cutover-plan.prompt.md`
- `wow-viewer-cli-gui-surface-plan.prompt.md`
- `wow-viewer-tool-migration-sequence-plan.prompt.md`

## Routing Rules

- Use `wow-viewer-pm4-library-implementation.prompt.md` when the problem is the next `Core.PM4` extraction, such as `pm4 inspect`, `pm4 audit`, `pm4 linkage`, `pm4 mscn`, `pm4 unknowns`, or `pm4 export-json`, a PM4 regression coverage slice, or a narrow PM4 consumer or solver seam that should stay library-first.
- If the PM4 ask is specifically about terminology reconciliation, decode-confidence, or whether current field names are documentation-backed versus local aliases, still route to `wow-viewer-pm4-library-implementation.prompt.md` and require a raw-doc-name versus local-alias breakdown.
- Use `wow-viewer-shared-io-implementation.prompt.md` when the problem is the next real `Core` or `Core.IO` shared-format slice, such as ADT root or split-ADT (`_tex0.adt`, `_obj0.adt`, `_lod.adt`) work, WDT summary work, WMO or BLP or DBC or DB2 detection work, a file detector, a chunk reader, a non-PM4 inspect verb, a converter command, or a shared-format regression update.
- Use `wow-viewer-world-runtime-plan-set.prompt.md` when the problem is splitting `WorldScene`, extracting world-runtime services, suppressing repeated asset-miss churn such as `.skin` lookup spam, or sequencing terrain/WMO/MDX/overlay runtime ownership into `wow-viewer`.
- Use `wow-viewer-bootstrap-layout-plan.prompt.md` when the problem is repo shape, solution structure, bootstrap scripts, project layout, or where each tool/app/lib should live.
- Use `wow-viewer-shared-io-library-plan.prompt.md` when the problem is the broader ownership plan for reading/writing ADT, WDT, M2, MDX, WMO, PM4, DBC, BLP, placement, or related formats.
- Use `wow-viewer-tool-inventory-cutover-plan.prompt.md` when the problem is inventorying old tools, deciding what migrates, what merges, what stays archaeology-only, and what becomes a first-class tool in the new repo.
- Use `wow-viewer-cli-gui-surface-plan.prompt.md` when the problem is making tools available both as CLI commands and existing GUI panels over the same shared services.
- Use `wow-viewer-tool-migration-sequence-plan.prompt.md` when the problem is phasing, dependencies, vertical slices, or what order to migrate tool families without exploding scope.

## Deliverables

Return all items:

1. the best prompt to run next
2. why it is the right prompt for the current ask
3. which companion prompts should follow after it
4. what concrete repo/tool/file-type scope the next prompt should include
5. what not to waste time on yet
6. whether the current terminology in that area is raw-doc-backed or only a local research alias layer

## First Output

Start with:

1. the exact planning problem you think the user is actually trying to solve
2. the single best next prompt from the set above
3. the concrete migration/tool scope that prompt should cover first
