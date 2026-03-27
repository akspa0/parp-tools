# Skill Registry

Codex-oriented skill docs live here. They are reusable workflow playbooks converted from the active workspace guidance and normalized for cross-tool use.

## Active skills

- `wow-viewer-pm4-library/SKILL.md`
  - Use for `Core.PM4` extraction, PM4 inspect or audit verbs, PM4 analyzer work, placement math, and PM4-focused regression updates.

- `wow-viewer-shared-io-library/SKILL.md`
  - Use for shared non-PM4 format ownership in `WowViewer.Core` or `WowViewer.Core.IO`, including ADT, WDT, WMO, BLP, DBC, DB2, file detection, and top-level summary seams.

- `wow-viewer-migration-continuation/SKILL.md`
  - Use for continuation routing, next-slice selection, migration regrouping, and deciding whether work belongs in `wow-viewer`, `gillijimproject_refactor`, PM4, or shared I/O.

- `terrain-alpha-regression/SKILL.md`
  - Use for terrain alpha-mask, MCAL, MCLY, split ADT texture, and blending regressions in `gillijimproject_refactor`.

## How to use this registry

- Start with `AGENTS.md` for the always-on workspace rules.
- Use `.codex/prompts/README.md` when you need a reusable prompt or planning template.
- Use the skill files here when you need an area-specific playbook with focused guardrails, validation commands, and file ownership guidance.

## Notes

- These skill docs are shared workflow assets for Codex, Copilot, and OpenCode.
- Keep this registry aligned with `AGENTS.md`, `OPENCODE.md`, `.codex/README.md`, and the underlying skill files when the workflow surface changes.
