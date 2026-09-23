# Plan — Epic 251 Viewer UX, Shell & Code Health

**Status**: Implementation approach **not yet selected** (operator directive 2026-09-23: triage first).

## Dependencies between backlog items

```text
U-10 sidebar standard (was 227-T004 inventory gate) ──> U-01 extraction ──> U-30 WoW shell
U-10 ──> U-11/U-13/U-14/U-15 dedupe work
U-18, U-21 independent small fixes
```

The 223 → 227 → 231 chain was three successive consolidation attempts, each superseding the last's
unfinished part; U-10–U-19 are its single merged residue.

## Design documents adopted by reference

| Item | Adopted design |
|---|---|
| U-01 | [228 plan](../archived/228-source-decomposition/plan.md) · [research](../archived/228-source-decomposition/research.md) |
| U-02 | [224 spec](../archived/224-speckit-governance/spec.md); rules live in [AGENTS.md](../../../AGENTS.md) §9 |
| U-03 | [213 spec](../archived/213-mcp-tooling-harness/spec.md) · [178 spec](../archived/178-mcp-automation-surface/spec.md) |
| U-10–U-12 | [227 plan](../archived/227-ui-reaudit/plan.md) · [inventory v2](../archived/227-ui-reaudit/surface-inventory-v2.md) (the §11 enforcement artifact) |
| U-13–U-16 | [223 plan](../archived/223-ui-consolidation-audit/plan.md) · [surface inventory](../archived/223-ui-consolidation-audit/surface-inventory.md) |
| U-17, U-18 | [231 plan](../archived/231-editor-archaeology-ui-overhaul/plan.md) |
| U-30 | [229 spec](../archived/229-wow-shell-keybind-profiles/spec.md) |
| U-31 | [225 spec](../archived/225-overhead-ortho-view/spec.md) |
| U-32 | [212 plan](../archived/212-spatial-ui-shell/plan.md) US6 |

## Standing constraints

AGENTS.md §10 (no new `WorldScene`/`ViewerApp` members, ~2,000-line budget, owned services) and §11
(`SharedUiWidgets`, one authoritative home per surface, inventory row per new surface).
