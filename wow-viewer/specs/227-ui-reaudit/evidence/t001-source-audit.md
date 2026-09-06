# T001 Receipt — Source-Audit Inventory Baseline

**Date**: 2026-09-06
**Task**: `T001` in [tasks.md](../tasks.md)
**Scope**: documentation-only source audit. No viewer source, UI behavior, or runtime claim changed.

## Files changed

| File | Change |
|---|---|
| `specs/227-ui-reaudit/plan.md` | Filled implementation plan and phased gates. |
| `specs/227-ui-reaudit/spec.md` | Updated status from unstarted draft to the receipted source-audit phase. |
| `specs/227-ui-reaudit/research.md` | Recorded source-backed planning decisions. |
| `specs/227-ui-reaudit/data-model.md` | Defined inventory record fields and replacement rule. |
| `specs/227-ui-reaudit/quickstart.md` | Added source/build/operator validation boundaries. |
| `specs/227-ui-reaudit/tasks.md` | Added dependency-ordered implementation controls. |
| `specs/227-ui-reaudit/surface-inventory-v2.md` | Added current source baseline and screenshot/interaction matrix. |
| `specs/227-ui-reaudit/evidence/t001-source-audit.md` | This receipt. |
| `memory-bank/activeContext.md` | Advanced the dashboard to the Spec 227 T002 handoff and retained the separate Spec 223 operator gate. |
| `memory-bank/progress.md` | Added the compact session ledger entry. |

## Verification

| Command | Exit status | Result |
|---|---:|---|
| `rg -n -g "*.cs" "GetBottomTabLabels|GetArchaeologyWorkbenchLabels|DrawMinimapWindow|DrawUtilitiesMinimap|DrawUnifiedInspectorContent|DrawTerrainControlsAdjustmentWeakSignalContent|DrawTerrainLabSubTab" wow-viewer/src/viewer/WoWViewer` | 0 | Located current workbench, minimap, Inspector, weak-signal, and Terrain Lab source anchors. |
| `$matches = rg -n "\[FEATURE\]|NEEDS CLARIFICATION|\[REMOVE IF UNUSED\]|\[###-feature" wow-viewer/specs/227-ui-reaudit; if ($LASTEXITCODE -eq 1) { Write-Output 'No unfilled SpecKit placeholders found.'; exit 0 }; $matches; exit 1` | 0 | No unfilled SpecKit placeholders found. |
| `$matches = rg -n "\[FEATURE\]|NEEDS CLARIFICATION|\[REMOVE IF UNUSED\]|\[###-feature" wow-viewer/specs/227-ui-reaudit/plan.md wow-viewer/specs/227-ui-reaudit/research.md wow-viewer/specs/227-ui-reaudit/data-model.md wow-viewer/specs/227-ui-reaudit/quickstart.md wow-viewer/specs/227-ui-reaudit/tasks.md wow-viewer/specs/227-ui-reaudit/surface-inventory-v2.md; if ($LASTEXITCODE -eq 1) { Write-Output 'Spec 227 implementation artifacts have no template placeholders.'; exit 0 }; $matches; exit 1` | 0 | The final implementation artifacts have no template placeholders; the receipt itself retains the first command as historical evidence. |
| `git -C wow-viewer diff --check -- specs/227-ui-reaudit` | 0 | No whitespace diagnostics for tracked changes; new Spec 227 artifacts remain intentionally unstaged. |
| `$whitespace = rg -n "[ \t]+$" wow-viewer/specs/227-ui-reaudit wow-viewer/memory-bank/activeContext.md wow-viewer/memory-bank/progress.md; if ($LASTEXITCODE -eq 1) { Write-Output 'No trailing whitespace found in changed documentation.'; exit 0 }; $whitespace; exit 1` | 0 | No trailing whitespace found in the new Spec 227 and continuity documentation. |

## Criterion-to-evidence

| Task criterion | Evidence | Result |
|---|---|---|
| Dated source-audit inventory exists. | [surface-inventory-v2.md](../surface-inventory-v2.md) sections A-C, sourced from the listed `rg` results. | Met for the initial source baseline. |
| Every source-audited duplicate family has an authority decision or an explicit pending state. | [surface-inventory-v2.md](../surface-inventory-v2.md) section B: weak-signal candidate, minimap pending runtime decision, Inspector pending per-object mapping. | Met; no unobserved authority was asserted. |
| Screenshot and interaction matrix exists without fabricated runtime evidence. | [surface-inventory-v2.md](../surface-inventory-v2.md) section D. | Met; all captures are explicitly pending operator evidence. |
| The full US1 inventory acceptance requirement is satisfied. | Spec 227 requires actual screenshots and full v1 reconciliation. | **Not met**; T002-T004 remain open and no source consolidation is authorized. |
