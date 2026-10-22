# Planning Receipt — Spec 284

**Historical initial snapshot**: The operator subsequently rebased this plan onto current harvesting
and v65, retaining placement metadata and excluding object geometry/masks, with explicit CASC input.
Legacy-command parity/adapters in the original task meanings are superseded. See the current
[rebase receipt](rebase-v65-receipt.md), plan and tasks; this initial receipt is retained as history.

**Date**: 2026-10-21  
**Delivered layer**: Documentation and implementation handoff only. No implementation task closed.

## Files authored/updated

- `specs/284-map-object-converter-revitalization/spec.md`: operator intent, stories, requirements and criteria.
- `specs/284-map-object-converter-revitalization/plan.md`: architecture, ownership, phases and gates.
- `specs/284-map-object-converter-revitalization/tasks.md`: 74 pending tasks across 11 bounded phases.
- `specs/284-map-object-converter-revitalization/research.md`: code-backed baseline and explicit gaps.
- `specs/284-map-object-converter-revitalization/data-model.md`: client, route, identity and candidate contracts.
- `specs/284-map-object-converter-revitalization/contracts/conversion-tools.md`: separate hosts/shared APIs.
- `specs/284-map-object-converter-revitalization/quickstart.md`: future PowerShell commands/witness matrix.
- `specs/284-map-object-converter-revitalization/evidence/planning-receipt.md`: this receipt.
- `specs/STATUS.md`, `memory-bank/activeContext.md`, `memory-bank/progress.md`: narrow routing/handoff entries.

Existing branch `280-ui-rework`, Spec 283 documents, UI source/registry/reference WIP and `imgui.ini`
were retained. No executable source, project file, client data or persisted Spec Kit feature selection
was edited. The script's logical feature override does not represent a real branch switch.

## Verification commands and actual results

From `I:/parp/parp-tools/wow-viewer`, setup used the repository PowerShell workflow:

```powershell
$env:SPECIFY_FEATURE_DIRECTORY = 'I:/parp/parp-tools/wow-viewer/specs/284-map-object-converter-revitalization'
$env:SPECIFY_FEATURE = '284-map-object-converter-revitalization'
& ./.specify/scripts/powershell/setup-plan.ps1 -Json
```

Exit **0**. Returned `FEATURE_SPEC`, `IMPL_PLAN`, `SPECS_DIR` under Spec 284 and `PLAN_PRESERVED=false`.
The copied template was replaced with the authored plan. No persisted `.specify/feature.json` edit.

After authoring:

```powershell
$env:SPECIFY_FEATURE_DIRECTORY = 'I:/parp/parp-tools/wow-viewer/specs/284-map-object-converter-revitalization'
$env:SPECIFY_FEATURE = '284-map-object-converter-revitalization'
& ./.specify/scripts/powershell/check-prerequisites.ps1 -Json -RequireTasks -IncludeTasks
```

Exit **0**; Spec 284 resolved with research, data-model, contracts, quickstart and tasks available.

Document validation (from any working directory):

```powershell
$specRoot = 'I:/parp/parp-tools/wow-viewer/specs/284-map-object-converter-revitalization'
$taskText = Get-Content -LiteralPath (Join-Path $specRoot 'tasks.md')
$taskLines = @($taskText | Where-Object { $_ -match '^- \[' })
$badTasks = @($taskLines | Where-Object { $_ -notmatch '^- \[ \] T\d{3}( \[P\])?( \[US[1-7]\])? .+`[^`]+`' })
$taskIds = @($taskLines | ForEach-Object { if ($_ -match 'T(\d{3})') { [int]$Matches[1] } })
if ($taskLines.Count -ne 74 -or $badTasks.Count -gt 0 -or @(Compare-Object (1..74) $taskIds).Count -gt 0) { throw 'Task format/count/sequence failed' }
$phaseCounts = [ordered]@{}
$phase = $null
foreach ($line in $taskText) {
    if ($line -match '^## Phase (\d+)') { $phase = $Matches[1]; $phaseCounts[$phase] = 0 }
    if ($line -match '^- \[ \] T' -and $null -ne $phase) { $phaseCounts[$phase]++ }
}
if (@($phaseCounts.Values | Where-Object { $_ -gt 10 }).Count -gt 0) { throw 'Oversized phase' }
$missingLinks = @()
$markdownFiles = @(Get-ChildItem -LiteralPath $specRoot -Filter '*.md' -Recurse)
foreach ($file in $markdownFiles) {
    $content = Get-Content -LiteralPath $file.FullName -Raw
    if (([regex]::Matches($content, '(?m)^```')).Count % 2 -ne 0) { throw "Unclosed fence: $($file.Name)" }
    foreach ($match in [regex]::Matches($content, '\[[^\]]+\]\(([^)]+)\)')) {
        $target = $match.Groups[1].Value.Split('#')[0]
        if ($target -and $target -notmatch '^(https?://|mailto:)') {
            if (-not (Test-Path -LiteralPath (Join-Path $file.DirectoryName $target))) { $missingLinks += "$($file.Name): $target" }
        }
    }
}
if ($missingLinks.Count -gt 0) { throw ($missingLinks -join '; ') }
[pscustomobject]@{ tasks = $taskLines.Count; checked = 0; phases = $phaseCounts; markdownFiles = $markdownFiles.Count; missingLinks = $missingLinks.Count } | ConvertTo-Json -Compress
```

Exit **0**: 74 sequential pending tasks; phase counts **3, 6, 7, 7, 6, 8, 7, 7, 8, 8, 7**;
all 8 package Markdown files have closed fences and zero missing local Markdown links.

Final containment check: `git diff --check -- wow-viewer/specs/284-map-object-converter-revitalization wow-viewer/specs/STATUS.md wow-viewer/memory-bank/activeContext.md wow-viewer/memory-bank/progress.md`
from the workspace root returned exit **0**. `git status --short` showed the new spec folder and
the three continuity files alongside the original unrelated WIP; no implementation edit was made.

## Criterion-to-evidence boundary

| Criterion | Planning coverage | Actual implementation/acceptance |
|---|---|---|
| SC-001 client routes | AD-2, US1 T010–T016; real-client evidence gate | Pending; no client inspected this session. |
| SC-002 uniqueIDs | AD-4, identity model, US2 T017–T023 | Pending; no object IDs rewritten. |
| SC-003 modern→Alpha pilot | AD-5–7, US3–5 and target task T071 | Pending; no output or client witness. |
| SC-004 independent tools/parity | AD-1/9, separate CLI contracts and US3–6 | Pending; no CLI or Editor code added. |
| SC-005 generation/remeshing | AD-7/8, US7 T060–T067 | Pending; no mesh generated. |
| SC-006 guided Editor flow | AD-9, US6 and operator T059 | Pending; no UI/runtime proof. |
| SC-007 base preservation | Explicit protected boundaries and T068 | No implementation changes this session; future parity proof pending. |

Read-only source review established existing WMO regrouping and saved known-good entries, plus
model fidelity and modern-route gaps. Independent planning review caught and corrected one roadmap
wording error: the Editor phase closes structural integration, while SC-003 target acceptance remains
at T071. No build/tests, conversion, training/GPU job, full-client scan, viewer or target run occurred.
