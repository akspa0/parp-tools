# Receipt: Layer Opacity Metadata and Canonical Writer Reuse

**2026-10-21. Planning only.** Operator requested opacity reports/metadata for MCAL/MCLY layers
and refreshed output using the viewer's current New Map writers. No implementation task completed.

## Changed files

Under `specs/285-v65-terrain-interchange/`: `spec.md`, `plan.md`, `tasks.md`, `data-model.md`,
`research.md`, `quickstart.md`, `checklists/requirements.md`, `contracts/layer-opacity-report.md`,
the historical receipt pointer and this receipt. 62 pending tasks; final tasks remapped T058–T062.

Under `specs/284-map-object-converter-revitalization/`: `spec.md`, `plan.md`, `tasks.md`,
`data-model.md`, `research.md`, `quickstart.md`, `contracts/conversion-tools.md`,
`contracts/map-writer-reuse.md` and historical rebase receipt pointer. 74 pending tasks, with
writer inventory/materialization/readback refinements in existing US3 tasks.

Continuity: `specs/STATUS.md`, `memory-bank/activeContext.md`, `memory-bank/progress.md`.
Only planning/continuity Markdown was changed by this session. Concurrent UI/Spellbook WIP retained.

## Source-backed findings

- `NewMapCreatorService` calls `TemplatedTerrainGenerator`, `LkAdtWriter`, `LkWdtWriter` and
  `WdlWriter`. `AdtTerrainWriter` patches existing MCVT/MCNR; it is not fresh-map serialization.
- Current canonical writers have explicit metadata/default/native-WDL limitations. Reuse is not
  a full-fidelity claim. Alpha writer/working readers remain protected.
- Existing alpha decoding and ordinary ordered Lerp/contribution logic is reusable. Existing
  summaries do not provide complete per-layer opacity statistics; texture aggregation loses
  repeated-layer identity and a pixel-loop failure count is not a layer count.
- Unknown/fixed/filled samples, implicit base and height-blend semantics require distinct states.
  Source report generation does not depend on later target writer implementation.

## Criterion-to-evidence boundary

| Criterion | Planning evidence | Actual implementation/acceptance |
|---|---|---|
| 285 FR-015 / SC-011 | US7 T050–T057 and per-layer opacity contract | Pending; no reports generated. |
| 284 FR-014 / SC-009 | Canonical writer contract; US3 writer/data/readback tasks | Pending; no output written or read back. |
| Every layer retained | Stable per-layer rows, source/decoded/effective separation and validity | Contract only; real source/target proof pending. |
| Protected behavior | Explicit library/frozen-file boundaries and fresh-write/patch distinctions | No executable source changes this session. |

## Verification

Spec Kit prerequisites, task/phase formats, links/fences and PowerShell parsing are documentation
checks only. No build/tests, harvesting, map creation, GPU/training, visual or client run occurred.

All documentation checks exited **0**: Spec Kit prerequisites for both packages; task format/
sequence/phase sizes (**74 + 62 = 136 pending**, zero malformed, max eight tasks/phase);
**22 Markdown files**, zero broken local links/unclosed fences and **19 PowerShell snippets**
parse after recording this command. Scoped `git diff --check` returned exit 0. Concurrent source
WIP was observed but not edited by this session.

The exact prerequisite command is in each quickstart. The exact link/fence/PowerShell parser
command is retained in the [earlier rebase receipt](../../284-map-object-converter-revitalization/evidence/rebase-v65-receipt.md)
and was run against the current files. Task validation used:

```powershell
$base = 'I:/parp/parp-tools/wow-viewer/specs'
foreach ($pair in @(@('284-map-object-converter-revitalization',74),@('285-v65-terrain-interchange',62))) {
    $lines = @(Get-Content -LiteralPath (Join-Path (Join-Path $base $pair[0]) 'tasks.md'))
    $tasks = @($lines | Where-Object { $_ -match '^- \[' })
    $bad = @($tasks | Where-Object { $_ -notmatch '^- \[ \] T\d{3}( \[P\])?( \[US\d+\])? .+`[^`]+`' })
    $ids = @($tasks | ForEach-Object { if ($_ -match 'T(\d{3})') { [int]$Matches[1] } })
    $counts = [ordered]@{}
    $phase = ''
    foreach ($line in $lines) {
        if ($line -match '^## Phase (\d+)') { $phase = $Matches[1]; $counts[$phase]=0 }
        if ($line -match '^- \[ \] T' -and $phase) { $counts[$phase]++ }
    }
    if ($tasks.Count -ne $pair[1] -or $bad.Count -ne 0 -or @(Compare-Object (1..$pair[1]) $ids).Count -ne 0 -or @($counts.Values | Where-Object { $_ -gt 10 }).Count -ne 0) { throw "Task validation failed: $($pair[0])" }
    [pscustomobject]@{spec=$pair[0];tasks=$tasks.Count;checked=0;malformed=$bad.Count;phaseCounts=$counts} | ConvertTo-Json -Compress
}
```

Scoped whitespace command from workspace root:

`git diff --check -- wow-viewer/specs/284-map-object-converter-revitalization wow-viewer/specs/285-v65-terrain-interchange wow-viewer/specs/STATUS.md wow-viewer/memory-bank/activeContext.md wow-viewer/memory-bank/progress.md`

Review clarified that Spec 285 source-report readiness precedes Spec 284 comparisons/reduction;
adapter code can start earlier, and target comparison follows actual output. No runtime claims.
