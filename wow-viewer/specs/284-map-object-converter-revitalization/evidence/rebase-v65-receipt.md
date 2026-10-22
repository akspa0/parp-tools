# Receipt: Spec 284 Rebased on v65

**Historical rebase snapshot**: Current writer reuse and layer-opacity additions are recorded in
the [latest receipt](../../285-v65-terrain-interchange/evidence/opacity-writer-planning-receipt.md).
Spec 284 remains 74 pending tasks; Spec 285 now has 62. Earlier counts/checks below are historical.

**2026-10-21 — planning revision only.** Original planning receipt is historical. All **74 tasks
remain pending**, with their meanings amended by the operator's current-tooling/data-interchange direction.

## Changes

- Map path: current MPQ/loose/CASC harvest → v65 native terrain/typed placements → modern export.
- Spec 285 owns missing chunks, durable unknown/native metadata, cell/layer appearance indexes,
  complete referenced source/PNG texture resources and Python-owned Zarr/reverse transport.
- Placements retain exact integer IDs; geometry/masks stay outside the dataset. Spec 284 owns
  target re-itemization, object asset conversion/remeshing/regrouping/generation and Editor/CLI tools.
- Old Converter executable and mandatory aliases/parity work removed as implementation obligations.
  Useful original algorithms remain reference logic with fresh real-data checks.
- Adapter work begins at native/placement/protocol readiness; complete appearance and target
  acceptance follow, preventing a circular dependency on the adapter's own eventual output.

Changed paths and operator clarifications are enumerated in the
[Spec 285 receipt](../../285-v65-terrain-interchange/evidence/planning-receipt.md).
No executable source/project/client files changed; no builds or ingestion/conversion runs.

## Documentation verification

Spec Kit prerequisite checks for both packages: exit **0**, research/data-model/contracts/
quickstart/tasks available. Task format/sequence/phase checks: exit **0**, **74 + 54 = 128 pending**,
no malformed IDs/checklists and maximum eight tasks in any phase.

Validation results: exit **0** for both Spec Kit prerequisites and scoped `git diff --check`;
**19 Markdown files**, zero missing local links, zero unclosed fences, and **17 PowerShell
blocks** parse successfully after recording the command below. No future command was executed.
Git status included concurrent UI/Spellbook work; this session wrote only the named planning
and continuity Markdown files.

Exact link/fence/PowerShell syntax validation command:

```powershell
$base = 'I:/parp/parp-tools/wow-viewer/specs'
$summary = @()
$failures = @()
foreach ($name in @('284-map-object-converter-revitalization','285-v65-terrain-interchange')) {
    $root = Join-Path $base $name
    $files = @(Get-ChildItem -LiteralPath $root -Recurse -Filter '*.md')
    $blocks = 0
    foreach ($file in $files) {
        $content = Get-Content -LiteralPath $file.FullName -Raw
        if (([regex]::Matches($content, '(?m)^```')).Count % 2 -ne 0) { $failures += "Unclosed fence: $($file.FullName)" }
        foreach ($link in [regex]::Matches($content, '\[[^\]]+\]\(([^)]+)\)')) {
            $target = $link.Groups[1].Value.Split('#')[0]
            if ($target -and $target -notmatch '^(https?://|mailto:)' -and -not (Test-Path -LiteralPath (Join-Path $file.DirectoryName $target))) { $failures += "Missing link: $($file.Name) $target" }
        }
        foreach ($block in [regex]::Matches($content, '(?ms)^```powershell\r?\n(.*?)^```')) {
            $blocks++
            $tokens = $null
            $parseErrors = $null
            [void][System.Management.Automation.Language.Parser]::ParseInput($block.Groups[1].Value, [ref]$tokens, [ref]$parseErrors)
            foreach ($errorItem in $parseErrors) { $failures += "PowerShell syntax: $($file.Name) $($errorItem.Message)" }
        }
    }
    $summary += [pscustomobject]@{spec=$name;markdownFiles=$files.Count;powershellBlocks=$blocks}
}
if ($failures.Count -gt 0) { throw ($failures -join '; ') }
[pscustomobject]@{packages=$summary;missingLinks=0;unclosedFences=0;powershellParseErrors=0} | ConvertTo-Json -Depth 4 -Compress
```

From workspace root, scoped whitespace/containment command:

```powershell
git diff --check -- wow-viewer/specs/284-map-object-converter-revitalization wow-viewer/specs/285-v65-terrain-interchange wow-viewer/specs/STATUS.md wow-viewer/specs/254-epic-datasets-and-terrain-ml/plan.md wow-viewer/memory-bank/activeContext.md wow-viewer/memory-bank/progress.md
git status --short
```

Source/format/runtime/visual/CASC/PNG acceptance remains **pending**. Documentation checks do not
close Spec 284 SC-001–SC-008 or Spec 285 SC-001–SC-010.
