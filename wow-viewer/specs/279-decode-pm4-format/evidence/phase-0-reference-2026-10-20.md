# Phase 0 receipt — current PM4 reference, 2026-10-20

## Delivered

[PM4: A Measured Format Reference](../../../docs/research/pm4-format-reference.md) now provides the
readable overview, evidence legend, graph relationships, all **16** currently parsed chunk
signatures/raw layouts, parser preservation limits, viewer/matching implications and **17**
unresolved question groups with next tests. The old wiki draft is retained with a supersession
notice. It is not published or re-certified as a fully decoded format.

Spec Kit created branch **279-decode-pm4-format** and this feature directory using the repository's
PowerShell script. The specification passed one quality-review iteration; plan/research/data/
contracts/quickstart/tasks are present. Existing Spec 270 evidence is adopted with its measured
scope, not presented as new proof. No reader, renderer or production matcher behavior changed.

## Scope and adoption

Operator explicitly requested continued full-format comprehension, the remaining unknowns, a nice
current writeup and Spec Kit. Epic 253 P-01/P-02 and P-03's documentation portion are routed here;
semantic code renames and consumer behavior changes remain gated. Prior PM4 code/evidence work
and unrelated dirty work were preserved; no staging, reset or commit occurred.

Fetched remote refs successfully (exit 0), queried remote heads successfully (exit 0), checked local
branches/spec numbers and found highest spec 278. Created the feature once:

```powershell
Set-Location 'wow-viewer'
& '.specify/scripts/powershell/create-new-feature.ps1' -Json -Number 279 -ShortName 'decode-pm4-format' 'Fully comprehend the PM4 format, including the things we do not understand, to improve the viewer and matching tooling; continue evidence-driven whole-map investigation and provide a clear current format writeup using Spec Kit.'
```

Exit 0; returned feature branch `279-decode-pm4-format`, spec path
`wow-viewer/specs/279-decode-pm4-format/spec.md`, number 279, Git enabled.

## Validation

[Actual validation output](reference-validation-2026-10-20.txt), exit **0**:

- 16/16 reader switch signatures appear in the reference inventory.
- 35 named fields in the five multi-field header/surface/link/point/reference sections are present
  within their appropriate section; shared vector/index and auxiliary layouts were source-reviewed.
- 17 unknown-register rows exist, covering the remaining supported families/semantic gaps.
- 34 local documentation links resolve at this validation snapshot; no unfilled template markers.
- Both adopted real-data reports contain all 616 files, have identical map fingerprints, and that
  fingerprint appears in the writeup.

The validation command was a repository-root `pwsh -NoProfile -Command` body, with strict errors,
captured through `2>&1 | Tee-Object -FilePath 'wow-viewer/specs/279-decode-pm4-format/evidence/reference-validation-2026-10-20.txt'`.
Its exact check body (line wrapping only) is:

```powershell
$doc='wow-viewer/docs/research/pm4-format-reference.md'
$text=Get-Content $doc -Raw
$reader=Get-Content 'wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4ResearchReader.cs' -Raw
$pattern='case '+[char]34+'([A-Z0-9]{4})'+[char]34
$chunks=@([regex]::Matches($reader,$pattern) | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
if($chunks.Count -ne 16){throw 'Reader inventory changed.'}
foreach($c in $chunks){if($text -notmatch ('(?m)^\| '+$c+' \|')){throw ('Missing chunk inventory row: '+$c)}}
$fields=@{MSHD=@('Field00','Field04','Field08','Field0C','Field10','Field14','Field18','Field1C');MSUR=@('GroupKey','IndexCount','AttributeMask','Padding','Normal','Height','MsviFirstIndex','_0x18','PackedParams');MSLK=@('TypeFlags','Subtype','Padding','GroupObjectId','MspiFirstIndex','MspiIndexCount','LinkId','RefIndex','SystemFlag');MPRL=@('Unk00','Unk02','Unk04','Unk06','Position','Unk14','Unk16');MPRR=@('Value1','Value2')}
foreach($c in $fields.Keys){$section=[regex]::Match($text,('(?ms)^### '+$c+'.*?(?=^### |\z)')).Value;foreach($field in $fields[$c]){if($section -notmatch [regex]::Escape($field)){throw ('Missing raw-field alias '+$c+'.'+$field)}}}
$unknowns=@([regex]::Matches($text,'(?m)^\| U[0-9]{2} \|'))
if($unknowns.Count -ne 17){throw 'Unknown register incomplete.'}
$docs=@($doc)+@(Get-ChildItem 'wow-viewer/specs/279-decode-pm4-format' -Filter '*.md' -Recurse | ForEach-Object FullName)
foreach($file in $docs){$content=Get-Content $file -Raw;if($content -match '\[NEEDS CLARIFICATION:|\[FEATURE NAME\]|\$ARGUMENTS'){throw ('Template marker in '+$file)};foreach($m in [regex]::Matches($content,'\[[^\]]+\]\(([^)]+)\)')){$target=$m.Groups[1].Value;if($target -match '^(https?://|file:|mailto:|#)'){continue};$target=$target.Split('#')[0];if(-not(Test-Path -LiteralPath (Join-Path (Split-Path $file -Parent) $target))){throw ('Broken local link '+$file+' -> '+$target)}}}
$a=Get-Content 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/map-cross-tile-paired-2026-10-20.json' -Raw | ConvertFrom-Json
$b=Get-Content 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/map-path-contact-2026-10-20.json' -Raw | ConvertFrom-Json
if($a.FileCount -ne 616 -or $b.FileCount -ne 616 -or $a.MapSha256 -ne $b.MapSha256 -or -not $text.Contains($a.MapSha256)){throw 'Corpus identity mismatch.'}
```

The first attempt exited 1 before running checks because the evidence directory was absent.
Created that repository-contained directory and reran successfully. No data/source repair occurred.
This is documentation/source/evidence validation, not new semantic or runtime behavior proof.

## Files changed

- This spec's requirements, quality checklist, plan, research/data/contracts/quickstart/tasks,
  validation output and this receipt; Spec Kit's feature pin points at this feature.
- [Current reference](../../../docs/research/pm4-format-reference.md).
- [Docs index](../../../docs/README.md), [wiki draft index](../../../docs/wowdev-wiki/README.md),
  [historical wiki draft notice](../../../docs/wowdev-wiki/pm4-pd4-draft.md).
- Epic 253 spec/plan/tasks, TRIAGE and STATUS routing; current memory-bank handoff/history.

## Criterion → evidence

| Criterion | Evidence | Limit |
|---|---|---|
| SC-001 supported layout coverage | Reader/source review + 16-signature and field-section checks | Layout coverage is not semantic completion |
| SC-002 claim levels/unknowns | Legend, evidence table, scoped relationship sections, 17 unknown groups | Existing reports are adopted, not universally revalidated |
| SC-003 traceable addressed example | Actual 0xFFFF0201 / surface-753 example and linked paired report | One corpus/build identity remains incomplete |
| SC-004 phase receipts | This receipt + exact commands/status and adopted real-file reports | Later research phases need their own receipts |
| SC-005 honest limits | Reference/spec explicitly leave point identity, ownership and full decode open | No viewer/accuracy improvement claimed |

**Gate R0:** documentation phase validated; next bounded phase is the raw link +0x04 domain/
reverse-link experiment. Future unchecked research and consumer tasks are not certified by this gate.
