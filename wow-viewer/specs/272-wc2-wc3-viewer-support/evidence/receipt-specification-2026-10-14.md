# Specification-Only Receipt — 2026-10-14

**Historical receipt — superseded scope:** This records the initial [WC2-first pass](../prior-scope.md), before the operator corrected priority to WC3 assets in the File Browser. Its command counts describe that earlier document, not the [current specification](../spec.md). Do not rerun its old count assertions as a current acceptance gate; use the [WC3 planning receipt](receipt-wc3-planning-2026-10-14.md).

**Outcome**: [Spec 272](../spec.md) is ready for bounded DOS WC2 implementation planning. No runtime implementation, game-compatibility result, or release commitment is claimed. [Quality checklist](../checklists/requirements.md): 16/16 pass by document review, supported by the automated checks below.

## Files Changed

- [Specification](../spec.md)
- [Operator addendum and GDS lineage clarification](../operator-addendum.md)
- [Local discovery evidence](discovery-2026-10-14.md)
- [Quality checklist](../checklists/requirements.md)
- [This receipt](receipt-specification-2026-10-14.md)
- [Feature pin](../../../.specify/feature.json)
- [Status registry](../../STATUS.md)
- [Continuity dashboard](../../../memory-bank/activeContext.md)
- [Progress ledger](../../../memory-bank/progress.md)

The original draft and pre-existing dirty viewer settings remain operator-owned and were not edited/staged. No runtime source or test files changed; no proprietary source/assets were copied. The feature pin previously targeted Epic 248 and now points to this spec, as required by the feature-creation script.

## Commands and Exit Status

1. Remote fetch and local/remote branch inventory: exit **0**. No exact short-name collision found. Highest active numbered spec was 271; 272 follows the workspace's global-number rule rather than restarting at 001.
2. From the viewer project directory, feature creation ran exactly once, exit **0**:

```powershell
pwsh -NoProfile -File .specify/scripts/powershell/create-new-feature.ps1 -Json -Number 272 -ShortName wc2-wc3-viewer-support 'Plan eventual Warcraft II original DOS asset and map support in the viewer first, with the operator WC3 viewing, texture ancestry, restoration, and coordinate bridging addendum as later milestones; investigate WC1 reuse and use local PlayStation source read-only as a reference.'
```

Actual script output identified branch 272-wc2-wc3-viewer-support, feature number 272, the specification in this directory, and Git availability as true. The branch is based on the starting v0.6.1-dev worktree.

3. Initial source search: exit **1** because a guessed map source filename did not exist. Corrected bounded archive/map/source discovery: exit **0**. [Discovery](discovery-2026-10-14.md) records the actual results and limitations.
4. Documentation and input-fingerprint validation from the workspace root ran through PowerShell 7, exit **0**. The exact command body passed to the child shell is shown below, directly runnable at a PowerShell 7 prompt:

```powershell
$ErrorActionPreference = 'Stop'; $feature = 'wow-viewer/specs/272-wc2-wc3-viewer-support'; git diff --check -- wow-viewer/.specify/feature.json wow-viewer/specs/STATUS.md wow-viewer/memory-bank/activeContext.md wow-viewer/memory-bank/progress.md; if ($LASTEXITCODE -ne 0) { throw 'Tracked whitespace check failed' }; $docs = @(Get-ChildItem -LiteralPath $feature -Recurse -File -Filter '*.md'); $linkCount = 0; foreach ($doc in $docs) { $text = Get-Content -LiteralPath $doc.FullName -Raw; if ($text -match '(?m)[ \t]+$') { throw ('Trailing whitespace: ' + $doc.Name) }; foreach ($m in [regex]::Matches($text,'\[[^\]]+\]\(([^)]+)\)')) { $target = $m.Groups[1].Value; if ($target -match '^(https?://|#)') { continue }; $line = 0; if ($target -match ':(\d+)$') { $line = [int]$Matches[1]; $target = $target -replace ':\d+$','' }; $path = Join-Path $doc.DirectoryName ($target -split '#')[0]; if (-not (Test-Path -LiteralPath $path)) { throw ('Broken link: ' + $doc.Name + ' -> ' + $target) }; if ($line -gt 0 -and (Get-Content -LiteralPath $path).Count -lt $line) { throw ('Invalid line reference: ' + $target) }; $linkCount++ } }; $spec = Get-Content -LiteralPath ($feature + '/spec.md') -Raw; foreach ($heading in @('## User Scenarios & Testing','## Requirements','## Success Criteria','## Assumptions')) { if (-not $spec.Contains($heading)) { throw ('Missing section: ' + $heading) } }; if ($spec -match 'NEEDS CLARIFICATION|\[FEATURE NAME\]|\$ARGUMENTS|ACTION REQUIRED') { throw 'Unresolved template text' }; $fr = [regex]::Matches($spec,'(?m)^- \*\*FR-\d{3}\*\*:').Count; $sc = [regex]::Matches($spec,'(?m)^- \*\*SC-\d{3}\*\*:').Count; $stories = [regex]::Matches($spec,'(?m)^### User Story \d+ -').Count; if ($fr -ne 15 -or $sc -ne 8 -or $stories -ne 6) { throw 'Unexpected requirement/scenario counts' }; $pin = Get-Content -LiteralPath 'wow-viewer/.specify/feature.json' -Raw | ConvertFrom-Json; if ($pin.feature_directory -ne 'specs/272-wc2-wc3-viewer-support') { throw 'Wrong feature pin' }; $evidence = Get-Content -LiteralPath ($feature + '/evidence/discovery-2026-10-14.md') -Raw; $data = 'C:\WarcraftResearch\WC2\War2'; $inputs = @(Get-ChildItem -LiteralPath $data -File -Filter '*.PUD'); $inputs += Get-Item -LiteralPath ($data + '\DATA\MAINDAT.WAR'),($data + '\DATA\REZDAT.WAR'); foreach ($inputFile in $inputs) { if (-not $evidence.Contains((Get-FileHash -LiteralPath $inputFile.FullName -Algorithm SHA256).Hash)) { throw ('Input fingerprint mismatch: ' + $inputFile.Name) } }; [pscustomobject]@{MarkdownFiles=$docs.Count;LocalLinks=$linkCount;UserStories=$stories;FunctionalRequirements=$fr;SuccessCriteria=$sc;InputFingerprints=$inputs.Count;TrackedWhitespace='PASS';Template='PASS';FeaturePin='PASS'} | ConvertTo-Json; git branch --show-current; git status --short
```

The host invoked that body using PowerShell 7's no-profile command option from the Windows command shell. Machine-local data paths here are receipt inputs only, not portable application/test defaults.

Actual output before this receipt/checklist completion was added:

```json
{
  "MarkdownFiles": 4,
  "LocalLinks": 27,
  "UserStories": 6,
  "FunctionalRequirements": 15,
  "SuccessCriteria": 8,
  "InputFingerprints": 10,
  "TrackedWhitespace": "PASS",
  "Template": "PASS",
  "FeaturePin": "PASS"
}
```

Git printed normal LF-to-CRLF notices, not whitespace errors. Hash rechecks matched both archive inputs and all eight loose maps. Document review additionally checked template order, operator-originated scope, every requirement's scenario/outcome mapping, the DOS correction, the GDS clarification, and the separation of technical assertions from user-facing requirements. No builds/tests/clients were run for this documentation-only change.

## Criterion → Current Evidence

| Feature criterion | Actual evidence available | Status / owner of missing proof |
|---|---|---|
| SC-001 DOS maps/terrain | Eight real map envelopes and dimensions; two real archive headers; ten matching input hashes | Open: no resource decode or terrain image; agent implementation, then operator visual check |
| SC-002 sprites | No decoded sprite | Open: agent implementation, then operator frame/color comparison |
| SC-003 safe failure/read-only | Read-only discovery and matching rechecked input hashes | Open: no viewer failure/recovery exercise or malformed-input suite yet |
| SC-004 classic WC3 | Scoped existing-code search and operator requirements only | Open: real WC3 inputs and viewing proof required |
| SC-005 matching | Requested comparison method retained in the addendum | Open: no comparison run/report |
| SC-006 synthesis/swapping | Requested modes retained; no generated textures | Open: real output, independent consumer, and operator seam/swap proof |
| SC-007 coordinates/bridging | Existing DAT height divisor and operator-supplied coordinate examples | Open: screenshots/origins/anchors and separate converted output needed |
| SC-008 WoW preservation/compatibility claims | No runtime source changes; unsupported variants remain explicitly gated | Open: runtime regression proof belongs to future implementation |

## Handoff

Use Spec Kit planning next for a bounded original-DOS WC2 archive/resource-discovery and terrain-preview slice, with shared-GDS WC1 reuse kept visible. Then author tasks with receipts and implement one validated phase at a time. WC3/restoration/coordinate work remains later scope; no release scheduling was changed.
