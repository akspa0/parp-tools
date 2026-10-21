# Spec 280 P0 / T002 — UI string inventory receipt

Date: 2026-10-21 UTC. Branch: `280-ui-rework`.
Source HEAD: `5336aacba74f0839b0332f040bd09aa6a4e1c9de`.

**Result:** pre-refactor static baseline captured from **353 viewer C# files**:
**3,489 rows / 3,300 callsites**, including **117 rows with lexical menu paths**.
This is source evidence, **not runtime completeness or functional verification**.
Only the inventory portion of P0 is delivered; the P0 gate is not claimed closed.

## Changed paths

- [Reusable PowerShell script](../../../eng/ui/Get-UiStringInventory.ps1)
- [Pre-refactor baseline](ui-strings-baseline.txt)
- [This receipt](p0-inventory.md)

No viewer source was edited, no commits were made, and no client was launched.
The parent's build/test run was neither restarted nor modified. Its results belong
to the separate T001 receipt. Existing unrelated dirty worktree files were preserved.
Task checkboxes, the spec, and continuity dashboards were not edited because they
are outside this assignment's explicit write scope.

## Capture and fingerprints

PowerShell **7.6.6**, its bundled Roslyn **5.0.0.0**. No Python, package restore,
SDK compilation, or network access is used by the inventory script.

- Source fingerprint: `D95BFAFCE9562DAA613B893AE71D522E6E390147553C0864CB41F647DA6DC44C`
- Baseline SHA-256: `F7C00CF3EB74DB10F4B294BF90721B830718AF3A8F4D06900344E009A759F5C1`
- Baseline: **3,499 physical lines** = 9 comment lines + 1 TSV header + 3,489 data rows.
- Kinds: **2,250 literal**, **736 interpolated**, **501 dynamic**, **2 unlabelled**.
- **63 inferred wrapper definitions**; **0 conditional-compilation directives** in this capture.

The source fingerprint hashes an ordinal-sorted manifest of repository-relative
file paths and each file's SHA-256, separated by tabs and terminated by LF. File
sets and hashes are checked again before output is written. Viewer source had no
diff against HEAD at final validation. The initial capture and final capture have
the same source fingerprint: the only intervening correction was to the extractor,
which had mistaken 21 two-argument menu enablement flags for shortcut strings.
Those false rows were removed and a regression self-check added before handoff.

## Commands run and outputs

All paths below are relative to the workspace root. Each command is PowerShell 7
compatible. Commands 1 and 2 were standalone CLI invocations; command 3 is the
exact final read-only payload passed to `pwsh -NoProfile -Command`.

### 1. In-memory extraction self-checks — exit 0

```powershell
pwsh -NoProfile -File wow-viewer/eng/ui/Get-UiStringInventory.ps1 -SelfTest
```

```text
SELF_TEST_PASS checks=16 rows=22
```

Covers comment/string-content exclusion; multiline/interpolated, verbatim and raw
strings; named arguments; nested menu ancestry and else-branch exclusion; hidden
ID suffixes; transitive wrappers/local initializers; hint escaping; draw-list and
tree overloads; boolean menu enablement; combo options; unlabelled images;
same-line callsites; repeatable sorting. These checks use an in-memory synthetic
fixture, not a real client.

### 2. Final baseline write — exit 0

```powershell
pwsh -NoProfile -File wow-viewer/eng/ui/Get-UiStringInventory.ps1 -OutputPath wow-viewer/specs/280-ui-audit-refactor/evidence/ui-strings-baseline.txt -Force
```

```text
UI_INVENTORY_WRITTEN files=353 rows=3489 callsites=3300 wrapper_definitions=63 conditional_directives=0
SOURCE_SHA256=D95BFAFCE9562DAA613B893AE71D522E6E390147553C0864CB41F647DA6DC44C
OUTPUT=wow-viewer/specs/280-ui-audit-refactor/evidence/ui-strings-baseline.txt
```

Force was used only to replace this assignment's earlier extractor output, with
viewer source unchanged. The default refuses overwriting an existing inventory.

### 3. Byte-for-byte regeneration and structural/source checks — exit 0

```powershell
$ErrorActionPreference = 'Stop'
& ./wow-viewer/eng/ui/Get-UiStringInventory.ps1 -OutputPath wow-viewer/specs/280-ui-audit-refactor/evidence/ui-strings-baseline.txt -Verify
$path = 'wow-viewer/specs/280-ui-audit-refactor/evidence/ui-strings-baseline.txt'
$lines = @(Get-Content $path)
$rows = @($lines | Where-Object { -not $_.StartsWith('#') } | ConvertFrom-Csv -Delimiter ([char]9))
Write-Output ('DATA_ROWS=' + $rows.Count + '; FILE_LINES=' + $lines.Count)
Write-Output ('BASELINE_SHA256=' + (Get-FileHash $path).Hash)
$rows | Group-Object kind | Sort-Object Name | ForEach-Object { 'KIND ' + $_.Name + '=' + $_.Count }
Write-Output ('MENU_PATH_ROWS=' + @($rows | Where-Object menu_path -ne '-').Count)
$prior = $null
foreach ($line in $lines | Select-Object -Skip 10) {
    if ($null -ne $prior -and [StringComparer]::Ordinal.Compare($prior,$line) -ge 0) { throw 'Not unique ordinal order' }
    $prior = $line
}
if (@($rows | Where-Object file -Match '(?i)/(?:[^/]*Defunct[^/]*|bin|obj)/').Count -ne 0) { throw 'Excluded path present' }
if (@($rows | Where-Object { $_.control -eq 'ImGui.MenuItem' -and $_.label -eq 'hasTerrain' }).Count -ne 0) { throw 'Boolean menu flag present' }
Write-Output 'STRUCTURE_PASS ordinal_unique=true excluded_rows=0 boolean_menu_rows=0'
git diff --exit-code -- wow-viewer/src/viewer
if ($LASTEXITCODE -ne 0) { throw 'Viewer source differs from HEAD' }
Write-Output 'VIEWER_SOURCE_DIFF_EXIT=0'
git status --short --untracked-files=all
```

Relevant real output:

```text
UI_INVENTORY_VERIFIED files=353 rows=3489 callsites=3300 wrapper_definitions=63 conditional_directives=0
SOURCE_SHA256=D95BFAFCE9562DAA613B893AE71D522E6E390147553C0864CB41F647DA6DC44C
OUTPUT=wow-viewer/specs/280-ui-audit-refactor/evidence/ui-strings-baseline.txt
DATA_ROWS=3489; FILE_LINES=3499
BASELINE_SHA256=F7C00CF3EB74DB10F4B294BF90721B830718AF3A8F4D06900344E009A759F5C1
KIND dynamic=501
KIND interpolated=736
KIND literal=2250
KIND none=2
MENU_PATH_ROWS=117
STRUCTURE_PASS ordinal_unique=true excluded_rows=0 boolean_menu_rows=0
VIEWER_SOURCE_DIFF_EXIT=0
```

Additional negative probes invoked the script with the existing baseline path
without Force, and with `-OutputPath ../ui-inventory-must-not-write.txt`.
Both child invocations exited **1**, as expected: respectively `Output already
exists` and `Output must be inside RepositoryRoot`. The asserting parent command
exited **0** and reported `BASELINE_UNCHANGED=true`. Neither probe wrote output.

## Criterion → actual evidence

| Bounded T002 criterion | Evidence |
|---|---|
| Inventory all active viewer source files, excluding Defunct/bin/obj | Recursive directory-pruned scan of 353 files under the viewer source tree; 0 excluded-path rows; aggregate source fingerprint above. "Active" here means non-excluded source, not runtime reachability. |
| Literal and dynamic visible-label/control callsites | 2,250 decoded literal rows, 736 interpolated rows and 501 dynamic-expression rows from the actual source; 3,300 callsites. Direct controls/text, 42 draw-list text rows, and inferred wrapper arguments are included. |
| Menu paths where feasible | 117 lexical-path rows. Actual examples: `File > Open File...`; `Tools > Export > Terrain > Alpha Masks > Current Tile Atlas (PNG)...`. Saved-client menu labels retain their interpolated source expressions. |
| Stable file/line/control/label ordering | Ordinal unique-order assertion passed. Line and column numbers are zero-padded; columns disambiguate same-line calls. UTF-8 without BOM, LF line endings, escaped TSV fields. In-memory regeneration matched output bytes. |
| Baseline before viewer refactor | Source fingerprint unchanged throughout extraction correction; viewer source diff exit 0; no viewer source writes. |
| Reusable for final inventory | Parameterized output path, protected overwrite, read-only verification, source-drift rejection, optional preprocessor symbols, and self-checks. Exact usage below. |

## Output contract and limitations

Columns: file, line, control, label, column, role, kind, menu_path. Sorting prioritizes
the first four columns, then the remaining columns. Locations are one-based at
the invocation start. The label column contains a decoded literal or the original
unevaluated expression; hidden ImGui ID suffixes are retained. Escaping is
backslash → double backslash, then tabs/CR/LF → backslash-t/r/n.

- This is an explicitly selected set of direct ImGui string/control sinks plus
  syntactically inferred wrappers, not a semantic/runtime UI registry. New APIs,
  aliases, indirect delegates, external/custom drawing and rendering outside this
  source tree may require extending the extractor.
- Wrapper inference follows string parameters, local initializers and forwarding
  calls to a fixed point. Matching is by method name/parameter shape, without type
  binding; same-name overloads, nested functions and ID-only parameters can produce
  conservative extra rows. Wrapper arguments and underlying sinks both appear.
- Dynamic values, returned labels, resource strings and data-fed values are not
  evaluated. Option arrays are recorded as expressions, not expanded. Numeric
  display-format strings and implicit wrapper-default values are not separately
  enumerated. Ambiguous dynamic second arguments on two-argument menu calls are
  omitted as possible boolean enablement flags, not asserted to be shortcuts.
- Container IDs and hidden labels are explicitly retained for deletion tracking;
  rows are not counts of distinct visible strings or actionable runtime controls.
  Unreachable/dead/duplicate code can remain in the scan; project compile inclusion
  and reachability are not evaluated.
- Menu paths only reflect positive enclosing menu-if branches in the same file;
  helper-generated menu ancestry, early-return patterns and call-graph parentage
  are not reconstructed. A dash means no inferred ancestry, not no runtime parent.
- Conditional branches follow supplied preprocessor symbols, none by default;
  there were no conditional directives in this source snapshot. Different sources
  may require additional runs with the relevant symbols.
- No claims about visible/control completeness, ownership, working/dead status,
  real files exported/imported, runtime appearance, FPS, or client compatibility.
  T003 and full-spec visible-control acceptance remain operator-owned.

## Exact reuse

From the repository root, retain the baseline and write a **new final artifact**:

```powershell
pwsh -NoProfile -File wow-viewer/eng/ui/Get-UiStringInventory.ps1 -OutputPath wow-viewer/specs/280-ui-audit-refactor/evidence/ui-strings-final.txt
```

Verify any existing artifact without writing:

```powershell
pwsh -NoProfile -File wow-viewer/eng/ui/Get-UiStringInventory.ps1 -OutputPath wow-viewer/specs/280-ui-audit-refactor/evidence/ui-strings-final.txt -Verify
```

Do not overwrite the baseline during later phases. Compare data columns and menu
paths, not just row counts: moved callsites change locations, and any source edit
changes the source fingerprint even if labels do not change. The final artifact
was not generated in this assignment.
