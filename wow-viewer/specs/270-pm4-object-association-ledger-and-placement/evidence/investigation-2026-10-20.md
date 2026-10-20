# PM4 continuation — sparse M2 evidence, not a WMO rewrite

Date: 2026-10-20. Baseline: `ea3ed2b6` on `v0.6.1-dev`; .NET SDK `10.0.303`, Windows.
Operator direction: continue PM4 investigation; treat old research as potentially coincidental,
preserve the working WMO behavior, and focus on M2's much sparser pathfinding geometry.

**Operator correction after the probe:** all PM4s must be read as a whole-map object, not only
individual tiles. Existing WMO matching is ~70% accurate (operator-reported), and important context
may remain undecoded even in parsed chunks. These five-file measurements are local diagnostics,
not map-wide matching accuracy or proof that connected fragments are complete M2 objects.

## Result

No production code, shared reader, source data, or restored map changed. A bounded evidence-only
probe read five explicitly selected PM4/companion ADT pairs. Input lengths and SHA-256 hashes,
all candidate records and measurements are in [the raw report](height-kind-2026-10-20.json).
The exact client build of this local `original_development` corpus is not established by this
pass; filenames are not a build fingerprint. No installed client root was consumed.

### Sparse M2 candidates

Only the unkeyed remainder is measured here. Components reproduce the current resolver's shared,
rounded XYZ cells at 0.25 yards, **not** a true Euclidean-epsilon weld. Components can include
terrain or unknown geometry and need not correspond one-to-one to placements.

| Tile | Authored M2s | Components | ≤2 triangles | ≤4 triangles | Distinct histograms, bins 1 / 0.25 | Within 24 yd of an M2 origin, real / reflected |
|---|---:|---:|---:|---:|---:|---:|
| 00_00 | 10 | 9 | 3 | 7 | 5 / 5 | 9 / 4 |
| 00_02 | 8 | 2 | 1 | 2 | 2 / 2 | 2 / 2 |
| 16_32 | 241 | 181 | 71 | 135 | 133 / 158 | 180 / 134 |
| 16_33 | 265 | 234 | 89 | 170 | 171 / 205 | 233 / 192 |
| 50_38 | 1 | 3 | 3 | 3 | 3 / 3 | 0 / 0 |

Tile 16_33 has zero authored WMOs. **170/234 components (72.6%) contain at most four triangles.**
Its default edge/area histogram has only 171 distinct values for 234 components. Finer bins
separate some observations (205 values), but that is **not** evidence of improved asset accuracy:
different instances/scales of the same asset can also separate.

The reflected control mirrors each component centroid about the remainder's XY bounds center,
keeping the authored placement set fixed. **192/234 (82.1%)** reflected centroids still fall within
24 yards of an M2 origin, compared with 233/234 observed centroids. This high null baseline rules
out treating simple 24-yard proximity as 99.6% identity accuracy. Reflection is one deterministic
control, not a calibrated false-positive rate or a proof that the observed correspondence is random.

Concrete follow-up sample: tile 16_33, component 937 is one triangle; its nearest mushroom origin
is 0.368 yards away, but **19 origins** lie within 24 yards. Its default histogram is shared by
nine components. Component 323 is also one triangle with nine default-bin peers; nearby tree and
mushroom paths disagree. These are ambiguous candidates, not confirmed mushroom/tree identities.

### Current kind-routing contradiction: observed, not a global format conclusion

Across these five files, 21 nonzero raw-word groups have bit-exact placement-Z candidates inside
their canonical XY footprint (±1 yard); all candidates are WMOs. Twenty groups have one such
candidate; one has two same-path WMO candidates. Sixteen groups would enter the current keyed-M2
branch because their top byte is 0x40 or 0x41. The report retains every candidate; it does not
choose an asset identity merely from height.

This is fresh local evidence against treating that byte as a universal M2 classifier. It does
not prove a complete field interpretation, extend to all builds, or authorize changing working
WMO matching.

The existing real-data test's premise is specifically contradicted: tile 50_38 has **two authored
WMOs and one M2**, not zero WMOs. Its four keyed surfaces (eight fan triangles) carry raw word
`0x41DE5F3F`, float 27.796507, identical to both `WALLPOST02.WMO` placement Z values. Thus the passing
assertion of an empty WMO result does not establish correct M2 classification. Neither wall post
is uniquely identified by that shared value.

Ten separately labeled synthetic controls reuse one real fingerprint, changing only declared
kind/top byte. They demonstrate current gate mechanics, **not** real asset matching: a query
declared M2 accepts a WMO self-reference at byte 0x42/0xC2; byte 0 permits either kind. No production
behavior was changed to make these controls pass.

## Verification receipts

All commands run from the repository root through `pwsh -NoProfile -Command`. The following are
the exact successful command bodies; pipelines were followed by `exit $LASTEXITCODE`.

### Probe — exit 0

```powershell
$ErrorActionPreference = 'Stop'
dotnet run --no-restore --project 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/height-kind-probe/HeightKindProbe.csproj' -c Debug -- 'wow-viewer/test_data/original_development/World/Maps/development' 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/height-kind-2026-10-20.json' 'development_00_00.pm4' 'development_00_02.pm4' 'development_16_32.pm4' 'development_16_33.pm4' 'development_50_38.pm4' 2>&1 | Tee-Object -FilePath 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/height-kind-probe-2026-10-20.log'
exit $LASTEXITCODE
```

Real output: “Wrote 5 paired tile reports and 10 synthetic gate controls”. Raw stdout is preserved
in [probe output](height-kind-probe-2026-10-20.txt); [compact measurements](sparse-m2-summary-2026-10-20.txt)
come from the report. Existing dependency vulnerability warnings were emitted; dependencies were
not changed or warnings disabled.

### Focused tests — exit 0, four passed

```powershell
dotnet test 'wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj' -c Debug --no-restore --filter 'FullyQualifiedName~Pm4M2SurfaceCorrelationTests' --logger 'console;verbosity=normal' 2>&1 | Tee-Object -FilePath 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/correlation-tests-2026-10-20.log'
exit $LASTEXITCODE
```

[Test output](correlation-tests-2026-10-20.txt): four tests passed, zero failed. Three tests use
synthetic geometry; the fourth asserts only an empty WMO resolution on 50_38. Its input exists
here, so the file-absence early return was not taken. The handoff's “7/7” claim is not supported
by the current suite. No real PM4-to-M2 asset identification was validated.

### Investigation setup failures (resolved, not hidden)

- Initial read-only Git command used a semicolon in the CMD shell: exit 129; rerun through
  PowerShell succeeded (exit 0). Only pre-existing `imgui.ini` was dirty at baseline.
- First probe build: exit 1, incorrect access to a candidate's kind in the new probe only;
  corrected to access its fingerprint. Shared code was untouched.
- First data run: exit 3762504530, missing companion for selected 00_01. It stopped without
  emitting a report or guessing a fallback. Replaced that sample with verified paired 00_02.
- `git diff --check`: exit 0 after probe/source/spec edits; final continuity check recorded below.
- Archive/continuity pass preserved the old progress detail but its first whitespace check exited 2
  for a new blank line at the live ledger's EOF; removed that blank line before the final check.
- The first archive-content verification exited 1 on Unicode differences from the Git baseline.
  Regenerated the archived tail directly from `HEAD`, preserving Unicode and rebasing relative links;
  the exact-content comparison then passed (exit 0). No historical content difference is accepted.

### Final containment/continuity checks

```powershell
git diff --check
git diff --exit-code HEAD -- wow-viewer/src wow-viewer/tests wow-viewer/tools
$r = Get-Content 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/height-kind-2026-10-20.json' -Raw | ConvertFrom-Json
$n = 0
foreach ($t in $r.Tiles) {
    foreach ($f in @($t.Pm4, $t.Adt)) {
        if ((Get-FileHash -LiteralPath $f.Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $f.Sha256) {
            throw ('Input hash changed: ' + $f.Path)
        }
        $n++
    }
}
if ($n -ne 10) { throw 'Expected ten input fingerprints.' }
$baseline = (@(git show HEAD:wow-viewer/memory-bank/progress.md) -join [char]10).Replace(([char]13).ToString(), '')
$expected = $baseline.Substring($baseline.IndexOf('## 2026-10-12', [StringComparison]::Ordinal)).TrimEnd().Replace('](../', '](../../').Replace('](archive/', '](')
$archive = (Get-Content 'wow-viewer/memory-bank/archive/2026-10-20-progress-pre-2026-10-14.md' -Raw).Replace(([char]13).ToString(), '')
$actual = $archive.Substring($archive.IndexOf('## 2026-10-12', [StringComparison]::Ordinal)).TrimEnd()
if ($expected -cne $actual) { throw 'Archived history differs beyond relative-link rebasing.' }
```

The commands are also run with explicit failure checks around the native Git invocations.
[Final output](final-verification-2026-10-20.txt) records source containment, ten unchanged source
hashes, archive preservation, and the compact live-ledger line count. The archive contains the
older 2026-10-12-and-earlier entries, not an audit/endorsement of their historical claims.

## Files changed

- [spec](../spec.md), [plan](../plan.md), [tasks](../tasks.md): dated continuation/correction and receipts.
- [probe project](height-kind-probe/HeightKindProbe.csproj), [driver](height-kind-probe/Program.cs),
  [sparse diagnostic](height-kind-probe/SparseM2Probe.cs): bounded, read-only evidence harness.
- [raw report](height-kind-2026-10-20.json), probe/test stdout and compact summary linked above.
- [active context](../../../memory-bank/activeContext.md), [progress](../../../memory-bank/progress.md),
  [PM4 workstream](../../../memory-bank/workstream-pm4-decode.md): current findings and operator correction.
- Older progress detail is retained in the indexed memory-bank archive; it is not deleted history.
- [Status router](../../STATUS.md) records the resumed investigation and separates it from shipped claims.

## Criterion → evidence

| Investigation criterion | Evidence | Limit |
|---|---|---|
| I01: audit code and coverage | ExtractWmoGroups/ExtractDoodadComponents top-byte branches; matcher gate; four named tests in test stdout | Source review, not a global field decode |
| I02: bounded real-file measurement | Five input pairs with ten SHA-256 hashes; all raw groups/candidates; per-component geometry and reflection controls in report | Candidate correspondences, not confirmed asset identities |
| I03: focused verification and proof gaps | Probe exit 0; four tests pass; raw outputs and setup failures above | No full-solution, GPU, visual or real asset-collision match proof |
| I04: synchronized handoff | Spec/tasks and memory-bank updates linked above | Production fixes remain separate from this evidence pass |

Original AC-1.x/2.x/3.x/4.x/5.x/6.x are **not re-certified** by this investigation. In particular,
original visual criteria and real surface-to-asset identity accuracy remain unproved here.

## Corrected next investigation — map context first

The earlier proposed isolated-component comparison was too narrow. First account for the entire
PM4 corpus as a map object, preserving per-file provenance and local reference domains. Audit which
parsed chunk relationships actually reach matching and which are discarded or still unexplained.
The new PM4 surface extractor accepts only MSVT/MSVI/MSUR plus caller labels; its triangle
histogram does not retain topology or relationships to MSLK/MSPV/MSPI/MSCN/MPRL/MPRR. Merely joining
all tile histograms would not restore that lost context.

Use working WMO correspondences as controls/anchors where independently supported, without treating
every current match as truth. Retained 16_33 components 937/323 are local diagnostic samples, not
confirmed independent objects. Revisit partial M2 collision overlap only with map-wide context,
exact-build assets, authored scale/rotation and wrong-asset controls. Preserve WMO matching; no
map-wide scan, new semantic decode, placement regeneration or real asset-identity proof occurred here.
