# Tasks: PM4 Correlation to World Assets & Generator

**Input**: `specs/065-pm4-correlation-to-world-assets/`

## Phase 1: Fingerprint Extraction Library — NOT STARTED

### Task 1.1 — Pm4FingerprintContracts.cs
Create `wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4FingerprintContracts.cs`:
- `Pm4FingerprintRecord` (AssetId, AssetPath, AssetKind, Ck24Type, SortedDim0/1/2, BoundsMin/Max, Center, FootprintHull, FootprintArea, SurfaceCount, VertexCount, IndexCount, TypeFlagsProfile, GroupCount, SourceLabel)
- `Pm4FingerprintDatabase` (records list + BuildDate + ArchiveRoot + WmoCount)
- `Pm4FingerprintMatchCandidate` (fingerprint record + Pm4CorrelationMetrics + rank + status)
- `Pm4FingerprintMatchResult` (CK24 group info + ranked candidates)
- JSON-serializable (System.Text.Json).

### Task 1.2 — Pm4FingerprintExtractor.cs
Create `wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4FingerprintExtractor.cs`:
- `ExtractFromGeometry(Vector3[] vertices, ushort[] indices, surfaceCount, ck24Type, typeFlagsProfile, assetId, assetPath, assetKind)` → `Pm4FingerprintRecord`
- PCA normalization: center at centroid, covariance of XY-projected points, eigen-decomposition → principal axes, rotation matrix to align. Try both flip candidates, keep the one that produces higher hull area (canonical orientation).
- Use `Pm4CorrelationMath.BuildFootprintHull` on PCA-normalized points.
- Compute sorted dims from AABB of PCA-normalized points.
- Handle degenerate cases: <3 unique points → null, zero-area hull → null, NaN → null.

### Task 1.3 — PCA normalization unit tests
Create `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4FingerprintExtractorTests.cs`:
- Test: known box (10×20×30) → sorted dims = [10,20,30], hull area = 200.
- Test: same box rotated 45° → PCA-normalized hull matches original hull (overlap ≥0.95).
- Test: L-shape → hull has correct vertex count and area.
- Test: degenerate (<3 points) → returns null.
- Test: near-symmetric square → both flip candidates tried, canonical one kept.

### Task 1.4 — Build + test
`dotnet build wow-viewer/WowViewer.slnx -c Debug` && `dotnet test wow-viewer/WowViewer.slnx -c Debug --filter FingerprintExtractor`
- [ ] Build passes
- [ ] Tests pass

---

## Phase 2: WMO Fingerprint Database — NOT STARTED

### Task 2.1 — WMO collision geometry loader
In `Pm4FingerprintExtractor.cs` (or a new `WmoFingerprintBuilder.cs` in Core.PM4):
- `BuildWmoFingerprints(string wmoRootPath, byte[] rootBytes, assetReader)` → root fingerprint + per-group fingerprints.
- Use `WmoRenderDocumentReader.Read` to get embedded groups with MOVT/MOVI.
- Merge all group vertices+indices for root fingerprint.
- Extract per-group fingerprint for multi-group WMOs.
- Skip WMOs with empty MOVT (warn).

### Task 2.2 — Fix WMO enumeration (506/1985 gap)
The archive catalog `GetAllKnownFiles()` misses ~75% of WMOs. Fix:
- Use listfile-based enumeration: read `componentfile.txt` or a provided listfile, filter `.wmo` entries that don't contain `_` (root files).
- Add `--listfile <path>` option to the CLI command as fallback.
- Verify: enumeration finds ≥1900 WMO roots (up from 506).

### Task 2.3 — CLI: pm4 build-wmo-fingerprint-db
In `Program.cs`, add `case "build-wmo-fingerprint-db"`:
- Args: `--archive-root <staged> --output <db.json> [--listfile <path>] [--limit <n>]`
- Enumerate WMO roots (archive catalog + listfile fallback).
- For each WMO: read root + groups, extract fingerprint, add to DB.
- Serialize DB to JSON. Report: total WMOs, successful, skipped, build time.

### Task 2.4 — Validate WMO DB
Run: `pm4 build-wmo-fingerprint-db --archive-root output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft --output wow-viewer/output/wmo-fingerprint-db.json`
- [ ] ≥500 WMO fingerprints (target: ≥1900 with listfile fix)
- [ ] GoldshireInn.wmo fingerprint has sorted dims ~30×32×60
- [ ] No crashes on malformed WMOs (skipped with warnings)
- [ ] Build time <10 minutes

---

## Phase 3: PM4 Fingerprint Extraction — NOT STARTED

### Task 3.1 — PM4 CK24 fingerprint extraction
In `Pm4FingerprintExtractor.cs`:
- `BuildPm4Fingerprints(Pm4ResearchDocument document, string sourcePath)` → list of `Pm4FingerprintRecord` per CK24 group.
- Group MSUR by CK24, collect MSVT/MSVI per group (reuse `Pm4CorrelateModelsSupport.ExportCk24GroupGeometry` logic, ported to library).
- Extract fingerprint via `ExtractFromGeometry` for each group.
- Compute TypeFlags profile from MSLK.TypeFlags per surface.

### Task 3.2 — CLI: pm4 extract-pm4-fingerprints
In `Program.cs`, add `case "extract-pm4-fingerprints"`:
- Args: `--input <dir> --output <fp.json> [--tiles <filter>]`
- Read each PM4, extract fingerprints, serialize.
- Report: total PM4s, total CK24 groups, fingerprints extracted, skipped.

### Task 3.3 — Validate PM4 fingerprints
Run: `pm4 extract-pm4-fingerprints --input test_data/development/World/Maps/development --output wow-viewer/output/pm4-fingerprints.json`
- [ ] 1604 CK24 group fingerprints
- [ ] Multi-tile OID 52202: cross-tile PCA-normalized hull overlap ≥0.90
- [ ] No crashes on degenerate groups
- [ ] TypeFlags profiles populated for groups with MSLK data

---

## Phase 4: Fingerprint Matching — NOT STARTED

### Task 4.1 — Pm4FingerprintMatcher.cs
Create `wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4FingerprintMatcher.cs`:
- `Match(IReadOnlyList<Pm4FingerprintRecord> pm4Fingerprints, Pm4FingerprintDatabase wmoDb, MatchOptions options)` → list of `Pm4FingerprintMatchResult`.
- Type-filter: 0x42/0x43/0xC0-0xC3 → WMO; 0x40/0x41 → M2 (or Ineligible if no M2 DB).
- Sorted-dim prefilter: reject >25% dim mismatch on any axis.
- For survivors: `Pm4CorrelationMath.EvaluateMetrics` on PCA-normalized hulls + bounds.
- `Pm4CorrelationMath.CompareCandidateScores` for ranking.
- Status: Matched (≥0.45, margin >0.03), Ambiguous (margin ≤0.03), Unresolved (<0.45), Ineligible.
- `MatchOptions`: MinScore, AmbiguousWindow, DimPrefilterTolerance, MaxCandidates.

### Task 4.2 — CLI: pm4 match-fingerprints
In `Program.cs`, add `case "match-fingerprints"`:
- Args: `--pm4-fingerprints <fp.json> --wmo-db <db.json> [--min-score <0.0-1.0>] [--max-candidates <n>] [--output <matches.json>]`
- Load both JSONs, run matcher, serialize results.
- Report: total groups, matched, ambiguous, unresolved, ineligible. Top matches table.

### Task 4.3 — Matcher unit tests
Create `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4FingerprintMatcherTests.cs`:
- Test: identical PM4 + WMO fingerprint → top-1 match, footprint overlap = 1.0.
- Test: PM4 fingerprint with no dim-compatible WMO → Unresolved.
- Test: two WMOs with same dims, different hulls → Ambiguous or correct disambiguation by hull overlap.
- Test: type mismatch (0x42 PM4 vs M2-only DB) → Ineligible.

### Task 4.4 — Validate matching on tile 24_35
Run: `pm4 match-fingerprints --pm4-fingerprints wow-viewer/output/pm4-fingerprints.json --wmo-db wow-viewer/output/wmo-fingerprint-db.json --output wow-viewer/output/pm4-wmo-matches.json`
- [ ] GoldshireInn.wmo is top-1 for the ~30×32×60 CK24 group on tile 24_35
- [ ] Footprint overlap ≥0.80 for that match
- [ ] Full 616-tile run completes in <60 seconds
- [ ] Match rate on PM4-only tiles ≈ match rate on ADT-backed tiles

---

## Phase 5: Validation Against ADT Ground Truth — NOT STARTED

### Task 5.1 — CLI: pm4 validate-matches
In `Program.cs`, add `case "validate-matches"`:
- Args: `--matches <matches.json> --pm4-dir <dir> --archive-root <staged> [--output <report.json>]`
- For each ADT-backed tile: read ADT placements, compute ground-truth CK24↔WMO pairs via `Pm4CorrelateModelsSupport.Correlate` (ADT-based, used ONLY here).
- Compare fingerprint-DB top-1/top-3 against ground truth.
- Report: precision@1, precision@3, coverage, failure categories (dim collision, PCA flip, wrong type, degenerate hull).

### Task 5.2 — Run validation + tune
Run: `pm4 validate-matches --matches wow-viewer/output/pm4-wmo-matches.json --pm4-dir test_data/development/World/Maps/development --archive-root output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft --output wow-viewer/output/match-validation-report.json`
- [ ] precision@1 ≥ 0.40 (baseline)
- [ ] precision@3 ≥ 0.60 (baseline)
- [ ] Failure categories documented
- [ ] If below baseline, tune prefilter/PCA/scoring and re-run

---

## Phase 6: PM4 Generator (downstream) — DONE (existing)

### Task 6.1 — Pm4Generator — DONE
`Pm4Generator.cs` exists with plane clustering + convex hull. `pm4 generate-from-wmo` CLI exists.

### Task 6.2 — Revisit if correlation findings require updates
Deferred until Phase 5 validation confirms the PM4↔WMO geometric relationship. If vertex-level comparison reveals a transform, update generator accordingly. Not blocking Phases 1-5.

---

## Legacy commands (kept, not primary)

- `pm4 identify-models` — sorted-dimension-only matching. KEPT for comparison. Not the primary matcher.
- `pm4 correlate-models` / `sweep-correlate` — ADT-based correlation. KEPT for validation ground truth (Phase 5). Not the primary matcher.
- `pm4 match-assets` — ADT-dependent scorer. KEPT but deprecated. `pm4 match-fingerprints` supersedes it.
- `pm4 fingerprint-scan` — old PM4 fingerprint scan (surfaces/indices/vertices only, no hull). Superseded by `pm4 extract-pm4-fingerprints`.
