# Spec 052: PM4 Signature Matcher

**Status**: Draft | **Priority**: P1 | **Owner**: WoWViewer

## Problem

After spec 050 and spec 051, we have two ways to identify a PM4 object:

- **ADT placement lookup** (spec 050): reliable but only works if the WMO/M2 was placed in the tile's `_obj0.adt` for the same CK24+object.
- **PM4 signature** (spec 051): always present, contains MSCN/MSPV point counts, bounding box, and structural ratio (cyan:magenta). The signature is unique to each WMO/M2 model.

But we still don't have a **direct matching tool** that takes a selected PM4 object and returns the most likely WMO/M2 file by comparing signatures. Users have to manually flip between the viewer and a file explorer, eyeball MSCN/MSPV point clouds, and guess which `.wmo` or `.m2` file the data came from.

The PM4 signature is **stable per-model**: every instance of `World/wmo/transports/GOLDSHIREINN.wmo` has the same MSCN/MSPV counts and the same spatial distribution (up to the camera-window subset). So we can build a signature index of all WMO/M2 files in a staged client once, then for any selected PM4 object, return the best match.

## User Stories

### US1: Per-Object Signature Display (P1)

As a user, when I select a PM4 object and open the match panel, I want to see the object's signature: MSCN count, MSPV count, ratio, bounding box, surface density. This is the input to the matcher.

**Why this priority**: Without seeing the signature, the user can't understand what the matcher is doing. The signature is also useful on its own for diagnosing WMO/M2 structure (see spec 051 US7).

**Independent Test**: Click any PM4 object → match panel shows its signature. Move camera to a different tile with the same model → signature is the same.

**Acceptance Scenarios**:
1. **Given** a PM4 object is selected, **When** the user opens the match panel, **Then** it displays `MSCN: N, MSPV: M, ratio: R, bounds: (min) .. (max), density: D points/m³`.
2. **Given** two PM4 objects of the same model instance, **When** the user inspects both signatures, **Then** they have identical MSCN/MSPV counts (up to subset differences).
3. **Given** a PM4 object with MSCN=847, MSPV=234, **When** the user clicks "Refresh Signature", **Then** the panel re-computes the values from current `_pm4TileMscnPoints` and `_pm4TileMspvPoints` membership.

### US2: Client WMO/M2 Signature Index (P1)

As a user, when I set a staged client root, I want the viewer to build a signature index of every WMO and M2 file in that client once, cached to disk, so the matcher can search the index instantly.

**Why this priority**: Computing signatures for thousands of WMO/M2 files on every match would be too slow. The index must be built once, cached, and re-used across sessions.

**Independent Test**: Point viewer at a staged client → "Index" button → progress bar → "Index built: N WMOs, M M2s, K min". Subsequent matches return results in <100ms.

**Acceptance Scenarios**:
1. **Given** a staged client root is set, **When** user clicks "Build Signature Index", **Then** the viewer walks `world/wmo/**/*.wmo` and `world/m2/**/*.m2` and reads each file's MSCN+MSPV-like data (or approximate from MOGI/MSUR/M2 chunks), then writes a JSON index to `output/tmp/wmo_signature_index.json`.
2. **Given** the index exists for a client, **When** user reopens the viewer with the same client, **Then** the index is loaded from disk (no rebuild needed).
3. **Given** the index is loaded, **When** user clicks "Find Match" on a PM4 object, **Then** the matcher searches the in-memory index and returns top N candidates in <500ms.

### US3: Top-N Match Results (P1)

As a user, when I click "Find Match" on a PM4 object, I want to see the top N candidate WMO/M2 files ranked by signature similarity, with a confidence score for each.

**Why this priority**: This is the core deliverable. The user has been asking for "a matching tool" — this is it.

**Independent Test**: Click "Find Match" on a known WMO-type PM4 object → see top 5 candidates with scores. The top result should match the actual WMO file for that placement in most cases.

**Acceptance Scenarios**:
1. **Given** a PM4 object with known signature, **When** user clicks "Find Match", **Then** the match panel shows a ranked list of WMO/M2 candidates with model path, model name, and a similarity score 0.0-1.0.
2. **Given** the top match, **When** the user clicks "Frame" on a candidate, **Then** the camera frames the candidate's WMO/M2 asset (or its center if not loaded).
3. **Given** a match result, **When** the user clicks "Confirm Match", **Then** the match is saved to `wow-viewer/output/pm4_wmo_matches.json` (reusing spec 050's match store) and the panel shows the confirmed match as "✓ matched".

### US4: Signature Similarity Scoring (P1)

As a developer building the matcher, I want a multi-factor signature similarity score that combines:
- **MSCN count ratio** — `min(pm4Mscn, candidateMscn) / max(pm4Mscn, candidateMscn)` (penalizes wildly different polygon counts)
- **MSPV count ratio** — same formula on MSPV counts (penalizes different vertex sharing patterns)
- **MSCN:MSPV ratio similarity** — `1 - |pm4Ratio - candidateRatio| / max(pm4Ratio, candidateRatio)` (penalizes different structural classes)
- **Bounding box volume ratio** — `min(pm4Vol, candVol) / max(pm4Vol, candVol)` (penalizes very different object sizes)
- **Bounding box aspect similarity** — `1 - |pm4Aspect - candAspect| / max(pm4Aspect, candAspect)` (penalizes different object shapes)

**Why this priority**: The scoring formula is the core algorithm. Get this right and the matcher works; get this wrong and it returns garbage.

**Independent Test**: For a known PM4 object that came from `GOLDSHIREINN.wmo`, the matcher's top result should be that file with score > 0.7.

**Acceptance Scenarios**:
1. **Given** two signatures with identical MSCN, MSPV, ratio, and bounds, **When** the matcher scores them, **Then** the similarity is 1.0.
2. **Given** a PM4 object with MSCN=1000 and a WMO with MSCN=500, **When** the matcher scores them, **Then** the MSCN count ratio is 0.5, contributing a weighted 0.5 * weight_mscn to the final score.
3. **Given** a PM4 object with ratio 1.5 (connected WMO) and a WMO with ratio 0.1 (decoration), **When** the matcher scores them, **Then** the ratio similarity is low, penalizing the final score.

### US5: ADT Placement Pre-Filter (P2)

As a user, when I click "Find Match" on a WMO-type PM4 object, the matcher should first try ADT placement lookup (spec 050 FR-001) before falling back to signature search. If ADT placement returns a single match with high confidence, use that; otherwise fall through to signature search.

**Why this priority**: ADT placement is more reliable than signature matching when it's available. The ADT path tells us exactly which WMO file the PM4 object came from. The signature search is the fallback for when ADT placement doesn't exist (e.g., for PM4 objects in tiles without `_obj0.adt`, or for object types not in the ADT).

**Independent Test**: Click "Find Match" on a PM4 object whose tile has `_obj0.adt` with a matching WMO placement → the top result is the ADT-placement WMO, not a signature match.

**Acceptance Scenarios**:
1. **Given** a PM4 object in a tile with `_obj0.adt`, **When** user clicks "Find Match", **Then** the matcher first calls `Pm4WmoGroupMatchService.MatchFromPlacement()` and uses its top result if confidence > 0.7.
2. **Given** a PM4 object in a tile without `_obj0.adt`, **When** user clicks "Find Match", **Then** the matcher skips ADT lookup and goes directly to signature search.
3. **Given** ADT returns a low-confidence match (< 0.7), **When** user clicks "Find Match", **Then** the matcher falls through to signature search and the ADT result is shown as a secondary candidate.

### US6: M2 Support (P2)

As a user, when the PM4 object is an M2 type (Ck24Type 0x40 or 0x41), the matcher should also search the M2 signature index, not just WMO. The result panel should distinguish WMO matches from M2 matches.

**Why this priority**: Spec 050 was WMO-only. But spec 051 found that the M2 totem and tree stumps have rich PM4 signatures. M2s are just as matchable as WMOs by signature.

**Independent Test**: Click "Find Match" on the M2 totem (DRAGNOTOTEM01.M2) → top result is the actual M2 file in `world/m2/` with score > 0.6.

**Acceptance Scenarios**:
1. **Given** a PM4 object with Ck24Type 0x40 (M2), **When** user clicks "Find Match", **Then** the matcher searches both WMO and M2 indices, returning the top candidates from each.
2. **Given** a match result, **When** the user looks at the candidate, **Then** it's labeled `WMO` or `M2` based on which index it came from.
3. **Given** an M2 match, **When** user clicks "Confirm Match", **Then** the saved match entry includes the model kind (WMO/M2).

### US7: Match Result Confidence Levels (P2)

As a user, when I see match results, I want to know how confident the matcher is:
- **HIGH** (>0.8): the top match is very likely correct; auto-suggest it
- **MEDIUM** (0.5-0.8): the top match is plausible but could be wrong; show 3 candidates
- **LOW** (<0.5): no clear winner; show top 5 but flag for manual review

**Why this priority**: Confidence levels help the user know when to trust the matcher and when to look at the candidates themselves.

**Independent Test**: Run the matcher on a set of known PM4 objects → check that the confidence levels correlate with actual match accuracy.

**Acceptance Scenarios**:
1. **Given** a PM4 object whose signature has a near-exact match in the index, **When** the matcher returns, **Then** the top result has confidence HIGH (>0.8) and the panel shows a "✓ likely match" badge.
2. **Given** a PM4 object whose signature has only loose matches, **When** the matcher returns, **Then** the top result has confidence LOW (<0.5) and the panel shows 5 candidates.
3. **Given** a HIGH-confidence match, **When** the user clicks "Auto-confirm", **Then** the match is saved automatically without manual intervention.

### US8: Match Persistence Reuse (P3)

As a user, when I confirm a match (manual or auto), the confirmation is stored in `wow-viewer/output/pm4_wmo_matches.json` and shown as "✓ matched" the next time I select that PM4 object. On subsequent viewer sessions, the matcher checks this store first and returns the saved match as the top result.

**Why this priority**: Reuse spec 050's match store. New users get to start curating from saved matches.

**Independent Test**: Confirm a match → reload viewer → re-select the same PM4 object → match panel shows the saved match as the top result.

**Acceptance Scenarios**:
1. **Given** a match has been confirmed and saved, **When** the user reloads the viewer and re-selects the PM4 object, **Then** the match panel shows the saved match at the top with a "✓ saved" badge.
2. **Given** a saved match, **When** user clicks "Forget Match", **Then** the match is removed from `pm4_wmo_matches.json` and the next match attempt uses the live matcher.
3. **Given** no saved match, **When** user clicks "Find Match", **Then** the live matcher runs and saves nothing automatically.

## Functional Requirements

### FR-001: Pm4ObjectSignature Data Type
- `Pm4ObjectSignature` struct in `WowViewer.Core` or `WowViewer.Core.PM4`:
  - `MscnCount: int`
  - `MspvCount: int`
  - `MscnMspvRatio: float`
  - `MscnBoundsMin: Vector3`, `MscnBoundsMax: Vector3`
  - `Volume: float` (computed from bounds)
  - `SurfaceDensity: float` (MscnCount / max(1, Volume))
  - `AspectXY: float` (bounds size X/Y)
  - `AspectXZ: float` (bounds size X/Z)
  - `SignatureType: Pm4SignatureType` enum (`DisjointDecoration`, `SimpleMesh`, `ConnectedWmo`, `ContiguousM2`)
- Computed from PM4 object membership in `_pm4TileMscnPoints` and `_pm4TileMspvPoints` via spatial bounds containment

### FR-002: WorldScene.Signature API
- `bool TryGetPm4ObjectSignature((tileX, tileY, ck24, objectPart) key, out Pm4ObjectSignature signature)` on WorldScene
- Implementation: for each MSCN/MSPV point in the tile, check if the point is inside the object's bounds; if yes, increment the corresponding count
- Cache the signature per object part on first computation; invalidate on overlay reload

### FR-003: SignatureIndexBuilder Tool
- New tool: `WowViewer.Tool.Inspect pm4 build-index --client-root <path> --output <json>`
- Walks `world/wmo/**/*.wmo` and `world/m2/**/*.m2` under the client root
- For each file, reads MOHD/MOGI (WMO) or equivalent M2 vertex data to estimate:
  - Surface count (from MOGI entry count for WMO, or M2 polygon count for M2)
  - Vertex count (from MOHB or equivalent)
  - Overall bounds
- Writes JSON index with file path → signature mapping
- Index format:
  ```json
  {
    "clientPath": "...",
    "builtAt": "2026-06-08T...",
    "wmos": { "world/wmo/.../foo.wmo": { mscnCount, mspvCount, boundsMin, boundsMax, ... }, ... },
    "m2s": { "world/m2/.../bar.m2": { ... }, ... }
  }
  ```

### FR-004: Pm4SignatureMatcher Service
- `static class Pm4SignatureMatcher` in viewer project (similar pattern to Pm4WmoGroupMatchService)
- `static IReadOnlyList<Pm4MatchCandidate> FindMatches(Pm4ObjectSignature signature, SignatureIndex index, int topN = 5)`
- For each WMO and M2 in the index, compute similarity score using the formula in US4
- Return top N sorted by score descending
- `Pm4MatchCandidate` includes: file path, kind (WMO/M2), score, score breakdown per factor

### FR-005: UI Integration
- The "Find WMO Match" button (already in `DrawPm4SelectionWorkbenchContent` from spec 050) is extended to:
  1. Show the selected object's signature in the panel first
  2. Call ADT placement lookup (spec 050) — use if high confidence
  3. Call `Pm4SignatureMatcher.FindMatches` with the signature and loaded index — merge with ADT results
  4. Display top N candidates in a table with model path, kind, score
  5. "Confirm Match" button per candidate saves to `pm4_wmo_matches.json` (reuse spec 050's Pm4WmoMatchStore)
- "Build Index" button in the workbench runs the index builder tool against the active client root

### FR-006: Index Caching
- Index loaded from `output/tmp/wmo_signature_index.json` on viewer startup if it exists
- "Rebuild Index" button in the workbench
- Index staleness check: if the client root's mtime is newer than the index mtime, suggest rebuild

### FR-007: ADT Pre-Filter Integration
- The "Find WMO Match" flow first calls `Pm4WmoGroupMatchService.MatchFromPlacement`
- If ADT result has confidence > 0.7, use it as the top result
- Otherwise, fall through to signature search
- Both results are shown in the panel (ADT result labeled "From ADT", signature results labeled "From Signature")

## Success Criteria

1. The matcher's top-1 result is correct for >70% of PM4 objects with known WMO/M2 placements in a test corpus
2. Signature index builds in <5 minutes for a typical staged client (1500+ WMO files, 5000+ M2 files)
3. Index loads from disk in <1 second on viewer startup
4. "Find Match" returns top 5 candidates in <500ms against a loaded index
5. The signature similarity score formula is documented with weights tuned on at least 20 ground-truth PM4→WMO pairs

## Out of Scope

- Training a learned embedding for the signature (this is a hand-crafted formula for v1; learned comes later)
- Real-time WMO/M2 content extraction beyond MOHD/MOGI for the index (e.g., reading every M2 chunk is too slow)
- Cross-client signature matching (each client has its own index)
- Automated match correction when the user discovers the matcher is wrong
- Per-instance WMO scaling (the index stores the base WMO; scaled instances will have different bounds — fallback to ADT placement for those)
