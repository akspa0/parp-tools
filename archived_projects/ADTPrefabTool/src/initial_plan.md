# ADT Prefab Mining: Reverse-Subdivision and Neighborhood Pattern Discovery

This plan documents the approach to recover prefab-like terrain building blocks from WoW ADT tiles by treating chunks as tokens and mining recurring multi-chunk constellations (irregular shapes, cross-tile). It captures all context needed to implement without compilation issues.

## Objectives
- Identify recurring chunk constellations (2x2, 3x2, 3x3, 4x2, 4x3, etc.) across tiles, not limited to perfect squares.
- Be invariant to rotation/reflection when desired, robust to small local differences (wildcards).
- Support cross-tile detection and export candidates in normalized local space for visual QA.
- Provide diagnostics (counts, distributions) and NDJSON/JSON outputs for tooling.

## Current State Summary (Program.cs)
- Directory: `src/ADTPreFabTool.Console/Program.cs`
- Already implemented:
  - `ComputeChunkGrid8(MCNK)` returns normalized 8x8 height patch per chunk.
  - Similarity-only 64-bit chunk signature; prefab scan with 128-bit 2-channel block signature (height+gradient), 8-way canonicalization.
  - Global grid build in `ProcessPrefabDirectoryGlobal()` across tiles.
  - Ruggedness/edge-density filters; pairwise Hamming; OBJ exports in world coords.
- Gaps:
  - Rectangular windows only; duplicates; world-space exports hard to compare; features skew to slopes; no tokenization or neighborhood mining; no grouping/NMS.

## High-Level Approach
1) Tokenize each chunk into a compact code ("chunk token") derived from robust per-chunk features.
2) Build rolling neighborhood descriptors around each chunk for multiple small sizes (2x2, 3x2, 3x3, 4x2, 4x3), optionally canonicalized for 8 symmetries and with k wildcards.
3) Index descriptors and mine frequent/repeating patterns; seed candidates.
4) Region-grow seeds to recover irregular shapes spanning tiles; merge overlaps (NMS).
5) Export representative candidates in local coordinates; write detailed stats.

## CLI Additions
- `--prefab-patterns`                               Activate the new pattern-mining pass.
- `--pattern-sizes 2x2,3x2,3x3,4x2,4x3`            Neighborhood sizes.
- `--pattern-wildcards 1`                          Number of wildcard positions allowed per neighborhood.
- `--pattern-codebook 512`                         Chunk token codebook size (k-means clusters).
- `--pattern-canonicalize`/`--no-pattern-canonicalize`  8-way rotation/mirror canonicalization.
- `--prefab-local-export`                          Export meshes in normalized local space.
- `--prefab-export-grouped`                        One export per template/group (avoid pair duplicates).
- Reuse existing: `--tiles`, `--tile-range`, `--export-max`.

## Data Structures (C#)
- `struct ChunkFeat { int Gx, Gy; float[,] H8; float HeightRange; float AvgSlope; float CurvMean; float[] OriHist8; }`
- `struct ChunkToken { int Gx, Gy; int Code; }`
- `struct NeighborhoodKey { string Size; bool Canon; int MaskId; byte[] KeyBytes; }` // KeyBytes from token IDs + relative pos
- `class NeighborhoodIndex { Dictionary<NeighborhoodKey, List<(int Gx,int Gy)>> Occ; }`
- `class Codebook { int K; float[][] Centroids; }` // simple k-means over feature vectors
- `class PrefabCandidate { HashSet<(int Gx,int Gy)> Chunks; NeighborhoodKey SeedKey; int Frequency; float Score; }`

Feature vector for tokenization (per chunk):
- 8x8 height patch summary: mean, std, p10/p90 (4)
- Height range (1)
- Avg slope from 8x8 (1)
- Curvature mean via Laplacian on 8x8 (1)
- 8-bin gradient orientation histogram (8)
- Total dims ≈ 15 (kept small for speed). Keep raw 8x8 grid separately for export/embeddings.

## Algorithms

### 1) Build Global Grid
- Reuse logic from `ProcessPrefabDirectoryGlobal()`:
  - Load selected tiles → per-chunk `ComputeChunkGrid8()`; put into `Dictionary<(int gx,int gy), float[,]>`.

### 2) Compute Per-Chunk Features
- From 8x8 `H8`:
  - HeightRange = max-min.
  - AvgSlope: mean |dx|+|dy| using central differences.
  - CurvMean: mean Laplacian (dxx + dyy).
  - OriHist8: Sobel or simple dx/dy → atan2 bins (8 bins, L2-normalized).

### 3) Tokenization (Codebook)
- Collect feature vectors for all chunks in scan; run k-means (K=`--pattern-codebook`, default 512).
- Assign each chunk to nearest centroid → `ChunkToken` with Code ∈ [0..K-1].
- Persist centroids to run folder: `token_codebook.json` for reuse/debug.

### 4) Neighborhood Hashing
- For each `size ∈ --pattern-sizes` and for each origin (gx,gy), if all positions present:
  - Gather token IDs in row-major relative to origin.
  - Apply up to `--pattern-wildcards` omissions: generate 1..k masked variants (deterministic subset selection: center-first, then corners, etc.).
  - Canonicalization (optional): choose minimal hash among 8 symmetries (rot 0/90/180/270; mirror X + those).
  - Build `NeighborhoodKey` = (size, canonFlag, maskId, hashBytes).
  - Index: `Occ[key].Add((gx,gy))`.

HashBytes construction:
- Start with: [sizeId, maskId, tokenId0, tokenId1, ...] (byte[] with token IDs modulo 256; for K>256, use ushort[] and serialize to bytes).
- Apply a fast hash (e.g., FNV-1a 64-bit) for bucket; store full byte array as the canonical key to avoid collisions in-memory.

### 5) Seed Selection and Region Growing
- For each `key` with frequency ≥ 2 (and within top-N per size):
  - Seeds = `Occ[key]`.
  - For each seed, initialize group with its neighborhood footprint positions.
  - Greedy growth: iteratively add adjacent chunks if their neighborhoods (in any overlap with the current group boundary) have matching keys within wildcard tolerance; stop when no improvement.
  - Score = size (unique chunks) × average neighborhood support.

### 6) Non-Maximum Suppression (NMS)
- Sort candidates by Score desc; accept if Jaccard overlap with any accepted group < `0.3` (tunable), else discard.
- Optionally merge nearly identical groups (high overlap and similar extents).

### 7) Export and Reports
- Local-space export (`--prefab-local-export`):
  - Origin at (min gx, min gy), X-right (chunks), Z-down (chunks), Y=height (meters from chunk data).
  - Scale X/Z by `CHUNK_SIZE`, place per-chunk vertices using same triangulation logic as existing exporters but translated to local origin.
- Grouped export (`--prefab-export-grouped`):
  - Export one representative occurrence per unique `NeighborhoodKey` (or per grown group template), limit by `--export-max`.
- Write:
  - `prefab_stats.json`: counts for chunks, tokens, neighborhoods considered, seeds, grown, NMS kept, exports.
  - `prefab_candidates.ndjson`: one JSON per candidate with fields: id, size, seedKey (hex), chunkCount, bbox, frequency, score, occurrences (limited).

## File and Function Additions (Program.cs)
- Add new entry path in `Main()`:
  - If `--prefab-patterns` → call `ProcessPrefabPatternsGlobal(...)`.
- New helpers:
  - `static ChunkFeat ComputeChunkFeatures(float[,] h8)`
  - `static Codebook TrainKMeans(List<float[]> feats, int K, int maxIters=30)`
  - `static int AssignToken(float[] feat, Codebook cb)`
  - `static IEnumerable<(NeighborhoodKey key,(int gx,int gy) org)> EnumerateNeighborhoods(Dictionary<(int,int),int> tokens, List<(int w,int h)> sizes, int wildcards, bool canonicalize)`
  - `static List<PrefabCandidate> GrowAndSuppressCandidates(...)`
  - `static void ExportPrefabGroupLocal(... PrefabCandidate c, string outPath)`
  - `static byte[] BuildNeighborhoodBytes(...)` and `static NeighborhoodKey CanonicalizeNeighborhood(...)`

All new code will be placed in `Program.cs` near existing prefab scan functions to reuse context and avoid new files (ensures compilation).

## Diagnostics and Robustness
- Always output per-phase counters to console and `prefab_stats.json`.
- Guard all IO with try/catch; continue on per-tile/neighbor errors; log warnings.
- If KMeans gets too few points or K > points, reduce K automatically and warn.
- Cap neighborhood enumeration by scan bounds; skip at edges where positions are missing.

## Defaults and Tuning
- Default sizes: `2x2,3x2,3x3,4x2,4x3`.
- Default wildcards: `1`.
- Default codebook: `512` (reduce to 256 for small scans).
- Canonicalization: ON by default.
- Export grouped: ON by default when `--prefab-patterns` is used.
- Local export: ON by default when `--prefab-patterns` is used.

## Test Commands
- Small tile subset (developer toolbox regions):
```powershell
# From src/
dotnet run --project ADTPreFabTool.Console -- "..\test_data\060-maps\World\Maps\Azeroth" ^
  --timestamped --tiles 29_40,30_40,31_40,32_40,29_41,30_41,31_41,32_41 ^
  --prefab-patterns --pattern-sizes 2x2,3x2,3x3,4x2,4x3 --pattern-wildcards 1 --pattern-codebook 512 ^
  --prefab-export-grouped --export-max 50 --prefab-local-export
```

- Wider range:
```powershell
dotnet run --project ADTPreFabTool.Console -- "..\test_data\060-maps\World\Maps\Azeroth" ^
  --timestamped --tile-range 29,40,32,42 ^
  --prefab-patterns --pattern-sizes 2x2,3x2,3x3,4x2,4x3 --pattern-wildcards 1 --pattern-codebook 256 ^
  --prefab-export-grouped --export-max 80 --prefab-local-export
```

## Acceptance Criteria
- CLI parses and routes to new pass without breaking existing modes.
- Runs on sample datasets without exceptions; writes stats and candidates files.
- Exports at least several distinct local-space OBJ groups from toolbox regions.
- Diagnostics show non-zero neighborhoods, seeds, grown groups, and NMS-kept results.

## Future Enhancements
- Learned embeddings via TorchSharp (tiny CNN over 8x8 to 16–32D), contrastive training from frequent neighborhoods.
- Template library ingestion to match known prefabs; nearest-neighbor search per location.
- Mask library for common shapes; masked hashing and matching.
- Optional approximate indexing (LSH/ANN) for larger scans.

## Notes
- Respect ADT coordinate conventions already in use for triangulation; local export only translates to a normalized origin; vertex heights preserved.
- Keep rotation/reflection canonicalization consistent across hashing and export labels.
- Avoid exporting both sides of pairs; prefer group-wise representative exports.
