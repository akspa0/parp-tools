# CASC-1 receipt — local CASC read path

Date: 2026-10-22 · Spec 275 Phase 0 (T022–T026) · Owner: agent (source, tests, real-install read benchmark)

## Origin

Operator: "stream data from casc a bit better, as that is holding the CASC support back in terms of
performance." Operator-originated; recorded as an amendment in [spec.md §5](../spec.md).

## Where the cost was

`BuildInstance.OpenFileByFDID` → `CDN.GetFileFromArchive` → `DownloadFileFromArchive` →
`CDN.TryGetLocalFile` (TACTSharp, `libs/wowdev/TACTSharp/TACTSharp/CDN.cs:326`). Per file it does a
`FileInfo.Length` stat, `MemoryMappedFile.CreateFromFile` over the **whole** `data.NNN` archive (up to
~1 GB), `CreateViewAccessor(0, 0)`, a copy of the entry, unmap and close; `DownloadFileFromArchive`
then `ToArray()`s it again before BLTE decode. The upstream comment says caching the mappings was
avoided on purpose ("battle.net scares me"). TACTSharp is the upstream `wowdev` submodule and is
left untouched.

Synthetic probe (`evidence/read-cost-probe/`, 768 MB scratch file, warm OS cache, 24 logical processors):

| Read | Mapped per read (TACTSharp pattern) | Cached handle + positional read |
|---|---|---|
| 4 KB, 1 thread | 385.6 µs | 7.0 µs |
| 4 KB, 8 threads | 260.5 µs (3,839 reads/s) | 8.6 µs (116,501 reads/s) |
| 64 KB, 8 threads | 252.2 µs | 18.5 µs |
| 1 MB, 8 threads | 368.4 µs | 169.1 µs |

Slices were identical in 200 random comparisons. This is a mechanism measurement, not a viewer FPS result.

## What changed

| File | Change |
|---|---|
| `src/core/WowViewer.Core.IO/Casc/LocalCascArchiveReader.cs` (new) | Opens the newest `.idx` per bucket with TACTSharp's public `CASCIndexInstance` (same bucket rule and offset/size arithmetic), reads entries through one cached read-only handle per `data.NNN` (`FileShare.ReadWrite \| FileShare.Delete`, `RandomAccess.Read`). Returns false on any miss or short read. |
| `src/core/WowViewer.Core.IO/Casc/CascStorage.cs` | Fast path first (root → content key → encoding key → local read → `BLTE.Decode`, the same chain as `OpenFileByFDID`), stock path on any miss or exception. The first 64 fast reads are compared byte-for-byte with the stock read; a difference disables the fast path, logs once and returns the stock bytes. `PARP_CASC_FAST_READS=0` starts it off. `CascReadStats`, `IDisposable`. |
| `src/viewer/WoWViewer/DataSources/CascDataSource.cs` | `ReadStats` (loader misses, prefetch read time), log hook, disposes its storages. |
| `src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.Sidebars.cs` | Perf panel: "CASC reads" and "loader misses" lines. |
| `tools/inspect/WowViewer.Tool.Inspect/CascCommandSupport.cs` | `casc bench` read A/B (stock vs fast, sequential and parallel, byte-for-byte pass) and `--read-ab-only`. |
| `tests/WowViewer.Core.Tests/LocalCascArchiveReaderTests.cs` (new) | 9 tests on a synthetic install with the real `.idx` layout. |

The harvest tool and every other `CascStorage` user get the fast path automatically; the shadow check
protects them the same way.

## Verification

Run from `wow-viewer`, PowerShell 7.

| Command | Result |
|---|---|
| `dotnet build ./src/core/WowViewer.Core.IO/WowViewer.Core.IO.csproj -c Debug` | Exit 0, 0 errors |
| `dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-build --filter 'FullyQualifiedName~LocalCascArchiveReaderTests'` | Exit 0, **9/9**. First run failed 5 tests: TACTSharp's `GetIndexInfo` returns "not found" when the lower bound is the first entry of an index (upstream quirk, one entry per bucket); the synthetic indices now start with a sentinel entry, as real indices effectively do. The reader calls the same lookup, so both behave identically. |
| `dotnet build ./src/viewer/WoWViewer/WoWViewer.CrossPlatform.csproj -c Debug -p:OutDir=<scratchpad>` | Exit 0, 0 errors (the operator's viewer was running and locks `bin\`) |
| `dotnet build ./tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Release` | Exit 0, 0 errors |
| `dotnet test ... WowViewer.Core.Tests.csproj -c Debug --no-build` (full suite) | Run 1: 8 failed / 1,917 passed. Run 2: **7 failed / 1,918 passed / 1 skipped** — the 7 are exactly the pre-existing set in `specs/280-ui-audit-refactor/evidence/p0-test.log`. The 8th in run 1, `MccvTerrainShadowServiceTests.ExtractMccvTileLuminance_RealClassicBetaTile…`, passed in all 5 reruns (alone, with its class, and 3 more times with `TerrainHoleInvestigationTests`) and is **unexplained**; its failure text was not captured. |
| the 17 real-install tests (`MccvTerrainShadowServiceTests` + `TerrainHoleInvestigationTests`), fast path on ×3 / off ×1 | All pass; 109–116 s on vs 170 s off (end-to-end, includes non-read work). |
| `WowViewer.Tool.Inspect.exe casc bench --install <wow_classic_beta install> --product wow_classic_beta --cache <viewer cache> --wdt-id 775971 --listfile <community listfile> --threads 8 --read-ab-only` | Exit 0. Output below and in [casc-read-ab-azeroth-2026-10-22.txt](casc-read-ab-azeroth-2026-10-22.txt). |

### Real-install read A/B (`wow_classic_beta`, Azeroth file set, Release, warm OS cache)

```text
file set: 521 WMO roots + groups = 3179 files, 3111 M2 + skins; total 9401
read A/B stock (map archive per read) sequential     2407 ms (   3905 files/s, 406.3 MB)   parallel x8      941 ms (   9989 files/s)
read A/B fast (cached handles)        sequential      844 ms (  11143 files/s, 406.3 MB)   parallel x8      154 ms (  61220 files/s)
read A/B byte-for-byte over 9401 files: 9401 identical, 0 DIFFERENT, 0 unreadable by both
read A/B counters: fast 41973 (avg 0.118 ms), stock 28203 (avg 0.432 ms), shadow-checked 64, shadow mismatches 0, fast path on
```

Sequential 2.9×, parallel ×8 6.1×. The remaining ~0.12 ms per fast read is index/encoding lookup, BLTE
(zlib) decode and allocation, not file I/O.

## Criterion → evidence

| Criterion | Evidence | Status |
|---|---|---|
| FR-CASC-1 byte-identical, falls back on miss | 9 synthetic tests incl. parity with `CDN.TryGetLocalFile` (220 entries, 2 MB entries, 8 threads); 9,401 real files identical; fallback is the unchanged stock path | PASS (CDN-fill fallback not exercised offline) |
| FR-CASC-2 self-check and switch | Shadow check of 64 reads, 0 mismatches on real data; disable-on-mismatch and the env switch are code paths **not exercised by a test** (no injection seam) | Partial: read the code, no test |
| FR-CASC-3 visible in Perf panel | Perf panel lines added; compiles | **Operator to confirm on screen (T027)** |
| AC-009 | A/B above | PASS on this install and file set |
| SC-CASC-1 | Not claimed | **OPEN — operator (T027)** |

## Known limits

- The A/B used a warm OS cache; a cold cache is dominated by disk reads, which this change does not touch.
- Viewer FPS/hitch effect is not measured. The 2.9×/6.1× are read-path numbers on one file set.
- Entries larger than 512 MB, or archives the index points at but the disk lacks, fall back to the stock path.
- File handles close on `CascStorage.Dispose` (called from `CascDataSource.Dispose`); standalone tool use relies on process exit.
- Build warnings were not triaged.
