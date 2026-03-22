# Pm4Research.Core

Standalone PM4 reading library for fresh-format exploration.

Goals:
- stay isolated from `MdxViewer` PM4 reconstruction heuristics
- stay isolated from `WoWMapConverter.Core` legacy/compatibility view models
- preserve raw chunk payloads and offsets for re-analysis
- expose typed decoders only for chunk layouts that are currently well understood

Non-goals:
- no viewer/world transform policy
- no CK24 object reconstruction
- no placement solving
- no reuse of existing PM4 reader classes as dependencies

Current typed coverage:
- `MVER`
- `MSHD`
- `MSLK`
- `MSPV`
- `MSPI`
- `MSVT`
- `MSVI`
- `MSUR`
- `MSCN`
- `MPRL`
- `MPRR`
- `MDBH`
- `MDBI`
- `MDBF`
- `MDOS`
- `MDSF`

Unknown or not-yet-re-decoded chunks remain available as raw payloads through `Pm4ChunkRecord`.

Current decode-confidence workflow:
- use `inspect-audit` on a single PM4 to verify chunk size/stride consistency and cross-chunk references before making reconstruction claims
- use `scan-audit` across the full PM4 corpus to distinguish recurring format structure from tile-specific payloads
- use `inspect-mslk-refindex` and `scan-mslk-refindex` when chasing the specific open seam where `MSLK.RefIndex` does not fit `MSUR`
- use `scan-mslk-refindex-classifier` to break the unresolved `MSLK.RefIndex` mismatch population into likely target-domain families instead of relying on raw fit counts alone
- use `scan-msur-geometry` when checking whether the current `MSUR` float fields really behave like plane normals and a plane-distance term
- use `scan-structure-confidence` when you need one corpus-backed answer to the deeper question "which chunk layouts are byte-level real, and which field meanings are still inherited guesses or conflicts?"
- use `scan-linkage` when checking whether the UI `Ck24ObjectId`, `MSLK.GroupObjectId`, and mismatch-family patterns are exposing a real hierarchy/member layer or only derived identity slices
- use `scan-mscn` when checking whether `MSCN` is acting as a real CK24-linked collision/ownership layer or only as an auxiliary point cloud
- use `scan-unknowns` to build one corpus-wide map of verified relationships, partial fits, field distributions, and still-open PM4 unknowns
- treat `00_00` as the populated destructible-building reference tile, not as the general PM4 chunk-distribution baseline

Companion workflow:
- `src/Pm4Research.Cli` provides `inspect`, `export-json`, and `scan-dir` commands on top of this library
- `src/Pm4Research.Cli` also provides `inspect-audit` and `scan-audit` for raw decode-confidence work over one file or the full corpus
- `src/Pm4Research.Cli` also provides `inspect-hypotheses`, `export-hypotheses`, and `scan-hypotheses` to generate object-candidate partitions across the whole PM4 corpus
- `src/Pm4Research.Cli` also provides `scan-hypotheses-ndjson` for line-oriented corpus export that is safe for very large PM4 datasets
- `scripts/analyze_pm4_reports.py` can consume `scan-dir` JSON or invoke the CLI directly over a PM4 directory
- `scripts/build_pm4_sqlite.py` ingests compact hypothesis reports into normalized SQLite tables for cross-tile correlation queries
- prefer `test_data/development/World/Maps/development/development_00_00.pm4` as the trusted reference tile for PM4 rediscovery work; it is dense, real-data backed, and the user has matching original ADT placements outside the repo for visual cross-checks

Important corpus finding from the new audit path:
- `MDBI` and `MDBF` are genuinely one-tile chunks in the current development corpus
- `MDBH`, `MDOS`, and `MDSF` appear as chunk containers in many files, but only one tile currently carries populated destructible-building payloads; the rest are empty or placeholder stubs
- do not generalize Wintergrasp destructible-building semantics from chunk presence alone; use populated-data counts instead
- targeted `MSLK.RefIndex` audit result:
	- `development_00_00.pm4` has zero `MSLK.RefIndex -> MSUR` mismatches, so it is not the general linkage-problem tile
	- `150` files in the development corpus contain `4553` total `MSLK.RefIndex` mismatches
	- those bad values almost never fit `MPRL`, so `RefIndex` does not look like a general hidden `MPRL` index
	- many bad values still fit within `MSLK`, `MSPI`, `MSVI`, and `MSCN` counts on the affected tiles, which makes those domains stronger next-step candidates than `MPRL`
- broad unknowns report result:
	- `MSUR -> MSVI`, `MSVI -> MSVT`, `MSLK -> MSPI`, `MSPI -> MSPV`, and `MDSF -> {MSUR, MDOS}` are currently the strongest verified raw PM4 relationships
	- `LinkId` is uniformly `0xFFFFYYXX` in the current corpus run
	- `MPRL.Unk02` is always `-1`, `Unk06` is always `0x8000`, `Unk14` spans `-1..15`, and `Unk16` collapses to `0x0000` / `0x3FFF`
	- `MSLK.MspiIndexCount` has no triangles-only evidence in the current corpus, but still has overlap cases where both interpretations fit
	- `MPRR` remains mixed/open; it fits both `MPRL` and `MSVT` heavily and cannot yet be named confidently
- MSCN relationship report result:
	- `MSUR.MdosIndex -> MSCN` is strong but not closed (`511891` fits, `6201` misses)
	- `1886 / 1895` CK24 groups have valid MSCN-backed node coverage
	- in the standalone raw path, raw MSCN bounds overlap CK24 mesh bounds far more often than swapped-XY MSCN bounds (`1162` vs `10` fits)
	- the current corpus does not support the old blanket claim that standalone MSCN is simply world-space plus XY swap
	- `MSLK.GroupObjectId` does not behave like a direct full CK24 key, and only weakly touches CK24 low-16 object ids
- linkage report result:
	- the UI `Ck24ObjectId` is just the low 16 bits of `MSUR.PackedParams -> CK24`
	- in current corpus data it is usually a near one-to-one slice of full CK24 inside a file, not a broadly reused hierarchy id (`2` reuse cases out of `1601` analyzed non-zero object-id groups)
	- both reuse cases occur on tile `36_24`, where one low16 object id survives across two full CK24 values and two type bytes
	- `MSLK.GroupObjectId` remains weak as the missing hierarchy/ownership answer for the unresolved `RefIndex` population (`16` low16 matches and `15` low24 matches across `4553` mismatches)
	- `58` files carry bad `MSUR.MdosIndex` references, including several large non-zero CK24 families, not only `CK24=0` aggregates
- structure-confidence report result:
	- current standalone PM4 chunk layouts are much stronger than their field names: `13` audited chunk families currently land in `high` layout confidence on the fixed corpus, with zero stride-remainder files across the active typed set
	- semantic confidence is still weaker than the layout story, but the new geometry audit materially improved one major seam: `2` tracked fields now land in `high` semantic confidence, `4` in `medium`, `9` in `low`, and `4` in `very-low`
	- the highest current hallucination-risk fields are now `MSLK.RefIndex`, `MPRR.Value1`, `MPRL.Unk04/14/16`, and sparse destructible payload fields such as `MDOS.DestructibleBuildingIndex`
	- the strongest current byte+semantic anchors are the vector/index streams (`MSPV`, `MSPI`, `MSVT`, `MSVI`), `MSUR -> MSVI`, `MDSF -> {MSUR, MDOS}`, and the newly validated `MSUR` plane fields
	- explicit conflict inventory now lives in the standalone report for fields where older notes overstated closure, including `MSLK.LinkId`, `MSLK.RefIndex`, `MSUR.MdosIndex`, `MSUR.Normal + Height`, the MSCN coordinate-frame claim, and `MPRR.Value1`
- MSUR geometry report result:
	- `518092 / 518092` analyzed surfaces have unit-length stored normals and strong positive alignment with geometry-derived polygon normals
	- the current `MSUR.Height` float behaves like the negative plane-distance term along the stored normal, not like a generic vertical height: best candidate `storedPlane.-` has mean absolute error `0.00367829`
	- this materially upgrades trust in `MSUR` bytes `4..19`, while also showing that the final float is semantically misdescribed by the current name `Height`
- RefIndex classifier result:
	- `505` mismatch families are now classified beyond pure ambiguity, covering `2651` of the `4553` mismatch rows
	- the largest resolved family population is `probable-MSVT` (`293` families), followed by smaller but important `probable/possible` families in `MSPI`, `MSPV`, `MSVI`, `MSCN`, and a small real `MPRL`-specific slice (`36` probable families)
	- this does not close `MSLK.RefIndex`, but it replaces the old all-or-nothing ambiguity with concrete target-family buckets
- placement-truth validation result using the existing active `pm4-validate-coords` path:
	- `206` tiles validated against real `_obj0.adt` placements on the fixed development dataset
	- `114301 / 114301` `MPRL` refs landed inside their expected tile bounds (`100.0%`)
	- `107907 / 114301` landed within `32` units of a nearest real placement (`94.4%`), with average nearest placement distance `10.98`
	- this materially strengthens `MPRL.Position` as real placement/footprint data; it does not by itself close `MPRR`

Current object-hypothesis families:
- `ck24`
- `ck24_mslk_refindex`
- `ck24_mdos`
- `ck24_connectivity`
- `ck24_mslk_refindex_mdos`
- `ck24_mslk_refindex_connectivity`
- `ck24_mdos_connectivity`
- `ck24_mslk_refindex_mdos_connectivity`

These are candidate object partitions, not a claim that any one family is the final PM4 truth.

Example commands:
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- inspect --input test_data/development/World/Maps/development/development_00_00.pm4`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- inspect-audit --input test_data/development/World/Maps/development/development_00_00.pm4`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- inspect-mslk-refindex --input test_data/development/World/Maps/development/development_00_00.pm4`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- inspect-hypotheses --input test_data/development/World/Maps/development/development_00_00.pm4`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-dir --input test_data/development/World/Maps/development --output output/pm4_reports/development_scan.json`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-audit --input test_data/development/World/Maps/development --output output/pm4_reports/development_decode_audit.json`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mslk-refindex --input test_data/development/World/Maps/development --output output/pm4_reports/development_mslk_refindex_audit.json`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mslk-refindex-classifier --input test_data/development/World/Maps/development --output output/pm4_reports/development_mslk_refindex_classifier_report.json`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-msur-geometry --input test_data/development/World/Maps/development --output output/pm4_reports/development_msur_geometry_report.json`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-structure-confidence --input test_data/development/World/Maps/development --output output/pm4_reports/development_structure_confidence_report.json`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-linkage --input test_data/development/World/Maps/development --output output/pm4_reports/development_linkage_report.json`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mscn --input test_data/development/World/Maps/development --output output/pm4_reports/development_mscn_relationship_report.json`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-unknowns --input test_data/development/World/Maps/development --output output/pm4_reports/development_unknowns_report.json`
- `dotnet run --project src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- pm4-validate-coords --input test_data/development/World/Maps/development --json output/pm4_reports/development_pm4_coordinate_validation_report.json`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-hypotheses --input test_data/development/World/Maps/development --output output/pm4_reports/development_object_hypotheses.json`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-hypotheses-ndjson --input test_data/development/World/Maps/development --output output/pm4_reports/development_object_hypotheses.ndjson`
- `python scripts/build_pm4_sqlite.py --pm4-dir test_data/development/World/Maps/development --sqlite output/pm4_reports/development_object_hypotheses.sqlite`
- `python scripts/analyze_pm4_reports.py --reports-json output/pm4_reports/development_scan.json --top 20`