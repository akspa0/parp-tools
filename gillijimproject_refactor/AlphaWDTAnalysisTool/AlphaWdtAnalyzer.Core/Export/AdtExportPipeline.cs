using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GillijimProject.WowFiles.Alpha;
using DBCTool.V2.Core;
using AlphaWdtAnalyzer.Core.Dbc;
using AlphaWdtAnalyzer.Core.Assets;

namespace AlphaWdtAnalyzer.Core.Export;

public static class AdtExportPipeline
{
    public sealed class Options
    {
        public string? SingleWdtPath { get; init; }
        public string? InputRoot { get; init; }
        public string? CommunityListfilePath { get; init; }
        public string? LkListfilePath { get; init; }
        public required string ExportDir { get; init; }
        public required string FallbackTileset { get; init; }
        public required string FallbackNonTilesetBlp { get; init; }
        public required string FallbackWmo { get; init; }
        public required string FallbackM2 { get; init; }
        public bool ConvertToMh2o { get; init; } = true;
        public bool AssetFuzzy { get; init; } = true;
        public bool UseFallbacks { get; init; } = true;
        public bool EnableFixups { get; init; } = true;
        public string?[]? AssetRoots { get; init; }
        public bool LogExact { get; init; } = false;
        public string? RemapPath { get; init; }
        public bool Verbose { get; init; } = false;
        public bool TrackAssets { get; init; } = false;
        // Optional DBCTool.V2 integration (when provided, enables AreaIdMapperV2)
        public string? DbdDir { get; init; }
        public string? DbctoolSrcAlias { get; init; } // e.g., 0.5.3 | 0.5.5 | 0.6.0
        public string? DbctoolSrcDir { get; init; }   // folder with source DBCs
        public string? DbctoolLkDir { get; init; }    // folder with 3.3.5 DBCs
        // Optional: Use precomputed DBCTool.V2 patch CSV(s)
        public string? DbctoolPatchDir { get; init; } // directory containing Area_patch_crosswalk_*.csv
        public string? DbctoolPatchFile { get; init; } // specific patch CSV file
    }

    public static void ExportSingle(Options opts)
    {
        if (string.IsNullOrWhiteSpace(opts.SingleWdtPath)) throw new ArgumentException("SingleWdtPath required", nameof(opts.SingleWdtPath));
        Directory.CreateDirectory(opts.ExportDir);

        var resolver = MultiListfileResolver.FromFiles(opts.LkListfilePath, opts.CommunityListfilePath);
        var mapName = Path.GetFileNameWithoutExtension(opts.SingleWdtPath!);
        var logDir = Path.Combine(opts.ExportDir, "csv", "maps", mapName);
        Directory.CreateDirectory(logDir);
        using var fixupLogger = new FixupLogger(Path.Combine(logDir, "asset_fixups.csv"));
        var inventory = new AssetInventory(opts.AssetRoots);
        var fixup = new AssetFixupPolicy(
            resolver,
            opts.FallbackTileset,
            opts.FallbackNonTilesetBlp,
            opts.FallbackWmo,
            opts.FallbackM2,
            opts.AssetFuzzy,
            opts.UseFallbacks,
            opts.EnableFixups,
            fixupLogger,
            inventory,
            opts.LogExact);

        var areaMapper = AreaIdMapper.TryCreate(null, null, null, opts.RemapPath);
        AreaIdMapperV2? areaMapperV2 = null;
        if (!string.IsNullOrWhiteSpace(opts.DbdDir) && !string.IsNullOrWhiteSpace(opts.DbctoolSrcDir) && !string.IsNullOrWhiteSpace(opts.DbctoolLkDir))
        {
            var alias = string.IsNullOrWhiteSpace(opts.DbctoolSrcAlias) ? "0.5.3" : opts.DbctoolSrcAlias!;
            areaMapperV2 = AreaIdMapperV2.TryCreate(opts.DbdDir!, alias, opts.DbctoolSrcDir!, opts.DbctoolLkDir!);
        }

        var wdt = new WdtAlphaScanner(opts.SingleWdtPath!);
        // Load DBCTool.V2 patch mapping if provided
        DbcPatchMapping? patchMap = null;
        if (!string.IsNullOrWhiteSpace(opts.DbctoolPatchFile) || !string.IsNullOrWhiteSpace(opts.DbctoolPatchDir))
        {
            patchMap = new DbcPatchMapping();
            if (!string.IsNullOrWhiteSpace(opts.DbctoolPatchFile) && File.Exists(opts.DbctoolPatchFile!))
                patchMap.LoadFile(opts.DbctoolPatchFile!);
            if (!string.IsNullOrWhiteSpace(opts.DbctoolPatchDir) && Directory.Exists(opts.DbctoolPatchDir!))
            {
                foreach (var f in Directory.EnumerateFiles(opts.DbctoolPatchDir!, "Area_patch_crosswalk_*.csv", SearchOption.TopDirectoryOnly))
                    patchMap.LoadFile(f);
            }
        }
        var adtScanner = new AdtScanner();
        var result = adtScanner.Scan(wdt);

        // Build union of tiles from placements and from WDT offsets (non-zero)
        var placementsByTile = result.Placements
            .GroupBy(p => (p.TileX, p.TileY))
            .ToDictionary(g => g.Key, g => (IEnumerable<PlacementRecord>)g);

        var candidateTiles = new HashSet<(int tx, int ty)>(placementsByTile.Keys);
        for (int adtNum = 0; adtNum < wdt.AdtMhdrOffsets.Count; adtNum++)
        {
            if (wdt.AdtMhdrOffsets[adtNum] > 0)
            {
                int x = adtNum % 64;
                int y = adtNum / 64;
                candidateTiles.Add((x, y));
            }
        }

        foreach (var (x, y) in candidateTiles.OrderBy(t => t.tx).ThenBy(t => t.ty))
        {
            var hasGroup = placementsByTile.TryGetValue((x, y), out var group);
            var g = hasGroup ? group! : Array.Empty<PlacementRecord>();

            IReadOnlyList<int>? alphaAreaIds = null;
            int adtNum = (y * 64) + x;
            int offset = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
            if (offset > 0)
            {
                var alpha = new AdtAlpha(wdt.WdtPath, offset, adtNum);
                alphaAreaIds = alpha.GetAlphaMcnkAreaIds();

                int currentMapId = ResolveMapIdByName(wdt.MapName);

                var ctx = new AdtWotlkWriter.WriteContext
                {
                    ExportDir = opts.ExportDir,
                    MapName = wdt.MapName,
                    TileX = x,
                    TileY = y,
                    Placements = g,
                    Fixup = fixup,
                    ConvertToMh2o = opts.ConvertToMh2o,
                    AreaMapper = areaMapper,
                    AreaMapperV2 = areaMapperV2,
                    PatchMapping = patchMap,
                    AlphaAreaIds = alphaAreaIds,
                    WdtPath = wdt.WdtPath,
                    AdtNumber = adtNum,
                    AdtOffset = offset,
                    MdnmFiles = wdt.MdnmFiles,
                    MonmFiles = wdt.MonmFiles,
                    Verbose = opts.Verbose,
                    TrackAssets = opts.TrackAssets,
                    CurrentMapId = currentMapId
                };
                AdtWotlkWriter.WriteBinary(ctx);
            }
        }
    }

    public static void ExportBatch(Options opts)
    {
        if (string.IsNullOrWhiteSpace(opts.InputRoot)) throw new ArgumentException("InputRoot required", nameof(opts.InputRoot));
        Directory.CreateDirectory(opts.ExportDir);

        var resolver = MultiListfileResolver.FromFiles(opts.LkListfilePath, opts.CommunityListfilePath);

        var wdts = Directory.EnumerateFiles(opts.InputRoot!, "*.wdt", SearchOption.AllDirectories)
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase);

        var inventory = new AssetInventory(opts.AssetRoots);

        foreach (var wdtPath in wdts)
        {
            try
            {
                var wdt = new WdtAlphaScanner(wdtPath);

                var mapName = Path.GetFileNameWithoutExtension(wdtPath);
                var logDir = Path.Combine(opts.ExportDir, "csv", "maps", mapName);
                Directory.CreateDirectory(logDir);
                using var fixupLogger = new FixupLogger(Path.Combine(logDir, "asset_fixups.csv"));

                var fixup = new AssetFixupPolicy(
                    resolver,
                    opts.FallbackTileset,
                    opts.FallbackNonTilesetBlp,
                    opts.FallbackWmo,
                    opts.FallbackM2,
                    opts.AssetFuzzy,
                    opts.UseFallbacks,
                    opts.EnableFixups,
                    fixupLogger,
                    inventory,
                    opts.LogExact);

                var areaMapper = AreaIdMapper.TryCreate(null, null, null, opts.RemapPath);
                AreaIdMapperV2? areaMapperV2 = null;
                if (!string.IsNullOrWhiteSpace(opts.DbdDir) && !string.IsNullOrWhiteSpace(opts.DbctoolSrcDir) && !string.IsNullOrWhiteSpace(opts.DbctoolLkDir))
                {
                    var alias = string.IsNullOrWhiteSpace(opts.DbctoolSrcAlias) ? "0.5.3" : opts.DbctoolSrcAlias!;
                    areaMapperV2 = AreaIdMapperV2.TryCreate(opts.DbdDir!, alias, opts.DbctoolSrcDir!, opts.DbctoolLkDir!);
                }
                // Load DBCTool.V2 patch mapping if provided
                DbcPatchMapping? patchMap = null;
                if (!string.IsNullOrWhiteSpace(opts.DbctoolPatchFile) || !string.IsNullOrWhiteSpace(opts.DbctoolPatchDir))
                {
                    patchMap = new DbcPatchMapping();
                    if (!string.IsNullOrWhiteSpace(opts.DbctoolPatchFile) && File.Exists(opts.DbctoolPatchFile!))
                        patchMap.LoadFile(opts.DbctoolPatchFile!);
                    if (!string.IsNullOrWhiteSpace(opts.DbctoolPatchDir) && Directory.Exists(opts.DbctoolPatchDir!))
                    {
                        foreach (var f in Directory.EnumerateFiles(opts.DbctoolPatchDir!, "Area_patch_crosswalk_*.csv", SearchOption.TopDirectoryOnly))
                            patchMap.LoadFile(f);
                    }
                }

                var adtScanner = new AdtScanner();
                var result = adtScanner.Scan(wdt);

                // Union of placements and WDT-present tiles
                var placementsByTile = result.Placements
                    .GroupBy(p => (p.TileX, p.TileY))
                    .ToDictionary(g => g.Key, g => (IEnumerable<PlacementRecord>)g);

                var candidateTiles = new HashSet<(int tx, int ty)>(placementsByTile.Keys);
                for (int adtNum = 0; adtNum < wdt.AdtMhdrOffsets.Count; adtNum++)
                {
                    if (wdt.AdtMhdrOffsets[adtNum] > 0)
                    {
                        int x = adtNum % 64;
                        int y = adtNum / 64;
                        candidateTiles.Add((x, y));
                    }
                }

                foreach (var (x, y) in candidateTiles.OrderBy(t => t.tx).ThenBy(t => t.ty))
                {
                    var hasGroup = placementsByTile.TryGetValue((x, y), out var group);
                    var g = hasGroup ? group! : Array.Empty<PlacementRecord>();

                    IReadOnlyList<int>? alphaAreaIds = null;
                    int adtNum = (y * 64) + x;
                    int offset = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
                    if (offset > 0)
                    {
                        var alpha = new AdtAlpha(wdt.WdtPath, offset, adtNum);
                        alphaAreaIds = alpha.GetAlphaMcnkAreaIds();

                        int currentMapId = ResolveMapIdByName(wdt.MapName);

                        var ctx = new AdtWotlkWriter.WriteContext
                        {
                            ExportDir = opts.ExportDir,
                            MapName = wdt.MapName,
                            TileX = x,
                            TileY = y,
                            Placements = g,
                            Fixup = fixup,
                            ConvertToMh2o = opts.ConvertToMh2o,
                            AreaMapper = areaMapper,
                            AreaMapperV2 = areaMapperV2,
                            PatchMapping = patchMap,
                            AlphaAreaIds = alphaAreaIds,
                            WdtPath = wdt.WdtPath,
                            AdtNumber = adtNum,
                            AdtOffset = offset,
                            MdnmFiles = wdt.MdnmFiles,
                            MonmFiles = wdt.MonmFiles,
                            Verbose = opts.Verbose,
                            TrackAssets = opts.TrackAssets,
                            CurrentMapId = currentMapId
                        };
                        AdtWotlkWriter.WriteBinary(ctx);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Export failed for {wdtPath}: {ex}");
            }
        }
    }

    private static int ResolveMapIdByName(string mapName)
    {
        // Minimal classic mapping; extend as needed
        if (mapName.Equals("Azeroth", StringComparison.OrdinalIgnoreCase)) return 0;
        if (mapName.Equals("Kalimdor", StringComparison.OrdinalIgnoreCase)) return 1;
        // fallback unknown
        return -1;
    }
}
