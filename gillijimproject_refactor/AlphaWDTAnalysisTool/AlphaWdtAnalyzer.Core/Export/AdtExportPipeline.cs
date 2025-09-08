using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GillijimProject.WowFiles.Alpha;
using AlphaWdtAnalyzer.Core.Dbc;

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
        public string? AreaAlphaPath { get; init; }
        public string? AreaLkPath { get; init; }
        public string? DbcDir { get; init; }
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
        var fixup = new AssetFixupPolicy(
            resolver,
            opts.FallbackTileset,
            opts.FallbackNonTilesetBlp,
            opts.FallbackWmo,
            opts.FallbackM2,
            opts.AssetFuzzy,
            opts.UseFallbacks,
            opts.EnableFixups,
            fixupLogger);

        var areaMapper = AreaIdMapper.TryCreate(opts.AreaAlphaPath, opts.AreaLkPath, opts.DbcDir);

        // Auto-export DBCs to CSV when provided
        if (!string.IsNullOrWhiteSpace(opts.DbcDir) &&
            !string.IsNullOrWhiteSpace(opts.AreaAlphaPath) &&
            !string.IsNullOrWhiteSpace(opts.AreaLkPath))
        {
            var outCsvDir = Path.Combine(opts.ExportDir, "csv", "dbc");
            AreaTableDbcExporter.ExportAlphaAndLkToCsv(opts.AreaAlphaPath!, opts.AreaLkPath!, opts.DbcDir!, outCsvDir);
        }

        var wdt = new WdtAlphaScanner(opts.SingleWdtPath!);
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
                    AlphaAreaIds = alphaAreaIds,
                    WdtPath = wdt.WdtPath,
                    AdtNumber = adtNum,
                    AdtOffset = offset,
                    MdnmFiles = wdt.MdnmFiles,
                    MonmFiles = wdt.MonmFiles
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

        // Auto-export DBCs to CSV when provided (once per batch)
        if (!string.IsNullOrWhiteSpace(opts.DbcDir) &&
            !string.IsNullOrWhiteSpace(opts.AreaAlphaPath) &&
            !string.IsNullOrWhiteSpace(opts.AreaLkPath))
        {
            var outCsvDir = Path.Combine(opts.ExportDir, "csv", "dbc");
            AreaTableDbcExporter.ExportAlphaAndLkToCsv(opts.AreaAlphaPath!, opts.AreaLkPath!, opts.DbcDir!, outCsvDir);
        }

        var wdts = Directory.EnumerateFiles(opts.InputRoot!, "*.wdt", SearchOption.AllDirectories)
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase);

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
                    fixupLogger);

                var areaMapper = AreaIdMapper.TryCreate(opts.AreaAlphaPath, opts.AreaLkPath, opts.DbcDir);

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
                            AlphaAreaIds = alphaAreaIds,
                            WdtPath = wdt.WdtPath,
                            AdtNumber = adtNum,
                            AdtOffset = offset,
                            MdnmFiles = wdt.MdnmFiles,
                            MonmFiles = wdt.MonmFiles
                        };
                        AdtWotlkWriter.WriteBinary(ctx);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Export failed for {wdtPath}: {ex.Message}");
            }
        }
    }
}
