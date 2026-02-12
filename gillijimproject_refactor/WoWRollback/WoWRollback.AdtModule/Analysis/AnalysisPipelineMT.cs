using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using AlphaWdtAnalyzer.Core; // WdtAlphaScanner, ListfileLoader, PlacementRecord
using AlphaWdtAnalyzer.Core.Terrain; // Terrain extractors

namespace WoWRollback.AdtModule.Analysis;

internal static class AnalysisPipelineMT
{
    internal sealed class Options
    {
        public required string WdtPath { get; init; }
        public required string OutDir { get; init; }
        public string? ListfilePath { get; init; }
        public string? LkAdtDirectory { get; init; }
        public bool ExtractMcnkTerrain { get; init; } = true;
        public bool ExtractMcnkShadows { get; init; } = true;
        public int ClusterThreshold { get; init; } = 10;
        public int ClusterGap { get; init; } = 1000;
        public int DegreeOfParallelism { get; init; } = 0; // 0 -> auto
    }

    public static void Run(Options opts)
    {
        Directory.CreateDirectory(opts.OutDir);
        var csvDir = Path.Combine(opts.OutDir, "csv");
        Directory.CreateDirectory(csvDir);

        var wdt = new WdtAlphaScanner(opts.WdtPath);
        var dop = opts.DegreeOfParallelism > 0 ? opts.DegreeOfParallelism : Math.Max(1, Environment.ProcessorCount - 1);

        // Scan ADTs (multithreaded)
        var scanner = new AdtScannerMT();
        var scan = scanner.Scan(wdt, dop);

        // Missing assets by simple presence in the scan (listfile optional at this stage)
        var listfile = new ListfileLoader();
        // Prefer explicit path
        if (!string.IsNullOrWhiteSpace(opts.ListfilePath) && File.Exists(opts.ListfilePath))
        {
            listfile.Load(opts.ListfilePath);
        }
        else
        {
            // Try load a listfile next to OutDir if present (optional); safe if missing
            var possibleTxt = Path.Combine(Path.GetDirectoryName(opts.OutDir) ?? ".", "listfile.txt");
            var possibleCsv = Path.Combine(Path.GetDirectoryName(opts.OutDir) ?? ".", "listfile.csv");
            if (File.Exists(possibleTxt)) listfile.Load(possibleTxt);
            else if (File.Exists(possibleCsv)) listfile.Load(possibleCsv);
        }

        var missingWmo = scan.WmoAssets.Where(p => !listfile.TryGetIdByPath(p, out _)).ToList();
        var missingM2 = scan.M2Assets.Where(p => !listfile.TryGetIdByPath(p, out _)).ToList();
        var missingBlp = scan.BlpAssets.Where(p => !listfile.TryGetIdByPath(p, out _)).ToList();

        // Per-map CSV directory
        var mapCsvDir = Path.Combine(csvDir, wdt.MapName);
        Directory.CreateDirectory(mapCsvDir);

        // Write CSVs using existing writer for now [PORT]
        CsvReportWriter.WriteAssetsByType(mapCsvDir, scan.WmoAssets, scan.M2Assets, scan.BlpAssets);
        CsvReportWriter.WritePlacements(mapCsvDir, scan.Placements);
        CsvReportWriter.WriteMissing(mapCsvDir, missingWmo, missingM2, missingBlp);

        var ids = scan.Placements.Where(p => p.UniqueId.HasValue).Select(p => p.UniqueId!.Value);
        var clusters = UniqueIdClusterer.FindClusters(ids, opts.ClusterThreshold, opts.ClusterGap);
        CsvReportWriter.WriteIdRanges(mapCsvDir, clusters);
        CsvReportWriter.WriteUniqueIds(mapCsvDir, scan.Placements);

        // Terrain/shadows
        if (opts.ExtractMcnkTerrain)
        {
            var terrainEntries = !string.IsNullOrEmpty(opts.LkAdtDirectory) && Directory.Exists(opts.LkAdtDirectory)
                ? McnkTerrainExtractor.ExtractTerrainWithLkAreaIds(wdt, opts.LkAdtDirectory!)
                : McnkTerrainExtractor.ExtractTerrain(wdt);
            var terrainCsvPath = Path.Combine(csvDir, wdt.MapName, $"{wdt.MapName}_mcnk_terrain.csv");
            McnkTerrainCsvWriter.WriteCsv(terrainEntries, terrainCsvPath);
        }

        if (opts.ExtractMcnkShadows)
        {
            var shadowEntries = McnkShadowExtractor.ExtractShadows(wdt);
            var shadowCsvPath = Path.Combine(csvDir, wdt.MapName, $"{wdt.MapName}_mcnk_shadows.csv");
            McnkShadowCsvWriter.WriteCsv(shadowEntries, shadowCsvPath);
        }

        // Per-map index.json
        var mapOutDir = Path.Combine(opts.OutDir, wdt.MapName);
        Directory.CreateDirectory(mapOutDir);
        var idx = new AnalysisIndex
        {
            MapName = wdt.MapName,
            Tiles = scan.Tiles,
            WmoAssets = scan.WmoAssets.OrderBy(x => x).ToList(),
            M2Assets = scan.M2Assets.OrderBy(x => x).ToList(),
            BlpAssets = scan.BlpAssets.OrderBy(x => x).ToList(),
            Placements = scan.Placements,
            MissingWmo = missingWmo.OrderBy(x => x).ToList(),
            MissingM2 = missingM2.OrderBy(x => x).ToList(),
            MissingBlp = missingBlp.OrderBy(x => x).ToList()
        };
        File.WriteAllText(Path.Combine(mapOutDir, "index.json"), JsonSerializer.Serialize(idx, new JsonSerializerOptions { WriteIndented = true }));
    }
}
