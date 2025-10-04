using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using AlphaWdtAnalyzer.Core.Terrain;

namespace AlphaWdtAnalyzer.Core;

public static class AnalysisPipeline
{
    public sealed class Options
    {
        public required string WdtPath { get; init; }
        public required string ListfilePath { get; init; }
        public required string OutDir { get; init; }
        public int ClusterThreshold { get; init; } = 10;
        public int ClusterGap { get; init; } = 1000;
        public string? DbcDir { get; init; }
        public string? AreaAlphaPath { get; init; }
        public string? AreaLkPath { get; init; }
        public bool ExtractMcnkTerrain { get; init; } = false;
        public bool ExtractMcnkShadows { get; init; } = false;
    }

    public static void Run(Options opts)
    {
        Directory.CreateDirectory(opts.OutDir);
        var csvDir = Path.Combine(opts.OutDir, "csv");
        Directory.CreateDirectory(csvDir);

        // Load listfile
        var listfile = new ListfileLoader();
        listfile.Load(opts.ListfilePath);

        // Scan WDT
        var wdt = new WdtAlphaScanner(opts.WdtPath);

        // Scan ADTs
        var adtScanner = new AdtScanner();
        var adt = adtScanner.Scan(wdt);

        // Compute missing
        var missingWmo = adt.WmoAssets.Where(p => !listfile.TryGetIdByPath(p, out _)).ToList();
        var missingM2 = adt.M2Assets.Where(p => !listfile.TryGetIdByPath(p, out _)).ToList();
        var missingBlp = adt.BlpAssets.Where(p => !listfile.TryGetIdByPath(p, out _)).ToList();

        // UniqueID clusters (global, from any placements that have UniqueId)
        var ids = adt.Placements
            .Where(p => p.UniqueId.HasValue)
            .Select(p => p.UniqueId!.Value);
        var clusters = UniqueIdClusterer.FindClusters(ids, opts.ClusterThreshold, opts.ClusterGap);

        // Write CSVs
        CsvReportWriter.WriteAssetsByType(csvDir, adt.WmoAssets, adt.M2Assets, adt.BlpAssets);
        CsvReportWriter.WritePlacements(csvDir, adt.Placements);
        CsvReportWriter.WriteMissing(csvDir, missingWmo, missingM2, missingBlp);
        CsvReportWriter.WriteIdRanges(csvDir, clusters);
        CsvReportWriter.WriteUniqueIds(csvDir, adt.Placements);

        // Per-map ID clusters and range summaries
        var perMapClusters = new List<(string MapName, UniqueIdClusterer.Cluster Cluster)>();
        var perMapSummaries = new List<(string MapName, int MinId, int MaxId, int Count)>();
        var byMap = adt.Placements
            .Where(p => p.UniqueId.HasValue)
            .GroupBy(p => p.MapName, StringComparer.OrdinalIgnoreCase);
        foreach (var g in byMap)
        {
            var idList = g.Select(p => p.UniqueId!.Value);
            var mapClusters = UniqueIdClusterer.FindClusters(idList, opts.ClusterThreshold, opts.ClusterGap);
            perMapClusters.AddRange(mapClusters.Select(c => (g.Key, c)));
            if (idList.Any())
            {
                var minId = idList.Min();
                var maxId = idList.Max();
                var count = idList.Count();
                perMapSummaries.Add((g.Key, minId, maxId, count));
            }
        }
        CsvReportWriter.WriteIdRangesByMap(csvDir, perMapClusters);
        CsvReportWriter.WriteIdRangeSummaryByMap(csvDir, perMapSummaries);
        if (ids.Any())
        {
            CsvReportWriter.WriteIdRangeSummaryGlobal(csvDir, ids.Min(), ids.Max(), ids.Count());
        }

        // Optional DBC scanning and crosswalk have been removed from this pipeline.
        // They are performed (when needed) via the DBCD-backed export flow.

        // Extract MCNK terrain data if requested
        if (opts.ExtractMcnkTerrain)
        {
            var terrainEntries = McnkTerrainExtractor.ExtractTerrain(wdt);
            var terrainCsvPath = Path.Combine(csvDir, wdt.MapName, $"{wdt.MapName}_mcnk_terrain.csv");
            McnkTerrainCsvWriter.WriteCsv(terrainEntries, terrainCsvPath);
        }

        // Extract MCNK shadow maps if requested
        if (opts.ExtractMcnkShadows)
        {
            var shadowEntries = McnkShadowExtractor.ExtractShadows(wdt);
            var shadowCsvPath = Path.Combine(csvDir, wdt.MapName, $"{wdt.MapName}_mcnk_shadows.csv");
            McnkShadowCsvWriter.WriteCsv(shadowEntries, shadowCsvPath);
        }

        // Write index.json for web UI
        var idx = new AnalysisIndex
        {
            MapName = wdt.MapName,
            Tiles = adt.Tiles,
            WmoAssets = adt.WmoAssets.OrderBy(x => x).ToList(),
            M2Assets = adt.M2Assets.OrderBy(x => x).ToList(),
            BlpAssets = adt.BlpAssets.OrderBy(x => x).ToList(),
            Placements = adt.Placements,
            MissingWmo = missingWmo.OrderBy(x => x).ToList(),
            MissingM2 = missingM2.OrderBy(x => x).ToList(),
            MissingBlp = missingBlp.OrderBy(x => x).ToList(),
        };

        var idxPath = Path.Combine(opts.OutDir, "index.json");
        var json = JsonSerializer.Serialize(idx, new JsonSerializerOptions{ WriteIndented = true });
        File.WriteAllText(idxPath, json);
    }
}
