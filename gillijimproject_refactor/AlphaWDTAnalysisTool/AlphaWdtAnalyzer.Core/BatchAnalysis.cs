using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AlphaWdtAnalyzer.Core;

public static class BatchAnalysis
{
    public sealed class Options
    {
        public required string InputRoot { get; init; }
        public required string ListfilePath { get; init; }
        public required string OutDir { get; init; }
        public int ClusterThreshold { get; init; } = 10;
        public int ClusterGap { get; init; } = 1000;
        public string? DbcDir { get; init; }
        public bool Web { get; init; } = false;
    }

    public static void Run(Options opts)
    {
        Directory.CreateDirectory(opts.OutDir);
        var csvRoot = Path.Combine(opts.OutDir, "csv");
        var csvGlobal = Path.Combine(csvRoot, "global");
        var csvMapsRoot = Path.Combine(csvRoot, "maps");
        Directory.CreateDirectory(csvRoot);
        Directory.CreateDirectory(csvGlobal);
        Directory.CreateDirectory(csvMapsRoot);

        // Load listfile (normalize helper still used elsewhere)
        var listfile = new ListfileLoader();
        listfile.Load(opts.ListfilePath);

        // Discover all WDT files
        var wdts = Directory.EnumerateFiles(opts.InputRoot, "*.wdt", SearchOption.AllDirectories)
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
            .ToList();
        if (wdts.Count == 0)
        {
            Console.WriteLine($"No WDT files found under {opts.InputRoot}");
            return;
        }

        var adtScanner = new AdtScanner();
        var allPlacements = new List<PlacementRecord>(capacity: 100_000);
        var byMap = new Dictionary<string, List<PlacementRecord>>(StringComparer.OrdinalIgnoreCase);

        foreach (var wdtPath in wdts)
        {
            try
            {
                var wdt = new WdtAlphaScanner(wdtPath);
                var result = adtScanner.Scan(wdt);

                // aggregate placements
                allPlacements.AddRange(result.Placements);
                if (!byMap.TryGetValue(wdt.MapName, out var list))
                {
                    list = new List<PlacementRecord>();
                    byMap[wdt.MapName] = list;
                }
                list.AddRange(result.Placements);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failed to analyze {wdtPath}: {ex.Message}");
            }
        }

        // Global outputs
        CsvReportWriter.WriteUniqueIds(csvGlobal, allPlacements);
        CsvReportWriter.WriteTimeline(csvGlobal, allPlacements);
        var globalIds = allPlacements.Where(p => p.UniqueId.HasValue).Select(p => p.UniqueId!.Value);
        var globalClusters = UniqueIdClusterer.FindClusters(globalIds, opts.ClusterThreshold, opts.ClusterGap);
        CsvReportWriter.WriteIdRanges(Path.Combine(csvGlobal), globalClusters);
        if (globalIds.Any())
        {
            CsvReportWriter.WriteIdRangeSummaryGlobal(csvGlobal, globalIds.Min(), globalIds.Max(), globalIds.Count());
        }

        // Per-map outputs and aggregate per-map cluster indices
        var perMapClusters = new List<(string MapName, UniqueIdClusterer.Cluster Cluster)>();
        var perMapSummaries = new List<(string MapName, int MinId, int MaxId, int Count)>();

        foreach (var kv in byMap.OrderBy(k => k.Key, StringComparer.OrdinalIgnoreCase))
        {
            var mapName = kv.Key;
            var plist = kv.Value;
            var mapDir = Path.Combine(csvMapsRoot, mapName);
            Directory.CreateDirectory(mapDir);

            // per-map unique ids and timeline
            CsvReportWriter.WriteMapUniqueIds(mapDir, mapName, plist);
            CsvReportWriter.WriteMapTimeline(mapDir, mapName, plist);

            var mapIds = plist.Where(p => p.UniqueId.HasValue).Select(p => p.UniqueId!.Value);
            var clusters = UniqueIdClusterer.FindClusters(mapIds, opts.ClusterThreshold, opts.ClusterGap);
            CsvReportWriter.WriteIdRanges(mapDir, clusters);
            perMapClusters.AddRange(clusters.Select(c => (mapName, c)));
            if (mapIds.Any())
            {
                perMapSummaries.Add((mapName, mapIds.Min(), mapIds.Max(), mapIds.Count()));
            }
        }

        // Aggregate per-map indices into csv root
        CsvReportWriter.WriteIdRangesByMap(csvRoot, perMapClusters);
        CsvReportWriter.WriteIdRangeSummaryByMap(csvRoot, perMapSummaries);

        // optional web UI (global index.json only)
        if (opts.Web)
        {
            WebAssetsWriter.Write(opts.OutDir);
            Console.WriteLine($"Web UI written to {Path.Combine(opts.OutDir, "web")}. Open index.html in a browser.");
        }
    }
}
