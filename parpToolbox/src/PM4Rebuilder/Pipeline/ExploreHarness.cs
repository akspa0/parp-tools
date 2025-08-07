using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using ParpToolbox.Formats.PM4;

namespace PM4Rebuilder.Pipeline
{
    /// <summary>
    /// Fully automated exploration harness that brute-forces groupings over Unknown* fields
    /// and writes summary metrics + optional OBJ export.
    /// NOTE: This is an initial skeleton to get real metrics out quickly; optimisation and
    /// additional heuristics can be added incrementally without changing the CLI.
    /// </summary>
    internal static class ExploreHarness
    {
        public static void Run(Pm4Scene scene, string outDir, bool exportObj = true)
        {
            string root = Path.Combine(outDir, $"explore_{DateTime.Now:yyyyMMdd_HHmmss}");
            Directory.CreateDirectory(root);

            PipelineLogger.Log($"[EXPLORE] Writing outputs to {root}");

            // Collect candidate records: links, surfaces, placements, etc.
            var records = new List<object>();
            records.AddRange(scene.Links);
            records.AddRange(scene.Surfaces);
            records.AddRange(scene.Placements);
            records.AddRange(scene.Vertices); // vertices rarely have Unknowns but include anyway

            // index records by type for later correlation
            var recordsByType = records
                .GroupBy(r => r.GetType())
                .ToDictionary(g => g.Key, g => g.ToList());

            // Identify all Unknown* fields across record types via reflection once.
            var candidatePropNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var rec in records)
            {
                foreach (var pi in rec.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance))
                {
                    if (pi.Name.StartsWith("Unknown", StringComparison.OrdinalIgnoreCase))
                    {
                        candidatePropNames.Add(pi.Name);
                    }
                }
            }
            PipelineLogger.Log($"[EXPLORE] Found {candidatePropNames.Count} candidate Unknown* fields across all chunks.");

            // Build per-field grouping metrics (single-key only for v1) – evaluated PER record type so
            // Unknown16 in MSLK is distinct from Unknown16 in MSUR.
            var groupingSummaries = new List<GroupingSummary>();
            var perTypeSummaries = new Dictionary<Type, List<GroupingSummary>>();
            var recordTypes = recordsByType.Keys;
            foreach (var recType in recordTypes)
            {
                var typeRecords = recordsByType[recType];
                foreach (var keyName in candidatePropNames)
                {
                    var pi = recType.GetProperty(keyName, BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase);
                    if (pi == null) continue;

                    var groups = new Dictionary<object, int>();
                    foreach (var rec in typeRecords)
                    {
                        object? keyVal = pi.GetValue(rec);
                        keyVal ??= "<null>";
                        if (!groups.TryGetValue(keyVal, out int count))
                            groups[keyVal] = 1;
                        else
                            groups[keyVal] = count + 1;
                    }



                    var labelPrefix = recType.DeclaringType != null ? $"{recType.DeclaringType.Name}.{recType.Name}" : recType.Name;
                    var summary = new GroupingSummary
                    {
                        KeyCombination = $"{labelPrefix}.{keyName}",
                        GroupCount = groups.Count,
                        TotalRecords = typeRecords.Count,
                        MinPerGroup = groups.Values.Min(),
                        MaxPerGroup = groups.Values.Max(),
                        AvgPerGroup = groups.Values.Average()
                    };
                    groupingSummaries.Add(summary);
                    if (!perTypeSummaries.TryGetValue(recType, out var list))
                    {
                        list = new List<GroupingSummary>();
                        perTypeSummaries[recType] = list;
                    }
                    list.Add(summary);
                }
            }

            // Rank by smaller group count (but >1), so candidate for object grouping
            var top = groupingSummaries
                .Where(g => g.GroupCount > 1 && g.GroupCount < 5000)
                .OrderBy(g => g.GroupCount)
                .ThenByDescending(g => g.TotalRecords)
                .Take(5)
                .ToList();

            // Write summary file
            string summaryPath = Path.Combine(root, "summary.txt");
            using (var sw = new StreamWriter(summaryPath))
            {
                sw.WriteLine("ChunkField\tGroups\tMinPerGroup\tAvgPerGroup\tMaxPerGroup\tTotalRecords");
                foreach (var g in top)
                {
                    sw.WriteLine($"{g.KeyCombination}\t{g.GroupCount}\t{g.MinPerGroup}\t{g.AvgPerGroup:F1}\t{g.MaxPerGroup}\t{g.TotalRecords}");
                }
            }
            PipelineLogger.Log($"[EXPLORE] Master summary written: {summaryPath}");

            // Write per-chunk summaries
            foreach (var kv in perTypeSummaries)
            {
                var typeLabel = kv.Key.DeclaringType != null ? $"{kv.Key.DeclaringType.Name}_{kv.Key.Name}" : kv.Key.Name;
                string chunkFile = Path.Combine(root, $"{typeLabel}_summary.txt");
                using var swType = new StreamWriter(chunkFile);
                swType.WriteLine("Field\tGroups\tMin\tAvg\tMax\tTotal");
                foreach (var g in kv.Value.OrderBy(s => s.GroupCount))
                {
                    swType.WriteLine($"{g.KeyCombination}\t{g.GroupCount}\t{g.MinPerGroup}\t{g.AvgPerGroup:F1}\t{g.MaxPerGroup}\t{g.TotalRecords}");
                }
            }
            PipelineLogger.Log($"[EXPLORE] Per-chunk summaries written: {perTypeSummaries.Count}");

            // Cross-chunk correlation on matching Unknown* fields
            WriteCrossChunkCorrelation(root, perTypeSummaries, recordsByType);

            // TODO: implement OBJ export for top groupings in future iterations.
            PipelineLogger.Log("[EXPLORE] OBJ export not implemented yet – summary only.");
        }

        private class GroupingSummary
        {
            public string KeyCombination { get; set; } = string.Empty;
            public int GroupCount { get; set; }
            public int TotalRecords { get; set; }
            public int MinPerGroup { get; set; }
            public int MaxPerGroup { get; set; }
            public double AvgPerGroup { get; set; }
        }
        private static void WriteCrossChunkCorrelation(string root, Dictionary<Type, List<GroupingSummary>> perType, Dictionary<Type,List<object>> recordsByType)
        {
            var correlationPath = Path.Combine(root, "cross_chunk_links.txt");
            using var sw = new StreamWriter(correlationPath);
            sw.WriteLine("Field\tOtherField\tSharedValue\tCountTypeA\tCountTypeB");

            // Build value->count maps per (type, field)
            var valueMaps = new Dictionary<(Type,string), Dictionary<object,int>>();
            foreach (var kv in perType)
            {
                var recType = kv.Key;
                var recordsOfType = recordsByType.TryGetValue(recType, out var listRec) ? listRec : new List<object>();

                foreach (var g in kv.Value)
                {
                    var fieldName = g.KeyCombination.Split('.').Last();
                    var pi = recType.GetProperty(fieldName, BindingFlags.Public|BindingFlags.Instance|BindingFlags.IgnoreCase);
                    if (pi == null) continue;

                    var dict = new Dictionary<object,int>();
                    foreach (var rec in recordsOfType)
                    {
                        var val = pi.GetValue(rec) ?? "<null>";
                        if (!dict.TryGetValue(val, out int c)) c = 0;
                        dict[val] = c + 1;
                    }
                    valueMaps[(recType, fieldName)] = dict;
                }
            }

            // Cross pair comparisons – naive O(n^2) but small field count.
            var keys = valueMaps.Keys.ToList();
            for (int i = 0; i < keys.Count; i++)
            {
                for (int j = i + 1; j < keys.Count; j++)
                {
                    var aKey = keys[i];
                    var bKey = keys[j];
                    if (aKey.Item1 == bKey.Item1) continue; // same type

                    var mapA = valueMaps[aKey];
                    var mapB = valueMaps[bKey];
                    var shared = mapA.Keys.Intersect(mapB.Keys).ToList();
                    if (shared.Count == 0) continue;

                    foreach (var val in shared.Take(20)) // cap output
                    {
                        sw.WriteLine($"{aKey.Item1.Name}.{aKey.Item2}\t{bKey.Item1.Name}.{bKey.Item2}\t{val}\t{mapA[val]}\t{mapB[val]}");
                    }
                }
            }

            PipelineLogger.Log("[EXPLORE] cross_chunk_links.txt generated.");
        }
    }
}
