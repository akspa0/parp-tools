using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using GillijimProject.Next.Core.Adapters.Dbcd;
using GillijimProject.Next.Core.Services;

namespace GillijimProject.Next.Cli.Commands;

public static class FixAreaIdsCommand
{
    public static int Run(string[] args)
    {
        var opts = ParseOptions(args);

        if (!opts.TryGetValue("--dbc-alpha", out var alphaDbc) || !File.Exists(alphaDbc))
        {
            Console.Error.WriteLine("[fix-areaids] Missing or invalid --dbc-alpha <path> (AreaTable.dbc).");
            return 2;
        }
        if (!opts.TryGetValue("--dbc-wotlk", out var lkDbc) || !File.Exists(lkDbc))
        {
            Console.Error.WriteLine("[fix-areaids] Missing or invalid --dbc-wotlk <path> (AreaTable.dbc).");
            return 2;
        }

        opts.TryGetValue("--build-alpha", out var alphaBuild);
        opts.TryGetValue("--build-wotlk", out var lkBuild);
        opts.TryGetValue("--areaid-overrides", out var overridesPath);
        opts.TryGetValue("--map-overrides", out var mapOverridesPath);
        opts.TryGetValue("--out", out var outDir);

        try
        {
            var provider = new DbcdAreaTableProvider(alphaDbc, lkDbc, alphaBuild, string.IsNullOrWhiteSpace(lkBuild) ? "3.3.5.12340" : lkBuild);
            var translator = new AreaIdTranslator(provider);
            // Warm up loads
            provider.EnsureLoaded();
            Console.WriteLine($"[fix-areaids] Loaded DBC: AreaTable alpha={provider.AlphaRowCount}, lk={provider.LkRowCount}; Map alpha={provider.AlphaMapRowCount}, lk={provider.LkMapRowCount}.");

            // Build mapping and report
            translator.BuildMapping(string.IsNullOrWhiteSpace(overridesPath) ? null : overridesPath,
                                   string.IsNullOrWhiteSpace(mapOverridesPath) ? null : mapOverridesPath);
            Console.WriteLine($"[fix-areaids] Map crosswalk: matched={translator.MapMatchedCount}, ambiguous={translator.MapAmbiguousCount}, unmatched={translator.MapUnmatchedCount}");
            Console.WriteLine($"[fix-areaids] AreaID mapping: matched={translator.MatchedCount}, ambiguous={translator.AmbiguousCount}, unmatched={translator.UnmatchedCount}");

            if (!string.IsNullOrWhiteSpace(outDir))
            {
                Directory.CreateDirectory(outDir);
                var jsonPath = Path.Combine(outDir, "areaid_mapping.json");
                var summaryPath = Path.Combine(outDir, "areaid_mapping_summary.txt");
                var reportObj = new
                {
                    matchedCount = translator.MatchedCount,
                    ambiguousCount = translator.AmbiguousCount,
                    unmatchedCount = translator.UnmatchedCount,
                    mapping = translator.GetMapping(),
                    ambiguousAlpha = translator.GetAmbiguousAlpha(),
                    unmatchedAlpha = translator.GetUnmatchedAlpha(),
                    maps = new {
                        matchedCount = translator.MapMatchedCount,
                        ambiguousCount = translator.MapAmbiguousCount,
                        unmatchedCount = translator.MapUnmatchedCount,
                        crosswalk = translator.GetMapCrosswalk(),
                        ambiguousAlpha = translator.GetMapAmbiguousAlpha(),
                        unmatchedAlpha = translator.GetMapUnmatchedAlpha()
                    }
                };
                var json = JsonSerializer.Serialize(reportObj, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(jsonPath, json);
                File.WriteAllText(summaryPath,
                    $"maps_matched={translator.MapMatchedCount}\nmaps_ambiguous={translator.MapAmbiguousCount}\nmaps_unmatched={translator.MapUnmatchedCount}\n" +
                    $"areas_matched={translator.MatchedCount}\nareas_ambiguous={translator.AmbiguousCount}\nareas_unmatched={translator.UnmatchedCount}\n");
                Console.WriteLine($"[fix-areaids] Wrote mapping report: {jsonPath}");
            }

            // Placeholder dry-run preview
            var sampleIds = new[] { 1, 8, 12, 33, 85 };
            foreach (var alphaId in sampleIds)
            {
                if (translator.TryTranslate(alphaId, out var lkId))
                    Console.WriteLine($"[fix-areaids] preview: alpha {alphaId} -> lk {lkId}");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[fix-areaids] Failed: {ex.Message}");
            return 3;
        }

        Console.WriteLine("[fix-areaids] TODO: implement AreaID correction re-emission.");
        return 0;
    }

    private static Dictionary<string, string> ParseOptions(string[] args)
    {
        var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length; i++)
        {
            var key = args[i];
            if (key.StartsWith("--", StringComparison.Ordinal))
            {
                if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
                {
                    dict[key] = args[++i];
                }
                else
                {
                    dict[key] = string.Empty;
                }
            }
        }
        return dict;
    }
}
