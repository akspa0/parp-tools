using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Linq;
using GillijimProject.Next.Core.Adapters.Dbcd;
using GillijimProject.Next.Core.Services;
using GillijimProject.Next.Core.Domain.Liquids;
using GillijimProject.Next.Core.Domain;
using GillijimProject.Next.Core.Transform;
using GillijimProject.Next.Core.IO;
using GillijimProject.Next.Core.WowFiles.Alpha;

namespace GillijimProject.Next.Cli.Commands;

public static class ConvertCommand
{
    public static int Run(string[] args)
    {
        // Options (subset for bootstrap):
        // --wdt-alpha <path> --out <dir> --dbc-alpha <path> --dbc-wotlk <path> [--build-alpha <ver>] [--build-wotlk <ver>]

        var opts = ParseOptions(args);

        if (!opts.TryGetValue("--dbc-alpha", out var alphaDbc) || !File.Exists(alphaDbc))
        {
            Console.Error.WriteLine("[convert] Missing or invalid --dbc-alpha <path> (AreaTable.dbc).");
            return 2;
        }
        if (!opts.TryGetValue("--dbc-wotlk", out var lkDbc) || !File.Exists(lkDbc))
        {
            Console.Error.WriteLine("[convert] Missing or invalid --dbc-wotlk <path> (AreaTable.dbc).");
            return 2;
        }

        DbcdAreaTableProvider? provider = null;
        AreaIdTranslator? translator = null;

        opts.TryGetValue("--build-alpha", out var alphaBuild);
        opts.TryGetValue("--build-wotlk", out var lkBuild);
        opts.TryGetValue("--areaid-overrides", out var overridesPath);
        opts.TryGetValue("--map-overrides", out var mapOverridesPath);
        opts.TryGetValue("--out", out var outDir);

        // Liquids CLI options
        opts.TryGetValue("--liquids", out var liquidsFlag);
        opts.TryGetValue("--liquid-precedence", out var precedenceRaw);
        opts.TryGetValue("--liquid-id-map", out var liquidMapPath);
        opts.TryGetValue("--green-lava", out var greenLavaFlag);

        try
        {
            provider = new DbcdAreaTableProvider(alphaDbc, lkDbc, alphaBuild, string.IsNullOrWhiteSpace(lkBuild) ? "3.3.5.12340" : lkBuild);
            provider.EnsureLoaded();
            Console.WriteLine($"[convert] Loaded DBC: AreaTable alpha={provider.AlphaRowCount}, lk={provider.LkRowCount}; Map alpha={provider.AlphaMapRowCount}, lk={provider.LkMapRowCount}.");

            // Build AreaID mapping
            translator = new AreaIdTranslator(provider);
            translator.BuildMapping(string.IsNullOrWhiteSpace(overridesPath) ? null : overridesPath,
                                    string.IsNullOrWhiteSpace(mapOverridesPath) ? null : mapOverridesPath);
            Console.WriteLine($"[convert] Map crosswalk: matched={translator.MapMatchedCount}, ambiguous={translator.MapAmbiguousCount}, unmatched={translator.MapUnmatchedCount}");
            Console.WriteLine($"[convert] AreaID mapping: matched={translator.MatchedCount}, ambiguous={translator.AmbiguousCount}, unmatched={translator.UnmatchedCount}");

            // Optional: write mapping report to --out
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
                Console.WriteLine($"[convert] Wrote mapping report: {jsonPath}");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[convert] Failed to load DBCD: {ex.Message}");
            return 3;
        }

        // Validate and enumerate input
        if (!opts.TryGetValue("--wdt-alpha", out var wdtPath) || !File.Exists(wdtPath))
        {
            Console.Error.WriteLine("[convert] Missing or invalid --wdt-alpha <path>.");
            return 2;
        }

        // Use modern AlphaWdtReader for WDT introspection
        var wdtInfo = AlphaWdtReader.Read(wdtPath);
        Console.WriteLine($"[convert] WDT: wmoBased={wdtInfo.WmoBased} mdnm={wdtInfo.MdnmFiles.Count} monm={wdtInfo.MonmFiles.Count}");
        int presentTiles = 0;
        for (int i = 0; i < wdtInfo.AdtOffsets.Count; i++) if (wdtInfo.AdtOffsets[i] != 0) presentTiles++;
        Console.WriteLine($"[convert] MAIN present tiles={presentTiles}");

        // Domain record for existing converter pipeline
        var wdt = new WdtAlpha(wdtPath);
        var mapName = Path.GetFileNameWithoutExtension(wdtPath);
        var adtDir = Path.GetDirectoryName(wdtPath) ?? ".";
        var adtFiles = Directory.EnumerateFiles(adtDir, $"{mapName}_*.adt", SearchOption.TopDirectoryOnly).ToList();
        var adts = adtFiles.Select(p => new AdtAlpha(p)).ToList();
        if (adts.Count == 0)
        {
            Console.WriteLine($"[convert] No ADT files found next to WDT for map '{mapName}'.");
        }

        // Build LiquidsOptions from CLI flags
        var liquids = BuildLiquidsOptions(opts);
        var precStr = string.Join('>', liquids.Precedence.Select(p => p.ToString()));
        Console.WriteLine($"[convert] Liquids: enabled={liquids.EnableLiquids}, precedence={precStr}, greenLava={liquids.GreenLava}, map={(opts.ContainsKey("--liquid-id-map") ? liquidMapPath : "default")} ");

        IAlphaLiquidsExtractor extractor = new AlphaMclqExtractor();
        var outputs = AlphaToLkConverter.Convert(wdt, adts, translator!, liquids, extractor).ToList();
        int liquidChunks = outputs.Sum(o => o.Mh2oByChunk?.Count(c => c is not null) ?? 0);
        Console.WriteLine($"[convert] Converted {outputs.Count} ADTs; liquid-bearing chunks={liquidChunks}.");
        return 0;
    }

    private static LiquidsOptions BuildLiquidsOptions(Dictionary<string, string> opts)
    {
        bool enabled = GetFlag(opts, "--liquids", defaultValue: true);
        bool greenLava = GetFlag(opts, "--green-lava", defaultValue: false);

        IReadOnlyList<MclqLiquidType> precedence;
        if (opts.TryGetValue("--liquid-precedence", out var rawPrec) && !string.IsNullOrWhiteSpace(rawPrec))
        {
            precedence = ParsePrecedence(rawPrec);
        }
        else
        {
            precedence = new[] { MclqLiquidType.Magma, MclqLiquidType.Slime, MclqLiquidType.River, MclqLiquidType.Ocean };
        }

        LiquidTypeMapping mapping = LiquidTypeMapping.CreateDefault();
        if (opts.TryGetValue("--liquid-id-map", out var mapPath) && !string.IsNullOrWhiteSpace(mapPath))
        {
            var loaded = LoadLiquidTypeMappingFromJson(mapPath);
            if (loaded is not null) mapping = loaded;
        }

        return new LiquidsOptions
        {
            EnableLiquids = enabled,
            GreenLava = greenLava,
            Precedence = precedence,
            Mapping = mapping,
        };
    }

    private static bool GetFlag(Dictionary<string, string> opts, string key, bool defaultValue)
    {
        if (!opts.TryGetValue(key, out var val)) return defaultValue;
        if (string.IsNullOrEmpty(val)) return true;
        if (bool.TryParse(val, out var b)) return b;
        if (string.Equals(val, "1", StringComparison.OrdinalIgnoreCase) || string.Equals(val, "yes", StringComparison.OrdinalIgnoreCase) || string.Equals(val, "on", StringComparison.OrdinalIgnoreCase)) return true;
        if (string.Equals(val, "0", StringComparison.OrdinalIgnoreCase) || string.Equals(val, "no", StringComparison.OrdinalIgnoreCase) || string.Equals(val, "off", StringComparison.OrdinalIgnoreCase)) return false;
        return defaultValue;
    }

    private static IReadOnlyList<MclqLiquidType> ParsePrecedence(string raw)
    {
        var tokens = raw.Split(new[] { '>', ',', ';' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var list = new List<MclqLiquidType>();
        foreach (var t in tokens)
        {
            if (TryParseTypeToken(t, out var type) && !list.Contains(type))
                list.Add(type);
        }
        return list.Count > 0 ? list.ToArray() : new[] { MclqLiquidType.Magma, MclqLiquidType.Slime, MclqLiquidType.River, MclqLiquidType.Ocean };
    }

    private static bool TryParseTypeToken(string token, out MclqLiquidType type)
    {
        token = token.Trim();
        if (token.Equals("lava", StringComparison.OrdinalIgnoreCase)) { type = MclqLiquidType.Magma; return true; }
        return Enum.TryParse<MclqLiquidType>(token, true, out type);
    }

    private static LiquidTypeMapping? LoadLiquidTypeMappingFromJson(string path)
    {
        try
        {
            if (!File.Exists(path))
            {
                Console.Error.WriteLine($"[convert] --liquid-id-map not found: {path}");
                return null;
            }

            var text = File.ReadAllText(path);
            var dict = JsonSerializer.Deserialize<Dictionary<string, ushort>>(text);
            if (dict is null)
            {
                Console.Error.WriteLine($"[convert] --liquid-id-map invalid JSON: {path}");
                return null;
            }

            var map = new Dictionary<MclqLiquidType, ushort>();
            foreach (var kv in dict)
            {
                if (!TryParseTypeToken(kv.Key, out var t))
                {
                    Console.Error.WriteLine($"[convert] Unknown liquid type in mapping: '{kv.Key}'");
                    continue;
                }
                map[t] = kv.Value;
            }
            if (map.Count == 0)
            {
                Console.Error.WriteLine($"[convert] --liquid-id-map produced empty mapping; using defaults");
                return null;
            }
            return new LiquidTypeMapping(map);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[convert] Failed to load --liquid-id-map: {ex.Message}");
            return null;
        }
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
                    // flags without value not used yet
                    dict[key] = string.Empty;
                }
            }
        }
        return dict;
    }
}
