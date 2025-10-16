using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using WoWRollback.LkToAlphaModule;
using WoWRollback.LkToAlphaModule.Utils;
using WoWRollback.LkToAlphaModule.Inspectors;
using WoWRollback.LkToAlphaModule.Validators;

namespace WoWRollback.AdtConverter;

internal static class Program
{
    private static int Main(string[] args)
    {
        var root = new RootCommand("WoWRollback ADT Converter (LK -> Alpha terrain-only MVP)");

        var wdtCmd = new Command("convert-wdt", "Convert LK WDT to Alpha (MVER+MAIN)");
        var wdtPath = new Option<string>("--lk-wdt", description: "Path to LK WDT file") { IsRequired = true };
        var mapOpt = new Option<string>("--map", description: "Map name (e.g., Azeroth)") { IsRequired = true };
        var outOpt = new Option<string>("--out", () => string.Empty, "Output directory (default: project_output/<map>_<ts>)");
        wdtCmd.AddOption(wdtPath);
        wdtCmd.AddOption(mapOpt);
        wdtCmd.AddOption(outOpt);
        wdtCmd.SetHandler((string wdt, string map, string outDir) =>
        {
            var orch = new LkToAlphaOrchestrator();
            var resolvedOut = string.IsNullOrWhiteSpace(outDir) ? OutputPathResolver.GetDefaultRoot(map) : outDir;
            var res = orch.ConvertLkToAlpha(wdt, resolvedOut, resolvedOut, map, new LkToAlphaOptions());
            Console.WriteLine(res.Success ? $"WDT OK: {res.AlphaOutputDirectory}" : $"WDT FAILED: {res.ErrorMessage}");
        }, wdtPath, mapOpt, outOpt);

        var mapCmd = new Command("convert-map-terrain", "Convert LK map (root ADTs only) to Alpha terrain-only ADTs and write Alpha WDT");
        var lkDirOpt = new Option<string>("--lk-dir", description: "LK map directory or root containing World/Maps/<map>/") { IsRequired = true };
        mapCmd.AddOption(wdtPath);
        mapCmd.AddOption(lkDirOpt);
        mapCmd.AddOption(mapOpt);
        mapCmd.AddOption(outOpt);
        mapCmd.SetHandler((string wdt, string lkDir, string map, string outDir) =>
        {
            var orch = new LkToAlphaOrchestrator();
            var resolvedOut = string.IsNullOrWhiteSpace(outDir) ? OutputPathResolver.GetDefaultRoot(map) : outDir;
            var res = orch.ConvertLkToAlphaTerrainOnlyAdts(wdt, lkDir, resolvedOut, map, new LkToAlphaOptions());
            Console.WriteLine(res.Success ? $"MAP OK: {res.TilesProcessed} tiles -> {res.AlphaOutputDirectory}" : $"MAP FAILED: {res.ErrorMessage}");
        }, wdtPath, lkDirOpt, mapOpt, outOpt);

        root.Add(wdtCmd);
        root.Add(mapCmd);

        var packCmd = new Command("pack-monolithic", "Pack a monolithic Alpha WDT (header + embedded terrain-only ADTs)");
        var wdtPathPack = new Option<string>("--lk-wdt", description: "Path to LK WDT file") { IsRequired = true };
        var lkDirPack = new Option<string>("--lk-dir", description: "LK map directory or root containing World/Maps/<map>/") { IsRequired = true };
        var mapOptPack = new Option<string>("--map", description: "Map name (e.g., Azeroth)") { IsRequired = true };
        var outOptPack = new Option<string?>("--out", description: "Output directory (default: project_output/<map>_<ts>)");
        var forceArea = new Option<int?>("--force-area-id", description: "Override AreaID when LK AreaId==0");
        var debugFlat = new Option<float?>("--debug-flat-mcvt", description: "Force constant terrain height for all MCVT samples");
        var baseTex = new Option<string?>("--base-texture", description: "Override MTEX path for base texture");
        var mainPointToData = new Option<bool>("--main-point-to-data", () => false, "MAIN offsets point to MHDR.data (+8) instead of letters");
        packCmd.AddOption(wdtPathPack);
        packCmd.AddOption(lkDirPack);
        packCmd.AddOption(mapOptPack);
        packCmd.AddOption(outOptPack);
        packCmd.AddOption(forceArea);
        packCmd.AddOption(debugFlat);
        packCmd.AddOption(baseTex);
        packCmd.AddOption(mainPointToData);
        packCmd.SetHandler((string wdt, string lkDir, string map, string? outDir, int? forceAreaId, float? debugFlatMcvt, string? baseTexture, bool pointToData) =>
        {
            var orch = new LkToAlphaOrchestrator();
            var resolvedOut = string.IsNullOrWhiteSpace(outDir) ? OutputPathResolver.GetDefaultRoot(map) : outDir;
            var res = orch.PackMonolithicAlphaWdt(wdt, lkDir, resolvedOut, map, new LkToAlphaOptions
            {
                ForceAreaId = forceAreaId,
                DebugFlatMcvt = debugFlatMcvt,
                BaseTexture = baseTexture,
                MainPointToMhdrData = pointToData
            });
            Console.WriteLine(res.Success ? $"PACK OK: {res.TilesProcessed} tiles -> {Path.Combine(res.AlphaOutputDirectory, map + ".wdt")}" : $"PACK FAILED: {res.ErrorMessage}");
        }, wdtPathPack, lkDirPack, mapOptPack, outOptPack, forceArea, debugFlat, baseTex, mainPointToData);

        root.Add(packCmd);

        var inspCmd = new Command("inspect-alpha", "Inspect a real Alpha WDT and report chunk order, MHDR/MCIN alignment, and sample MAIN tiles");
        var inspWdt = new Option<string>("--wdt", description: "Path to Alpha WDT file") { IsRequired = true };
        var inspTiles = new Option<int>("--tiles", () => 3, "Number of sample tiles to inspect from MAIN");
        var inspJson = new Option<string?>("--json", description: "Optional path to write a JSON report");
        inspCmd.AddOption(inspWdt);
        inspCmd.AddOption(inspTiles);
        inspCmd.AddOption(inspJson);
        inspCmd.SetHandler((string wdt, int tiles, string? json) =>
        {
            AlphaWdtInspector.Inspect(wdt, tiles, json);
        }, inspWdt, inspTiles, inspJson);

        root.Add(inspCmd);

        var compareCmd = new Command("compare-alpha", "Compare two Alpha WDT files byte-by-byte");
        var compareRef = new Option<string>("--reference", description: "Reference Alpha WDT file (known good)") { IsRequired = true };
        var compareTest = new Option<string>("--test", description: "Test Alpha WDT file (to check)") { IsRequired = true };
        var compareBytes = new Option<int>("--max-bytes", () => 100000, "Maximum bytes to compare");
        compareCmd.AddOption(compareRef);
        compareCmd.AddOption(compareTest);
        compareCmd.AddOption(compareBytes);
        compareCmd.SetHandler((string refFile, string testFile, int maxBytes) =>
        {
            AlphaWdtInspector.CompareFiles(refFile, testFile, maxBytes);
        }, compareRef, compareTest, compareBytes);

        root.Add(compareCmd);

        var validateCmd = new Command("validate-wdt", "Validate WDT structure against reference (automated test)");
        var validateTest = new Option<string>("--test", description: "WDT file to validate") { IsRequired = true };
        var validateRef = new Option<string>("--reference", description: "Reference WDT (known good)") { IsRequired = true };
        validateCmd.AddOption(validateTest);
        validateCmd.AddOption(validateRef);
        validateCmd.SetHandler((string testFile, string refFile) =>
        {
            var result = WdtStructureValidator.ValidateAgainstReference(testFile, refFile);
            Console.WriteLine();
            if (result.IsValid)
            {
                Console.WriteLine("✓ VALIDATION PASSED");
            }
            else
            {
                Console.WriteLine("✗ VALIDATION FAILED");
                Console.WriteLine("\nErrors:");
                foreach (var err in result.Errors)
                {
                    Console.WriteLine($"  - {err}");
                }
            }
            if (result.Warnings.Count > 0)
            {
                Console.WriteLine("\nWarnings:");
                foreach (var warn in result.Warnings)
                {
                    Console.WriteLine($"  - {warn}");
                }
            }
        }, validateTest, validateRef);

        root.Add(validateCmd);

        return root.Invoke(args);
    }
}
