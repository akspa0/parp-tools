using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using WoWRollback.LkToAlphaModule;
using WoWRollback.LkToAlphaModule.Utils;
using WoWRollback.LkToAlphaModule.Inspectors;

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

        return root.Invoke(args);
    }
}
