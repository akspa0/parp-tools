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
        var lkDirPack = new Option<string>("--lk-dir", description: "LK map directory or root containing World/Maps/<map>/") { IsRequired = true };
        packCmd.AddOption(wdtPath);
        packCmd.AddOption(lkDirPack);
        packCmd.AddOption(mapOpt);
        packCmd.AddOption(outOpt);
        packCmd.SetHandler((string wdt, string lkDir, string map, string outDir) =>
        {
            var orch = new LkToAlphaOrchestrator();
            var resolvedOut = string.IsNullOrWhiteSpace(outDir) ? OutputPathResolver.GetDefaultRoot(map) : outDir;
            var res = orch.PackMonolithicAlphaWdt(wdt, lkDir, resolvedOut, map, new LkToAlphaOptions());
            Console.WriteLine(res.Success ? $"PACK OK: {res.TilesProcessed} tiles -> {Path.Combine(res.AlphaOutputDirectory, map + ".wdt")}" : $"PACK FAILED: {res.ErrorMessage}");
        }, wdtPath, lkDirPack, mapOpt, outOpt);

        root.Add(packCmd);

        var inspCmd = new Command("inspect-alpha", "Inspect a real Alpha WDT and report chunk order, MHDR/MCIN alignment, and sample MAIN tiles");
        var inspWdt = new Option<string>("--wdt", description: "Path to Alpha WDT file") { IsRequired = true };
        var inspTiles = new Option<int>("--tiles", () => 3, "Number of sample tiles to inspect from MAIN");
        inspCmd.AddOption(inspWdt);
        inspCmd.AddOption(inspTiles);
        inspCmd.SetHandler((string wdt, int tiles) =>
        {
            AlphaWdtInspector.Inspect(wdt, tiles);
        }, inspWdt, inspTiles);

        root.Add(inspCmd);

        return root.Invoke(args);
    }
}
