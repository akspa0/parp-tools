using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using WoWRollback.LkToAlphaModule;
using WoWRollback.LkToAlphaModule.Utils;

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

        return root.Invoke(args);
    }
}
