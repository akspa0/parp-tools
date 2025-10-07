using System;
using System.Collections.Generic;
using System.IO;
using WoWRollback.AdtModule;

namespace WoWRollback.Orchestrator;

internal sealed class AdtStageRunner
{
    public IReadOnlyList<AdtStageResult> Run(SessionContext session)
    {
        if (session is null)
        {
            throw new ArgumentNullException(nameof(session));
        }

        var results = new List<AdtStageResult>();

        foreach (var map in session.Options.Maps)
        {
            foreach (var version in session.Options.Versions)
            {
                var runResult = RunExport(session, version, map);
                results.Add(runResult);
            }
        }

        return results;
    }

    private static AdtStageResult RunExport(SessionContext session, string version, string map)
    {
        var options = session.Options;
        var sourceAlphaDir = Path.Combine(options.AlphaRoot, version);
        var wdtPath = Path.Combine(sourceAlphaDir, "tree", "World", "Maps", map, map + ".wdt");

        if (!File.Exists(wdtPath))
        {
            return new AdtStageResult(
                Map: map,
                Version: version,
                Success: false,
                TilesProcessed: 0,
                AreaIdsPatched: 0,
                AdtOutputDirectory: session.Paths.AdtDir,
                Error: $"Missing WDT: {wdtPath}");
        }

        var exportDir = Path.Combine(session.Paths.AdtDir, version);
        Directory.CreateDirectory(exportDir);

        var alias = DbcStageRunner.DeriveAlias(version);

        // Create AdtOrchestrator and conversion options
        var orchestrator = new AdtOrchestrator();
        
        // CRITICAL: Match the path structure DbcStageRunner uses
        // DbcOrchestrator creates: 02_crosswalks/{version}/{alias}/compare/v2/
        // So we need to point to: 02_crosswalks/{version}/{alias}/compare/v2/
        var crosswalkPatchDir = Path.Combine(session.Paths.CrosswalkDir, version, alias, "compare", "v2");
        
        var conversionOptions = new ConversionOptions
        {
            CommunityListfilePath = TryLocateCommunityListfile(options.AlphaRoot),
            LkListfilePath = TryLocateLkListfile(options.AlphaRoot),
            DbdDir = options.DbdDirectory,
            CrosswalkDir = crosswalkPatchDir,
            LkDbcDir = ResolveLkDbcDirectory(options),
            ConvertToMh2o = true,
            AssetFuzzy = true,
            UseFallbacks = true,
            EnableFixups = true,
            Verbose = options.Verbose,
        };

        // Call AdtOrchestrator API
        var result = orchestrator.ConvertAlphaToLk(
            wdtPath: wdtPath,
            exportDir: exportDir,
            mapName: map,
            srcAlias: alias,
            opts: conversionOptions);

        if (!result.Success)
        {
            return new AdtStageResult(
                Map: map,
                Version: version,
                Success: false,
                TilesProcessed: 0,
                AreaIdsPatched: 0,
                AdtOutputDirectory: exportDir,
                Error: result.ErrorMessage);
        }

        return new AdtStageResult(
            Map: map,
            Version: version,
            Success: true,
            TilesProcessed: result.TilesProcessed,
            AreaIdsPatched: result.AreaIdsPatched,
            AdtOutputDirectory: result.AdtOutputDirectory,
            Error: null);
    }

    private static string? TryLocateCommunityListfile(string alphaRoot)
    {
        var candidate = Path.Combine(alphaRoot, "listfile.csv");
        return File.Exists(candidate) ? candidate : null;
    }

    private static string? TryLocateLkListfile(string alphaRoot)
    {
        var candidate = Path.Combine(alphaRoot, "lk_listfile.txt");
        return File.Exists(candidate) ? candidate : null;
    }

    private static string? ResolveLkDbcDirectory(PipelineOptions options)
    {
        if (!string.IsNullOrWhiteSpace(options.LkDbcDirectory))
        {
            return options.LkDbcDirectory;
        }

        var candidate = Path.Combine(options.AlphaRoot, "3.3.5", "tree", "DBFilesClient");
        return Directory.Exists(candidate) ? candidate : null;
    }

}

internal sealed record AdtStageResult(
    string Map,
    string Version,
    bool Success,
    int TilesProcessed,
    int AreaIdsPatched,
    string AdtOutputDirectory,
    string? Error);
