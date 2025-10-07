using System;
using System.Collections.Generic;
using System.IO;
using AlphaWdtAnalyzer.Core.Export;

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
        var build = DbcStageRunner.ResolveBuildIdentifier(alias);

        var pipelineOptions = new AdtExportPipeline.Options
        {
            SingleWdtPath = wdtPath,
            CommunityListfilePath = TryLocateCommunityListfile(options.AlphaRoot),
            LkListfilePath = TryLocateLkListfile(options.AlphaRoot),
            ExportDir = exportDir,
            FallbackTileset = "Tileset\\Generic\\Checkers.blp",
            FallbackNonTilesetBlp = "Dungeons\\Textures\\temp\\64.blp",
            FallbackWmo = "wmo\\Dungeon\\test\\missingwmo.wmo",
            FallbackM2 = "World\\Scale\\HumanMaleScale.mdx",
            ConvertToMh2o = true,
            AssetFuzzy = true,
            UseFallbacks = true,
            EnableFixups = true,
            RemapPath = null,
            Verbose = false,
            TrackAssets = false,
            DbdDir = options.DbdDirectory,
            DbctoolOutRoot = session.SharedCrosswalkRoot,
            DbctoolSrcAlias = alias,
            DbctoolSrcDir = Path.Combine(options.AlphaRoot, version, "tree", "DBFilesClient"),
            DbctoolLkDir = ResolveLkDbcDirectory(options),
            DbctoolPatchDir = Path.Combine(session.SharedCrosswalkRoot, alias, build, "compare"),
            DbctoolPatchFile = null,
            VizSvg = false,
            VizHtml = false,
            VizDir = null,
            PatchOnly = false,
            NoZoneFallback = false,
        };

        try
        {
            AdtExportPipeline.ExportSingle(pipelineOptions);
        }
        catch (Exception ex)
        {
            return new AdtStageResult(
                Map: map,
                Version: version,
                Success: false,
                TilesProcessed: 0,
                AreaIdsPatched: 0,
                AdtOutputDirectory: exportDir,
                Error: ex.Message);
        }

        var tilesProcessed = CountGeneratedTiles(exportDir, map);
        var areaIdsPatched = CountPatchedAreaIds(exportDir, map);

        return new AdtStageResult(
            Map: map,
            Version: version,
            Success: true,
            TilesProcessed: tilesProcessed,
            AreaIdsPatched: areaIdsPatched,
            AdtOutputDirectory: exportDir,
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

    private static int CountGeneratedTiles(string exportDir, string map)
    {
        var mapDir = Path.Combine(exportDir, "World", "Maps", map);
        if (!Directory.Exists(mapDir))
        {
            return 0;
        }

        return Directory.EnumerateFiles(mapDir, "*.adt", SearchOption.TopDirectoryOnly).Count();
    }

    private static int CountPatchedAreaIds(string exportDir, string map)
    {
        var csvDir = Path.Combine(exportDir, "csv", "maps", map);
        if (!Directory.Exists(csvDir))
        {
            return 0;
        }

        var areaPatchPath = Path.Combine(csvDir, "area_patch_crosswalk.csv");
        if (!File.Exists(areaPatchPath))
        {
            return 0;
        }

        var lines = File.ReadAllLines(areaPatchPath);
        return Math.Max(0, lines.Length - 1);
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
