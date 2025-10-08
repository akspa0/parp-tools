using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using AlphaWdtAnalyzer.Core;
using WoWRollback.Core.IO;
using WoWRollback.Core.Models;
using WoWRollback.Core.Services.Viewer;

namespace WoWRollback.Orchestrator;

internal sealed class ViewerStageRunner
{
    private const string ViewerAssetsSourcePath = "WoWRollback.Viewer/assets";

    public ViewerStageResult Run(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
    {
        if (session is null)
        {
            throw new ArgumentNullException(nameof(session));
        }

        try
        {
            Directory.CreateDirectory(session.Paths.ViewerDir);

            // Copy existing viewer assets (index.html, JS, CSS, etc.)
            CopyViewerAssets(session);

            // Generate viewer data files (index.json, config.json)
            GenerateViewerDataFiles(session, adtResults);

            // Generate overlay metadata
            var overlayCount = GenerateOverlayMetadata(session, adtResults);

            return new ViewerStageResult(
                Success: true,
                ViewerDirectory: session.Paths.ViewerDir,
                OverlayCount: overlayCount,
                Notes: $"Copied viewer assets and generated {overlayCount} overlay(s)");
        }
        catch (Exception ex)
        {
            return new ViewerStageResult(
                Success: false,
                ViewerDirectory: session.Paths.ViewerDir,
                OverlayCount: 0,
                Notes: $"Viewer generation failed: {ex.Message}");
        }
    }

    private static void CopyViewerAssets(SessionContext session)
    {
        var sourceDir = Path.GetFullPath(ViewerAssetsSourcePath);
        
        if (!Directory.Exists(sourceDir))
        {
            throw new DirectoryNotFoundException($"Viewer assets not found at: {sourceDir}");
        }

        // Copy all viewer assets to session viewer directory
        FileHelpers.CopyDirectory(sourceDir, session.Paths.ViewerDir, overwrite: true);
    }

    private static void GenerateViewerDataFiles(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
    {
        // Generate index.json - catalog of maps/tiles/versions
        var indexData = new
        {
            maps = session.Options.Maps.ToArray(),
            versions = session.Options.Versions.Select(v => new
            {
                version = v,
                alias = DbcStageRunner.DeriveAlias(v)
            }).ToArray(),
            tiles = adtResults
                .Where(r => r.Success)
                .GroupBy(r => r.Map)
                .ToDictionary(
                    g => g.Key,
                    g => g.Select(r => new { version = r.Version, tiles = r.TilesProcessed }).ToArray()
                )
        };

        var indexPath = Path.Combine(session.Paths.ViewerDir, "index.json");
        File.WriteAllText(indexPath, JsonSerializer.Serialize(indexData, new JsonSerializerOptions { WriteIndented = true }));

        // Generate config.json - viewer configuration
        var configData = new
        {
            default_version = session.Options.Versions.FirstOrDefault() ?? "0.5.3",
            default_map = session.Options.Maps.FirstOrDefault() ?? "Kalimdor",
            tile_size = 512,
            diff_thresholds = new
            {
                proximity = 10.0,
                moved_epsilon = 0.005
            }
        };

        var configPath = Path.Combine(session.Paths.ViewerDir, "config.json");
        File.WriteAllText(configPath, JsonSerializer.Serialize(configData, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static int GenerateOverlayMetadata(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
    {
        var overlaysDir = Path.Combine(session.Paths.ViewerDir, "overlays");
        Directory.CreateDirectory(overlaysDir);

        var metadata = new
        {
            session_id = session.SessionId,
            maps = session.Options.Maps,
            versions = session.Options.Versions,
            adt_results = adtResults.Select(r => new
            {
                map = r.Map,
                version = r.Version,
                success = r.Success,
                tiles_processed = r.TilesProcessed,
                area_ids_patched = r.AreaIdsPatched
            }).ToList()
        };

        var metadataPath = Path.Combine(overlaysDir, "metadata.json");
        var options = new JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(metadataPath, JsonSerializer.Serialize(metadata, options));

        return 1; // metadata.json counts as 1 overlay
    }
}

internal sealed record ViewerStageResult(
    bool Success,
    string ViewerDirectory,
    int OverlayCount,
    string? Notes);
