using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AlphaWdtAnalyzer.Core;
using WoWRollback.Core.IO;
using WoWRollback.Core.Logging;
using WoWRollback.Core.Models;
using WoWRollback.Core.Services.Viewer;

namespace WoWRollback.Orchestrator;

internal sealed class ViewerStageRunner
{
    private const string ViewerAssetsSourcePath = "WoWRollback.Viewer/assets2d";

    public ViewerStageResult Run(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
    {
        if (session is null)
        {
            throw new ArgumentNullException(nameof(session));
        }

        try
        {
            Directory.CreateDirectory(session.Paths.ViewerDir);

            // Build a slim viewer pack directly (tiles + minimal overlays scaffolds + index.json)
            var packOut = session.Paths.ViewerDir; // pack root contains index.json and tiles/
            var versions = session.Options.Versions;
            var mapSet = new HashSet<string>(adtResults.Where(r => r.Success).Select(r => r.Map), StringComparer.OrdinalIgnoreCase);

            var builder = new ViewerPackBuilder();
            var result = builder.Build(
                sessionRoot: session.Options.AlphaRoot,
                minimapRoot: session.Options.MinimapRoot ?? session.Options.AlphaRoot,
                outRoot: packOut,
                versionsFilter: versions,
                mapsFilter: mapSet,
                label: session.Options.ViewerLabel ?? (versions.FirstOrDefault() ?? "dev"));

            // Auto-harvest overlays from converted ADT outputs (no external adt-root needed)
            var inputs = new List<(string Version, string Map, string MapDir)>();
            foreach (var r in adtResults.Where(r => r.Success))
            {
                var mapDir = Path.Combine(session.Paths.AdtDir, r.Version, "World", "Maps", r.Map);
                if (Directory.Exists(mapDir)) inputs.Add((r.Version, r.Map, mapDir));
            }

            var overlaysRoot = Path.Combine(packOut, "data", "overlays");
            var builder2 = new ViewerPackBuilder();
            builder2.HarvestFromConvertedAdts(inputs, session.Options.CommunityListfile, session.Options.LkListfile, overlaysRoot);

            // Ensure the UI assets (index.html, js, css) are present at the pack root for the server
            TryCopyViewerAssetsTo(packOut);

            // Optional: basic overlays metadata for bookkeeping
            var overlayCount = GenerateOverlayMetadata(session, adtResults);

            return new ViewerStageResult(
                Success: true,
                ViewerDirectory: packOut,
                OverlayCount: overlayCount,
                Notes: $"viewer-pack: tiles={result.TilesWritten}, maps={result.MapsWritten}, defaultMap={result.DefaultMap}");
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

    // Legacy viewer assets copy removed for slim 2D viewer (UI is served separately by viewer-serve)

    private static void GenerateViewerDataFiles(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
    {
        // No-op: index.json is written by ViewerPackBuilder at pack root
    }

    // Minimap tiles are composed by ViewerPackBuilder; legacy generator removed

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

    private static void TryCopyViewerAssetsTo(string destinationRoot)
    {
        try
        {
            var candidates = new List<string>();
            // 1) From current working directory when running `dotnet run` inside WoWRollback
            var cwd = Directory.GetCurrentDirectory();
            var cand1 = Path.Combine(cwd, "WoWRollback.Viewer", "assets2d");
            candidates.Add(cand1);
            // 2) From bin path back to repo structure: Orchestrator/bin/Debug/netX -> up 4 -> WoWRollback
            var baseDir = AppContext.BaseDirectory;
            var repoRoot = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", ".."));
            var cand2 = Path.Combine(repoRoot, "WoWRollback", "WoWRollback.Viewer", "assets2d");
            candidates.Add(cand2);
            // 3) Parent of CWD (solution root) -> WoWRollback/WoWRollback.Viewer/assets2d
            var parent = Directory.GetParent(cwd)?.FullName;
            if (!string.IsNullOrEmpty(parent))
            {
                var cand3 = Path.Combine(parent, "WoWRollback", "WoWRollback.Viewer", "assets2d");
                candidates.Add(cand3);
            }

            ConsoleLogger.Info($"Viewer assets copy: candidates:\n  - {string.Join("\n  - ", candidates)}");

            string? src = candidates.FirstOrDefault(Directory.Exists);
            if (src is null)
            {
                ConsoleLogger.Warn("Viewer assets not found; UI will 404. Ensure WoWRollback.Viewer/assets2d exists.");
                // Write a minimal fallback index.html to avoid 404 and provide guidance
                var fallback = "<!doctype html><html><head><meta charset=\"utf-8\"><title>WoWRollback Viewer</title></head>" +
                               "<body style=\"font-family:system-ui,Arial,sans-serif;padding:20px;color:#eee;background:#121212\">" +
                               "<h2>Viewer UI assets not found</h2>" +
                               "<p>The server is running, but index.html and JS were not copied to the viewer pack.</p>" +
                               "<p>Expected source: WoWRollback/WoWRollback.Viewer/assets2d</p>" +
                               "<p>Please ensure the assets exist and re-run. Data endpoints like <code>/data/index.json</code> should still work.</p>" +
                               "</body></html>";
                Directory.CreateDirectory(destinationRoot);
                File.WriteAllText(Path.Combine(destinationRoot, "index.html"), fallback);
                return;
            }

            CopyDirectoryRecursive(src, destinationRoot);

            var indexHtml = Path.Combine(destinationRoot, "index.html");
            if (File.Exists(indexHtml))
            {
                ConsoleLogger.Success($"Viewer UI copied: {indexHtml}");
            }
            else
            {
                ConsoleLogger.Warn($"Viewer UI copy attempted from '{src}' but index.html not found at destination.");
            }
        }
        catch
        {
            // Non-fatal; server will 404 if UI missing
        }
    }

    private static void CopyDirectoryRecursive(string sourceDir, string destinationDir)
    {
        Directory.CreateDirectory(destinationDir);
        foreach (var file in Directory.EnumerateFiles(sourceDir, "*", SearchOption.TopDirectoryOnly))
        {
            var dst = Path.Combine(destinationDir, Path.GetFileName(file));
            File.Copy(file, dst, overwrite: true);
        }
        foreach (var dir in Directory.EnumerateDirectories(sourceDir, "*", SearchOption.TopDirectoryOnly))
        {
            var name = Path.GetFileName(dir);
            if (string.IsNullOrWhiteSpace(name)) continue;
            CopyDirectoryRecursive(dir, Path.Combine(destinationDir, name));
        }
    }
}

internal sealed record ViewerStageResult(
    bool Success,
    string ViewerDirectory,
    int OverlayCount,
    string? Notes);

internal sealed class TileInfo
{
    [System.Text.Json.Serialization.JsonPropertyName("row")]
    public int Row { get; set; }
    
    [System.Text.Json.Serialization.JsonPropertyName("col")]
    public int Col { get; set; }
    
    [System.Text.Json.Serialization.JsonPropertyName("versions")]
    public string[] Versions { get; set; } = Array.Empty<string>();
}
