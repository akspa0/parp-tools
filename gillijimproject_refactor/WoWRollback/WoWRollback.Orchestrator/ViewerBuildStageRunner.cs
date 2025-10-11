using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using WoWRollback.Core.Logging;
using WoWRollback.Core.Services.Viewer;

namespace WoWRollback.Orchestrator;

internal sealed class ViewerBuildStageRunner
{
    private const string AssetsFolderName = "assets2d";

    public ViewerStageResult Run(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
    {
        if (session is null) throw new ArgumentNullException(nameof(session));

        try
        {
            var packOut = session.Paths.ViewerDir;
            Directory.CreateDirectory(packOut);

            // Open persistent stage log
            var logPath = Path.Combine(session.Paths.LogsDir, "viewer_stage.log");
            Directory.CreateDirectory(session.Paths.LogsDir);
            using var log = new StreamWriter(logPath, append: true);
            log.WriteLine($"[viewer] session={session.SessionId} packOut={packOut}");
            log.WriteLine($"[viewer] alphaRoot={session.Options.AlphaRoot}");
            log.WriteLine($"[viewer] minimapRoot={session.Options.MinimapRoot ?? session.Options.AlphaRoot}");
            log.WriteLine($"[viewer] versions={string.Join(",", session.Options.Versions)} mapsFilter={string.Join(",", adtResults.Where(r=>r.Success).Select(r=>r.Map).Distinct())}");

            // Build data pack under data/
            var mapsFilter = new HashSet<string>(adtResults.Where(r => r.Success).Select(r => r.Map), StringComparer.OrdinalIgnoreCase);
            var builder = new ViewerPackBuilder();
            var buildResult = builder.Build(
                sessionRoot: session.Options.AlphaRoot,
                minimapRoot: session.Options.MinimapRoot ?? session.Options.AlphaRoot,
                outRoot: packOut,
                versionsFilter: session.Options.Versions,
                mapsFilter: mapsFilter,
                label: session.Options.ViewerLabel ?? (session.Options.Versions.FirstOrDefault() ?? "dev"));

            log.WriteLine($"[viewer] build: tiles={buildResult.TilesWritten} maps={buildResult.MapsWritten} defaultMap={buildResult.DefaultMap}");

            // Harvest overlays (LK-only) from converted ADTs into data/overlays
            var inputs = new List<(string Version, string Map, string MapDir)>();
            foreach (var r in adtResults.Where(r => r.Success))
            {
                var mapDir = Path.Combine(session.Paths.AdtDir, r.Version, "World", "Maps", r.Map);
                if (Directory.Exists(mapDir)) inputs.Add((r.Version, r.Map, mapDir));
            }
            var overlaysRoot = Path.Combine(packOut, "data", "overlays");
            builder.HarvestFromConvertedAdts(inputs, session.Options.CommunityListfile, session.Options.LkListfile, overlaysRoot);
            log.WriteLine($"[viewer] overlays harvested to: {overlaysRoot}");

            // Copy UI assets (assets2d) into pack root (prefer explicit --viewer-assets when provided)
            TryCopyViewerAssetsToPackRoot(packOut, session.Options.ViewerAssetsPath, log);

            // Write metadata for diagnostics
            var overlayCount = GenerateOverlayMetadata(session, adtResults);

            // Hard-verify pack structure and log example URLs
            HardVerifyPack(packOut, buildResult.DefaultMap, log);
            log.Flush();

            return new ViewerStageResult(
                Success: true,
                ViewerDirectory: packOut,
                OverlayCount: overlayCount,
                Notes: $"viewer-pack: tiles={buildResult.TilesWritten}, maps={buildResult.MapsWritten}, defaultMap={buildResult.DefaultMap}");
        }
        catch (Exception ex)
        {
            try
            {
                var logPath = Path.Combine(session.Paths.LogsDir, "viewer_stage.log");
                Directory.CreateDirectory(session.Paths.LogsDir);
                File.AppendAllText(logPath, $"[viewer][error] {ex}\n");
            }
            catch { /* ignore logging failures */ }
            return new ViewerStageResult(false, session.Paths.ViewerDir, 0, $"Viewer build failed: {ex.Message}");
        }
    }

    private static int GenerateOverlayMetadata(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
    {
        var overlaysDir = Path.Combine(session.Paths.ViewerDir, "data");
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
        return 1;
    }

    private static void TryCopyViewerAssetsToPackRoot(string packRoot, string? explicitAssetsPath, StreamWriter? log)
    {
        // 1) Use explicit path if provided
        if (!string.IsNullOrWhiteSpace(explicitAssetsPath))
        {
            var srcPath = Path.GetFullPath(explicitAssetsPath);
            if (Directory.Exists(srcPath))
            {
                ConsoleLogger.Info($"Viewer assets copy (explicit): '{srcPath}' -> '{packRoot}'");
                log?.WriteLine($"[viewer] assets explicit: {srcPath} -> {packRoot}");
                CopyDirectoryRecursive(srcPath, packRoot);
                var indexHtmlExp = Path.Combine(packRoot, "index.html");
                if (File.Exists(indexHtmlExp))
                {
                    ConsoleLogger.Success($"Viewer UI copied (explicit): {indexHtmlExp}");
                    log?.WriteLine($"[viewer] index.html present: {indexHtmlExp}");
                }
                else
                {
                    ConsoleLogger.Warn($"Explicit assets copied from '{srcPath}' but index.html not found in destination.");
                    log?.WriteLine($"[viewer][warn] explicit assets missing index.html");
                }
                return;
            }
            else
            {
                ConsoleLogger.Warn($"Explicit --viewer-assets path does not exist: {srcPath}. Falling back to auto-detection.");
                log?.WriteLine($"[viewer][warn] explicit assets path missing: {srcPath}");
            }
        }

        var candidates = new List<string>();
        var cwd = Directory.GetCurrentDirectory();
        candidates.Add(Path.Combine(cwd, "WoWRollback.Viewer", AssetsFolderName));
        var baseDir = AppContext.BaseDirectory;
        var repoRoot = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", ".."));
        candidates.Add(Path.Combine(repoRoot, "WoWRollback", "WoWRollback.Viewer", AssetsFolderName));
        var parent = Directory.GetParent(cwd)?.FullName;
        if (!string.IsNullOrEmpty(parent))
            candidates.Add(Path.Combine(parent, "WoWRollback", "WoWRollback.Viewer", AssetsFolderName));

        ConsoleLogger.Info($"Viewer assets copy: candidates:\n  - {string.Join("\n  - ", candidates)}");
        log?.WriteLine($"[viewer] asset candidates: {string.Join("; ", candidates)}");
        var src = candidates.FirstOrDefault(Directory.Exists);
        if (src is null)
        {
            ConsoleLogger.Warn("Viewer assets not found; writing fallback index.html.");
            log?.WriteLine("[viewer][warn] no assets found; writing fallback index.html");
            var fallback = "<!doctype html><html><head><meta charset=\"utf-8\"><title>WoWRollback Viewer</title></head>" +
                           "<body style=\"font-family:system-ui,Arial,sans-serif;padding:20px;color:#eee;background:#121212\">" +
                           "<h2>Viewer UI assets not found</h2>" +
                           "<p>The server is running, but index.html and JS were not copied to the viewer pack.</p>" +
                           "<p>Expected source: WoWRollback/WoWRollback.Viewer/assets2d</p>" +
                           "<p>Data endpoints like <code>/data/index.json</code> should still work.</p>" +
                           "</body></html>";
            Directory.CreateDirectory(packRoot);
            File.WriteAllText(Path.Combine(packRoot, "index.html"), fallback);
            ConsoleLogger.Info($"Fallback index.html written: {Path.Combine(packRoot, "index.html")}");
            log?.WriteLine($"[viewer] fallback index.html -> {Path.Combine(packRoot, "index.html")}");
            return;
        }

        CopyDirectoryRecursive(src, packRoot);
        var indexHtml = Path.Combine(packRoot, "index.html");
        if (File.Exists(indexHtml))
        {
            ConsoleLogger.Success($"Viewer UI copied: {indexHtml}");
            log?.WriteLine($"[viewer] index.html present: {indexHtml}");
        }
        else
        {
            ConsoleLogger.Warn($"Assets copied from '{src}' but index.html not found in destination. Writing fallback.");
            log?.WriteLine($"[viewer][warn] auto assets missing index.html; writing fallback");
            var fallback = "<!doctype html><html><head><meta charset=\"utf-8\"><title>WoWRollback Viewer</title></head>" +
                           "<body style=\"font-family:system-ui,Arial,sans-serif;padding:20px;color:#eee;background:#121212\">" +
                           "<h2>Viewer UI assets missing index.html</h2>" +
                           "<p>The server is running, but index.html was not present in the copied assets.</p>" +
                           "<p>Data endpoints like <code>/data/index.json</code> should still work.</p>" +
                           "</body></html>";
            File.WriteAllText(indexHtml, fallback);
            ConsoleLogger.Info($"Fallback index.html written: {indexHtml}");
            log?.WriteLine($"[viewer] fallback index.html -> {indexHtml}");
        }
    }

    private static void HardVerifyPack(string packRoot, string defaultMap, StreamWriter? log)
    {
        // Verify existence of index.html and data/index.json
        var indexHtml = Path.Combine(packRoot, "index.html");
        var dataIndex = Path.Combine(packRoot, "data", "index.json");
        var tilesDir = Path.Combine(packRoot, "data", "tiles", defaultMap);
        var overlaysManifest = Path.Combine(packRoot, "data", "overlays", defaultMap, "manifest.json");

        if (File.Exists(indexHtml)) { ConsoleLogger.Success($"[verify] index.html found: {indexHtml}"); log?.WriteLine($"[verify] index.html: OK"); }
        else { ConsoleLogger.Warn($"[verify] index.html missing at pack root: {indexHtml}"); log?.WriteLine($"[verify] index.html: MISSING"); }

        if (File.Exists(dataIndex)) { ConsoleLogger.Success($"[verify] data/index.json found: {dataIndex}"); log?.WriteLine($"[verify] data/index.json: OK"); }
        else { ConsoleLogger.Warn($"[verify] data/index.json missing: {dataIndex}"); log?.WriteLine($"[verify] data/index.json: MISSING"); }

        var zeroZero = Path.Combine(tilesDir, "0_0.webp");
        if (File.Exists(zeroZero))
        {
            ConsoleLogger.Success($"[verify] Tile found: {zeroZero}");
            log?.WriteLine($"[verify] tile 0_0.webp: OK");
        }
        else
        {
            // Fallback: any tile present
            var anyTile = Directory.Exists(tilesDir) ? Directory.EnumerateFiles(tilesDir, "*.webp").FirstOrDefault() : null;
            if (!string.IsNullOrEmpty(anyTile)) { ConsoleLogger.Success($"[verify] Tile present: {anyTile}"); log?.WriteLine($"[verify] tile any: OK"); }
            else { ConsoleLogger.Warn($"[verify] No tiles found under: {tilesDir}"); log?.WriteLine($"[verify] tiles: NONE"); }
        }

        if (File.Exists(overlaysManifest)) { ConsoleLogger.Success($"[verify] overlays manifest found: {overlaysManifest}"); log?.WriteLine($"[verify] overlays manifest: OK"); }
        else { ConsoleLogger.Warn($"[verify] overlays manifest missing: {overlaysManifest}"); log?.WriteLine($"[verify] overlays manifest: MISSING"); }

        // Example URLs
        ConsoleLogger.Info($"[verify] Example URLs:\n  / -> viewer UI\n  /data/index.json\n  /data/tiles/{defaultMap}/0_0.webp\n  /data/overlays/{defaultMap}/coords/0_0.json");
        log?.WriteLine($"[verify] urls: /, /data/index.json, /data/tiles/{defaultMap}/0_0.webp, /data/overlays/{defaultMap}/coords/0_0.json");
    }

    private static void CopyDirectoryRecursive(string sourceDir, string destinationDir)
    {
        Directory.CreateDirectory(destinationDir);
        foreach (var file in Directory.EnumerateFiles(sourceDir))
        {
            File.Copy(file, Path.Combine(destinationDir, Path.GetFileName(file)), overwrite: true);
        }
        foreach (var dir in Directory.EnumerateDirectories(sourceDir))
        {
            var name = Path.GetFileName(dir);
            if (string.IsNullOrWhiteSpace(name)) continue;
            CopyDirectoryRecursive(dir, Path.Combine(destinationDir, name));
        }
    }
}
