using System;
using System.IO;

namespace WoWRollback.WDLtoGLB;

/// <summary>
/// First-class WDL → GLB converter entry point.
/// Ports parsing and GLB generation with optional texture bake.
/// </summary>
public static class WdlToGlbConverter
{
    public static int Convert(string mapName, string wdlPath, string outputGlb, WdlToGlbOptions options)
    {
        if (string.IsNullOrWhiteSpace(wdlPath)) throw new ArgumentException("wdlPath is required", nameof(wdlPath));
        if (string.IsNullOrWhiteSpace(outputGlb)) throw new ArgumentException("outputGlb is required", nameof(outputGlb));

        if (!File.Exists(wdlPath))
        {
            Console.Error.WriteLine($"[error] WDL file not found: {wdlPath}");
            return 2;
        }

        var outDir = Path.GetDirectoryName(Path.GetFullPath(outputGlb)) ?? Directory.GetCurrentDirectory();
        if (!Directory.Exists(outDir)) Directory.CreateDirectory(outDir);

        // Read WDL
        Console.WriteLine("[info] Reading WDL...");
        var wdl = WdlReader.Parse(wdlPath);

        // Choose export mode: per-tile if TRS (minimap-root) or a minimap folder is provided; otherwise merged GLB
        var hasPerTile = !string.IsNullOrWhiteSpace(options.MinimapRoot) || !string.IsNullOrWhiteSpace(options.MinimapFolder);
        if (hasPerTile)
        {
            // Resolve tiles output directory
            string tilesDir;
            if (string.Equals(Path.GetExtension(outputGlb), ".glb", StringComparison.OrdinalIgnoreCase))
            {
                var baseDir = Path.GetDirectoryName(Path.GetFullPath(outputGlb)) ?? Directory.GetCurrentDirectory();
                var stem = Path.GetFileNameWithoutExtension(outputGlb);
                tilesDir = Path.Combine(baseDir, stem + "_tiles");
            }
            else
            {
                tilesDir = Path.GetFullPath(outputGlb);
            }

            Console.WriteLine($"[info] Exporting per-tile GLBs to: {tilesDir}");
            if (!string.IsNullOrWhiteSpace(options.MinimapRoot))
            {
                // TRS auto-detection is handled by exporter, but log provided roots/paths here for visibility
                if (!string.IsNullOrWhiteSpace(options.TrsPath)) Console.WriteLine($"[info] Using TRS: {options.TrsPath}");
                else Console.WriteLine($"[info] Will auto-detect md5translate.(trs|txt) under: {options.MinimapRoot}");
            }
            var perTileOpts = new WdlGltfExporter.ExportOptions(
                Scale: options.Scale,
                SkipHoles: true,
                NormalizeWorld: true,
                HeightScale: 1.0,
                TexturePath: null, // per-tile images override this
                MapName: options.MapName,
                MinimapFolder: options.MinimapFolder,
                MinimapRoot: options.MinimapRoot,
                TrsPath: options.TrsPath
            );
            var stats = WdlGltfExporter.ExportPerTile(wdl, tilesDir, perTileOpts);
            Console.WriteLine($"[ok] Per-tile GLBs written under: {tilesDir}");
            Console.WriteLine($"[ok] Tiles={stats.TilesExported}, ApproxVertices={stats.VerticesApprox}, Faces={stats.FacesWritten}");
        }
        else
        {
            // Export merged GLB (single texture if provided)
            Console.WriteLine("[info] Exporting merged GLB terrain...");
            var exportOpts = new WdlGltfExporter.ExportOptions(
                Scale: options.Scale,
                SkipHoles: true,
                NormalizeWorld: true,
                HeightScale: 1.0,
                TexturePath: options.TextureOverridePath
            );
            var stats = WdlGltfExporter.ExportMerged(wdl, outputGlb, exportOpts);
            Console.WriteLine($"[ok] GLB written: {outputGlb}");
            Console.WriteLine($"[ok] Tiles={stats.TilesExported}, ApproxVertices={stats.VerticesApprox}, Faces={stats.FacesWritten}");
        }
        return 0;
    }
}
