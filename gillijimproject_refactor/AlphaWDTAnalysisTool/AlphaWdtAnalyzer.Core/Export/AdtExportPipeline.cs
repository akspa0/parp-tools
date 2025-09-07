using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GillijimProject.WowFiles.Alpha;
using GillijimProject.WowFiles.LichKing;

namespace AlphaWdtAnalyzer.Core.Export;

public static class AdtExportPipeline
{
    public sealed class Options
    {
        public string? SingleWdtPath { get; init; }
        public string? InputRoot { get; init; }
        public string? CommunityListfilePath { get; init; }
        public string? LkListfilePath { get; init; }
        public required string ExportDir { get; init; }
        public required string FallbackTileset { get; init; }
        public required string FallbackNonTilesetBlp { get; init; }
        public required string FallbackWmo { get; init; }
        public required string FallbackM2 { get; init; }
        public bool ConvertToMh2o { get; init; } = true;
        public bool AssetFuzzy { get; init; } = true;
        public string? AreaAlphaPath { get; init; }
        public string? AreaLkPath { get; init; }
    }

    public static void ExportSingle(Options opts)
    {
        if (string.IsNullOrWhiteSpace(opts.SingleWdtPath)) throw new ArgumentException("SingleWdtPath required", nameof(opts.SingleWdtPath));
        Directory.CreateDirectory(opts.ExportDir);

        var resolver = MultiListfileResolver.FromFiles(opts.LkListfilePath, opts.CommunityListfilePath);
        var fixup = new AssetFixupPolicy(
            resolver,
            opts.FallbackTileset,
            opts.FallbackNonTilesetBlp,
            opts.FallbackWmo,
            opts.FallbackM2,
            opts.AssetFuzzy);

        var areaMapper = AreaIdMapper.TryCreate(opts.AreaAlphaPath, opts.AreaLkPath);

        var wdt = new WdtAlphaScanner(opts.SingleWdtPath!);
        var adtScanner = new AdtScanner();
        var result = adtScanner.Scan(wdt);

        // group placements per tile
        var byTile = result.Placements.GroupBy(p => (p.TileX, p.TileY));
        foreach (var g in byTile)
        {
            var (x, y) = g.Key;

            IReadOnlyList<int>? alphaAreaIds = null;
            // Always try to compute alpha area ids so we can emit CSV even without LK mapper
            int adtNum = (y * 64) + x;
            int offset = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
            if (offset > 0)
            {
                var alpha = new AdtAlpha(wdt.WdtPath, offset, adtNum);
                alphaAreaIds = alpha.GetAlphaMcnkAreaIds();

                // Build fixed model name tables (preserve indices) for MMDX/MWMO
                var fixedM2 = wdt.MdnmFiles.Select(n => fixup.Resolve(AssetType.MdxOrM2, n)).ToList();
                var fixedWmo = wdt.MonmFiles.Select(n => fixup.Resolve(AssetType.Wmo, n)).ToList();

                // Convert to LK ADT via existing writer, then patch
                var adtLk = alpha.ToAdtLk(fixedM2, fixedWmo);

                // Prepare patched area IDs (decoded-only). Keep alpha decoded id; do not remap.
                var patched = new int[alphaAreaIds.Count];
                for (int i = 0; i < alphaAreaIds.Count; i++)
                {
                    patched[i] = alphaAreaIds[i]; // may be -1 for empty MCNK; PatchMcnkAreaIds will skip negatives
                }
                adtLk.PatchMcnkAreaIds(patched);

                // Replace MTEX with normalized + fixed textures
                var alphaTextures = alpha.GetMtexTextureNames();
                var fixedTextures = alphaTextures
                    .Select(t => fixup.ResolveTexture(t))
                    .ToList();
                adtLk.ReplaceMtexFromNames(fixedTextures);

                // Write binary ADT next to placeholder artifacts
                var outFile = Path.Combine(opts.ExportDir, $"{wdt.MapName}_{x}_{y}.adt");
                adtLk.ToFile(outFile);
            }

            var ctx = new AdtWotlkWriter.WriteContext
            {
                ExportDir = opts.ExportDir,
                MapName = wdt.MapName,
                TileX = x,
                TileY = y,
                Placements = g,
                Fixup = fixup,
                ConvertToMh2o = opts.ConvertToMh2o,
                AreaMapper = areaMapper,
                AlphaAreaIds = alphaAreaIds
            };
            AdtWotlkWriter.WritePlaceholder(ctx);
        }
    }

    public static void ExportBatch(Options opts)
    {
        if (string.IsNullOrWhiteSpace(opts.InputRoot)) throw new ArgumentException("InputRoot required", nameof(opts.InputRoot));
        Directory.CreateDirectory(opts.ExportDir);

        var resolver = MultiListfileResolver.FromFiles(opts.LkListfilePath, opts.CommunityListfilePath);
        var fixup = new AssetFixupPolicy(
            resolver,
            opts.FallbackTileset,
            opts.FallbackNonTilesetBlp,
            opts.FallbackWmo,
            opts.FallbackM2,
            opts.AssetFuzzy);

        var areaMapper = AreaIdMapper.TryCreate(opts.AreaAlphaPath, opts.AreaLkPath);

        var wdts = Directory.EnumerateFiles(opts.InputRoot!, "*.wdt", SearchOption.AllDirectories)
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase);

        foreach (var wdtPath in wdts)
        {
            try
            {
                var wdt = new WdtAlphaScanner(wdtPath);
                var adtScanner = new AdtScanner();
                var result = adtScanner.Scan(wdt);

                var byTile = result.Placements.GroupBy(p => (p.TileX, p.TileY));
                foreach (var g in byTile)
                {
                    var (x, y) = g.Key;

                    IReadOnlyList<int>? alphaAreaIds = null;
                    int adtNum = (y * 64) + x;
                    int offset = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
                    if (offset > 0)
                    {
                        var alpha = new AdtAlpha(wdt.WdtPath, offset, adtNum);
                        alphaAreaIds = alpha.GetAlphaMcnkAreaIds();

                        // Fixed names for model tables (preserve size/order)
                        var fixedM2 = wdt.MdnmFiles.Select(n => fixup.Resolve(AssetType.MdxOrM2, n)).ToList();
                        var fixedWmo = wdt.MonmFiles.Select(n => fixup.Resolve(AssetType.Wmo, n)).ToList();

                        var adtLk = alpha.ToAdtLk(fixedM2, fixedWmo);

                        // Decoded-only area IDs (no remap). Keep alpha decoded id for all present chunks.
                        var patched = new int[alphaAreaIds.Count];
                        for (int i = 0; i < alphaAreaIds.Count; i++)
                        {
                            patched[i] = alphaAreaIds[i];
                        }
                        adtLk.PatchMcnkAreaIds(patched);

                        var alphaTextures = alpha.GetMtexTextureNames();
                        var fixedTextures = alphaTextures
                            .Select(t => fixup.ResolveTexture(t))
                            .ToList();
                        adtLk.ReplaceMtexFromNames(fixedTextures);

                        var outFile = Path.Combine(opts.ExportDir, $"{wdt.MapName}_{x}_{y}.adt");
                        adtLk.ToFile(outFile);
                    }

                    var ctx = new AdtWotlkWriter.WriteContext
                    {
                        ExportDir = opts.ExportDir,
                        MapName = wdt.MapName,
                        TileX = x,
                        TileY = y,
                        Placements = g,
                        Fixup = fixup,
                        ConvertToMh2o = opts.ConvertToMh2o,
                        AreaMapper = areaMapper,
                        AlphaAreaIds = alphaAreaIds
                    };
                    AdtWotlkWriter.WritePlaceholder(ctx);
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Export failed for {wdtPath}: {ex.Message}");
            }
        }
    }
}
