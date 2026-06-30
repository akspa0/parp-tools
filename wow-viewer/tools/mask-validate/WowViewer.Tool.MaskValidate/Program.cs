using System.Numerics;
using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Tools.MaskValidate;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length < 2)
        {
            PrintUsage();
            return 1;
        }

        // Two modes:
        //   1. Loose ADT:  <adt-file-path> <client-root> [output-dir] [listfilePath]
        //   2. Archive ADT: --archive <client-root> <adt-virtual-path> [output-dir] [listfilePath]
        //
        // The archive mode builds the ADT/texture-source bytes via ArchiveVirtualFileReader,
        // which works for MPQ-backed clients (3.3.5, Cataclysm). The asset reader path is
        // shared between both modes for model-file reads.

        bool archiveMode = args[0].Equals("--archive", StringComparison.OrdinalIgnoreCase);
        string adtPath;
        string clientRoot;
        string outputDir;
        string? listfilePath = null;

        if (archiveMode)
        {
            if (args.Length < 3)
            {
                PrintUsage();
                return 1;
            }

            clientRoot = args[1];
            adtPath = args[2];
            outputDir = args.Length > 3 ? args[3] : Path.Combine(Directory.GetCurrentDirectory(), "mask_validation_output");
            if (args.Length > 4)
                listfilePath = args[4];
        }
        else
        {
            adtPath = args[0];
            clientRoot = args[1];
            outputDir = args.Length > 2 ? args[2] : Path.Combine(Directory.GetCurrentDirectory(), "mask_validation_output");
            if (args.Length > 3)
                listfilePath = args[3];
        }

        Directory.CreateDirectory(outputDir);

        Console.WriteLine($"ADT:         {adtPath}");
        Console.WriteLine($"Client root: {clientRoot}");
        Console.WriteLine($"Output dir:  {outputDir}");
        Console.WriteLine($"Listfile:    {(listfilePath ?? "<none>")}");
        Console.WriteLine($"Mode:        {(archiveMode ? "archive (BuildFromBytes)" : "loose (Build from file)")}");

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"ERROR: Client root directory not found: {clientRoot}");
            return 1;
        }

        string clientRootFull = Path.GetFullPath(clientRoot);
        string? resolvedListfile = !string.IsNullOrWhiteSpace(listfilePath) && File.Exists(listfilePath)
            ? listfilePath
            : null;

        // Asset reader uses ArchiveVirtualFileReader so M2/MDX/WMO files can be read from
        // either loose folders or MPQ archives using the same code path as harvest/inspect.
        Func<string, byte[]?> assetReader = virtualPath => ReadAssetSafe(virtualPath, clientRootFull, resolvedListfile);

        Console.WriteLine("Building tensor pack...");
        TerrainTileTensorPack pack;
        try
        {
            if (archiveMode)
            {
                string adtVirtual = NormalizeVirtualPath(adtPath);
                byte[] adtBytes = ArchiveVirtualFileReader.ReadVirtualFile(adtVirtual, [clientRootFull], resolvedListfile);
                string adtSourcePath = Path.Combine(clientRootFull, adtVirtual.Replace('\\', Path.DirectorySeparatorChar));

                // Try to read tex0 companion for texture data.
                string texVirtual = Path.ChangeExtension(adtVirtual, null) + "_tex0.adt";
                byte[]? tex0Bytes = TryReadAssetSafe(texVirtual, clientRootFull, resolvedListfile);

                // Try to read obj0 companion for placement data (Cataclysm+ split-ADT).
                string objVirtual = Path.ChangeExtension(adtVirtual, null) + "_obj0.adt";
                byte[]? obj0Bytes = TryReadAssetSafe(objVirtual, clientRootFull, resolvedListfile);

                pack = AdtTensorPackBuilder.BuildFromBytes(
                    adtSourcePath,
                    adtBytes,
                    tex0Bytes,
                    obj0Bytes,
                    null,
                    Path.Combine(clientRootFull, texVirtual.Replace('\\', Path.DirectorySeparatorChar)),
                    Path.Combine(clientRootFull, objVirtual.Replace('\\', Path.DirectorySeparatorChar)),
                    assetReader);
            }
            else
            {
                if (!File.Exists(adtPath))
                {
                    Console.Error.WriteLine($"ERROR: ADT file not found: {adtPath}");
                    return 1;
                }

                pack = AdtTensorPackBuilder.Build(adtPath, null, null, assetReader);
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"ERROR: Tensor pack build failed: {ex}");
            return 2;
        }

        // Try to load minimap from archive if not already present in the pack
        if (pack.MinimapRgb256 is null && archiveMode)
        {
            Console.WriteLine($"  Trying minimap load: MapName='{pack.MapName}' TileX={pack.TileX} TileY={pack.TileY}");
            byte[,,]? minimapRgb = TryLoadMinimapFromArchive(clientRootFull, resolvedListfile, pack.MapName, pack.TileX, pack.TileY);
            if (minimapRgb is not null)
            {
                pack.MinimapRgb256 = minimapRgb;
                Console.WriteLine("  Loaded minimap from archive.");
            }
            else
            {
                Console.WriteLine("  Minimap load from archive returned null.");
            }
        }

        string tileName = pack.TileName;
        string tileOutputDir = Path.Combine(outputDir, tileName);
        Directory.CreateDirectory(tileOutputDir);

        Console.WriteLine($"Tile: {tileName}");
        Console.WriteLine($"Placements: MDDF (M2/MDX) = {pack.PlacementMddfCount}, MODF (WMO) = {pack.PlacementModfCount}");
        Console.WriteLine($"Available signals: {string.Join(", ", pack.AvailableSignals.OrderBy(s => s))}");

        WriteMaskPng(tileOutputDir, tileName, "object_mask", pack.ObjectMask257);
        WriteMaskPng(tileOutputDir, tileName, "object_precise_mask", pack.ObjectPreciseMask257);
        WriteMaskPng(tileOutputDir, tileName, "mddf_mask", pack.MddfMask257);
        WriteMaskPng(tileOutputDir, tileName, "modf_mask", pack.ModfMask257);
        WriteMaskPng(tileOutputDir, tileName, "object_filtered_mask", pack.ObjectFilteredMask257);
        WriteInstanceMaskPng(tileOutputDir, tileName, "object_instance_mask", pack.ObjectInstanceMask257);

        if (pack.MinimapRgb256 is { } minimap && minimap.GetLength(0) == 256 && minimap.GetLength(1) == 256)
        {
            WriteMinimapPng(tileOutputDir, tileName, minimap);
            Console.WriteLine("Wrote minimap.");
        }
        else
        {
            Console.WriteLine("Minimap not available for this tile.");
        }

        CreateComposite(tileOutputDir, tileName, pack);

        Console.WriteLine("Done.");
        return 0;
    }

    private static byte[,,]? TryLoadMinimapFromArchive(string clientRoot, string? listfile, string mapName, int? tileX, int? tileY)
    {
        if (string.IsNullOrWhiteSpace(mapName) || tileX is null || tileY is null)
            return null;

        string mapLower = mapName.ToLowerInvariant();
        string x2 = tileX.Value.ToString("00");
        string y2 = tileY.Value.ToString("00");

        string[] candidates =
        [
            $"textures/minimap/{mapLower}/map{x2}_{y2}.blp",
            $"textures/minimap/{mapLower}/map{y2}_{x2}.blp",
            $"textures\\minimap\\{mapLower}\\map{x2}_{y2}.blp",
            $"textures\\Minimap\\{mapLower}\\map{x2}_{y2}.blp",
            $"world\\minimaps\\{mapLower}\\map{x2}_{y2}.blp",
        ];

        foreach (string candidate in candidates)
        {
            Console.WriteLine($"    Trying minimap path: {candidate}");
            try
            {
                byte[] blpBytes = ArchiveVirtualFileReader.ReadVirtualFile(candidate, [clientRoot], listfile);
                Console.WriteLine($"    Found BLP ({blpBytes.Length} bytes), decoding...");

                using var ms = new MemoryStream(blpBytes, writable: false);
                using var blp = new BlpFile(ms);
                var bitmap = blp.GetBitmap(0);
                if (bitmap is null) { Console.WriteLine("    GetBitmap returned null"); continue; }

                int w = bitmap.Width;
                int h = bitmap.Height;
                Console.WriteLine($"    BLP decoded: {w}x{h}");
                if (w < 1 || h < 1) continue;

                var rgb = new byte[256, 256, 3];
                float scaleX = (float)(w - 1) / 255f;
                float scaleY = (float)(h - 1) / 255f;

                for (int y = 0; y < 256; y++)
                {
                    for (int x = 0; x < 256; x++)
                    {
                        int sx = Math.Clamp((int)(x * scaleX + 0.5f), 0, w - 1);
                        int sy = Math.Clamp((int)(y * scaleY + 0.5f), 0, h - 1);
                        var px = bitmap.GetPixel(sx, sy);
                        rgb[y, x, 0] = px.R;
                        rgb[y, x, 1] = px.G;
                        rgb[y, x, 2] = px.B;
                    }
                }

                return rgb;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"    Failed: {ex.GetType().Name}: {ex.Message}");
            }
        }

        return null;
    }

    private static void PrintUsage()
    {
        Console.Error.WriteLine("Usage (loose):  WowViewer.Tool.MaskValidate <adt-file-path> <client-root> [output-dir] [listfilePath]");
        Console.Error.WriteLine("Usage (archive): WowViewer.Tool.MaskValidate --archive <client-root> <adt-virtual-path> [output-dir] [listfilePath]");
        Console.Error.WriteLine();
        Console.Error.WriteLine("  adt-file-path        Loose .adt file on disk (alpha clients with extracted files).");
        Console.Error.WriteLine("  --archive            Read ADT/tex0/obj0/asset files from MPQ archives via ArchiveVirtualFileReader.");
        Console.Error.WriteLine("  client-root          Game client root directory.");
        Console.Error.WriteLine("  adt-virtual-path      Archive virtual path (e.g. world\\maps\\azeroth\\azeroth_32_48.adt).");
        Console.Error.WriteLine("  output-dir           Where to write PNG images. Default: ./mask_validation_output");
        Console.Error.WriteLine("  listfilePath         Optional community listfile for archive bootstrap.");
        Console.Error.WriteLine();
        Console.Error.WriteLine("Try a tile that has both M2 doodads and a WMO so you can compare the M2 masks (now");
        Console.Error.WriteLine("sized by model bounds) against WMO masks (already accurate via triangle rasterization).");
    }

    private static byte[]? ReadAssetSafe(string virtualPath, string clientRootFull, string? listfilePath)
    {
        try
        {
            return ArchiveVirtualFileReader.ReadVirtualFile(
                NormalizeVirtualPath(virtualPath),
                [clientRootFull],
                listfilePath);
        }
        catch
        {
            return null;
        }
    }

    private static byte[]? TryReadAssetSafe(string virtualPath, string clientRootFull, string? listfilePath)
        => ReadAssetSafe(virtualPath, clientRootFull, listfilePath);

    private static string NormalizeVirtualPath(string path) => path.Trim().Replace('/', '\\').TrimStart('\\');

    private static void WriteMaskPng(string outputDir, string tileName, string maskName, float[,]? mask)
    {
        if (mask is null)
        {
            Console.WriteLine($"  {maskName}: null (not available)");
            return;
        }

        int w = mask.GetLength(1);
        int h = mask.GetLength(0);

        using Image<L8> image = new(w, h);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float v = Math.Clamp(mask[y, x], 0f, 1f);
                image[x, y] = new L8((byte)(v * 255));
            }
        }

        string path = Path.Combine(outputDir, $"{tileName}_{maskName}.png");
        image.Save(path, new PngEncoder());
        Console.WriteLine($"  {maskName}: wrote {w}x{h} PNG -> {path}");
    }

    private static void WriteInstanceMaskPng(string outputDir, string tileName, string maskName, int[,]? mask)
    {
        if (mask is null)
        {
            Console.WriteLine($"  {maskName}: null (not available)");
            return;
        }

        int w = mask.GetLength(1);
        int h = mask.GetLength(0);

        int maxId = 0;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                maxId = Math.Max(maxId, mask[y, x]);

        if (maxId == 0)
        {
            Console.WriteLine($"  {maskName}: no instances painted");
            using Image<L8> image = new(w, h);
            string path = Path.Combine(outputDir, $"{tileName}_{maskName}.png");
            image.Save(path, new PngEncoder());
            Console.WriteLine($"    wrote blank {w}x{h} PNG -> {path}");
            return;
        }

        using Image<Rgba32> colorImage = new(w, h);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int id = mask[y, x];
                if (id == 0)
                {
                    colorImage[x, y] = new Rgba32(0, 0, 0, 255);
                }
                else
                {
                    HsvToRgb(id, out byte r, out byte g, out byte b);
                    colorImage[x, y] = new Rgba32(r, g, b, 255);
                }
            }
        }

        string path2 = Path.Combine(outputDir, $"{tileName}_{maskName}.png");
        colorImage.Save(path2, new PngEncoder());

        Console.WriteLine($"  {maskName}: wrote {w}x{h} color-coded PNG (max instance id = {maxId}) -> {path2}");
    }

    private static void WriteMinimapPng(string outputDir, string tileName, byte[,,] minimap)
    {
        int h = minimap.GetLength(0);
        int w = minimap.GetLength(1);
        using Image<Rgba32> image = new(w, h);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                image[x, y] = new Rgba32(minimap[y, x, 0], minimap[y, x, 1], minimap[y, x, 2], 255);

        string path = Path.Combine(outputDir, $"{tileName}_minimap.png");
        image.Save(path, new PngEncoder());
    }

    /// <summary>
    /// Combines the minimap with each mask as a red overlay so you can see which terrain is covered.
    /// </summary>
    private static void CreateComposite(string outputDir, string tileName, TerrainTileTensorPack pack)
    {
        if (pack.MinimapRgb256 is null)
        {
            Console.WriteLine("Minimap not available; skipping composite overlay images.");
            return;
        }

        int minimapH = pack.MinimapRgb256.GetLength(0);
        int minimapW = pack.MinimapRgb256.GetLength(1);

        float[,]? objectMask = ResizeMask256(pack.ObjectMask257, minimapW, minimapH);
        float[,]? preciseMask = ResizeMask256(pack.ObjectPreciseMask257, minimapW, minimapH);
        float[,]? mddfMask = ResizeMask256(pack.MddfMask257, minimapW, minimapH);
        float[,]? modfMask = ResizeMask256(pack.ModfMask257, minimapW, minimapH);
        float[,]? filteredMask = ResizeMask256(pack.ObjectFilteredMask257, minimapW, minimapH);

        WriteCompositePng(outputDir, tileName, "overlay_object_mask", pack.MinimapRgb256, objectMask, (255, 0, 0));
        WriteCompositePng(outputDir, tileName, "overlay_object_precise_mask", pack.MinimapRgb256, preciseMask, (0, 255, 0));
        WriteCompositePng(outputDir, tileName, "overlay_mddf_mask", pack.MinimapRgb256, mddfMask, (255, 100, 0));
        WriteCompositePng(outputDir, tileName, "overlay_modf_mask", pack.MinimapRgb256, modfMask, (0, 100, 255));
        WriteCompositePng(outputDir, tileName, "overlay_object_filtered_mask", pack.MinimapRgb256, filteredMask, (255, 255, 0));

        WriteSplitOverlayPng(outputDir, tileName, "overlay_split_mddf_modf", pack.MinimapRgb256, mddfMask, modfMask);
    }

    private static void WriteCompositePng(string outputDir, string tileName, string suffix, byte[,,] minimap, float[,]? mask, (byte r, byte g, byte b) overlay)
    {
        if (mask is null)
        {
            Console.WriteLine($"  {suffix}: mask is null, skipping composite");
            return;
        }

        int h = minimap.GetLength(0);
        int w = minimap.GetLength(1);

        using Image<Rgba32> image = new(w, h);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float mv = Math.Clamp(mask[y, x], 0f, 1f);
                byte r = (byte)(minimap[y, x, 0] * (1f - mv) + overlay.r * mv);
                byte g = (byte)(minimap[y, x, 1] * (1f - mv) + overlay.g * mv);
                byte b = (byte)(minimap[y, x, 2] * (1f - mv) + overlay.b * mv);
                image[x, y] = new Rgba32(r, g, b, 255);
            }
        }

        string path = Path.Combine(outputDir, $"{tileName}_{suffix}.png");
        image.Save(path, new PngEncoder());
        Console.WriteLine($"  {suffix}: wrote {w}x{h} composite PNG -> {path}");
    }

    private static void WriteSplitOverlayPng(string outputDir, string tileName, string suffix, byte[,,] minimap, float[,]? redMask, float[,]? blueMask)
    {
        if (redMask is null && blueMask is null)
        {
            Console.WriteLine($"  {suffix}: both masks null, skipping");
            return;
        }

        int h = minimap.GetLength(0);
        int w = minimap.GetLength(1);

        using Image<Rgba32> image = new(w, h);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float redV = Math.Clamp(redMask?[y, x] ?? 0f, 0f, 1f);
                float blueV = Math.Clamp(blueMask?[y, x] ?? 0f, 0f, 1f);

                byte r = (byte)(minimap[y, x, 0] * (1f - redV) + 255 * redV);
                byte g = (byte)(minimap[y, x, 1] * (1f - MathF.Max(redV, blueV)));
                byte b = (byte)(minimap[y, x, 2] * (1f - blueV) + 255 * blueV);

                image[x, y] = new Rgba32(r, g, b, 255);
            }
        }

        string path = Path.Combine(outputDir, $"{tileName}_{suffix}.png");
        image.Save(path, new PngEncoder());
        Console.WriteLine($"  {suffix}: wrote {w}x{h} split overlay PNG (M2=red, WMO=blue) -> {path}");
    }

    private static float[,]? ResizeMask256(float[,]? src, int targetW, int targetH)
    {
        if (src is null)
            return null;

        int srcH = src.GetLength(0);
        int srcW = src.GetLength(1);

        if (srcH == targetH && srcW == targetW)
            return src;

        float[,] dst = new float[targetH, targetW];
        for (int y = 0; y < targetH; y++)
        {
            int sy = (int)((float)y / targetH * srcH);
            sy = Math.Min(sy, srcH - 1);
            for (int x = 0; x < targetW; x++)
            {
                int sx = (int)((float)x / targetW * srcW);
                sx = Math.Min(sx, srcW - 1);
                dst[y, x] = src[sy, sx];
            }
        }

        return dst;
    }

    private static void HsvToRgb(int id, out byte r, out byte g, out byte b)
    {
        float hue = (id * 137.508f) % 360f;
        float sat = 0.75f;
        float val = 0.95f;
        HsvToRgb(hue, sat, val, out r, out g, out b);
    }

    private static void HsvToRgb(float h, float s, float v, out byte r, out byte g, out byte b)
    {
        float c = v * s;
        float x = c * (1 - MathF.Abs((h / 60f) % 2 - 1));
        float m = v - c;

        float rp, gp, bp;
        if (h < 60f) { rp = c; gp = x; bp = 0; }
        else if (h < 120f) { rp = x; gp = c; bp = 0; }
        else if (h < 180f) { rp = 0; gp = c; bp = x; }
        else if (h < 240f) { rp = 0; gp = x; bp = c; }
        else if (h < 300f) { rp = x; gp = 0; bp = c; }
        else { rp = c; gp = 0; bp = x; }

        r = (byte)Math.Clamp((rp + m) * 255, 0, 255);
        g = (byte)Math.Clamp((gp + m) * 255, 0, 255);
        b = (byte)Math.Clamp((bp + m) * 255, 0, 255);
    }
}