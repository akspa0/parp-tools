using System.Numerics;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Tools.Harvest;

static class Program
{
    private static Dictionary<string, string>? _md5Lookup;

    static int Main(string[] args)
    {
        if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
        {
            ShowUsage();
            return 0;
        }

        string command = args[0].ToLowerInvariant();
        string[] tail = args.Skip(1).ToArray();

        switch (command)
        {
            case "harvest-tile":
                RunHarvestTile(tail);
                break;
            case "harvest-map":
                RunHarvestMap(tail);
                break;
            case "synthetic-minimap":
                RunSyntheticMinimap(tail);
                break;
            case "extract-unified":
                RunExtractUnified(tail);
                break;
            default:
                Console.Error.WriteLine($"Unknown command '{command}'.");
                ShowUsage();
                return 1;
        }

        return 0;
    }

    static void ShowUsage()
    {
        Console.WriteLine("""
            WowViewer.Tool.Harvest — V14 dataset generation tool

            Usage: WowViewer.Tool.Harvest <command> [options]

            Commands:
              harvest-tile      Extract NPZ shard from a single ADT tile (disk path)
              harvest-map       Batch-extract all tiles from a map directory (disk path)
              extract-unified   Extract NPZ shard from a tile inside MPQ archives
                                (reads tileset BLP + ADT + WDL from MPQ, outputs NPZ shard)
              synthetic-minimap Composite tilesets + alpha → synthetic minimap

            Global options:
              --build, -b       Client build version (e.g. "4.3.4.15595") for
                               version-aware ADT profile selection. Auto-detected
                               from input path if not specified.
              --client-root     WoW client root directory (for extract-unified)
              --map, -m         Map name (e.g. "Azeroth") for extract-unified

            See --help on each command for options.
            """);
    }

    static void RunHarvestTile(string[] args)
    {
        string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(a => !a.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        string? textureSource = GetOption(args, "--texture-source", "-t");
        string? minimapRoot = GetOption(args, "--minimap-root", "-m");
        string? buildVersion = GetOption(args, "--build", "-b");
        int? tileXOpt = GetIntOption(args, "--tile-x", "-x");
        int? tileYOpt = GetIntOption(args, "--tile-y", "-y");

        if (string.IsNullOrWhiteSpace(input))
        {
            Console.Error.WriteLine("Error: --input <root.adt or alpha.wdt> is required.");
            Environment.ExitCode = 1;
            return;
        }

        if (string.IsNullOrWhiteSpace(output))
        {
            string dir = Path.GetDirectoryName(input) ?? ".";
            string stem = Path.GetFileNameWithoutExtension(input);
            output = Path.Combine(dir, $"{stem}_harvest.npz");
        }

        try
        {
            TerrainTileTensorPack pack;

            if (Path.GetExtension(input).Equals(".wdt", StringComparison.OrdinalIgnoreCase)
                && TryDetectAlphaWdt(input, out bool isAlpha, out bool isWmoBased))
            {
                if (isWmoBased)
                {
                    Console.Error.WriteLine($"Error: {input} is a WMO-based Alpha map (no terrain tiles). "
                        + "WMO-based Alpha maps are not yet supported for tensor pack export.");
                    Environment.ExitCode = 1;
                    return;
                }

                if (!isAlpha)
                {
                    Console.Error.WriteLine($"Error: {input} is a Retail WDT file. "
                        + "Use a root ADT file (e.g., mapname_XX_YY.adt) instead.");
                    Environment.ExitCode = 1;
                    return;
                }

                if (!tileXOpt.HasValue || !tileYOpt.HasValue)
                {
                    Console.Error.WriteLine("Error: Alpha WDT tiles require --tile-x <0-63> and --tile-y <0-63>.");
                    Environment.ExitCode = 1;
                    return;
                }

                int tileX = tileXOpt.Value;
                int tileY = tileYOpt.Value;
                if ((uint)tileX >= 64 || (uint)tileY >= 64)
                {
                    Console.Error.WriteLine("Error: tile coordinates must be 0-63.");
                    Environment.ExitCode = 1;
                    return;
                }

                if (!AlphaWdtReader.TryReadTile(input, tileX, tileY, out AlphaTileData? tileData))
                {
                    Console.Error.WriteLine($"Error: tile ({tileX},{tileY}) is not present in {input} "
                        + "(or the tile data could not be parsed).");
                    Environment.ExitCode = 1;
                    return;
                }

                pack = AlphaTensorPackBuilder.Build(tileData, tileX, tileY);
            }
            else
            {
                pack = AdtTensorPackBuilder.Build(input, textureSource, buildVersion);
            }

            if (!string.IsNullOrWhiteSpace(minimapRoot))
            {
                if (TryLoadMinimap(input, minimapRoot, out byte[,,]? minimap))
                {
                    pack.MinimapRgb256 = minimap;
                    pack.MinimapSourceTag = "raw";
                    pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals, StringComparer.OrdinalIgnoreCase)
                    {
                        "minimap_rgb_256"
                    };
                }
            }

            NpzTileSerializer.Serialize(pack, output);
            Console.WriteLine($"Harvested: {output}");
            Console.WriteLine($"Signals: {string.Join(", ", pack.AvailableSignals)}");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error harvesting {input}: {ex.Message}");
            Environment.ExitCode = 1;
        }
    }

    private static bool TryDetectAlphaWdt(string wdtPath, out bool isAlpha, out bool isWmoBased)
    {
        isAlpha = false;
        isWmoBased = false;
        try
        {
            byte[] bytes = File.ReadAllBytes(wdtPath);
            isAlpha = AlphaWdtReader.IsAlphaWdt(bytes);

            if (isAlpha && bytes.Length >= 16)
            {
                int mphdDataOffset = 8 + 4;
                if (mphdDataOffset + 8 <= bytes.Length)
                {
                    int mdnmOffset = BitConverter.ToInt32(bytes, mphdDataOffset + 4);
                    int monmOffset = BitConverter.ToInt32(bytes, mphdDataOffset + 12);
                    if (mdnmOffset == 2 || monmOffset == 2)
                        isWmoBased = true;
                }
            }
        }
        catch
        {
        }
        return true;
    }

    static void RunHarvestMap(string[] args)
    {
        string? inputDir = GetOption(args, "--input-dir", "-i") ?? args.FirstOrDefault(a => !a.StartsWith('-'));
        string? outputDir = GetOption(args, "--output-dir", "-o");
        string? minimapRoot = GetOption(args, "--minimap-root", "-m");
        string? buildVersion = GetOption(args, "--build", "-b");
        int limit = GetIntOption(args, "--limit", "-n") ?? int.MaxValue;
        bool overwrite = HasFlag(args, "--overwrite");

        if (string.IsNullOrWhiteSpace(inputDir) || string.IsNullOrWhiteSpace(outputDir))
        {
            Console.Error.WriteLine("Error: --input-dir <dir> and --output-dir <dir> are required.");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(inputDir))
        {
            Console.Error.WriteLine($"Error: input directory not found: {inputDir}");
            Environment.ExitCode = 1;
            return;
        }

        Directory.CreateDirectory(outputDir);

        int processed = 0;
        int skipped = 0;

        foreach (string adtPath in Directory.EnumerateFiles(inputDir, "*.adt")
            .Where(p => !p.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase)
                && !p.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase)
                && !p.EndsWith("_lod.adt", StringComparison.OrdinalIgnoreCase))
            .Take(limit))
        {
            string stem = Path.GetFileNameWithoutExtension(adtPath);
            string outputPath = Path.Combine(outputDir, $"{stem}_harvest.npz");

            if (!overwrite && File.Exists(outputPath))
            {
                skipped++;
                continue;
            }

            string? textureSource = File.Exists($"{inputDir}\\{stem}_tex0.adt")
                ? $"{inputDir}\\{stem}_tex0.adt"
                : null;

            try
            {
                TerrainTileTensorPack pack = AdtTensorPackBuilder.Build(adtPath, textureSource, buildVersion);

                if (!string.IsNullOrWhiteSpace(minimapRoot))
                {
                    if (TryLoadMinimap(adtPath, minimapRoot, out byte[,,]? minimap))
                    {
                        pack.MinimapRgb256 = minimap;
                        pack.MinimapSourceTag = "raw";
                        pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals, StringComparer.OrdinalIgnoreCase)
                        {
                            "minimap_rgb_256"
                        };
                    }
                }

                NpzTileSerializer.Serialize(pack, outputPath);
                processed++;
                Console.WriteLine($"Harvested: {stem}");
            }
            catch (Exception ex)
            {
                skipped++;
                Console.Error.WriteLine($"Skipped {stem}: {ex.Message}");
            }
        }

        Console.WriteLine($"Done. Processed={processed} Skipped={skipped}");
    }

    static void RunSyntheticMinimap(string[] args)
    {
        string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(a => !a.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");

        if (string.IsNullOrWhiteSpace(input))
        {
            Console.Error.WriteLine("Error: --input <npz> is required.");
            Environment.ExitCode = 1;
            return;
        }

        Console.WriteLine("Synthetic minimap compositor ready — texture pixel lookup not yet wired.");
    }

    static void RunExtractUnified(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapName = GetOption(args, "--map", "-m");
        string? output = GetOption(args, "--output", "-o");
        int? tileX = GetIntOption(args, "--tile-x", "-x");
        int? tileY = GetIntOption(args, "--tile-y", "-y");
        bool exportPlacements = HasFlag(args, "--export-placements");
        string? syntheticMinimap = GetOption(args, "--synthetic-minimap", "-s");
        bool dumpHex = HasFlag(args, "--dump-hex");

        if (string.IsNullOrWhiteSpace(clientRoot))
        {
            Console.Error.WriteLine("Error: --client-root <dir> is required.");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"Error: client root not found: {clientRoot}");
            Environment.ExitCode = 1;
            return;
        }

        if (string.IsNullOrWhiteSpace(mapName))
        {
            Console.Error.WriteLine("Error: --map <name> is required.");
            Environment.ExitCode = 1;
            return;
        }

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        LoadMd5Translate(clientRoot, catalog);

        string wdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
        byte[]? wdtBytes = catalog.ReadFile(wdtVirtual);
        if (wdtBytes is null)
        {
            Console.Error.WriteLine($"Error: Could not read WDT '{wdtVirtual}' from client.");
            Environment.ExitCode = 1;
            return;
        }

        if (tileX.HasValue && tileY.HasValue)
        {
            if (!RunExtractTileFromMpq(catalog, mapName, wdtBytes, tileX.Value, tileY.Value, output, exportPlacements, syntheticMinimap))
                Environment.ExitCode = 1;
        }
        else
        {
            using var ms = new MemoryStream(wdtBytes);
            MapFileSummary fileSummary = MapFileSummaryReader.Read(ms, wdtVirtual);
            var wdt = WdtSummaryReader.Read(ms, fileSummary);
            Console.WriteLine($"WDT: {mapName}");
            Console.WriteLine($"  IsWmoBased: {wdt.IsWmoBased}");
            Console.WriteLine($"  Tiles with data: {wdt.TilesWithData}/{wdt.TotalTiles}");
            Console.WriteLine();
            Console.WriteLine("Use --tile-x <0-63> --tile-y <0-63> to harvest a specific tile.");
        }
    }

    static bool RunExtractTileFromMpq(NativeMpqService catalog, string mapName, byte[] wdtBytes, int tileX, int tileY, string? outputPath, bool exportPlacements, string? syntheticMinimapPath = null)
    {
        TerrainTileTensorPack pack;
        AdtPlacementCatalog? placementCatalog = null;

        if (AlphaWdtReader.IsAlphaWdt(wdtBytes))
        {
            if (!AlphaWdtReader.TryReadTile(wdtBytes, tileX, tileY, out AlphaTileData? tileData) || tileData is null)
            {
                Console.Error.WriteLine($"Error: Alpha tile ({tileX},{tileY}) not present in WDT.");
                return false;
            }
            pack = AlphaTensorPackBuilder.Build(tileData, tileX, tileY);
            if (pack.MclqSurfaceHeight is null)
                TryAddWlLiquidFromMpq(catalog, mapName, tileX, tileY, pack);
            if (exportPlacements)
                placementCatalog = tileData.ToPlacementCatalog();
        }
        else
        {
            string adtVirtual = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}.adt";
            byte[]? adtBytes = catalog.ReadFile(adtVirtual);
            if (adtBytes is null)
            {
                Console.Error.WriteLine($"Error: Could not read ADT '{adtVirtual}' from client.");
                return false;
            }

            string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_harvest_{Guid.NewGuid():N}");
            Directory.CreateDirectory(tempDir);
            string adtDiskPath = Path.Combine(tempDir, Path.GetFileName(adtVirtual));
            string tex0DiskPath = Path.Combine(tempDir, $"{mapName}_{tileX}_{tileY}_tex0.adt");

            File.WriteAllBytes(adtDiskPath, adtBytes);

            string? tex0Virtual = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}_tex0.adt";
            byte[]? tex0Bytes = catalog.ReadFile(tex0Virtual);
            if (tex0Bytes != null)
                File.WriteAllBytes(tex0DiskPath, tex0Bytes);

            try
            {
                pack = AdtTensorPackBuilder.Build(adtDiskPath, tex0Bytes != null ? tex0DiskPath : null, null);
            }
            finally
            {
                try { Directory.Delete(tempDir, true); } catch { }
            }
        }

        if (pack.MinimapRgb256 is null)
        {
            byte[,,]? minimapRgb = TryLoadMinimapFromMpq(catalog, mapName, tileX, tileY);
            if (minimapRgb is not null)
            {
                pack.MinimapRgb256 = minimapRgb;
                pack.MinimapSourceTag = "mpq_blp";
                pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals) { "minimap_rgb" };
            }
        }

        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), $"{mapName}_{tileX}_{tileY}_v14.npz");

        NpzTileSerializer.Serialize(pack, outputPath);
        Console.WriteLine($"Harvested: {outputPath}");
        Console.WriteLine($"Signals: {string.Join(", ", pack.AvailableSignals)}");

        if (pack.MinimapRgb256 != null)
            Console.WriteLine($"  minimap: 256x256 RGB from MPQ");

        if (exportPlacements && placementCatalog is not null)
        {
            string placementPath = Path.ChangeExtension(outputPath, ".placement.json");
            string json = System.Text.Json.JsonSerializer.Serialize(placementCatalog, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(placementPath, json);
            Console.WriteLine($"Placements: {placementPath}");
        }

        if (!string.IsNullOrWhiteSpace(syntheticMinimapPath))
            GenerateSyntheticMinimap(catalog, pack, tileX, tileY, syntheticMinimapPath);

        return true;
    }

    static void GenerateSyntheticMinimap(NativeMpqService catalog, TerrainTileTensorPack pack, int tileX, int tileY, string outputPath)
    {
        if (pack.McalAlphaPack256 is null || pack.MclyTextureIds is null)
        {
            Console.Error.WriteLine("Error: tile has no MCAL/MCLY data, cannot generate synthetic minimap.");
            return;
        }

        var textureNames = pack.MclyTextureNames;
        if (textureNames.Count == 0)
        {
            Console.Error.WriteLine("Error: no texture names in tile metadata.");
            return;
        }

        var usedIds = new HashSet<int>();
        for (int cy = 0; cy < 16; cy++)
            for (int cx = 0; cx < 16; cx++)
                for (int l = 0; l < 4; l++)
                    if (pack.MclyTextureIds[cy, cx, l] >= 0)
                        usedIds.Add(pack.MclyTextureIds[cy, cx, l]);

        Console.WriteLine($"  Loading {usedIds.Count} unique textures for synthetic minimap...");

        var textures = new Dictionary<int, byte[,,]>();
        int maxId = usedIds.Max();
        for (int id = 0; id <= maxId && id < textureNames.Count; id++)
        {
            if (!usedIds.Contains(id)) continue;
            string texPath = textureNames[id];
            byte[,,]? rgb = LoadTextureFromMpq(catalog, texPath);
            if (rgb is not null)
                textures[id] = rgb;
            else
                Console.Error.WriteLine($"    Warning: Could not load texture [{id}] {texPath}");
        }

        if (textures.Count == 0)
        {
            Console.Error.WriteLine("Error: no textures could be loaded.");
            return;
        }

        Console.WriteLine($"  Loaded {textures.Count}/{usedIds.Count} textures, compositing...");

        const int size = 256;
        const float tileSize = 533.33333f;
        const float mapOrigin = 17066.666f;
        const float textureScale = 20f;

        using var image = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>(size, size);

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int chunkY = y / 16;
                int chunkX = x / 16;

                // World position for proper texture UV tiling
                float worldX = mapOrigin - tileX * tileSize + (x / (float)(size - 1)) * tileSize;
                float worldY = mapOrigin - tileY * tileSize + (y / (float)(size - 1)) * tileSize;

                float r = 0f, g = 0f, b = 0f;
                float implicitAlpha = 1f;

                for (int l = 1; l < 4; l++)
                {
                    float a = pack.McalAlphaPack256[y, x, l];
                    implicitAlpha -= a;
                    if (a <= 0.001f) continue;

                    int texId = pack.MclyTextureIds[chunkY, chunkX, l];
                    if (texId < 0 || !textures.TryGetValue(texId, out var tex)) continue;

                    int texH = tex.GetLength(0);
                    int texW = tex.GetLength(1);
                    float tu = (worldX / textureScale) % 1f;
                    float tv = (worldY / textureScale) % 1f;
                    if (tu < 0) tu += 1f;
                    if (tv < 0) tv += 1f;
                    int tx = (int)(tu * texW) % texW;
                    int ty = (int)(tv * texH) % texH;
                    r += tex[ty, tx, 0] * a;
                    g += tex[ty, tx, 1] * a;
                    b += tex[ty, tx, 2] * a;
                }

                if (implicitAlpha > 0.001f)
                {
                    int baseTexId = pack.MclyTextureIds[chunkY, chunkX, 0];
                    if (baseTexId >= 0 && textures.TryGetValue(baseTexId, out var baseTex))
                    {
                        int texH = baseTex.GetLength(0);
                        int texW = baseTex.GetLength(1);
                        float tu = (worldX / textureScale) % 1f;
                        float tv = (worldY / textureScale) % 1f;
                        if (tu < 0) tu += 1f;
                        if (tv < 0) tv += 1f;
                        int tx = (int)(tu * texW) % texW;
                        int ty = (int)(tv * texH) % texH;
                        r += baseTex[ty, tx, 0] * implicitAlpha;
                        g += baseTex[ty, tx, 1] * implicitAlpha;
                        b += baseTex[ty, tx, 2] * implicitAlpha;
                    }
                }

                image[x, y] = new SixLabors.ImageSharp.PixelFormats.Rgba32(
                    (byte)Math.Clamp((int)r, 0, 255),
                    (byte)Math.Clamp((int)g, 0, 255),
                    (byte)Math.Clamp((int)b, 0, 255),
                    255);
            }
        }

        string? dir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        using var fs = File.Create(outputPath);
        image.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
        Console.WriteLine($"Synthetic minimap: {outputPath}");
    }

    static void TryAddWlLiquidFromMpq(NativeMpqService catalog, string mapName, int tileX, int tileY, TerrainTileTensorPack pack)
    {
        string[] wlExtensions = [".wlw", ".wlm", ".wlq", ".wll"];
        string basePath = $"World\\Maps\\{mapName}\\{mapName}";

        bool any = false;
        float[,]? mask = null;
        float[,]? heights = null;

        const int size = 257;
        const float tileWorldSize = 533.33333f;
        const float mapOrigin = 17066.666f;

        foreach (string ext in wlExtensions)
        {
            byte[]? data = catalog.ReadFile(basePath + ext);
            if (data is null || data.Length == 0) continue;

            try
            {
                using var ms = new MemoryStream(data);
                var wl = WlFileReader.Read(ms);
                foreach (var block in wl.Blocks)
                {
                    Vector3 pos = block.WorldPosition;
                    // Project block world position to tile-local coordinates
                    int blockTileX = Math.Clamp((int)Math.Floor((mapOrigin - pos.Y) / tileWorldSize), 0, 63);
                    int blockTileY = Math.Clamp((int)Math.Floor((mapOrigin - pos.X) / tileWorldSize), 0, 63);
                    if (blockTileX != tileX || blockTileY != tileY) continue;

                    float avgH = block.Vertices.Average(v => v.Z);
                    float localX = (mapOrigin - pos.Y) - (tileX * tileWorldSize);
                    float localY = (mapOrigin - pos.X) - (tileY * tileWorldSize);
                    int cx = Math.Clamp((int)(localX / tileWorldSize * (size - 1)), 0, size - 1);
                    int cy = Math.Clamp((int)(localY / tileWorldSize * (size - 1)), 0, size - 1);

                    mask ??= new float[size, size];
                    heights ??= new float[size, size];

                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int px = Math.Clamp(cx + dx, 0, size - 1);
                            int py = Math.Clamp(cy + dy, 0, size - 1);
                            mask[py, px] = 1.0f;
                            heights[py, px] = avgH;
                        }
                    }
                    any = true;
                }
            }
            catch { }
        }

        if (any)
        {
            pack.WlLiquidMask = mask;
            pack.WlLiquidHeight = heights;
            pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals) { "wl_liquid_mask", "wl_liquid_height" };
        }
    }

    static bool TryLoadMinimap(string adtPath, string minimapRoot, out byte[,,]? minimap)
    {
        minimap = null;
        string stem = Path.GetFileNameWithoutExtension(adtPath);

        string[] candidates =
        [
            Path.Combine(minimapRoot, $"{stem}.png"),
            Path.Combine(minimapRoot, "images", $"{stem}.png"),
            Path.Combine(minimapRoot, "reference_minimaps", $"{stem}_reference_minimap.png"),
        ];

        foreach (string candidate in candidates)
        {
            if (File.Exists(candidate))
            {
                try
                {
                    using var img = SixLabors.ImageSharp.Image.Load<Rgba32>(candidate);
                    if (img.Width == 256 && img.Height == 256)
                    {
                        minimap = new byte[256, 256, 3];
                        for (int y = 0; y < 256; y++)
                        {
                            for (int x = 0; x < 256; x++)
                            {
                                var px = img[x, y];
                                minimap[y, x, 0] = px.R;
                                minimap[y, x, 1] = px.G;
                                minimap[y, x, 2] = px.B;
                            }
                        }
                        return true;
                    }
                }
                catch { }
            }
        }

        return false;
    }

    static void LoadMd5Translate(string clientRoot, NativeMpqService catalog)
    {
        if (Md5TranslateResolver.TryLoad(
            new[] { clientRoot },
            path => catalog.FileExists(path),
            path => catalog.ReadFile(path),
            out var md5Index))
        {
            _md5Lookup = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var kv in md5Index.PlainToHash)
                _md5Lookup[kv.Key] = kv.Value;
            Console.WriteLine($"  Loaded {_md5Lookup.Count} md5translate entries");
        }
    }

    static byte[,,]? TryLoadMinimapFromMpq(NativeMpqService catalog, string mapName, int tileX, int tileY)
    {
        string mapLower = mapName.ToLowerInvariant();
        string x2 = tileX.ToString("00");
        string y2 = tileY.ToString("00");
        string canonicalPath = $"textures/minimap/{mapLower}/map{x2}_{y2}.blp";

        // MD5 translate lookup
        if (_md5Lookup is not null && _md5Lookup.TryGetValue(canonicalPath, out string? md5Name))
        {
            byte[]? blpBytes = catalog.ReadFile(md5Name);
            if (blpBytes is not null && blpBytes.Length >= 8)
            {
                try
                {
                    byte[,,]? rgb = DecodeBlpToRgb(blpBytes);
                    if (rgb is not null) return rgb;
                }
                catch { }
            }
        }

        // Fallback candidates matching MdxViewer EnumerateTileCandidates
        string[] candidates =
        [
            canonicalPath.Replace('/', '\\'),                                        // textures\minimap\azeroth\map32_32.blp
            $"textures\\minimap\\{mapLower}\\map{x2}_{y2}.blp",                    // textures\minimap\azeroth\map32_32.blp
            $"textures\\Minimap\\{mapLower}\\map{x2}_{y2}.blp",                    // textures\Minimap\azeroth\map32_32.blp
            $"world\\minimaps\\{mapLower}\\map{x2}_{y2}.blp",                      // world\minimaps\azeroth\map32_32.blp
            $"world\\Minimaps\\{mapName}\\map{x2}_{y2}.blp",                       // world\Minimaps\Azeroth\map32_32.blp
            $"{mapLower}\\map{x2}_{y2}.blp",                                        // azeroth\map32_32.blp
        ];

        foreach (string candidate in candidates)
        {
            byte[]? blpBytes = catalog.ReadFile(candidate);
            if (blpBytes is null || blpBytes.Length < 8) continue;

            try
            {
                byte[,,]? rgb = DecodeBlpToRgb(blpBytes);
                if (rgb is not null) return rgb;
            }
            catch { }
        }

        return null;
    }

    static byte[,,]? LoadTextureFromMpq(NativeMpqService catalog, string virtualPath)
    {
        byte[]? blpBytes = catalog.ReadFile(virtualPath);
        if (blpBytes is null || blpBytes.Length < 8)
            return null;
        try { return DecodeBlpToRgb(blpBytes); }
        catch { return null; }
    }

    static byte[,,]? DecodeBlpToRgb(byte[] blpBytes)
    {
        using var ms = new MemoryStream(blpBytes, writable: false);
        using var blp = new SereniaBLPLib.BlpFile(ms);
        var bitmap = blp.GetBitmap(0);
        if (bitmap == null) return null;

        int w = bitmap.Width;
        int h = bitmap.Height;
        if (w < 1 || h < 1) return null;

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

    static string? GetOption(string[] args, string name, string shortName)
    {
        int idx = Array.IndexOf(args, name);
        if (idx >= 0 && idx + 1 < args.Length) return args[idx + 1];
        if (!string.IsNullOrEmpty(shortName))
        {
            idx = Array.IndexOf(args, shortName);
            if (idx >= 0 && idx + 1 < args.Length) return args[idx + 1];
        }
        return null;
    }

    static int? GetIntOption(string[] args, string name, string shortName)
    {
        string? val = GetOption(args, name, shortName);
        return int.TryParse(val, out int v) ? v : null;
    }

    static bool HasFlag(string[] args, string flag)
    {
        return args.Contains(flag, StringComparer.OrdinalIgnoreCase);
    }
}