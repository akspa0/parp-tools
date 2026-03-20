using System.Text;
using System.Text.Json;
using WoWMapConverter.Core.Formats.LichKing;
using WoWMapConverter.Core.Formats.Liquids;
using WoWMapConverter.Core.Formats.PM4;
using WoWRollback.PM4Module;
using WoWRollback.PM4Module.Services;

namespace WoWMapConverter.Core.Services;

public sealed record DevelopmentRepairOptions(
    string InputDirectory,
    string OutputDirectory,
    string Mode,
    int? TileLimit,
    string? Tile,
    bool SkipWl,
    bool SkipWdlGenerate,
    bool SkipPm4,
    string? ManifestPath)
{
    public static DevelopmentRepairOptions CreateDefault() => new(
        InputDirectory: DevelopmentMapAnalyzer.DefaultDevelopmentMapDirectory,
        OutputDirectory: Path.Combine("output", "development-repair"),
        Mode: "repair",
        TileLimit: null,
        Tile: null,
        SkipWl: false,
        SkipWdlGenerate: false,
        SkipPm4: true,
        ManifestPath: null);

    public bool IsRepairMode => string.Equals(Mode, "repair", StringComparison.OrdinalIgnoreCase);
}

public sealed class DevelopmentSourceFileEntry
{
    public string Kind { get; set; } = string.Empty;
    public string Path { get; set; } = string.Empty;
    public bool Exists { get; set; }
    public bool Used { get; set; }
}

public sealed class DevelopmentTileRepairManifest
{
    public string TileName { get; set; } = string.Empty;
    public int TileX { get; set; }
    public int TileY { get; set; }
    public string TileClass { get; set; } = string.Empty;
    public string RecommendedAction { get; set; } = string.Empty;
    public List<DevelopmentSourceFileEntry> SourceFiles { get; set; } = new();
    public bool SplitMergeRan { get; set; }
    public bool ChunkIndicesRepairRan { get; set; }
    public int ChunkIndexMismatchCount { get; set; }
    public string ChunkIndexRepairSource { get; set; } = "none";
    public bool WdlGenerationRan { get; set; }
    public bool Pm4MprlPatchingRan { get; set; }
    public bool WlLiquidsConverted { get; set; }
    public int WlLiquidChunkCount { get; set; }
    public bool MinimapMccvPaintingRan { get; set; }
    public bool NeedsManualReview { get; set; }
    public bool OutputWritten { get; set; }
    public string? OutputAdtPath { get; set; }
    public string RepairRoute { get; set; } = string.Empty;
    public DevelopmentTileTextureManifest TextureData { get; set; } = new();
    public List<string> Warnings { get; set; } = new();
}

public sealed class DevelopmentTileTextureManifest
{
    public List<string> MtexTextures { get; set; } = new();
    public List<DevelopmentChunkTextureManifest> ChunkLayers { get; set; } = new();
}

public sealed class DevelopmentChunkTextureManifest
{
    public int ChunkIndex { get; set; }
    public List<DevelopmentTextureLayerManifest> Layers { get; set; } = new();
}

public sealed class DevelopmentTextureLayerManifest
{
    public int LayerIndex { get; set; }
    public uint TextureId { get; set; }
    public string TexturePath { get; set; } = string.Empty;
    public uint Flags { get; set; }
    public uint AlphaMapOffset { get; set; }
    public uint EffectId { get; set; }
    public string? AlphaBitsBase64 { get; set; }
    public int? AlphaByteCount { get; set; }
}

public sealed class DevelopmentRepairSummaryManifest
{
    public string MapName { get; set; } = string.Empty;
    public string InputDirectory { get; set; } = string.Empty;
    public string OutputDirectory { get; set; } = string.Empty;
    public string Mode { get; set; } = string.Empty;
    public int TilesProcessed { get; set; }
    public int TilesWritten { get; set; }
    public int HealthySplitCount { get; set; }
    public int IndexCorruptCount { get; set; }
    public int ScanOnlyRootCount { get; set; }
    public int WdlRebuildCount { get; set; }
    public int ManualReviewCount { get; set; }
    public bool WdtWritten { get; set; }
    public bool WdlCopied { get; set; }
    public List<string> WlIndexWarnings { get; set; } = new();
    public List<DevelopmentTileRepairManifest> Tiles { get; set; } = new();
}

public sealed record DevelopmentRepairExecutionReport(
    string MapName,
    string InputDirectory,
    string OutputDirectory,
    string SummaryManifestPath,
    int TilesProcessed,
    int TilesWritten,
    bool WdtWritten,
    bool WdlCopied,
    List<DevelopmentTileRepairManifest> Tiles);

public static class DevelopmentRepairService
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    private sealed class WlTileIndexEntry
    {
        public WlMh2oTileData TileData { get; }
        public HashSet<string> SourcePaths { get; } = new(StringComparer.OrdinalIgnoreCase);

        public WlTileIndexEntry(WlMh2oTileData tileData)
        {
            TileData = tileData;
        }
    }

    public static DevelopmentRepairExecutionReport Execute(DevelopmentRepairOptions options)
    {
        if (!string.Equals(options.Mode, "audit", StringComparison.OrdinalIgnoreCase)
            && !string.Equals(options.Mode, "repair", StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported mode '{options.Mode}'. Expected audit|repair.");
        }

        string resolvedInputDirectory = Pm4CoordinateService.ResolveMapDirectory(options.InputDirectory);
        if (IsReference335Input(resolvedInputDirectory))
        {
            throw new ArgumentException(
                "development-repair does not accept reference WoWMuseum 3.3.5 inputs. "
                + "Use the original constituent source dataset at test_data/development/World/Maps/development.");
        }

        DevelopmentMapAnalysisReport analysis = DevelopmentMapAnalyzer.Analyze(resolvedInputDirectory, options.TileLimit);
        (int x, int y)? tileFilter = ParseTileFilter(options.Tile);

        string inputDirectory = analysis.MapDirectory;
        string mapName = analysis.MapName;
        string outputDirectory = Path.GetFullPath(options.OutputDirectory);
        string outputMapDirectory = Path.Combine(outputDirectory, "World", "Maps", mapName);
        string tileManifestDirectory = Path.Combine(outputDirectory, "manifests", "tiles");
        string summaryManifestPath = options.ManifestPath is { Length: > 0 }
            ? Path.GetFullPath(options.ManifestPath)
            : Path.Combine(outputDirectory, "manifests", "summary.json");

        Directory.CreateDirectory(outputDirectory);
        Directory.CreateDirectory(Path.GetDirectoryName(summaryManifestPath) ?? outputDirectory);
        Directory.CreateDirectory(tileManifestDirectory);
        if (options.IsRepairMode)
            Directory.CreateDirectory(outputMapDirectory);

        string preferredWdlPath = Path.Combine(inputDirectory, $"{mapName}.wdl");
        string fallbackWdlMpqPath = Path.Combine(inputDirectory, $"{mapName}.wdl.mpq");
        string? wdlPath = File.Exists(preferredWdlPath)
            ? preferredWdlPath
            : (File.Exists(fallbackWdlMpqPath) ? fallbackWdlMpqPath : null);

        var splitMerger = new SplitAdtMerger();
        var wdlService = new WdlService();
        Dictionary<(int tileX, int tileY), WlTileIndexEntry> wlTileIndex = new();
        List<string> wlIndexWarnings = new();

        if (!options.SkipWl)
            wlTileIndex = BuildWlTileIndex(inputDirectory, wlIndexWarnings);

        var tileManifests = new List<DevelopmentTileRepairManifest>();
        int tilesWritten = 0;

        foreach (DevelopmentTileAnalysisResult tile in analysis.Tiles.OrderBy(t => t.TileY).ThenBy(t => t.TileX))
        {
            if (tileFilter.HasValue && (tile.TileX != tileFilter.Value.x || tile.TileY != tileFilter.Value.y))
                continue;

            DevelopmentTileRepairManifest manifest = CreateTileManifest(inputDirectory, mapName, tile, wdlPath);
            byte[]? outputBytes = null;

            if (options.IsRepairMode)
            {
                outputBytes = BuildTileOutput(
                    options,
                    splitMerger,
                    wdlService,
                    wlTileIndex,
                    wdlPath,
                    tile,
                    manifest);

                if (outputBytes == null && TryGetSourcePath(manifest, "root-adt", out string? rootPath) && rootPath != null && File.Exists(rootPath))
                {
                    // Keep unresolved tiles visible in output while still marking manual review.
                    outputBytes = File.ReadAllBytes(rootPath);
                    MarkSourceUsed(manifest, "root-adt");
                    manifest.Warnings.Add("Copied root ADT without successful repair pipeline execution.");
                }

                if (outputBytes != null)
                {
                    string outAdtPath = Path.Combine(outputMapDirectory, tile.TileName + ".adt");
                    File.WriteAllBytes(outAdtPath, outputBytes);
                    manifest.OutputWritten = true;
                    manifest.OutputAdtPath = outAdtPath;
                    tilesWritten++;
                }
            }

            PopulateTextureDataManifest(manifest, outputBytes);

            manifest.NeedsManualReview = manifest.NeedsManualReview
                || string.Equals(manifest.TileClass, "manual-review", StringComparison.Ordinal)
                || (!manifest.OutputWritten && options.IsRepairMode);

            string tileManifestPath = Path.Combine(tileManifestDirectory, tile.TileName + ".json");
            File.WriteAllText(tileManifestPath, JsonSerializer.Serialize(manifest, JsonOptions));
            tileManifests.Add(manifest);
        }

        bool wdtWritten = false;
        bool wdlCopied = false;

        if (options.IsRepairMode)
        {
            string outputWdtPath = Path.Combine(outputDirectory, mapName + ".wdt");
            WriteWdtFromOutputTiles(outputMapDirectory, mapName, outputWdtPath);
            wdtWritten = File.Exists(outputWdtPath);

            if (wdlPath != null && File.Exists(wdlPath))
            {
                string outputWdlPath = Path.Combine(outputDirectory, Path.GetFileName(wdlPath));
                File.Copy(wdlPath, outputWdlPath, overwrite: true);
                wdlCopied = true;
            }
        }

        var summary = new DevelopmentRepairSummaryManifest
        {
            MapName = mapName,
            InputDirectory = inputDirectory,
            OutputDirectory = outputDirectory,
            Mode = options.Mode,
            TilesProcessed = tileManifests.Count,
            TilesWritten = tilesWritten,
            HealthySplitCount = tileManifests.Count(t => t.TileClass == "healthy-split"),
            IndexCorruptCount = tileManifests.Count(t => t.TileClass == "index-corrupt"),
            ScanOnlyRootCount = tileManifests.Count(t => t.TileClass == "scan-only-root"),
            WdlRebuildCount = tileManifests.Count(t => t.TileClass == "wdl-rebuild"),
            ManualReviewCount = tileManifests.Count(t => t.TileClass == "manual-review" || t.NeedsManualReview),
            WdtWritten = wdtWritten,
            WdlCopied = wdlCopied,
            WlIndexWarnings = wlIndexWarnings,
            Tiles = tileManifests
        };

        File.WriteAllText(summaryManifestPath, JsonSerializer.Serialize(summary, JsonOptions));

        return new DevelopmentRepairExecutionReport(
            MapName: mapName,
            InputDirectory: inputDirectory,
            OutputDirectory: outputDirectory,
            SummaryManifestPath: summaryManifestPath,
            TilesProcessed: tileManifests.Count,
            TilesWritten: tilesWritten,
            WdtWritten: wdtWritten,
            WdlCopied: wdlCopied,
            Tiles: tileManifests);
    }

    private static bool IsReference335Input(string inputDirectory)
    {
        string normalized = inputDirectory
            .Replace('/', '\\')
            .TrimEnd('\\')
            .ToLowerInvariant();

        return normalized.Contains("\\test_data\\wowmuseum\\335-dev\\world\\maps\\development")
            || normalized.EndsWith("\\335-dev")
            || normalized.Contains("\\wowmuseum\\335-dev\\");
    }

    private static DevelopmentTileRepairManifest CreateTileManifest(string inputDirectory, string mapName, DevelopmentTileAnalysisResult tile, string? wdlPath)
    {
        var manifest = new DevelopmentTileRepairManifest
        {
            TileName = tile.TileName,
            TileX = tile.TileX,
            TileY = tile.TileY,
            TileClass = tile.TileClass,
            RecommendedAction = tile.RecommendedAction,
            RepairRoute = tile.TileClass
        };

        AddSource(manifest, "root-adt", Path.Combine(inputDirectory, tile.TileName + ".adt"));
        AddSource(manifest, "obj0-adt", Path.Combine(inputDirectory, tile.TileName + "_obj0.adt"));
        AddSource(manifest, "tex0-adt", Path.Combine(inputDirectory, tile.TileName + "_tex0.adt"));
        AddSource(manifest, "pm4", Path.Combine(inputDirectory, tile.TileName + ".pm4"));
        if (!string.IsNullOrEmpty(wdlPath))
            AddSource(manifest, "wdl", wdlPath);

        return manifest;
    }

    private static byte[]? BuildTileOutput(
        DevelopmentRepairOptions options,
        SplitAdtMerger splitMerger,
        WdlService wdlService,
        Dictionary<(int tileX, int tileY), WlTileIndexEntry> wlTileIndex,
        string? wdlPath,
        DevelopmentTileAnalysisResult tile,
        DevelopmentTileRepairManifest manifest)
    {
        byte[]? bytes = null;

        TryGetSourcePath(manifest, "root-adt", out string? rootPath);
        TryGetSourcePath(manifest, "obj0-adt", out string? obj0Path);
        TryGetSourcePath(manifest, "tex0-adt", out string? tex0Path);

        bool hasRoot = !string.IsNullOrEmpty(rootPath) && File.Exists(rootPath);
        bool hasObj0 = !string.IsNullOrEmpty(obj0Path) && File.Exists(obj0Path);
        bool hasTex0 = !string.IsNullOrEmpty(tex0Path) && File.Exists(tex0Path);
        bool hasSplitSidecars = hasObj0 || hasTex0;

        // Constituent-first: always attempt to rebuild from available ADT pieces before WDL.
        if (hasRoot)
        {
            if (hasSplitSidecars)
            {
                SplitAdtMerger.MergeResult merge = splitMerger.Merge(
                    rootPath!,
                    hasObj0 ? obj0Path : null,
                    hasTex0 ? tex0Path : null);

                if (merge.Success && merge.Data != null)
                {
                    bytes = merge.Data;
                    manifest.SplitMergeRan = true;
                    manifest.RepairRoute = "constituent-split-merge";
                    MarkSourceUsed(manifest, "root-adt");
                    if (hasObj0)
                        MarkSourceUsed(manifest, "obj0-adt");
                    if (hasTex0)
                        MarkSourceUsed(manifest, "tex0-adt");
                }
                else
                {
                    bytes = File.ReadAllBytes(rootPath!);
                    manifest.RepairRoute = "constituent-root-fallback";
                    MarkSourceUsed(manifest, "root-adt");
                    manifest.Warnings.Add($"Split merge failed: {merge.Error ?? "unknown error"}. Falling back to root ADT bytes.");
                    manifest.NeedsManualReview = true;
                }
            }
            else
            {
                bytes = File.ReadAllBytes(rootPath!);
                manifest.RepairRoute = "constituent-root-only";
                MarkSourceUsed(manifest, "root-adt");
            }
        }
        else
        {
            manifest.Warnings.Add("Missing root ADT; attempting WDL generation fallback.");
        }

        if (bytes == null)
        {
            if (options.SkipWdlGenerate)
            {
                manifest.Warnings.Add("WDL generation skipped by option.");
                manifest.NeedsManualReview = true;
                return null;
            }

            if (string.IsNullOrEmpty(wdlPath) || !File.Exists(wdlPath))
            {
                manifest.Warnings.Add("WDL source is missing; cannot generate terrain tile.");
                manifest.NeedsManualReview = true;
                return null;
            }

            WdlToAdtGenerator.WdlTileData? wdlTile = wdlService.GetTileData(wdlPath, tile.TileX, tile.TileY);
            if (wdlTile == null)
            {
                manifest.Warnings.Add("WDL tile data is missing for this coordinate.");
                manifest.NeedsManualReview = true;
                return null;
            }

            bytes = WdlToAdtGenerator.GenerateAdt(wdlTile, tile.TileX, tile.TileY);
            manifest.WdlGenerationRan = true;
            manifest.RepairRoute = "wdl-generated-fallback";
            MarkSourceUsed(manifest, "wdl");
        }

        if (bytes == null)
            return null;

        DevelopmentMcnkIndexRepairReport indexRepair = DevelopmentMcnkIndexRepairService.RepairBytes(bytes, tile.TileName, writeChanges: true);
        manifest.ChunkIndicesRepairRan = true;
        manifest.ChunkIndexMismatchCount = indexRepair.MismatchCount;
        manifest.ChunkIndexRepairSource = indexRepair.UsedMcin ? "MCIN" : "chunk-scan";
        bytes = indexRepair.RepairedBytes ?? bytes;

        if (!options.SkipWl)
        {
            if (TryBuildTileMh2o(wlTileIndex, tile.TileX, tile.TileY, out WlMh2oTileData? mh2oTileData, out IReadOnlyCollection<string> sourcePaths))
            {
                byte[] mh2oData = SerializeMh2oChunk(mh2oTileData!);
                bytes = InjectMh2oChunk(bytes, mh2oData);
                manifest.WlLiquidsConverted = true;
                manifest.WlLiquidChunkCount = mh2oTileData!.ChunkCount;

                foreach (string sourcePath in sourcePaths)
                    MarkWlSourceUsed(manifest, sourcePath);
            }
        }

        manifest.Pm4MprlPatchingRan = false;
        manifest.MinimapMccvPaintingRan = false;

        return bytes;
    }

    private static bool TryBuildTileMh2o(
        Dictionary<(int tileX, int tileY), WlTileIndexEntry> wlTileIndex,
        int tileX,
        int tileY,
        out WlMh2oTileData? merged,
        out IReadOnlyCollection<string> sourcePaths)
    {
        merged = null;
        sourcePaths = Array.Empty<string>();

        if (!wlTileIndex.TryGetValue((tileX, tileY), out WlTileIndexEntry? entry))
            return false;

        if (entry.TileData.ChunkCount == 0)
            return false;

        merged = CloneWlTileData(entry.TileData);
        sourcePaths = entry.SourcePaths.ToArray();
        return true;
    }

    private static Dictionary<(int tileX, int tileY), WlTileIndexEntry> BuildWlTileIndex(string inputDirectory, List<string> warnings)
    {
        var index = new Dictionary<(int tileX, int tileY), WlTileIndexEntry>();
        string[] wlExtensions = { ".wlw", ".wlm", ".wlq", ".wll" };

        foreach (string filePath in Directory.EnumerateFiles(inputDirectory, "*.*", SearchOption.TopDirectoryOnly)
            .Where(path => wlExtensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase))
            .OrderBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase))
        {
            try
            {
                WlFile wl = WlFile.Read(filePath);
                WlToMh2oResult result = WlToLiquidConverter.ConvertToMh2o(wl);

                foreach (KeyValuePair<(int tileX, int tileY), WlMh2oTileData> tileData in result.TileData)
                {
                    if (tileData.Value.ChunkCount == 0)
                        continue;

                    if (!index.TryGetValue(tileData.Key, out WlTileIndexEntry? entry))
                    {
                        entry = new WlTileIndexEntry(CloneWlTileData(tileData.Value));
                        index[tileData.Key] = entry;
                    }
                    else
                    {
                        MergeWlTileData(entry.TileData, tileData.Value);
                    }

                    entry.SourcePaths.Add(filePath);
                }
            }
            catch (Exception ex)
            {
                warnings.Add($"Failed to parse WL source '{Path.GetFileName(filePath)}': {ex.Message}");
            }
        }

        return index;
    }

    private static void MarkWlSourceUsed(DevelopmentTileRepairManifest manifest, string sourcePath)
    {
        DevelopmentSourceFileEntry? existing = manifest.SourceFiles.FirstOrDefault(source => string.Equals(source.Path, sourcePath, StringComparison.OrdinalIgnoreCase));
        if (existing != null)
        {
            existing.Used = true;
            return;
        }

        manifest.SourceFiles.Add(new DevelopmentSourceFileEntry
        {
            Kind = $"wl-{Path.GetExtension(sourcePath).TrimStart('.').ToLowerInvariant()}",
            Path = sourcePath,
            Exists = File.Exists(sourcePath),
            Used = true
        });
    }

    private static void PopulateTextureDataManifest(DevelopmentTileRepairManifest manifest, byte[]? outputBytes)
    {
        try
        {
            var candidates = new List<DevelopmentTileTextureManifest>();
            if (outputBytes != null)
                candidates.Add(BuildTextureData(outputBytes));

            foreach (string kind in new[] { "tex0-adt", "root-adt" })
            {
                if (TryGetSourcePath(manifest, kind, out string? sourcePath)
                    && sourcePath != null
                    && File.Exists(sourcePath))
                {
                    candidates.Add(BuildTextureData(File.ReadAllBytes(sourcePath)));
                }
            }

            manifest.TextureData = ChooseRicherTextureData(candidates) ?? new DevelopmentTileTextureManifest();
        }
        catch (Exception ex)
        {
            manifest.Warnings.Add($"Failed to extract texture payload from ADT bytes: {ex.Message}");
        }
    }

    private static DevelopmentTileTextureManifest? ChooseRicherTextureData(IEnumerable<DevelopmentTileTextureManifest> candidates)
    {
        DevelopmentTileTextureManifest? selected = null;
        foreach (DevelopmentTileTextureManifest candidate in candidates)
        {
            if (selected == null)
            {
                selected = candidate;
                continue;
            }

            int candidateChunkCount = candidate.ChunkLayers.Count;
            int selectedChunkCount = selected.ChunkLayers.Count;
            if (candidateChunkCount > selectedChunkCount)
            {
                selected = candidate;
                continue;
            }

            if (candidateChunkCount == selectedChunkCount
                && candidate.MtexTextures.Count > selected.MtexTextures.Count)
            {
                selected = candidate;
            }
        }

        return selected;
    }

    private static DevelopmentTileTextureManifest BuildTextureData(byte[] adtBytes)
    {
        var textureData = new DevelopmentTileTextureManifest
        {
            MtexTextures = ReadMtexTexturePaths(adtBytes)
        };

        int mcinOffset = FindTopLevelChunkOffset(adtBytes, "MCIN");
        if (mcinOffset < 0)
            return textureData;

        int mcinDataStart = mcinOffset + 8;
        for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
        {
            int entryOffset = mcinDataStart + (chunkIndex * 16);
            if (entryOffset + 16 > adtBytes.Length)
                break;

            int mcnkOffset = BitConverter.ToInt32(adtBytes, entryOffset);
            if (mcnkOffset <= 0 || mcnkOffset + 8 > adtBytes.Length)
                continue;

            if (!string.Equals(ReadNormalizedFourCc(adtBytes, mcnkOffset), "MCNK", StringComparison.Ordinal))
                continue;

            int mcnkSize = BitConverter.ToInt32(adtBytes, mcnkOffset + 4);
            if (mcnkSize <= 0 || mcnkOffset + 8 + mcnkSize > adtBytes.Length)
                continue;

            byte[] mcnkBody = new byte[mcnkSize];
            Buffer.BlockCopy(adtBytes, mcnkOffset + 8, mcnkBody, 0, mcnkSize);

            var mcnk = new Mcnk(mcnkBody);
            if (mcnk.TextureLayers == null || mcnk.TextureLayers.Count == 0)
                continue;

            var chunkManifest = new DevelopmentChunkTextureManifest
            {
                ChunkIndex = chunkIndex
            };

            for (int layerIndex = 0; layerIndex < mcnk.TextureLayers.Count; layerIndex++)
            {
                MclyEntry layer = mcnk.TextureLayers[layerIndex];
                string texturePath = layer.TextureId < textureData.MtexTextures.Count
                    ? textureData.MtexTextures[(int)layer.TextureId]
                    : string.Empty;

                byte[]? alphaBytes = mcnk.AlphaMaps?.GetAlphaMapForLayer(layer, false);

                chunkManifest.Layers.Add(new DevelopmentTextureLayerManifest
                {
                    LayerIndex = layerIndex,
                    TextureId = layer.TextureId,
                    TexturePath = texturePath,
                    Flags = (uint)layer.Flags,
                    AlphaMapOffset = layer.AlphaMapOffset,
                    EffectId = layer.EffectId,
                    AlphaBitsBase64 = alphaBytes is { Length: > 0 } ? Convert.ToBase64String(alphaBytes) : null,
                    AlphaByteCount = alphaBytes is { Length: > 0 } ? alphaBytes.Length : null
                });
            }

            if (chunkManifest.Layers.Count > 0)
                textureData.ChunkLayers.Add(chunkManifest);
        }

        return textureData;
    }

    private static List<string> ReadMtexTexturePaths(byte[] adtBytes)
    {
        int mtexOffset = FindTopLevelChunkOffset(adtBytes, "MTEX");
        if (mtexOffset < 0 || mtexOffset + 8 > adtBytes.Length)
            return new List<string>();

        int size = BitConverter.ToInt32(adtBytes, mtexOffset + 4);
        if (size <= 0 || mtexOffset + 8 + size > adtBytes.Length)
            return new List<string>();

        var values = new List<string>();
        int start = mtexOffset + 8;
        int end = start + size;
        int position = start;

        while (position < end)
        {
            int nextNull = Array.IndexOf(adtBytes, (byte)0, position, end - position);
            if (nextNull < 0)
                break;

            if (nextNull > position)
            {
                string value = Encoding.UTF8.GetString(adtBytes, position, nextNull - position);
                if (!string.IsNullOrWhiteSpace(value))
                    values.Add(value);
            }

            position = nextNull + 1;
        }

        return values;
    }

    private static WlMh2oTileData CloneWlTileData(WlMh2oTileData source)
    {
        var copy = new WlMh2oTileData();
        for (int y = 0; y < 16; y++)
        {
            for (int x = 0; x < 16; x++)
            {
                WlMh2oChunkData? srcChunk = source.Chunks[x, y];
                if (srcChunk == null)
                    continue;

                copy.Chunks[x, y] = CloneWlChunk(srcChunk);
            }
        }

        return copy;
    }

    private static void MergeWlTileData(WlMh2oTileData target, WlMh2oTileData source)
    {
        for (int y = 0; y < 16; y++)
        {
            for (int x = 0; x < 16; x++)
            {
                WlMh2oChunkData? srcChunk = source.Chunks[x, y];
                if (srcChunk == null)
                    continue;

                if (target.Chunks[x, y] == null)
                {
                    target.Chunks[x, y] = CloneWlChunk(srcChunk);
                }
            }
        }
    }

    private static WlMh2oChunkData CloneWlChunk(WlMh2oChunkData chunk)
    {
        return new WlMh2oChunkData
        {
            LiquidTypeId = chunk.LiquidTypeId,
            VertexFormat = chunk.VertexFormat,
            MinHeight = chunk.MinHeight,
            MaxHeight = chunk.MaxHeight,
            XOffset = chunk.XOffset,
            YOffset = chunk.YOffset,
            Width = chunk.Width,
            Height = chunk.Height,
            ExistsBitmap = chunk.ExistsBitmap == null ? null : chunk.ExistsBitmap.ToArray(),
            Heights = chunk.Heights.ToArray(),
            DepthMap = chunk.DepthMap == null ? null : chunk.DepthMap.ToArray()
        };
    }

    private static byte[] SerializeMh2oChunk(WlMh2oTileData tileData)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        // Reserve 256 chunk headers (12 bytes each).
        writer.Write(new byte[256 * 12]);

        var headers = new Mh2oChunkHeader[256];

        for (int chunkY = 0; chunkY < 16; chunkY++)
        {
            for (int chunkX = 0; chunkX < 16; chunkX++)
            {
                WlMh2oChunkData? chunk = tileData.Chunks[chunkX, chunkY];
                if (chunk == null)
                    continue;

                int chunkIndex = chunkY * 16 + chunkX;
                headers[chunkIndex].OffsetInstances = (uint)stream.Position;
                headers[chunkIndex].LayerCount = 1;
                headers[chunkIndex].OffsetAttributes = 0;

                byte width = chunk.Width > 0 ? (byte)chunk.Width : (byte)8;
                byte height = chunk.Height > 0 ? (byte)chunk.Height : (byte)8;

                writer.Write(chunk.LiquidTypeId);
                writer.Write((ushort)chunk.VertexFormat);
                writer.Write(chunk.MinHeight);
                writer.Write(chunk.MaxHeight);
                writer.Write((byte)chunk.XOffset);
                writer.Write((byte)chunk.YOffset);
                writer.Write(width);
                writer.Write(height);
                writer.Write(0u); // exists bitmap offset (all exist)
                long vertexOffsetField = stream.Position;
                writer.Write(0u); // vertex data offset (patched below)

                uint vertexDataOffset = (uint)stream.Position;
                WriteMh2oVertexData(writer, chunk, width, height);

                long end = stream.Position;
                stream.Position = vertexOffsetField;
                writer.Write(vertexDataOffset);
                stream.Position = end;
            }
        }

        byte[] bytes = stream.ToArray();
        for (int i = 0; i < 256; i++)
        {
            int headerOffset = i * 12;
            WriteUInt32(bytes, headerOffset, headers[i].OffsetInstances);
            WriteUInt32(bytes, headerOffset + 4, headers[i].LayerCount);
            WriteUInt32(bytes, headerOffset + 8, headers[i].OffsetAttributes);
        }

        return bytes;
    }

    private static void WriteMh2oVertexData(BinaryWriter writer, WlMh2oChunkData chunk, byte width, byte height)
    {
        int vertexCount = (width + 1) * (height + 1);
        float[] heights = EnsureLength(chunk.Heights, vertexCount, chunk.MinHeight);

        switch (chunk.VertexFormat)
        {
            case Mh2oVertexFormat.HeightUv:
                for (int i = 0; i < vertexCount; i++)
                    writer.Write(heights[i]);

                for (int i = 0; i < vertexCount; i++)
                {
                    writer.Write((ushort)0);
                    writer.Write((ushort)0);
                }
                break;

            case Mh2oVertexFormat.DepthOnly:
                byte[] depthOnly = chunk.DepthMap ?? Enumerable.Repeat((byte)128, vertexCount).ToArray();
                if (depthOnly.Length < vertexCount)
                    Array.Resize(ref depthOnly, vertexCount);
                for (int i = 0; i < vertexCount; i++)
                    writer.Write(depthOnly[i]);
                break;

            case Mh2oVertexFormat.HeightUvDepth:
                for (int i = 0; i < vertexCount; i++)
                    writer.Write(heights[i]);

                for (int i = 0; i < vertexCount; i++)
                {
                    writer.Write((ushort)0);
                    writer.Write((ushort)0);
                }

                byte[] depthWithUv = chunk.DepthMap ?? Enumerable.Repeat((byte)128, vertexCount).ToArray();
                if (depthWithUv.Length < vertexCount)
                    Array.Resize(ref depthWithUv, vertexCount);
                for (int i = 0; i < vertexCount; i++)
                    writer.Write(depthWithUv[i]);
                break;

            case Mh2oVertexFormat.HeightDepth:
            default:
                for (int i = 0; i < vertexCount; i++)
                    writer.Write(heights[i]);

                byte[] depthMap = chunk.DepthMap ?? Enumerable.Repeat((byte)128, vertexCount).ToArray();
                if (depthMap.Length < vertexCount)
                    Array.Resize(ref depthMap, vertexCount);
                for (int i = 0; i < vertexCount; i++)
                    writer.Write(depthMap[i]);
                break;
        }
    }

    private static float[] EnsureLength(float[]? values, int requiredLength, float fallback)
    {
        if (values == null || values.Length == 0)
            return Enumerable.Repeat(fallback, requiredLength).ToArray();

        if (values.Length >= requiredLength)
            return values;

        float[] expanded = new float[requiredLength];
        Array.Copy(values, expanded, values.Length);
        for (int i = values.Length; i < requiredLength; i++)
            expanded[i] = values[^1];
        return expanded;
    }

    private static byte[] InjectMh2oChunk(byte[] adtBytes, byte[] mh2oData)
    {
        int mhdrOffset = FindTopLevelChunkOffset(adtBytes, "MHDR");
        if (mhdrOffset < 0)
            return adtBytes;

        int padding = (mh2oData.Length & 1) == 1 ? 1 : 0;
        int newChunkSize = 8 + mh2oData.Length + padding;
        byte[] output = new byte[adtBytes.Length + newChunkSize];

        Buffer.BlockCopy(adtBytes, 0, output, 0, adtBytes.Length);

        int chunkOffset = adtBytes.Length;
        WriteReversedFourCc(output, chunkOffset, "MH2O");
        WriteInt32(output, chunkOffset + 4, mh2oData.Length);
        Buffer.BlockCopy(mh2oData, 0, output, chunkOffset + 8, mh2oData.Length);

        int mhdrDataStart = mhdrOffset + 8;
        int mh2oFieldOffset = mhdrDataStart + 0x28;
        if (mh2oFieldOffset + 4 <= output.Length)
        {
            WriteInt32(output, mh2oFieldOffset, chunkOffset - mhdrDataStart);
        }

        return output;
    }

    private static int FindTopLevelChunkOffset(byte[] bytes, string chunkName)
    {
        int position = 0;
        while (position + 8 <= bytes.Length)
        {
            string chunk = ReadNormalizedFourCc(bytes, position);
            int chunkSize = BitConverter.ToInt32(bytes, position + 4);
            if (chunkSize < 0 || position + 8 + chunkSize > bytes.Length)
                return -1;

            if (string.Equals(chunk, chunkName, StringComparison.Ordinal))
                return position;

            int next = position + 8 + chunkSize + ((chunkSize & 1) == 1 ? 1 : 0);
            if (next <= position)
                return -1;

            position = next;
        }

        return -1;
    }

    private static string ReadNormalizedFourCc(byte[] bytes, int offset)
    {
        string sig = Encoding.ASCII.GetString(bytes, offset, 4);
        string reversed = new string(sig.Reverse().ToArray());
        return reversed switch
        {
            "MVER" or "MHDR" or "MCIN" or "MTEX" or "MMDX" or "MMID" or "MWMO" or "MWID" or "MDDF" or "MODF" or "MH2O" or "MCNK" => reversed,
            _ => sig
        };
    }

    private static void WriteWdtFromOutputTiles(string outputMapDirectory, string mapName, string wdtPath)
    {
        var tileFlags = new bool[64, 64];

        if (Directory.Exists(outputMapDirectory))
        {
            foreach (string file in Directory.EnumerateFiles(outputMapDirectory, mapName + "_*_*.adt", SearchOption.TopDirectoryOnly))
            {
                string name = Path.GetFileNameWithoutExtension(file);
                string[] parts = name.Split('_');
                if (parts.Length < 3)
                    continue;

                if (int.TryParse(parts[^2], out int tileX)
                    && int.TryParse(parts[^1], out int tileY)
                    && tileX is >= 0 and < 64
                    && tileY is >= 0 and < 64)
                {
                    tileFlags[tileY, tileX] = true;
                }
            }
        }

        using var stream = new FileStream(wdtPath, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);

        // MVER
        writer.Write(Encoding.ASCII.GetBytes("REVM"));
        writer.Write(4);
        writer.Write(18);

        // MPHD
        writer.Write(Encoding.ASCII.GetBytes("DHPM"));
        writer.Write(32);
        writer.Write(0x0E); // MCCV | BigAlpha | DoodadRefsSorted
        for (int i = 0; i < 28; i++)
            writer.Write((byte)0);

        // MAIN
        writer.Write(Encoding.ASCII.GetBytes("NIAM"));
        writer.Write(64 * 64 * 8);
        for (int tileY = 0; tileY < 64; tileY++)
        {
            for (int tileX = 0; tileX < 64; tileX++)
            {
                writer.Write(tileFlags[tileY, tileX] ? 1u : 0u);
                writer.Write(0u);
            }
        }

        // MWMO and MODF empty.
        writer.Write(Encoding.ASCII.GetBytes("OMWM"));
        writer.Write(0);
        writer.Write(Encoding.ASCII.GetBytes("FDOM"));
        writer.Write(0);
    }

    private static (int x, int y)? ParseTileFilter(string? tile)
    {
        if (string.IsNullOrWhiteSpace(tile))
            return null;

        string[] parts = tile.Split('_', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length != 2
            || !int.TryParse(parts[0], out int tileX)
            || !int.TryParse(parts[1], out int tileY)
            || tileX is < 0 or >= 64
            || tileY is < 0 or >= 64)
        {
            throw new ArgumentException($"Invalid --tile value '{tile}'. Expected x_y within 0..63.");
        }

        return (tileX, tileY);
    }

    private static void AddSource(DevelopmentTileRepairManifest manifest, string kind, string path)
    {
        manifest.SourceFiles.Add(new DevelopmentSourceFileEntry
        {
            Kind = kind,
            Path = path,
            Exists = File.Exists(path),
            Used = false
        });
    }

    private static void MarkSourceUsed(DevelopmentTileRepairManifest manifest, string kind)
    {
        DevelopmentSourceFileEntry? entry = manifest.SourceFiles.FirstOrDefault(source => source.Kind == kind);
        if (entry != null)
            entry.Used = true;
    }

    private static bool TryGetSourcePath(DevelopmentTileRepairManifest manifest, string kind, out string? path)
    {
        DevelopmentSourceFileEntry? entry = manifest.SourceFiles.FirstOrDefault(source => source.Kind == kind);
        if (entry == null)
        {
            path = null;
            return false;
        }

        path = entry.Path;
        return true;
    }

    private static void WriteInt32(byte[] bytes, int offset, int value)
    {
        byte[] valueBytes = BitConverter.GetBytes(value);
        Buffer.BlockCopy(valueBytes, 0, bytes, offset, sizeof(int));
    }

    private static void WriteUInt32(byte[] bytes, int offset, uint value)
    {
        byte[] valueBytes = BitConverter.GetBytes(value);
        Buffer.BlockCopy(valueBytes, 0, bytes, offset, sizeof(uint));
    }

    private static void WriteReversedFourCc(byte[] bytes, int offset, string fourCc)
    {
        string reversed = new string(fourCc.Reverse().ToArray());
        byte[] sig = Encoding.ASCII.GetBytes(reversed);
        Buffer.BlockCopy(sig, 0, bytes, offset, sig.Length);
    }
}
