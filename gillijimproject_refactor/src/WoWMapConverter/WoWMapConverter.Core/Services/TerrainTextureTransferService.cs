using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using WoWMapConverter.Core.Formats.PM4;
using WoWRollback.PM4Module;

namespace WoWMapConverter.Core.Services;

public sealed record TerrainTilePair(
    int SourceTileX,
    int SourceTileY,
    int TargetTileX,
    int TargetTileY)
{
    public string SourceLabel => $"{SourceTileX}_{SourceTileY}";
    public string TargetLabel => $"{TargetTileX}_{TargetTileY}";
}

public sealed record TerrainTextureTransferOptions(
    string SourceDirectory,
    string TargetDirectory,
    string OutputDirectory,
    string Mode,
    List<TerrainTilePair> Pairs,
    int? TileLimit,
    int? GlobalDeltaX,
    int? GlobalDeltaY,
    int ChunkOffsetX,
    int ChunkOffsetY,
    bool CopyMtex,
    bool CopyMcly,
    bool CopyMcal,
    bool CopyMcsh,
    bool CopyHoles,
    string? ManifestPath)
{
    public static TerrainTextureTransferOptions CreateDefault() => new(
        SourceDirectory: DevelopmentMapAnalyzer.DefaultDevelopmentMapDirectory,
        TargetDirectory: DevelopmentMapAnalyzer.DefaultDevelopmentMapDirectory,
        OutputDirectory: Path.Combine("output", "terrain-texture-transfer"),
        Mode: "dry-run",
        Pairs: new List<TerrainTilePair>(),
        TileLimit: null,
        GlobalDeltaX: null,
        GlobalDeltaY: null,
        ChunkOffsetX: 0,
        ChunkOffsetY: 0,
        CopyMtex: true,
        CopyMcly: true,
        CopyMcal: true,
        CopyMcsh: true,
        CopyHoles: true,
        ManifestPath: null);

    public bool IsApplyMode => string.Equals(Mode, "apply", StringComparison.OrdinalIgnoreCase);
}

public sealed class TerrainTextureTransferFileEntry
{
    public string Kind { get; set; } = string.Empty;
    public string Path { get; set; } = string.Empty;
    public bool Exists { get; set; }
    public bool Used { get; set; }
}

public sealed class TerrainTextureTransferTileManifest
{
    public string SourceTileName { get; set; } = string.Empty;
    public string TargetTileName { get; set; } = string.Empty;
    public int SourceTileX { get; set; }
    public int SourceTileY { get; set; }
    public int TargetTileX { get; set; }
    public int TargetTileY { get; set; }
    public List<TerrainTextureTransferFileEntry> SourceFiles { get; set; } = new();
    public List<TerrainTextureTransferFileEntry> TargetFiles { get; set; } = new();
    public bool MtexCopied { get; set; }
    public bool MclyCopied { get; set; }
    public bool McalCopied { get; set; }
    public bool McshCopied { get; set; }
    public bool HolesCopied { get; set; }
    public int SourceChunksUsed { get; set; }
    public int TargetChunksTouched { get; set; }
    public int MissingSourceChunkCount { get; set; }
    public int OutOfRangeChunkRemapCount { get; set; }
    public int McnkIndexMismatchCountAfterTransfer { get; set; }
    public bool OutputWritten { get; set; }
    public string? OutputAdtPath { get; set; }
    public bool NeedsManualReview { get; set; }
    public List<string> Warnings { get; set; } = new();
}

public sealed class TerrainTextureTransferSummaryManifest
{
    public string SourceDirectory { get; set; } = string.Empty;
    public string TargetDirectory { get; set; } = string.Empty;
    public string SourceMapName { get; set; } = string.Empty;
    public string TargetMapName { get; set; } = string.Empty;
    public string OutputDirectory { get; set; } = string.Empty;
    public string Mode { get; set; } = string.Empty;
    public int TilesPlanned { get; set; }
    public int TilesProcessed { get; set; }
    public int TilesWritten { get; set; }
    public int TilesNeedingManualReview { get; set; }
    public int ChunkPairsApplied { get; set; }
    public int MissingSourceChunkCount { get; set; }
    public int OutOfRangeChunkRemapCount { get; set; }
    public List<TerrainTextureTransferTileManifest> Tiles { get; set; } = new();
}

public sealed record TerrainTextureTransferExecutionReport(
    string SourceDirectory,
    string TargetDirectory,
    string SourceMapName,
    string TargetMapName,
    string OutputDirectory,
    string SummaryManifestPath,
    int TilesPlanned,
    int TilesProcessed,
    int TilesWritten,
    int TilesNeedingManualReview,
    int ChunkPairsApplied,
    List<TerrainTextureTransferTileManifest> Tiles);

public static partial class TerrainTextureTransferService
{
    private sealed class TopLevelChunk
    {
        public required byte[] RawSignature { get; init; }
        public required string Signature { get; init; }
        public required byte[] Data { get; set; }
        public required int Offset { get; init; }
    }

    private sealed class McnkSubChunk
    {
        public required byte[] RawSignature { get; init; }
        public required string Signature { get; init; }
        public required int DeclaredSize { get; set; }
        public required byte[] Data { get; set; }
        public required byte[] TailBytes { get; set; }
    }

    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    private static readonly HashSet<string> KnownTopLevelFourCc = new(StringComparer.Ordinal)
    {
        "MVER", "MHDR", "MCIN", "MTEX", "MMDX", "MMID", "MWMO", "MWID", "MDDF", "MODF", "MH2O", "MCNK"
    };

    private static readonly HashSet<string> KnownMcnkSubFourCc = new(StringComparer.Ordinal)
    {
        "MCVT", "MCNR", "MCLY", "MCRF", "MCAL", "MCSH", "MCSE", "MCLQ", "MCCV"
    };

    [GeneratedRegex(@"^(?<map>.+)_(?<x>\d+)_(?<y>\d+)\.adt$", RegexOptions.IgnoreCase | RegexOptions.Compiled)]
    private static partial Regex RootTilePattern();

    public static TerrainTextureTransferExecutionReport Execute(TerrainTextureTransferOptions options)
    {
        if (!string.Equals(options.Mode, "dry-run", StringComparison.OrdinalIgnoreCase)
            && !string.Equals(options.Mode, "apply", StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Unsupported mode '{options.Mode}'. Expected dry-run|apply.");
        }

        string sourceDirectory = Pm4CoordinateService.ResolveMapDirectory(options.SourceDirectory);
        string targetDirectory = Pm4CoordinateService.ResolveMapDirectory(options.TargetDirectory);
        string sourceMapName = InferMapName(sourceDirectory);
        string targetMapName = InferMapName(targetDirectory);

        string outputDirectory = Path.GetFullPath(options.OutputDirectory);
        string outputMapDirectory = Path.Combine(outputDirectory, "World", "Maps", targetMapName);
        string tileManifestDirectory = Path.Combine(outputDirectory, "manifests", "tiles");
        string summaryManifestPath = options.ManifestPath is { Length: > 0 }
            ? Path.GetFullPath(options.ManifestPath)
            : Path.Combine(outputDirectory, "manifests", "summary.json");

        Directory.CreateDirectory(outputDirectory);
        Directory.CreateDirectory(Path.GetDirectoryName(summaryManifestPath) ?? outputDirectory);
        Directory.CreateDirectory(tileManifestDirectory);
        if (options.IsApplyMode)
            Directory.CreateDirectory(outputMapDirectory);

        List<TerrainTilePair> plannedPairs = BuildPairList(options, sourceDirectory, sourceMapName);
        if (options.TileLimit.HasValue)
            plannedPairs = plannedPairs.Take(options.TileLimit.Value).ToList();

        var splitMerger = new SplitAdtMerger();
        var tileManifests = new List<TerrainTextureTransferTileManifest>();
        int tilesWritten = 0;
        int chunkPairsApplied = 0;
        int missingSourceChunks = 0;
        int outOfRangeChunkRemaps = 0;

        foreach (TerrainTilePair pair in plannedPairs)
        {
            var tileManifest = CreateTileManifest(sourceDirectory, targetDirectory, sourceMapName, targetMapName, pair);

            byte[]? sourceBytes = LoadMergedTileBytes(
                splitMerger,
                tileManifest.SourceFiles,
                tileManifest.Warnings,
                out bool sourceMergeUsedSplitInputs);

            byte[]? targetBytes = LoadMergedTileBytes(
                splitMerger,
                tileManifest.TargetFiles,
                tileManifest.Warnings,
                out bool targetMergeUsedSplitInputs);

            if (sourceBytes == null || targetBytes == null)
            {
                tileManifest.NeedsManualReview = true;
                WriteTileManifest(tileManifestDirectory, tileManifest);
                tileManifests.Add(tileManifest);
                continue;
            }

            if (!TryTransferTileTexturePayload(
                options,
                sourceBytes,
                targetBytes,
                tileManifest,
                out byte[]? transferredBytes))
            {
                tileManifest.NeedsManualReview = true;
                WriteTileManifest(tileManifestDirectory, tileManifest);
                tileManifests.Add(tileManifest);
                continue;
            }

            if (transferredBytes == null)
            {
                tileManifest.NeedsManualReview = true;
                WriteTileManifest(tileManifestDirectory, tileManifest);
                tileManifests.Add(tileManifest);
                continue;
            }

            DevelopmentMcnkIndexRepairReport repairReport = DevelopmentMcnkIndexRepairService.RepairBytes(
                transferredBytes,
                tileManifest.TargetTileName,
                writeChanges: true);

            transferredBytes = repairReport.RepairedBytes ?? transferredBytes;
            tileManifest.McnkIndexMismatchCountAfterTransfer = repairReport.MismatchCount;

            chunkPairsApplied += tileManifest.SourceChunksUsed;
            missingSourceChunks += tileManifest.MissingSourceChunkCount;
            outOfRangeChunkRemaps += tileManifest.OutOfRangeChunkRemapCount;

            if (options.IsApplyMode)
            {
                string outPath = Path.Combine(outputMapDirectory, tileManifest.TargetTileName + ".adt");
                File.WriteAllBytes(outPath, transferredBytes);
                tileManifest.OutputAdtPath = outPath;
                tileManifest.OutputWritten = true;
                tilesWritten++;
            }

            tileManifest.NeedsManualReview = tileManifest.NeedsManualReview
                || tileManifest.Warnings.Count > 0
                || tileManifest.SourceChunksUsed == 0;

            if (!sourceMergeUsedSplitInputs && !targetMergeUsedSplitInputs)
            {
                // no-op, but keep both values read to avoid warnings in future refactors
            }

            WriteTileManifest(tileManifestDirectory, tileManifest);
            tileManifests.Add(tileManifest);
        }

        if (options.IsApplyMode)
            CopyTargetMapCompanions(targetDirectory, targetMapName, outputDirectory);

        var summary = new TerrainTextureTransferSummaryManifest
        {
            SourceDirectory = sourceDirectory,
            TargetDirectory = targetDirectory,
            SourceMapName = sourceMapName,
            TargetMapName = targetMapName,
            OutputDirectory = outputDirectory,
            Mode = options.Mode,
            TilesPlanned = plannedPairs.Count,
            TilesProcessed = tileManifests.Count,
            TilesWritten = tilesWritten,
            TilesNeedingManualReview = tileManifests.Count(tile => tile.NeedsManualReview),
            ChunkPairsApplied = chunkPairsApplied,
            MissingSourceChunkCount = missingSourceChunks,
            OutOfRangeChunkRemapCount = outOfRangeChunkRemaps,
            Tiles = tileManifests
        };

        File.WriteAllText(summaryManifestPath, JsonSerializer.Serialize(summary, JsonOptions));

        return new TerrainTextureTransferExecutionReport(
            SourceDirectory: sourceDirectory,
            TargetDirectory: targetDirectory,
            SourceMapName: sourceMapName,
            TargetMapName: targetMapName,
            OutputDirectory: outputDirectory,
            SummaryManifestPath: summaryManifestPath,
            TilesPlanned: plannedPairs.Count,
            TilesProcessed: tileManifests.Count,
            TilesWritten: tilesWritten,
            TilesNeedingManualReview: tileManifests.Count(tile => tile.NeedsManualReview),
            ChunkPairsApplied: chunkPairsApplied,
            Tiles: tileManifests);
    }

    private static TerrainTextureTransferTileManifest CreateTileManifest(
        string sourceDirectory,
        string targetDirectory,
        string sourceMapName,
        string targetMapName,
        TerrainTilePair pair)
    {
        string sourceBase = $"{sourceMapName}_{pair.SourceTileX}_{pair.SourceTileY}";
        string targetBase = $"{targetMapName}_{pair.TargetTileX}_{pair.TargetTileY}";

        var tileManifest = new TerrainTextureTransferTileManifest
        {
            SourceTileName = sourceBase,
            TargetTileName = targetBase,
            SourceTileX = pair.SourceTileX,
            SourceTileY = pair.SourceTileY,
            TargetTileX = pair.TargetTileX,
            TargetTileY = pair.TargetTileY,
        };

        AddFileEntry(tileManifest.SourceFiles, "root-adt", Path.Combine(sourceDirectory, sourceBase + ".adt"));
        AddFileEntry(tileManifest.SourceFiles, "obj0-adt", Path.Combine(sourceDirectory, sourceBase + "_obj0.adt"));
        AddFileEntry(tileManifest.SourceFiles, "tex0-adt", Path.Combine(sourceDirectory, sourceBase + "_tex0.adt"));

        AddFileEntry(tileManifest.TargetFiles, "root-adt", Path.Combine(targetDirectory, targetBase + ".adt"));
        AddFileEntry(tileManifest.TargetFiles, "obj0-adt", Path.Combine(targetDirectory, targetBase + "_obj0.adt"));
        AddFileEntry(tileManifest.TargetFiles, "tex0-adt", Path.Combine(targetDirectory, targetBase + "_tex0.adt"));

        return tileManifest;
    }

    private static void AddFileEntry(List<TerrainTextureTransferFileEntry> entries, string kind, string path)
    {
        entries.Add(new TerrainTextureTransferFileEntry
        {
            Kind = kind,
            Path = path,
            Exists = File.Exists(path),
            Used = false
        });
    }

    private static void WriteTileManifest(string tileManifestDirectory, TerrainTextureTransferTileManifest tileManifest)
    {
        string tileManifestPath = Path.Combine(tileManifestDirectory, tileManifest.TargetTileName + ".json");
        File.WriteAllText(tileManifestPath, JsonSerializer.Serialize(tileManifest, JsonOptions));
    }

    private static void CopyTargetMapCompanions(string targetDirectory, string targetMapName, string outputDirectory)
    {
        string[] companionExtensions = [".wdt", ".wdl", ".wdl.mpq"];

        foreach (string extension in companionExtensions)
        {
            string src = Path.Combine(targetDirectory, targetMapName + extension);
            if (!File.Exists(src))
                continue;

            string dst = Path.Combine(outputDirectory, Path.GetFileName(src));
            File.Copy(src, dst, overwrite: true);
        }
    }

    private static byte[]? LoadMergedTileBytes(
        SplitAdtMerger splitMerger,
        List<TerrainTextureTransferFileEntry> files,
        List<string> warnings,
        out bool mergeUsedSplitInputs)
    {
        mergeUsedSplitInputs = false;

        TerrainTextureTransferFileEntry? root = files.FirstOrDefault(file => file.Kind == "root-adt");
        if (root == null || !root.Exists)
        {
            warnings.Add("Missing root ADT.");
            return null;
        }

        TerrainTextureTransferFileEntry? obj0 = files.FirstOrDefault(file => file.Kind == "obj0-adt");
        TerrainTextureTransferFileEntry? tex0 = files.FirstOrDefault(file => file.Kind == "tex0-adt");

        bool hasTex0Input = tex0?.Exists ?? false;
        if (!hasTex0Input)
        {
            root.Used = true;
            return File.ReadAllBytes(root.Path);
        }

        string? obj0Path = obj0 is { Exists: true } ? obj0.Path : null;
        string? tex0Path = tex0 is { Exists: true } ? tex0.Path : null;

        SplitAdtMerger.MergeResult mergeResult = splitMerger.Merge(root.Path, obj0Path, tex0Path);
        if (mergeResult.Success && mergeResult.Data != null)
        {
            root.Used = true;
            if (obj0Path != null && obj0 != null) obj0.Used = true;
            if (tex0Path != null && tex0 != null) tex0.Used = true;
            mergeUsedSplitInputs = true;
            return mergeResult.Data;
        }

        if (tex0Path != null && tex0 != null
            && TryComposeRootWithTex0TextureData(root.Path, tex0Path, warnings, out byte[]? composedBytes)
            && composedBytes != null)
        {
            root.Used = true;
            tex0.Used = true;
            mergeUsedSplitInputs = true;
            return composedBytes;
        }

        root.Used = true;
        warnings.Add($"Split merge failed ({mergeResult.Error ?? "unknown"}); using root ADT bytes.");
        return File.ReadAllBytes(root.Path);
    }

    private static bool TryComposeRootWithTex0TextureData(
        string rootPath,
        string tex0Path,
        List<string> warnings,
        out byte[]? mergedBytes)
    {
        mergedBytes = null;

        try
        {
            if (!File.Exists(rootPath) || !File.Exists(tex0Path))
                return false;

            List<TopLevelChunk> rootChunks = ParseTopLevelChunks(File.ReadAllBytes(rootPath));
            List<TopLevelChunk> tex0Chunks = ParseTopLevelChunks(File.ReadAllBytes(tex0Path));
            if (rootChunks.Count == 0 || tex0Chunks.Count == 0)
            {
                warnings.Add("root+tex0 fallback could not parse top-level chunks.");
                return false;
            }

            bool mtexCopied = TryCopyMtex(tex0Chunks, rootChunks);
            Dictionary<int, int> rootMap = BuildMcnkIndexMap(rootChunks);
            Dictionary<int, int> tex0Map = BuildMcnkIndexMap(tex0Chunks);

            if (rootMap.Count == 0 || tex0Map.Count == 0)
            {
                warnings.Add("root+tex0 fallback could not build MCNK index maps.");
                if (!mtexCopied)
                    return false;
            }

            int chunksPatched = 0;
            foreach ((int chunkIndex, int rootTopIndex) in rootMap)
            {
                if (!tex0Map.TryGetValue(chunkIndex, out int tex0TopIndex))
                    continue;

                if (TryOverlayTextureSubchunks(
                    sourceTexturePayload: tex0Chunks[tex0TopIndex].Data,
                    targetRootPayload: rootChunks[rootTopIndex].Data,
                    out byte[]? rebuiltPayload)
                    && rebuiltPayload != null)
                {
                    rootChunks[rootTopIndex].Data = rebuiltPayload;
                    chunksPatched++;
                }
            }

            if (chunksPatched == 0 && !mtexCopied)
            {
                warnings.Add("root+tex0 fallback found no transferable texture payload chunks.");
                return false;
            }

            mergedBytes = BuildTopLevelBytes(rootChunks);
            return true;
        }
        catch (Exception ex)
        {
            warnings.Add($"root+tex0 fallback failed: {ex.Message}");
            return false;
        }
    }

    private static bool TryOverlayTextureSubchunks(
        byte[] sourceTexturePayload,
        byte[] targetRootPayload,
        out byte[]? rebuiltPayload)
    {
        rebuiltPayload = null;

        if (targetRootPayload.Length < 128)
            return false;

        List<McnkSubChunk> sourceSubchunks = ParseMcnkSubchunks(sourceTexturePayload);
        if (sourceSubchunks.Count == 0)
            return false;

        List<McnkSubChunk> targetSubchunks = ParseMcnkSubchunks(targetRootPayload);
        if (targetSubchunks.Count == 0)
            return false;

        var sourceBySignature = sourceSubchunks
            .GroupBy(subChunk => subChunk.Signature)
            .ToDictionary(group => group.Key, group => group.First(), StringComparer.Ordinal);

        bool copiedMcly = false;
        bool copiedMcal = false;
        bool copiedMcsh = false;

        int targetMclyIndex = targetSubchunks.FindIndex(subChunk => string.Equals(subChunk.Signature, "MCLY", StringComparison.Ordinal));
        if (sourceBySignature.TryGetValue("MCLY", out McnkSubChunk? sourceMcly))
        {
            var replacement = new McnkSubChunk
            {
                RawSignature = (byte[])sourceMcly.RawSignature.Clone(),
                Signature = sourceMcly.Signature,
                DeclaredSize = sourceMcly.DeclaredSize,
                Data = (byte[])sourceMcly.Data.Clone(),
                TailBytes = Array.Empty<byte>()
            };

            if (targetMclyIndex >= 0)
                targetSubchunks[targetMclyIndex] = replacement;
            else
                targetSubchunks.Add(replacement);

            copiedMcly = true;
        }

        int targetMcalIndex = targetSubchunks.FindIndex(subChunk => string.Equals(subChunk.Signature, "MCAL", StringComparison.Ordinal));
        if (sourceBySignature.TryGetValue("MCAL", out McnkSubChunk? sourceMcal))
        {
            var replacement = new McnkSubChunk
            {
                RawSignature = (byte[])sourceMcal.RawSignature.Clone(),
                Signature = sourceMcal.Signature,
                DeclaredSize = sourceMcal.DeclaredSize,
                Data = (byte[])sourceMcal.Data.Clone(),
                TailBytes = Array.Empty<byte>()
            };

            if (targetMcalIndex >= 0)
                targetSubchunks[targetMcalIndex] = replacement;
            else
                targetSubchunks.Add(replacement);

            copiedMcal = true;
        }

        int targetMcshIndex = targetSubchunks.FindIndex(subChunk => string.Equals(subChunk.Signature, "MCSH", StringComparison.Ordinal));
        if (sourceBySignature.TryGetValue("MCSH", out McnkSubChunk? sourceMcsh))
        {
            var replacement = new McnkSubChunk
            {
                RawSignature = (byte[])sourceMcsh.RawSignature.Clone(),
                Signature = sourceMcsh.Signature,
                DeclaredSize = sourceMcsh.DeclaredSize,
                Data = (byte[])sourceMcsh.Data.Clone(),
                TailBytes = Array.Empty<byte>()
            };

            if (targetMcshIndex >= 0)
                targetSubchunks[targetMcshIndex] = replacement;
            else
                targetSubchunks.Add(replacement);

            copiedMcsh = true;
        }

        if (!copiedMcly && !copiedMcal && !copiedMcsh)
            return false;

        byte[] header = new byte[128];
        Buffer.BlockCopy(targetRootPayload, 0, header, 0, 128);
        if (copiedMcly && sourceBySignature.TryGetValue("MCLY", out McnkSubChunk? sourceMclyForCount))
        {
            int layerCount = sourceMclyForCount.DeclaredSize / 16;
            WriteUInt32(header, 0x0C, (uint)Math.Max(0, layerCount));
        }

        using var stream = new MemoryStream();
        stream.Write(header, 0, header.Length);

        var offsetBySignature = new Dictionary<string, int>(StringComparer.Ordinal);
        var sizeBySignature = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (McnkSubChunk subChunk in targetSubchunks)
        {
            int headerPos = checked((int)stream.Position);
            stream.Write(subChunk.RawSignature, 0, 4);
            stream.Write(BitConverter.GetBytes(subChunk.DeclaredSize), 0, 4);
            if (subChunk.Data.Length > 0)
                stream.Write(subChunk.Data, 0, subChunk.Data.Length);
            if (subChunk.TailBytes.Length > 0)
                stream.Write(subChunk.TailBytes, 0, subChunk.TailBytes.Length);

            if (!offsetBySignature.ContainsKey(subChunk.Signature))
            {
                offsetBySignature[subChunk.Signature] = headerPos + 8;
                sizeBySignature[subChunk.Signature] = subChunk.DeclaredSize;
            }
        }

        PatchMcnkHeaderOffsets(header, offsetBySignature, sizeBySignature);

        byte[] payload = stream.ToArray();
        Buffer.BlockCopy(header, 0, payload, 0, header.Length);
        rebuiltPayload = payload;
        return true;
    }

    private static bool TryTransferTileTexturePayload(
        TerrainTextureTransferOptions options,
        byte[] sourceBytes,
        byte[] targetBytes,
        TerrainTextureTransferTileManifest manifest,
        out byte[]? outputBytes)
    {
        outputBytes = null;

        List<TopLevelChunk> sourceChunks = ParseTopLevelChunks(sourceBytes);
        List<TopLevelChunk> targetChunks = ParseTopLevelChunks(targetBytes);
        if (sourceChunks.Count == 0 || targetChunks.Count == 0)
        {
            manifest.Warnings.Add("Failed to parse top-level chunks.");
            return false;
        }

        Dictionary<int, int> sourceMcnkMap = BuildMcnkIndexMap(sourceChunks);
        Dictionary<int, int> targetMcnkMap = BuildMcnkIndexMap(targetChunks);
        if (sourceMcnkMap.Count == 0 || targetMcnkMap.Count == 0)
        {
            manifest.Warnings.Add("Source or target has no usable MCNK map.");
            return false;
        }

        if (options.CopyMtex)
        {
            bool copiedMtex = TryCopyMtex(sourceChunks, targetChunks);
            manifest.MtexCopied = copiedMtex;
            if (!copiedMtex)
                manifest.Warnings.Add("MTEX copy requested but source MTEX was not found.");
        }

        int chunksTouched = 0;
        int sourceChunksUsed = 0;
        int missingSourceChunks = 0;
        int outOfRangeChunkRemaps = 0;
        bool anyMclyCopied = false;
        bool anyMcalCopied = false;
        bool anyMcshCopied = false;
        bool anyHolesCopied = false;

        foreach ((int targetChunkIndex, int targetTopIndex) in targetMcnkMap.OrderBy(entry => entry.Key))
        {
            int sourceChunkIndex = ResolveSourceChunkIndex(targetChunkIndex, options.ChunkOffsetX, options.ChunkOffsetY);
            if (sourceChunkIndex < 0)
            {
                outOfRangeChunkRemaps++;
                continue;
            }

            if (!sourceMcnkMap.TryGetValue(sourceChunkIndex, out int sourceTopIndex))
            {
                missingSourceChunks++;
                continue;
            }

            TopLevelChunk sourceChunk = sourceChunks[sourceTopIndex];
            TopLevelChunk targetChunk = targetChunks[targetTopIndex];
            if (!TryTransferMcnkChunk(
                    options,
                    sourceChunk.Data,
                    targetChunk.Data,
                    manifest,
                    out byte[]? rebuiltMcnkPayload,
                    out bool copiedMcly,
                    out bool copiedMcal,
                    out bool copiedMcsh,
                    out bool copiedHoles))
            {
                continue;
            }

            if (rebuiltMcnkPayload == null)
                continue;

            targetChunk.Data = rebuiltMcnkPayload;

            chunksTouched++;
            sourceChunksUsed++;
            anyMclyCopied |= copiedMcly;
            anyMcalCopied |= copiedMcal;
            anyMcshCopied |= copiedMcsh;
            anyHolesCopied |= copiedHoles;
        }

        manifest.TargetChunksTouched = chunksTouched;
        manifest.SourceChunksUsed = sourceChunksUsed;
        manifest.MissingSourceChunkCount = missingSourceChunks;
        manifest.OutOfRangeChunkRemapCount = outOfRangeChunkRemaps;
        manifest.MclyCopied = anyMclyCopied;
        manifest.McalCopied = anyMcalCopied;
        manifest.McshCopied = anyMcshCopied;
        manifest.HolesCopied = anyHolesCopied;

        if (chunksTouched == 0)
        {
            manifest.Warnings.Add("No chunks were transferred.");
            return false;
        }

        outputBytes = BuildTopLevelBytes(targetChunks);
        return true;
    }

    private static int ResolveSourceChunkIndex(int targetChunkIndex, int chunkOffsetX, int chunkOffsetY)
    {
        int targetX = targetChunkIndex % 16;
        int targetY = targetChunkIndex / 16;

        int sourceX = targetX + chunkOffsetX;
        int sourceY = targetY + chunkOffsetY;
        if (sourceX < 0 || sourceX > 15 || sourceY < 0 || sourceY > 15)
            return -1;

        return sourceY * 16 + sourceX;
    }

    private static bool TryTransferMcnkChunk(
        TerrainTextureTransferOptions options,
        byte[] sourcePayload,
        byte[] targetPayload,
        TerrainTextureTransferTileManifest manifest,
        out byte[]? rebuiltPayload,
        out bool copiedMcly,
        out bool copiedMcal,
        out bool copiedMcsh,
        out bool copiedHoles)
    {
        rebuiltPayload = null;
        copiedMcly = false;
        copiedMcal = false;
        copiedMcsh = false;
        copiedHoles = false;

        if (sourcePayload.Length < 128 || targetPayload.Length < 128)
        {
            manifest.Warnings.Add("MCNK payload shorter than 128-byte header.");
            return false;
        }

        List<McnkSubChunk> sourceSubchunks = ParseMcnkSubchunks(sourcePayload);
        List<McnkSubChunk> targetSubchunks = ParseMcnkSubchunks(targetPayload);
        if (targetSubchunks.Count == 0)
        {
            manifest.Warnings.Add("Target MCNK has no parseable subchunks.");
            return false;
        }

        var sourceBySignature = sourceSubchunks
            .GroupBy(subChunk => subChunk.Signature)
            .ToDictionary(group => group.Key, group => group.First(), StringComparer.Ordinal);

        var rebuiltSubchunks = targetSubchunks
            .Select(subChunk => new McnkSubChunk
            {
                RawSignature = (byte[])subChunk.RawSignature.Clone(),
                Signature = subChunk.Signature,
                DeclaredSize = subChunk.DeclaredSize,
                Data = (byte[])subChunk.Data.Clone(),
                TailBytes = (byte[])subChunk.TailBytes.Clone(),
            })
            .ToList();

        byte[] header = new byte[128];
        Buffer.BlockCopy(targetPayload, 0, header, 0, 128);

        if (options.CopyHoles)
        {
            header[0x3C] = sourcePayload[0x3C];
            header[0x3D] = sourcePayload[0x3D];
            copiedHoles = true;
        }

        copiedMcly = ReplaceOrAppendSubchunk(rebuiltSubchunks, sourceBySignature, "MCLY", options.CopyMcly, manifest);
        copiedMcal = ReplaceOrAppendSubchunk(rebuiltSubchunks, sourceBySignature, "MCAL", options.CopyMcal, manifest);
        copiedMcsh = ReplaceOrAppendSubchunk(rebuiltSubchunks, sourceBySignature, "MCSH", options.CopyMcsh, manifest);

        if (options.CopyMcly && sourceBySignature.TryGetValue("MCLY", out McnkSubChunk? sourceMcly))
        {
            int layerCount = sourceMcly.DeclaredSize / 16;
            WriteUInt32(header, 0x0C, (uint)Math.Max(0, layerCount));
        }

        using var stream = new MemoryStream();
        stream.Write(header, 0, header.Length);

        var offsetBySignature = new Dictionary<string, int>(StringComparer.Ordinal);
        var sizeBySignature = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (McnkSubChunk subChunk in rebuiltSubchunks)
        {
            int headerPos = checked((int)stream.Position);
            stream.Write(subChunk.RawSignature, 0, 4);
            stream.Write(BitConverter.GetBytes(subChunk.DeclaredSize), 0, 4);
            if (subChunk.Data.Length > 0)
                stream.Write(subChunk.Data, 0, subChunk.Data.Length);
            if (subChunk.TailBytes.Length > 0)
                stream.Write(subChunk.TailBytes, 0, subChunk.TailBytes.Length);

            if (!offsetBySignature.ContainsKey(subChunk.Signature))
            {
                offsetBySignature[subChunk.Signature] = headerPos + 8;
                sizeBySignature[subChunk.Signature] = subChunk.DeclaredSize;
            }
        }

        PatchMcnkHeaderOffsets(header, offsetBySignature, sizeBySignature);

        byte[] payload = stream.ToArray();
        Buffer.BlockCopy(header, 0, payload, 0, header.Length);
        rebuiltPayload = payload;
        return true;
    }

    private static bool ReplaceOrAppendSubchunk(
        List<McnkSubChunk> targetSubchunks,
        Dictionary<string, McnkSubChunk> sourceBySignature,
        string signature,
        bool enabled,
        TerrainTextureTransferTileManifest manifest)
    {
        if (!enabled)
            return false;

        if (!sourceBySignature.TryGetValue(signature, out McnkSubChunk? sourceSubchunk))
            return false;

        int targetIndex = targetSubchunks.FindIndex(subChunk => string.Equals(subChunk.Signature, signature, StringComparison.Ordinal));
        if (targetIndex >= 0)
        {
            targetSubchunks[targetIndex] = new McnkSubChunk
            {
                RawSignature = (byte[])sourceSubchunk.RawSignature.Clone(),
                Signature = signature,
                DeclaredSize = sourceSubchunk.DeclaredSize,
                Data = (byte[])sourceSubchunk.Data.Clone(),
                TailBytes = Array.Empty<byte>()
            };
            return true;
        }

        targetSubchunks.Add(new McnkSubChunk
        {
            RawSignature = (byte[])sourceSubchunk.RawSignature.Clone(),
            Signature = signature,
            DeclaredSize = sourceSubchunk.DeclaredSize,
            Data = (byte[])sourceSubchunk.Data.Clone(),
            TailBytes = Array.Empty<byte>()
        });

        return true;
    }

    private static void PatchMcnkHeaderOffsets(
        byte[] header,
        Dictionary<string, int> offsetBySignature,
        Dictionary<string, int> sizeBySignature)
    {
        WriteUInt32(header, 0x14, GetOffset("MCVT", offsetBySignature));
        WriteUInt32(header, 0x18, GetOffset("MCNR", offsetBySignature));
        WriteUInt32(header, 0x1C, GetOffset("MCLY", offsetBySignature));
        WriteUInt32(header, 0x20, GetOffset("MCRF", offsetBySignature));
        WriteUInt32(header, 0x24, GetOffset("MCAL", offsetBySignature));
        WriteUInt32(header, 0x2C, GetOffset("MCSH", offsetBySignature));
        WriteUInt32(header, 0x58, GetOffset("MCSE", offsetBySignature));
        WriteUInt32(header, 0x60, GetOffset("MCLQ", offsetBySignature));

        uint mcalSize = sizeBySignature.TryGetValue("MCAL", out int mcalDataSize)
            ? (uint)(mcalDataSize + 8)
            : 0;
        uint mcshSize = sizeBySignature.TryGetValue("MCSH", out int mcshDataSize)
            ? (uint)(mcshDataSize + 8)
            : 0;
        uint mclqSize = sizeBySignature.TryGetValue("MCLQ", out int mclqDataSize)
            ? (uint)(mclqDataSize + 8)
            : 0;

        WriteUInt32(header, 0x28, mcalSize);
        WriteUInt32(header, 0x30, mcshSize);
        WriteUInt32(header, 0x64, mclqSize);
    }

    private static uint GetOffset(string signature, Dictionary<string, int> offsets)
        => offsets.TryGetValue(signature, out int offset) ? (uint)offset : 0;

    private static bool TryCopyMtex(List<TopLevelChunk> sourceChunks, List<TopLevelChunk> targetChunks)
    {
        int sourceMtexIndex = sourceChunks.FindIndex(chunk => string.Equals(chunk.Signature, "MTEX", StringComparison.Ordinal));
        if (sourceMtexIndex < 0)
            return false;

        TopLevelChunk sourceMtex = sourceChunks[sourceMtexIndex];
        int targetMtexIndex = targetChunks.FindIndex(chunk => string.Equals(chunk.Signature, "MTEX", StringComparison.Ordinal));
        if (targetMtexIndex >= 0)
        {
            targetChunks[targetMtexIndex].Data = (byte[])sourceMtex.Data.Clone();
            return true;
        }

        int insertIndex = targetChunks.FindIndex(chunk => string.Equals(chunk.Signature, "MCNK", StringComparison.Ordinal));
        if (insertIndex < 0)
            insertIndex = targetChunks.Count;

        targetChunks.Insert(insertIndex, new TopLevelChunk
        {
            RawSignature = (byte[])sourceMtex.RawSignature.Clone(),
            Signature = "MTEX",
            Data = (byte[])sourceMtex.Data.Clone(),
            Offset = -1
        });

        return true;
    }

    private static Dictionary<int, int> BuildMcnkIndexMap(List<TopLevelChunk> chunks)
    {
        var map = new Dictionary<int, int>();
        int mcinIndex = chunks.FindIndex(chunk => string.Equals(chunk.Signature, "MCIN", StringComparison.Ordinal));

        if (mcinIndex >= 0)
        {
            TopLevelChunk mcinChunk = chunks[mcinIndex];
            if (mcinChunk.Data.Length >= 256 * 16)
            {
                var chunkOffsetToListIndex = new Dictionary<int, int>();
                for (int i = 0; i < chunks.Count; i++)
                {
                    TopLevelChunk chunk = chunks[i];
                    if (!string.Equals(chunk.Signature, "MCNK", StringComparison.Ordinal))
                        continue;

                    if (chunk.Offset >= 0)
                        chunkOffsetToListIndex[chunk.Offset] = i;
                }

                for (int i = 0; i < 256; i++)
                {
                    int entryOffset = i * 16;
                    int chunkOffset = BitConverter.ToInt32(mcinChunk.Data, entryOffset);
                    if (chunkOffset <= 0)
                        continue;

                    if (chunkOffsetToListIndex.TryGetValue(chunkOffset, out int listIndex))
                        map[i] = listIndex;
                }
            }
        }

        if (map.Count > 0)
            return map;

        int ordinal = 0;
        for (int i = 0; i < chunks.Count; i++)
        {
            if (!string.Equals(chunks[i].Signature, "MCNK", StringComparison.Ordinal))
                continue;

            map[ordinal] = i;
            ordinal++;
        }

        return map;
    }

    private static List<McnkSubChunk> ParseMcnkSubchunks(byte[] payload)
    {
        var chunks = new List<McnkSubChunk>();
        if (payload.Length < 8)
            return chunks;

        bool headerless = payload.Length >= 8
            && KnownMcnkSubFourCc.Contains(NormalizeFourCc(payload[0..4], KnownMcnkSubFourCc));

        if (!headerless && payload.Length < 128)
            return chunks;

        uint headerMcalSize = (!headerless && payload.Length >= 0x2C) ? BitConverter.ToUInt32(payload, 0x28) : 0;
        uint headerMcshSize = (!headerless && payload.Length >= 0x34) ? BitConverter.ToUInt32(payload, 0x30) : 0;

        int pos = headerless ? 0 : 0x80;
        while (pos + 8 <= payload.Length)
        {
            byte[] rawSignature = payload[pos..(pos + 4)];
            string signature = NormalizeFourCc(rawSignature, KnownMcnkSubFourCc);
            int declaredSize = BitConverter.ToInt32(payload, pos + 4);
            if (declaredSize < 0 || pos + 8 + declaredSize > payload.Length)
                break;

            int consumedSize = declaredSize;
            if (string.Equals(signature, "MCNR", StringComparison.Ordinal))
            {
                consumedSize = Math.Max(consumedSize, 0x1C0);
            }
            else if (string.Equals(signature, "MCAL", StringComparison.Ordinal) && headerMcalSize >= 8)
            {
                consumedSize = Math.Max(consumedSize, (int)headerMcalSize - 8);
            }
            else if (string.Equals(signature, "MCSH", StringComparison.Ordinal) && headerMcshSize >= 8)
            {
                consumedSize = Math.Max(consumedSize, (int)headerMcshSize - 8);
            }

            if (pos + 8 + consumedSize > payload.Length)
                break;

            byte[] data = payload[(pos + 8)..(pos + 8 + declaredSize)];
            byte[] tail = consumedSize > declaredSize
                ? payload[(pos + 8 + declaredSize)..(pos + 8 + consumedSize)]
                : Array.Empty<byte>();

            chunks.Add(new McnkSubChunk
            {
                RawSignature = rawSignature,
                Signature = signature,
                DeclaredSize = declaredSize,
                Data = data,
                TailBytes = tail
            });

            pos += 8 + consumedSize;
        }

        return chunks;
    }

    private static List<TopLevelChunk> ParseTopLevelChunks(byte[] bytes)
    {
        var chunks = new List<TopLevelChunk>();
        int pos = 0;

        while (pos + 8 <= bytes.Length)
        {
            byte[] rawSignature = bytes[pos..(pos + 4)];
            string signature = NormalizeFourCc(rawSignature, KnownTopLevelFourCc);
            int size = BitConverter.ToInt32(bytes, pos + 4);
            if (size < 0 || pos + 8 + size > bytes.Length)
                break;

            byte[] data = bytes[(pos + 8)..(pos + 8 + size)];
            chunks.Add(new TopLevelChunk
            {
                RawSignature = rawSignature,
                Signature = signature,
                Data = data,
                Offset = pos
            });

            int nextPos = pos + 8 + size;

            if (nextPos + 8 <= bytes.Length)
            {
                int nextSize = BitConverter.ToInt32(bytes, nextPos + 4);
                bool directLooksValid = nextSize >= 0 && nextPos + 8 + nextSize <= bytes.Length;

                if (!directLooksValid && nextPos + 1 + 8 <= bytes.Length)
                {
                    int alignedPos = nextPos + 1;
                    int alignedSize = BitConverter.ToInt32(bytes, alignedPos + 4);
                    bool alignedLooksValid = alignedSize >= 0 && alignedPos + 8 + alignedSize <= bytes.Length;
                    if (alignedLooksValid)
                        nextPos = alignedPos;
                }
            }

            if (nextPos <= pos)
                break;
            pos = nextPos;
        }

        return chunks;
    }

    private static byte[] BuildTopLevelBytes(List<TopLevelChunk> chunks)
    {
        using var stream = new MemoryStream();

        foreach (TopLevelChunk chunk in chunks)
        {
            stream.Write(chunk.RawSignature, 0, 4);
            stream.Write(BitConverter.GetBytes(chunk.Data.Length), 0, 4);
            if (chunk.Data.Length > 0)
                stream.Write(chunk.Data, 0, chunk.Data.Length);
        }

        return stream.ToArray();
    }

    private static string NormalizeFourCc(byte[] rawSignature, HashSet<string> knownSet)
    {
        string direct = Encoding.ASCII.GetString(rawSignature);
        if (knownSet.Contains(direct))
            return direct;

        byte[] reversed = [rawSignature[3], rawSignature[2], rawSignature[1], rawSignature[0]];
        string reversedText = Encoding.ASCII.GetString(reversed);
        return knownSet.Contains(reversedText) ? reversedText : direct;
    }

    private static void WriteUInt32(byte[] header, int offset, uint value)
    {
        byte[] bytes = BitConverter.GetBytes(value);
        Buffer.BlockCopy(bytes, 0, header, offset, 4);
    }

    private static string InferMapName(string mapDirectory)
    {
        foreach (string path in Directory.EnumerateFiles(mapDirectory, "*.adt").OrderBy(Path.GetFileName))
        {
            string fileName = Path.GetFileName(path);
            Match match = RootTilePattern().Match(fileName);
            if (match.Success)
                return match.Groups["map"].Value;
        }

        return new DirectoryInfo(mapDirectory).Name;
    }

    private static List<TerrainTilePair> BuildPairList(
        TerrainTextureTransferOptions options,
        string sourceDirectory,
        string sourceMapName)
    {
        if (options.Pairs.Count > 0)
            return options.Pairs.Distinct().ToList();

        if (!options.GlobalDeltaX.HasValue || !options.GlobalDeltaY.HasValue)
            throw new ArgumentException("No tile mappings were provided. Use --pair or --global-delta.");

        int dx = options.GlobalDeltaX.Value;
        int dy = options.GlobalDeltaY.Value;

        var pairs = new List<TerrainTilePair>();
        foreach ((int x, int y) in EnumerateRootTiles(sourceDirectory, sourceMapName))
        {
            pairs.Add(new TerrainTilePair(
                SourceTileX: x,
                SourceTileY: y,
                TargetTileX: x + dx,
                TargetTileY: y + dy));
        }

        return pairs.Distinct().ToList();
    }

    private static IEnumerable<(int x, int y)> EnumerateRootTiles(string mapDirectory, string mapName)
    {
        string escaped = Regex.Escape(mapName);
        var regex = new Regex($"^{escaped}_(?<x>\\d+)_(?<y>\\d+)\\.adt$", RegexOptions.IgnoreCase | RegexOptions.Compiled);

        foreach (string path in Directory.EnumerateFiles(mapDirectory, "*.adt").OrderBy(Path.GetFileName))
        {
            string fileName = Path.GetFileName(path);
            Match match = regex.Match(fileName);
            if (!match.Success)
                continue;

            if (!int.TryParse(match.Groups["x"].Value, out int x)
                || !int.TryParse(match.Groups["y"].Value, out int y))
            {
                continue;
            }

            yield return (x, y);
        }
    }
}