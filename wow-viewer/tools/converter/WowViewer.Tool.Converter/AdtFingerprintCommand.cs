using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Tool.Converter;

/// <summary>
/// Lightweight ADT fingerprint command for pre-extraction deduplication.
/// Reads root ADT files, extracts MCLY texture IDs, MCVT height range,
/// and MCNK flags, then computes a SHA256 fingerprint for each tile.
/// </summary>
internal static class AdtFingerprintCommand
{
    private const int RootMcnkHeaderSize = 128;
    private const int RootMcnkSubchunkOffset = 0x80;
    private const int McvtSampleCount = 145;
    private const int MclyEntrySize = 16;
    private const int McnrConsumedSize = 0x1C0;

    public static void Run(string[] args)
    {
        string? inputDir = GetOption(args, "--input-dir", "-i");
        string? outputPath = GetOption(args, "--output", "-o");

        if (string.IsNullOrWhiteSpace(inputDir))
        {
            Console.Error.WriteLine("Error: adt-fingerprint requires --input-dir <dir>.");
            Environment.ExitCode = 1;
            return;
        }

        string resolvedInputDir = Path.GetFullPath(inputDir);
        if (!Directory.Exists(resolvedInputDir))
        {
            Console.Error.WriteLine($"Error: input directory '{resolvedInputDir}' not found.");
            Environment.ExitCode = 1;
            return;
        }

        // Discover root ADT files (not _tex0, _obj0, _lod)
        string[] adtFiles = Directory.GetFiles(resolvedInputDir, "*.adt")
            .Where(static path =>
            {
                string name = Path.GetFileNameWithoutExtension(path);
                return !name.EndsWith("_tex0") && !name.EndsWith("_obj0") && !name.EndsWith("_lod");
            })
            .OrderBy(static path => path)
            .ToArray();

        if (adtFiles.Length == 0)
        {
            Console.Error.WriteLine("Error: no root ADT files found in input directory.");
            Environment.ExitCode = 1;
            return;
        }

        Console.Error.WriteLine($"Found {adtFiles.Length} root ADT files in '{resolvedInputDir}'.");

        List<AdtFingerprintEntry> entries = [];
        int failed = 0;

        foreach (string adtPath in adtFiles)
        {
            try
            {
                AdtFingerprintEntry entry = ComputeFingerprint(adtPath);
                entries.Add(entry);
                Console.Error.WriteLine($"  Fingerprinted: {entry.TileName}  {entry.Fingerprint[..16]}...");
            }
            catch (Exception ex)
            {
                string tileName = Path.GetFileNameWithoutExtension(adtPath);
                Console.Error.WriteLine($"  FAILED: {tileName} - {ex.Message}");
                failed++;
            }
        }

        // Build output
        var report = new AdtFingerprintReport
        {
            SchemaVersion = "1.0",
            CreatedAtUtc = DateTimeOffset.UtcNow,
            InputDirectory = resolvedInputDir,
            TotalFiles = adtFiles.Length,
            SuccessfulEntries = entries.Count,
            FailedCount = failed,
            Entries = entries,
        };

        string json = JsonSerializer.Serialize(report, new JsonSerializerOptions
        {
            WriteIndented = true,
        });

        if (!string.IsNullOrWhiteSpace(outputPath))
        {
            string resolvedOutputPath = Path.GetFullPath(outputPath);
            string? outputDir = Path.GetDirectoryName(resolvedOutputPath);
            if (!string.IsNullOrWhiteSpace(outputDir))
                Directory.CreateDirectory(outputDir);
            File.WriteAllText(resolvedOutputPath, json);
            Console.Error.WriteLine($"Wrote fingerprint report to '{resolvedOutputPath}'.");
        }
        else
        {
            Console.Out.WriteLine(json);
        }

        Console.Error.WriteLine($"AdtFingerprintCommand complete: {entries.Count} succeeded, {failed} failed.");
    }

    private static AdtFingerprintEntry ComputeFingerprint(string adtPath)
    {
        string tileName = Path.GetFileNameWithoutExtension(adtPath);
        long fileSize = new FileInfo(adtPath).Length;

        using FileStream stream = File.OpenRead(adtPath);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(adtPath));

        if (fileSummary.Kind != MapFileKind.Adt)
            throw new InvalidDataException($"Expected root ADT file, but found {fileSummary.Kind}.");

        // Resolve MCNK chunk locations
        List<MapChunkLocation> mcnkChunks = fileSummary.Chunks
            .Where(static chunk => chunk.Id == MapChunkIds.Mcnk)
            .ToList();

        if (mcnkChunks.Count == 0)
            throw new InvalidDataException("No MCNK chunks found in ADT file.");

        // Extract features from each MCNK chunk
        var chunkFeatures = new List<McnkFingerprintFeatures>(mcnkChunks.Count);
        var allTextureIds = new HashSet<uint>();
        float globalMinHeight = float.MaxValue;
        float globalMaxHeight = float.MinValue;
        int totalHoleChunks = 0;
        int totalLiquidFlagChunks = 0;

        foreach (MapChunkLocation mcnkChunk in mcnkChunks)
        {
            byte[] payload = ReadChunkPayload(stream, mcnkChunk);
            if (payload.Length < RootMcnkHeaderSize)
                continue;

            // Read MCNK header fields
            uint flags = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x00, 4));
            uint indexX = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x04, 4));
            uint indexY = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x08, 4));
            int layerCountFromHeader = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x0C, 4)));
            float baseHeight = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x14, 4)));
            ushort holes = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(0x3C, 2));

            if (holes != 0)
                totalHoleChunks++;

            if ((flags & 0x3Cu) != 0)
                totalLiquidFlagChunks++;

            // Scan subchunks for MCLY and MCVT
            var chunkTextureIds = new List<uint>();
            float chunkMinHeight = float.MaxValue;
            float chunkMaxHeight = float.MinValue;
            bool hasMcvt = false;

            int position = RootMcnkSubchunkOffset;
            while (position <= payload.Length - ChunkHeader.SizeInBytes)
            {
                if (!ChunkHeaderReader.TryRead(payload.AsSpan(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                    break;

                int declaredSize = checked((int)header.Size);
                int consumedSize = header.Id == AdtChunkIds.Mcnr
                    ? Math.Max(declaredSize, McnrConsumedSize)
                    : declaredSize;

                long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
                if (nextOffset > payload.Length)
                    break;

                int dataOffset = position + ChunkHeader.SizeInBytes;

                if (header.Id == AdtChunkIds.Mcly)
                {
                    // Read MCLY texture layer entries
                    int layerCount = declaredSize / MclyEntrySize;
                    for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
                    {
                        int entryOffset = dataOffset + (layerIndex * MclyEntrySize);
                        if (entryOffset + MclyEntrySize > payload.Length)
                            break;

                        uint textureId = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(entryOffset, 4));
                        chunkTextureIds.Add(textureId);
                        allTextureIds.Add(textureId);
                    }
                }
                else if (header.Id == AdtChunkIds.Mcvt)
                {
                    // Read MCVT height samples
                    hasMcvt = true;
                    int sampleCount = Math.Min(declaredSize / sizeof(float), McvtSampleCount);
                    for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
                    {
                        int sampleOffset = dataOffset + (sampleIndex * sizeof(float));
                        if (sampleOffset + sizeof(float) > payload.Length)
                            break;

                        float rawHeight = BitConverter.ToSingle(payload, sampleOffset);
                        float absoluteHeight = rawHeight + baseHeight;

                        if (absoluteHeight < chunkMinHeight)
                            chunkMinHeight = absoluteHeight;
                        if (absoluteHeight > chunkMaxHeight)
                            chunkMaxHeight = absoluteHeight;
                    }
                }

                position = checked((int)nextOffset);
            }

            if (chunkMinHeight < globalMinHeight)
                globalMinHeight = chunkMinHeight;
            if (chunkMaxHeight > globalMaxHeight)
                globalMaxHeight = chunkMaxHeight;

            chunkFeatures.Add(new McnkFingerprintFeatures
            {
                IndexX = checked((int)indexX),
                IndexY = checked((int)indexY),
                LayerCount = layerCountFromHeader,
                TextureIds = [..chunkTextureIds],
                HasMcvt = hasMcvt,
                ChunkMinHeight = chunkMinHeight,
                ChunkMaxHeight = chunkMaxHeight,
                Holes = holes,
                HasLiquidFlag = (flags & 0x3Cu) != 0,
            });
        }

        // Compute SHA256 fingerprint from concatenated features
        byte[] fingerprintBytes = ComputeFingerprintBytes(chunkFeatures, allTextureIds, globalMinHeight, globalMaxHeight);
        string fingerprint = Convert.ToHexStringLower(SHA256.HashData(fingerprintBytes));

        // Compute summary stats
        int totalLayers = chunkFeatures.Sum(static f => f.LayerCount);
        int maxLayers = chunkFeatures.Count > 0 ? chunkFeatures.Max(static f => f.LayerCount) : 0;
        int chunksWithMcvt = chunkFeatures.Count(static f => f.HasMcvt);

        return new AdtFingerprintEntry
        {
            TileName = tileName,
            SourcePath = Path.GetFullPath(adtPath),
            FileSize = fileSize,
            Fingerprint = fingerprint,
            McnkCount = mcnkChunks.Count,
            TotalLayerCount = totalLayers,
            MaxLayerCount = maxLayers,
            ChunksWithMcvt = chunksWithMcvt,
            ChunksWithHoles = totalHoleChunks,
            ChunksWithLiquidFlags = totalLiquidFlagChunks,
            UniqueTextureIds = [..allTextureIds.Order()],
            GlobalMinHeight = globalMinHeight,
            GlobalMaxHeight = globalMaxHeight,
        };
    }

    private static byte[] ComputeFingerprintBytes(
        List<McnkFingerprintFeatures> chunks,
        HashSet<uint> allTextureIds,
        float globalMinHeight,
        float globalMaxHeight)
    {
        // Build a deterministic byte sequence for hashing
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: false);

        // Write sorted unique texture IDs
        writer.Write(allTextureIds.Count);
        foreach (uint id in allTextureIds.Order())
            writer.Write(id);

        // Write global height range
        writer.Write(globalMinHeight);
        writer.Write(globalMaxHeight);

        // Write per-chunk features in order
        writer.Write(chunks.Count);
        foreach (McnkFingerprintFeatures chunk in chunks.OrderBy(static c => c.IndexY).ThenBy(static c => c.IndexX))
        {
            writer.Write(chunk.IndexX);
            writer.Write(chunk.IndexY);
            writer.Write(chunk.LayerCount);
            writer.Write(chunk.Holes);
            writer.Write(chunk.HasLiquidFlag);
            writer.Write(chunk.HasMcvt);
            writer.Write(chunk.ChunkMinHeight);
            writer.Write(chunk.ChunkMaxHeight);

            // Write texture IDs for this chunk
            writer.Write(chunk.TextureIds.Count);
            foreach (uint texId in chunk.TextureIds)
                writer.Write(texId);
        }

        return ms.ToArray();
    }

    private static byte[] ReadChunkPayload(Stream stream, MapChunkLocation chunk)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = chunk.DataOffset;
            byte[] payload = new byte[chunk.Size];
            stream.ReadExactly(payload);
            return payload;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static string? GetOption(string[] args, string longName, string shortName)
    {
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == longName || args[i] == shortName)
            {
                if (i + 1 < args.Length)
                    return args[i + 1];
                return null;
            }
        }
        return null;
    }

    private sealed class McnkFingerprintFeatures
    {
        public int IndexX { get; set; }
        public int IndexY { get; set; }
        public int LayerCount { get; set; }
        public List<uint> TextureIds { get; set; } = [];
        public bool HasMcvt { get; set; }
        public float ChunkMinHeight { get; set; } = float.MaxValue;
        public float ChunkMaxHeight { get; set; } = float.MinValue;
        public ushort Holes { get; set; }
        public bool HasLiquidFlag { get; set; }
    }
}

internal sealed record AdtFingerprintReport
{
    public string SchemaVersion { get; init; } = string.Empty;
    public DateTimeOffset CreatedAtUtc { get; init; }
    public string InputDirectory { get; init; } = string.Empty;
    public int TotalFiles { get; init; }
    public int SuccessfulEntries { get; init; }
    public int FailedCount { get; init; }
    public List<AdtFingerprintEntry> Entries { get; init; } = [];
}

internal sealed record AdtFingerprintEntry
{
    public string TileName { get; init; } = string.Empty;
    public string SourcePath { get; init; } = string.Empty;
    public long FileSize { get; init; }
    public string Fingerprint { get; init; } = string.Empty;
    public int McnkCount { get; init; }
    public int TotalLayerCount { get; init; }
    public int MaxLayerCount { get; init; }
    public int ChunksWithMcvt { get; init; }
    public int ChunksWithHoles { get; init; }
    public int ChunksWithLiquidFlags { get; init; }
    public List<uint> UniqueTextureIds { get; init; } = [];
    public float GlobalMinHeight { get; init; }
    public float GlobalMaxHeight { get; init; }
}
