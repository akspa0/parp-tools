using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtRawChunkBlobCollector
{
    private const int RootMcnkHeaderSize = 128;
    private const int RootMcnkSubchunkOffset = 0x80;

    private static readonly HashSet<FourCC> StructuralTopLevelChunkIds =
    [
        MapChunkIds.Mver,
        MapChunkIds.Mhdr,
        MapChunkIds.Mcin,
        MapChunkIds.Mcnk,
    ];

    private static readonly HashSet<FourCC> AlwaysDecodedRootTopLevelChunkIds =
    [
        MapChunkIds.Mfbo,
        MapChunkIds.Mh2o,
    ];

    private static readonly HashSet<FourCC> AlwaysDecodedRootMcnkChunkIds =
    [
        AdtChunkIds.Mcvt,
        AdtChunkIds.Mcnr,
        AdtChunkIds.Mccv,
        AdtChunkIds.Mclv,
        AdtChunkIds.Mclq,
        AdtChunkIds.Mcrf,
    ];

    private static readonly HashSet<FourCC> TextureDecodedTopLevelChunkIds =
    [
        MapChunkIds.Mamp,
        MapChunkIds.Mtex,
        MapChunkIds.Mtxf,
    ];

    private static readonly HashSet<FourCC> TextureDecodedMcnkChunkIds =
    [
        AdtChunkIds.Mcly,
        AdtChunkIds.Mcal,
        AdtChunkIds.Mcmt,
        AdtChunkIds.Mcsh,
    ];

    private static readonly HashSet<FourCC> PlacementDecodedTopLevelChunkIds =
    [
        MapChunkIds.Mmdx,
        MapChunkIds.Mmid,
        MapChunkIds.Mwmo,
        MapChunkIds.Mwid,
        MapChunkIds.Mddf,
        MapChunkIds.Modf,
    ];

    private static readonly HashSet<FourCC> PlacementDecodedMcnkChunkIds =
    [
        AdtChunkIds.Mcrd,
        AdtChunkIds.Mcrw,
    ];

    public static IReadOnlyList<TerrainRawChunkBlob> Collect(string adtPath, string? textureSourcePath = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(adtPath);

        AdtTileFamily family = AdtTileFamilyResolver.Resolve(adtPath);
        string? effectiveTexturePath = ResolveTextureSourcePath(family, textureSourcePath);
        string? effectivePlacementPath = family.PlacementSourcePath;

        Dictionary<string, RawChunkSourceContext> contexts = new(StringComparer.OrdinalIgnoreCase);

        if (family.HasRoot)
        {
            RawChunkSourceContext rootContext = GetOrAddContext(contexts, family.RootPath, "root");
            rootContext.ProcessedTopLevelChunkIds.UnionWith(StructuralTopLevelChunkIds);
            rootContext.ProcessedTopLevelChunkIds.UnionWith(AlwaysDecodedRootTopLevelChunkIds);
            rootContext.ProcessedMcnkChunkIds.UnionWith(AlwaysDecodedRootMcnkChunkIds);
        }

        if (!string.IsNullOrWhiteSpace(effectiveTexturePath))
        {
            RawChunkSourceContext textureContext = GetOrAddContext(contexts, effectiveTexturePath, ClassifySourceKind(effectiveTexturePath, family));
            textureContext.ProcessedTopLevelChunkIds.Add(MapChunkIds.Mver);
            textureContext.ProcessedTopLevelChunkIds.Add(MapChunkIds.Mcnk);
            textureContext.ProcessedTopLevelChunkIds.UnionWith(TextureDecodedTopLevelChunkIds);
            textureContext.ProcessedMcnkChunkIds.UnionWith(TextureDecodedMcnkChunkIds);
        }

        if (!string.IsNullOrWhiteSpace(effectivePlacementPath))
        {
            RawChunkSourceContext placementContext = GetOrAddContext(contexts, effectivePlacementPath, ClassifySourceKind(effectivePlacementPath, family));
            placementContext.ProcessedTopLevelChunkIds.Add(MapChunkIds.Mver);
            placementContext.ProcessedTopLevelChunkIds.Add(MapChunkIds.Mcnk);
            placementContext.ProcessedTopLevelChunkIds.UnionWith(PlacementDecodedTopLevelChunkIds);
            placementContext.ProcessedMcnkChunkIds.UnionWith(PlacementDecodedMcnkChunkIds);
        }

        List<TerrainRawChunkBlob> rawChunks = [];
        foreach (RawChunkSourceContext context in contexts.Values.OrderBy(static context => context.SourcePath, StringComparer.OrdinalIgnoreCase))
            CollectFromContext(context, rawChunks);

        return rawChunks;
    }

    private static RawChunkSourceContext GetOrAddContext(Dictionary<string, RawChunkSourceContext> contexts, string path, string sourceKind)
    {
        string fullPath = Path.GetFullPath(path);
        if (contexts.TryGetValue(fullPath, out RawChunkSourceContext? existing))
            return existing;

        RawChunkSourceContext created = new(fullPath, sourceKind);
        contexts.Add(fullPath, created);
        return created;
    }

    private static string? ResolveTextureSourcePath(AdtTileFamily family, string? textureSourcePath)
    {
        if (!string.IsNullOrWhiteSpace(textureSourcePath))
            return Path.GetFullPath(textureSourcePath);

        return family.TextureSourcePath;
    }

    private static string ClassifySourceKind(string path, AdtTileFamily family)
    {
        if (path.Equals(family.Tex0Path, StringComparison.OrdinalIgnoreCase))
            return "tex0";

        if (path.Equals(family.Obj0Path, StringComparison.OrdinalIgnoreCase))
            return "obj0";

        if (path.Equals(family.LodPath, StringComparison.OrdinalIgnoreCase))
            return "lod";

        return "root";
    }

    private static void CollectFromContext(RawChunkSourceContext context, List<TerrainRawChunkBlob> rawChunks)
    {
        if (!File.Exists(context.SourcePath))
            return;

        using FileStream stream = File.OpenRead(context.SourcePath);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, context.SourcePath);

        Dictionary<string, int> topLevelCounts = new(StringComparer.OrdinalIgnoreCase);
        foreach (MapChunkLocation chunk in fileSummary.Chunks)
        {
            if (chunk.Id == MapChunkIds.Mcnk)
                continue;

            if (context.ProcessedTopLevelChunkIds.Contains(chunk.Id))
                continue;

            byte[] payload = MapSummaryReaderCommon.ReadChunkPayload(stream, chunk);
            if (payload.Length == 0)
                continue;

            string chunkId = chunk.Id.ToString();
            int occurrence = topLevelCounts.TryGetValue(chunkId, out int count) ? count : 0;
            topLevelCounts[chunkId] = occurrence + 1;

            rawChunks.Add(new TerrainRawChunkBlob
            {
                EntryName = $"raw_chunks/{context.SourceKind}/top/{chunkId}_{occurrence:D3}",
                SourceKind = context.SourceKind,
                SourcePath = context.SourcePath,
                Scope = "top-level",
                ChunkId = chunkId,
                Data = payload,
            });
        }

        List<MapChunkLocation> mcnkChunks = fileSummary.Chunks.Where(static chunk => chunk.Id == MapChunkIds.Mcnk).ToList();
        for (int mcnkIndex = 0; mcnkIndex < mcnkChunks.Count; mcnkIndex++)
        {
            MapChunkLocation mcnkChunk = mcnkChunks[mcnkIndex];
            byte[] payload = MapSummaryReaderCommon.ReadChunkPayload(stream, mcnkChunk);
            if (payload.Length == 0)
                continue;

            int? chunkX = null;
            int? chunkY = null;
            int scanOffset = 0;

            if (fileSummary.Kind == MapFileKind.Adt && payload.Length >= RootMcnkHeaderSize)
            {
                chunkX = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x04, 4));
                chunkY = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x08, 4));
                scanOffset = RootMcnkSubchunkOffset;
            }
            else if (mcnkIndex < 256)
            {
                chunkX = mcnkIndex % 16;
                chunkY = mcnkIndex / 16;
            }

            CollectRawMcnkSubchunks(context, payload, mcnkIndex, chunkX, chunkY, scanOffset, rawChunks);
        }
    }

    private static bool IsValidAdtFourCC(FourCC id)
    {
        string s = id.ToString();
        foreach (char c in s)
        {
            if (c < 0x20 || c > 0x7E)
                return false;
        }
        return true;
    }

    private static void CollectRawMcnkSubchunks(
        RawChunkSourceContext context,
        byte[] payload,
        int mcnkIndex,
        int? chunkX,
        int? chunkY,
        int scanOffset,
        List<TerrainRawChunkBlob> rawChunks)
    {
        Dictionary<string, int> subchunkCounts = new(StringComparer.OrdinalIgnoreCase);

        uint headerMcalSize = payload.Length >= 0x2C ? BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x28, 4)) : 0;
        uint headerMcshSize = payload.Length >= 0x34 ? BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x30, 4)) : 0;

        int position = scanOffset;

        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.AsSpan(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            if (!IsValidAdtFourCC(header.Id))
                break;

            long consumedSize = (long)header.Size;
            if (header.Id == AdtChunkIds.Mcal && headerMcalSize >= ChunkHeader.SizeInBytes)
                consumedSize = Math.Max(consumedSize, (long)headerMcalSize - ChunkHeader.SizeInBytes);
            else if (header.Id == AdtChunkIds.Mcsh && headerMcshSize >= ChunkHeader.SizeInBytes)
                consumedSize = Math.Max(consumedSize, (long)headerMcshSize - ChunkHeader.SizeInBytes);

            long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
            if (nextOffset > payload.Length)
                break;

            long payloadSizeLong = consumedSize;
            if (!context.ProcessedMcnkChunkIds.Contains(header.Id) && payloadSizeLong > 0)
            {
                int chunkSize = (int)Math.Min(payloadSizeLong, int.MaxValue);
                string chunkId = header.Id.ToString();
                int occurrence = subchunkCounts.TryGetValue(chunkId, out int count) ? count : 0;
                subchunkCounts[chunkId] = occurrence + 1;

                byte[] chunkPayload = payload.AsSpan(position + ChunkHeader.SizeInBytes, chunkSize).ToArray();
                string chunkLabel = chunkX.HasValue && chunkY.HasValue
                    ? $"mcnk_{chunkX.Value:D2}_{chunkY.Value:D2}"
                    : $"mcnk_{mcnkIndex:D3}";

                rawChunks.Add(new TerrainRawChunkBlob
                {
                    EntryName = $"raw_chunks/{context.SourceKind}/{chunkLabel}/{chunkId}_{occurrence:D3}",
                    SourceKind = context.SourceKind,
                    SourcePath = context.SourcePath,
                    Scope = "mcnk-subchunk",
                    ChunkId = chunkId,
                    ChunkIndex = mcnkIndex,
                    ChunkX = chunkX,
                    ChunkY = chunkY,
                    Data = chunkPayload,
                });
            }

            position = (int)nextOffset;
        }
    }

    private sealed class RawChunkSourceContext
    {
        public RawChunkSourceContext(string sourcePath, string sourceKind)
        {
            SourcePath = sourcePath;
            SourceKind = sourceKind;
        }

        public string SourcePath { get; }

        public string SourceKind { get; }

        public HashSet<FourCC> ProcessedTopLevelChunkIds { get; } = [];

        public HashSet<FourCC> ProcessedMcnkChunkIds { get; } = [];
    }
}