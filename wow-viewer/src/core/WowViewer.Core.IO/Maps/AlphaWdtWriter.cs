using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AlphaWdtWriter
{
    private const int ChunkHeaderSize = 8;
    private const int MphdAlphaSize = 128;
    private const int McnkHeaderSize = 128;
    private const int McinEntryCount = 256;
    private const int McinEntrySize = 16;
    private const int MainEntrySize = 16;
    private const int TilesPerAxis = 64;
    private const int TileSize = 257;
    private const int AlphaClientMaxMcnkPayloadSize = 15000;
    private const int MddfEntrySize = 36;
    private const int ModfEntrySize = 64;
    private const int MclyEntrySize = 16;
    private const int AlphaMcvtSize = 580;
    private const int AlphaMcnrSize = 448;
    private const int AlphaTileAlphaSize = 256;
    private const int AlphaLegacyTileAlphaSize = 1024;
    private const int AlphaChunkAlphaSize = 64;
    private const int AlphaMclqTileFlagsOffset = 0x290;
    private const int AlphaMclqFlowCountOffset = 0x2D0;
    private const int AlphaMclqPayloadSize = 0x324;
    private const float MapOrigin = 17066.666f;
    private const float TileWorldSize = 533.33333f;
    private const float ChunkWorldSize = TileWorldSize / 16f;

    public static byte[] Build(string mapName, Dictionary<(int tileX, int tileY), AlphaTileData> tiles)
    {
        ArgumentNullException.ThrowIfNull(tiles);

        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        WriteChunk(bw, "MVER", 4, w => w.Write(18));

        long mphdPosition = ms.Position;
        WriteChunk(bw, "MPHD", MphdAlphaSize, static w => w.Write(new byte[MphdAlphaSize]));

        long mainPosition = ms.Position;
        byte[] mainData = BuildMainPayload(tiles);
        WriteChunk(bw, "MAIN", mainData.Length, w => w.Write(mainData));

        IReadOnlyList<string> allMdxNames = CollectMdxNames(tiles);
        IReadOnlyList<string> allWmoNames = CollectWmoNames(tiles);

        long mdnmStart = ms.Position;
        byte[] mdnmData = BuildStringTable(allMdxNames);
        WriteDataChunk(bw, "MDNM", mdnmData);

        long monmStart = ms.Position;
        byte[] monmData = BuildStringTable(allWmoNames);
        WriteDataChunk(bw, "MONM", monmData);

        PatchMphd(ms, mphdPosition, allMdxNames, mdnmStart, allWmoNames, monmStart);

        var mdxNameIndex = BuildNameIndex(allMdxNames);
        var wmoNameIndex = BuildNameIndex(allWmoNames);

        foreach (var kvp in tiles.OrderBy(t => t.Key.Item2 * TilesPerAxis + t.Key.Item1))
        {
            var (tileX, tileY) = kvp.Key;
            var tile = kvp.Value;

            int tileOffset = (int)ms.Position;
            int tileHeaderSize = WriteTileData(bw, tile, tileX, tileY, allMdxNames, allWmoNames, mdxNameIndex, wmoNameIndex);
            PatchMainEntry(mainData, tileY * TilesPerAxis + tileX, tileOffset, tileHeaderSize);
        }

        PatchMainPayload(ms, mainPosition, mainData);

        bw.Flush();
        return ms.ToArray();
    }

    private static int WriteTileData(BinaryWriter bw, AlphaTileData tile, int tileX, int tileY,
        IReadOnlyList<string> mdxNames, IReadOnlyList<string> wmoNames,
        Dictionary<string, int> mdxIndex, Dictionary<string, int> wmoIndex)
    {
        AlphaPlacementWritePlan placementPlan = BuildPlacementWritePlan(tile, tileX, tileY, mdxIndex, wmoIndex);

        var mcnkDataList = new List<byte[]>(256);
        for (int cy = 0; cy < 16; cy++)
        {
            for (int cx = 0; cx < 16; cx++)
            {
                int chunkIndex = cy * 16 + cx;
                mcnkDataList.Add(BuildMcnkData(tile, cx, cy, placementPlan.RefsByChunk[chunkIndex]));
            }
        }

        byte[] mtexData = BuildStringTable(tile.TextureNames);

        long mhdrStart = bw.Seek(0, SeekOrigin.Current);
        WriteChunk(bw, "MHDR", 64, static w => w.Write(new byte[64]));

        byte[] mcinData = new byte[McinEntryCount * McinEntrySize];

        long mcinStart = bw.Seek(0, SeekOrigin.Current);
        WriteChunk(bw, "MCIN", mcinData.Length, w => w.Write(mcinData));

        long afterMcin = bw.Seek(0, SeekOrigin.Current);
        long mhdrDataStart = mhdrStart + ChunkHeaderSize;

        int mtexRelative = (int)(bw.Seek(0, SeekOrigin.Current) - mhdrDataStart);
        WriteDataChunk(bw, "MTEX", mtexData);

        int mddfRelative = (int)(bw.Seek(0, SeekOrigin.Current) - mhdrDataStart);
        WriteDataChunk(bw, "MDDF", placementPlan.MddfData);

        int modfRelative = (int)(bw.Seek(0, SeekOrigin.Current) - mhdrDataStart);
        WriteDataChunk(bw, "MODF", placementPlan.ModfData);

        int tileHeaderSize = checked((int)(bw.Seek(0, SeekOrigin.Current) - mhdrStart));

        int[] mcnkOffsets = new int[McinEntryCount];
        for (int i = 0; i < McinEntryCount; i++)
        {
            if (mcnkDataList[i].Length > 0)
            {
                mcnkOffsets[i] = (int)bw.Seek(0, SeekOrigin.Current);
                bw.Write(mcnkDataList[i]);
            }
            else
            {
                mcnkOffsets[i] = 0;
            }
        }

        WriteMcinOffsets(bw, mcnkOffsets, mcnkDataList, mcinStart);
        WriteMhdrData(bw, mhdrStart, mtexRelative, mddfRelative, modfRelative);

        return tileHeaderSize;
    }

    private static byte[] BuildMcnkData(AlphaTileData tile, int cx, int cy, AlphaPlacementRefs placementRefs)
    {
        float[] heights = ExtractChunkHeights(tile.Heightmap, cx, cy);
        float chunkBaseHeight = heights[0];

        byte[] mcvtAlpha = BuildAlphaMcvt(heights);
        byte[] mcnrAlpha = BuildAlphaMcnr(tile.McnrNormalXyz, cx, cy);

        byte[] mcshRaw = tile.McshShadowMask1024 != null ? SliceChunkShadowAlpha(tile.McshShadowMask1024, cx, cy) : [];
        AlphaLiquidChunk? liquidChunk = FindLiquidChunk(tile, cx, cy);

        int nLayers = 0;
        for (int l = 0; l < 4; l++)
        {
            if (cx < tile.MclyLayerMask.GetLength(0) && cy < tile.MclyLayerMask.GetLength(1) && tile.MclyLayerMask[cx, cy, l])
                nLayers = l + 1;
        }

        byte[] mclyRaw = BuildAlphaMcly(tile, cx, cy, nLayers);
        byte[] mcalRaw = BuildAlphaMcal(tile, cx, cy, nLayers);
        byte[] mclqRaw = BuildAlphaMclq(liquidChunk);

        byte[] mcrfRaw = BuildMcrfData(placementRefs);
        int nDoodadRefs = placementRefs.DoodadIndices.Count;
        int nMapObjRefs = placementRefs.MapObjIndices.Count;

        byte[] mclyWhole = WrapChunk("MCLY", mclyRaw);
        byte[] mcrfWhole = WrapChunk("MCRF", mcrfRaw);

        int cursor = 0;
        int offsHeight = cursor;
        cursor += mcvtAlpha.Length;
        int offsNormal = cursor;
        cursor += mcnrAlpha.Length;
        int offsLayer = cursor;
        cursor += mclyWhole.Length;
        int offsRefs = cursor;
        cursor += mcrfWhole.Length;
        int offsShadow = mcshRaw.Length > 0 ? cursor : 0;
        cursor += mcshRaw.Length;
        int offsAlpha = cursor;
        cursor += mcalRaw.Length;
        int offsLiquid = mclqRaw.Length > 0 ? cursor : 0;
        cursor += mclqRaw.Length;

        uint flags = liquidChunk is not null ? NormalizeAlphaLiquidFlags(liquidChunk.McnkFlags) : 0u;
        if (mcshRaw.Length > 0) flags |= 0x01;

        float radius = CalculateRadius(heights);

        int chunkDataSize = cursor;
        int totalDataSize = McnkHeaderSize + chunkDataSize;

        using var ms = new MemoryStream();
        using var msw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        msw.Write(FourCC.FromString("MCNK").ToFileBytes());
        msw.Write(totalDataSize);

        byte[] header = new byte[McnkHeaderSize];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x00), flags);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x04), cx);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x08), cy);
        BinaryPrimitives.WriteSingleLittleEndian(header.AsSpan(0x0C), radius);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x10), nLayers);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x14), nDoodadRefs);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x18), offsHeight);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x1C), offsNormal);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x20), offsLayer);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x24), offsRefs);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x28), offsAlpha);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x2C), mcalRaw.Length);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x30), mcshRaw.Length > 0 ? offsShadow : 0);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x34), mcshRaw.Length);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x38), 0);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x3C), nMapObjRefs);
        ushort holeMaskValue = 0;
        if (tile.HoleFullMasks != null && cx < tile.HoleFullMasks.GetLength(0) && cy < tile.HoleFullMasks.GetLength(1))
            holeMaskValue = tile.HoleFullMasks[cx, cy];
        else if (cx < tile.HoleMask.GetLength(0) && cy < tile.HoleMask.GetLength(1) && tile.HoleMask[cx, cy])
            holeMaskValue = 0xFFFF;
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x40), holeMaskValue);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x5C), chunkDataSize);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x64), offsLiquid);

        // MCNK offsets 0x68/0x6C: chunk Position.Z for rendering-precision relativization.
        // The client (CMapChunk::CreateVertices) reads these as Position, then subtracts
        // Position from each vertex to maintain floating-point precision at large world coords.
        // MCVT heights are stored as absolute world-space Z; Position.Z = heights[0].
        BinaryPrimitives.WriteSingleLittleEndian(header.AsSpan(0x68), chunkBaseHeight);
        BinaryPrimitives.WriteSingleLittleEndian(header.AsSpan(0x6C), chunkBaseHeight);
        msw.Write(header);

        msw.Write(mcvtAlpha);
        msw.Write(mcnrAlpha);
        msw.Write(mclyWhole);
        msw.Write(mcrfWhole);
        if (mcshRaw.Length > 0) msw.Write(mcshRaw);
        if (mcalRaw.Length > 0) msw.Write(mcalRaw);
        if (mclqRaw.Length > 0) msw.Write(mclqRaw);

        return ms.ToArray();
    }

    private static AlphaPlacementWritePlan BuildPlacementWritePlan(AlphaTileData tile, int tileX, int tileY,
        Dictionary<string, int> mdxIndex, Dictionary<string, int> wmoIndex)
    {
        byte[] mddfData = BuildMddfData(tile, mdxIndex);
        byte[] modfData = BuildModfData(tile, wmoIndex);
        Dictionary<int, int> doodadIndexByUniqueId = BuildDoodadIndexByUniqueId(tile.ModelPlacements);
        Dictionary<int, int> worldIndexByUniqueId = BuildWorldModelIndexByUniqueId(tile.WorldModelPlacements);

        AlphaPlacementRefs[] refsByChunk = BuildPlacementReferences(
            tile,
            tileX,
            tileY,
            doodadIndexByUniqueId,
            worldIndexByUniqueId);

        ApplyClientMcnkSizeBudget(tile, tileX, tileY, refsByChunk);

        return new AlphaPlacementWritePlan(mddfData, modfData, refsByChunk);
    }

    private static void ApplyClientMcnkSizeBudget(
        AlphaTileData tile,
        int tileX,
        int tileY,
        AlphaPlacementRefs[] refsByChunk)
    {
        int[] mapObjRefChunkCounts = BuildMapObjRefChunkCounts(refsByChunk, tile.WorldModelPlacements.Count);
        int[] mapObjAnchorChunks = BuildMapObjAnchorChunks(tile.WorldModelPlacements, tileX, tileY);
        int[] fixedPayloadSizesByChunk = BuildFixedPayloadSizesByChunk(tile);

        for (int chunkIndex = 0; chunkIndex < refsByChunk.Length; chunkIndex++)
        {
            TrimChunkPlacementRefsToBudget(
                tile,
                tileX,
                tileY,
                chunkIndex,
                refsByChunk,
                mapObjRefChunkCounts,
                mapObjAnchorChunks,
                fixedPayloadSizesByChunk);
        }
    }

    private static AlphaPlacementRefs[] BuildPlacementReferences(AlphaTileData tile, int tileX, int tileY,
        Dictionary<int, int> doodadIndexByUniqueId,
        Dictionary<int, int> worldIndexByUniqueId)
    {
        var refsByChunk = new AlphaPlacementRefs[256];
        for (int index = 0; index < refsByChunk.Length; index++)
            refsByChunk[index] = new AlphaPlacementRefs();

        IReadOnlyList<int>[]? preservedDoodadUniqueIdsByChunk = tile.McrfDoodadUniqueIdsByChunk;
        IReadOnlyList<int>[]? preservedWorldUniqueIdsByChunk = tile.McrfWorldModelUniqueIdsByChunk;
        IReadOnlyList<int>[]? preservedDoodadRefsByChunk = tile.McrfDoodadRefsByChunk;
        IReadOnlyList<int>[]? preservedWorldModelRefsByChunk = tile.McrfWorldModelRefsByChunk;
        bool usePreservedDoodadUniqueIds = HasAnyChunkRefs(preservedDoodadUniqueIdsByChunk);
        bool usePreservedDoodadRefs = HasAnyChunkRefs(preservedDoodadRefsByChunk);
        bool usePreservedWorldUniqueIds = HasAnyChunkRefs(preservedWorldUniqueIdsByChunk);
        bool usePreservedWorldModelRefs = HasAnyChunkRefs(preservedWorldModelRefsByChunk);

        if (usePreservedDoodadUniqueIds)
        {
            AddMappedUniqueIdRefs(refsByChunk, preservedDoodadUniqueIdsByChunk!, doodadIndexByUniqueId, static (placementRefs, placementIndex) => placementRefs.DoodadIndices.Add(placementIndex));
        }
        else if (usePreservedDoodadRefs)
        {
            for (int chunkIndex = 0; chunkIndex < refsByChunk.Length; chunkIndex++)
            {
                if (preservedDoodadRefsByChunk![chunkIndex] is not { Count: > 0 } refs)
                    continue;

                foreach (int placementIndex in refs)
                {
                    if ((uint)placementIndex < (uint)tile.ModelPlacements.Count)
                        refsByChunk[chunkIndex].DoodadIndices.Add(placementIndex);
                }
            }
        }
        else
        {
            for (int index = 0; index < tile.ModelPlacements.Count; index++)
            {
                int chunkIndex = FindContainingChunk(tile.ModelPlacements[index].Position, tileX, tileY);
                refsByChunk[chunkIndex].DoodadIndices.Add(index);
            }
        }

        if (usePreservedWorldUniqueIds)
        {
            AddMappedUniqueIdRefs(refsByChunk, preservedWorldUniqueIdsByChunk!, worldIndexByUniqueId, static (placementRefs, placementIndex) => placementRefs.MapObjIndices.Add(placementIndex));
        }
        else if (usePreservedWorldModelRefs)
        {
            for (int chunkIndex = 0; chunkIndex < refsByChunk.Length; chunkIndex++)
            {
                if (preservedWorldModelRefsByChunk![chunkIndex] is { Count: > 0 } refs)
                    refsByChunk[chunkIndex].MapObjIndices.AddRange(refs);
            }
        }
        else
        {
            for (int index = 0; index < tile.WorldModelPlacements.Count; index++)
            {
                AddMapObjReference(refsByChunk, index, tile.WorldModelPlacements[index], tileX, tileY);
            }
        }

        return refsByChunk;
    }

    private static void AddMappedUniqueIdRefs(AlphaPlacementRefs[] refsByChunk, IReadOnlyList<int>[] refsByUniqueIdChunk,
        Dictionary<int, int> placementIndexByUniqueId, Action<AlphaPlacementRefs, int> addPlacementIndex)
    {
        for (int chunkIndex = 0; chunkIndex < refsByChunk.Length; chunkIndex++)
        {
            if (refsByUniqueIdChunk[chunkIndex] is not { Count: > 0 } refs)
                continue;

            foreach (int uniqueId in refs)
            {
                if (placementIndexByUniqueId.TryGetValue(uniqueId, out int placementIndex))
                    addPlacementIndex(refsByChunk[chunkIndex], placementIndex);
            }
        }
    }

    private static Dictionary<int, int> BuildDoodadIndexByUniqueId(IReadOnlyList<AlphaModelPlacement> placements)
    {
        var indexByUniqueId = new Dictionary<int, int>();
        for (int index = 0; index < placements.Count; index++)
            indexByUniqueId.TryAdd(placements[index].UniqueId, index);
        return indexByUniqueId;
    }

    private static Dictionary<int, int> BuildWorldModelIndexByUniqueId(IReadOnlyList<AlphaWorldModelPlacement> placements)
    {
        var indexByUniqueId = new Dictionary<int, int>();
        for (int index = 0; index < placements.Count; index++)
            indexByUniqueId.TryAdd(placements[index].UniqueId, index);
        return indexByUniqueId;
    }

    private static int FindContainingChunk(Vector3 position, int tileX, int tileY)
    {
        float tileOriginX = MapOrigin - tileX * TileWorldSize;
        float tileOriginY = MapOrigin - tileY * TileWorldSize;

        int cy = Math.Clamp((int)MathF.Floor((tileOriginX - position.X) / ChunkWorldSize), 0, 15);
        int cx = Math.Clamp((int)MathF.Floor((tileOriginY - position.Y) / ChunkWorldSize), 0, 15);
        return cy * 16 + cx;
    }

    private static void AddMapObjReference(AlphaPlacementRefs[] refsByChunk, int placementIndex, AlphaWorldModelPlacement placement, int tileX, int tileY)
    {
        float minX = MathF.Min(placement.BoundsMin.X, placement.BoundsMax.X);
        float maxX = MathF.Max(placement.BoundsMin.X, placement.BoundsMax.X);
        float minY = MathF.Min(placement.BoundsMin.Y, placement.BoundsMax.Y);
        float maxY = MathF.Max(placement.BoundsMin.Y, placement.BoundsMax.Y);

        bool added = false;
        for (int cy = 0; cy < 16; cy++)
        {
            for (int cx = 0; cx < 16; cx++)
            {
                GetChunkPlanarBounds(tileX, tileY, cx, cy, out float chunkMinX, out float chunkMaxX, out float chunkMinY, out float chunkMaxY);
                if (maxX < chunkMinX || minX > chunkMaxX || maxY < chunkMinY || minY > chunkMaxY)
                    continue;

                refsByChunk[cy * 16 + cx].MapObjIndices.Add(placementIndex);
                added = true;
            }
        }

        if (!added)
        {
            int chunkIndex = FindContainingChunk(placement.Position, tileX, tileY);
            refsByChunk[chunkIndex].MapObjIndices.Add(placementIndex);
        }
    }

    private static int[] BuildMapObjRefChunkCounts(AlphaPlacementRefs[] refsByChunk, int placementCount)
    {
        if (placementCount <= 0)
            return [];

        int[] counts = new int[placementCount];
        for (int chunkIndex = 0; chunkIndex < refsByChunk.Length; chunkIndex++)
        {
            foreach (int placementIndex in refsByChunk[chunkIndex].MapObjIndices)
            {
                if ((uint)placementIndex < (uint)counts.Length)
                    counts[placementIndex]++;
            }
        }

        return counts;
    }

    private static int[] BuildMapObjAnchorChunks(IReadOnlyList<AlphaWorldModelPlacement> placements, int tileX, int tileY)
    {
        int[] anchors = new int[placements.Count];
        for (int index = 0; index < placements.Count; index++)
            anchors[index] = FindContainingChunk(placements[index].Position, tileX, tileY);

        return anchors;
    }

    private static void TrimChunkPlacementRefsToBudget(
        AlphaTileData tile,
        int tileX,
        int tileY,
        int chunkIndex,
        AlphaPlacementRefs[] refsByChunk,
        int[] mapObjRefChunkCounts,
        int[] mapObjAnchorChunks,
        int[] fixedPayloadSizesByChunk)
    {
        AlphaPlacementRefs placementRefs = refsByChunk[chunkIndex];
        int cx = chunkIndex % 16;
        int cy = chunkIndex / 16;
        int fixedPayloadSize = fixedPayloadSizesByChunk[chunkIndex];

        while (EstimateMcnkPayloadSize(fixedPayloadSize, placementRefs) >= AlphaClientMaxMcnkPayloadSize)
        {
            int trimIndex = SelectLeastLocalDuplicateMapObjRef(tile, tileX, tileY, chunkIndex, placementRefs.MapObjIndices, mapObjRefChunkCounts, mapObjAnchorChunks);
            if (trimIndex >= 0)
            {
                int placementIndex = placementRefs.MapObjIndices[trimIndex];
                placementRefs.MapObjIndices.RemoveAt(trimIndex);
                if ((uint)placementIndex < (uint)mapObjRefChunkCounts.Length && mapObjRefChunkCounts[placementIndex] > 0)
                    mapObjRefChunkCounts[placementIndex]--;

                continue;
            }

            break;
        }

        int payloadSize = EstimateMcnkPayloadSize(fixedPayloadSize, placementRefs);
        if (payloadSize >= AlphaClientMaxMcnkPayloadSize)
        {
            throw new InvalidDataException(
                $"Alpha MCNK ({tileX},{tileY}) chunk ({cx},{cy}) payload size {payloadSize} exceeds the client limit {AlphaClientMaxMcnkPayloadSize - 1}. Doodads={placementRefs.DoodadIndices.Count}, WMOs={placementRefs.MapObjIndices.Count}.");
        }
    }

    private static int[] BuildFixedPayloadSizesByChunk(AlphaTileData tile)
    {
        int[] fixedPayloadSizesByChunk = new int[256];
        AlphaPlacementRefs emptyRefs = new();
        for (int chunkIndex = 0; chunkIndex < fixedPayloadSizesByChunk.Length; chunkIndex++)
        {
            int cx = chunkIndex % 16;
            int cy = chunkIndex / 16;
            fixedPayloadSizesByChunk[chunkIndex] = GetMcnkPayloadSize(BuildMcnkData(tile, cx, cy, emptyRefs));
        }

        return fixedPayloadSizesByChunk;
    }

    private static int EstimateMcnkPayloadSize(int fixedPayloadSize, AlphaPlacementRefs placementRefs)
    {
        int totalRefs = placementRefs.DoodadIndices.Count + placementRefs.MapObjIndices.Count;
        return fixedPayloadSize + (totalRefs * sizeof(uint));
    }

    private static int SelectLeastLocalDuplicateMapObjRef(
        AlphaTileData tile,
        int tileX,
        int tileY,
        int chunkIndex,
        IReadOnlyList<int> mapObjIndices,
        int[] mapObjRefChunkCounts,
        int[] mapObjAnchorChunks)
    {
        if (mapObjIndices.Count == 0)
            return -1;

        int cx = chunkIndex % 16;
        int cy = chunkIndex / 16;
        GetChunkCenter(tileX, tileY, cx, cy, out float centerX, out float centerY);

        int selectedIndex = -1;
        float farthestDistanceSquared = float.NegativeInfinity;
        for (int index = 0; index < mapObjIndices.Count; index++)
        {
            int placementIndex = mapObjIndices[index];
            if ((uint)placementIndex >= (uint)tile.WorldModelPlacements.Count
                || (uint)placementIndex >= (uint)mapObjRefChunkCounts.Length
                || (uint)placementIndex >= (uint)mapObjAnchorChunks.Length)
            {
                continue;
            }

            if (mapObjRefChunkCounts[placementIndex] <= 1 || mapObjAnchorChunks[placementIndex] == chunkIndex)
                continue;

            AlphaWorldModelPlacement placement = tile.WorldModelPlacements[placementIndex];
            float dx = placement.Position.X - centerX;
            float dy = placement.Position.Y - centerY;
            float distanceSquared = dx * dx + dy * dy;
            if (distanceSquared > farthestDistanceSquared)
            {
                farthestDistanceSquared = distanceSquared;
                selectedIndex = index;
            }
        }

        return selectedIndex;
    }

    private static int GetMcnkPayloadSize(byte[] mcnkBytes)
    {
        if (mcnkBytes.Length < 8)
            return 0;

        return BinaryPrimitives.ReadInt32LittleEndian(mcnkBytes.AsSpan(4, 4));
    }

    private static void GetChunkPlanarBounds(int tileX, int tileY, int cx, int cy, out float minX, out float maxX, out float minY, out float maxY)
    {
        float tileOriginX = MapOrigin - tileX * TileWorldSize;
        float tileOriginY = MapOrigin - tileY * TileWorldSize;

        maxX = tileOriginX - cy * ChunkWorldSize;
        minX = maxX - ChunkWorldSize;
        maxY = tileOriginY - cx * ChunkWorldSize;
        minY = maxY - ChunkWorldSize;
    }

    private static void GetChunkCenter(int tileX, int tileY, int cx, int cy, out float centerX, out float centerY)
    {
        GetChunkPlanarBounds(tileX, tileY, cx, cy, out float minX, out float maxX, out float minY, out float maxY);
        centerX = (minX + maxX) * 0.5f;
        centerY = (minY + maxY) * 0.5f;
    }

    private static float DistanceSquared(float x0, float y0, float x1, float y1)
    {
        float dx = x0 - x1;
        float dy = y0 - y1;
        return (dx * dx) + (dy * dy);
    }

    private static byte[] BuildMcrfData(AlphaPlacementRefs placementRefs)
    {
        int totalRefs = placementRefs.DoodadIndices.Count + placementRefs.MapObjIndices.Count;
        if (totalRefs == 0)
            return [];

        byte[] data = new byte[totalRefs * sizeof(uint)];
        int offset = 0;
        foreach (int index in placementRefs.DoodadIndices)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset), (uint)index);
            offset += sizeof(uint);
        }

        foreach (int index in placementRefs.MapObjIndices)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset), (uint)index);
            offset += sizeof(uint);
        }

        return data;
    }
    private static float[] ExtractChunkHeights(float[,] heightmap, int cx, int cy)
    {
        float[] heights = new float[145];
        int baseX = cx * 16;
        int baseY = cy * 16;
        int idx = 0;

        for (int row = 0; row < 17; row++)
        {
            bool isInner = (row & 1) != 0;
            int cols = isInner ? 8 : 9;
            for (int col = 0; col < cols; col++)
            {
                int sampleX = isInner ? (col * 2) + 1 : col * 2;
                int sampleY = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;
                int px = baseX + sampleX;
                int py = baseY + sampleY;

                if ((uint)px < TileSize && (uint)py < TileSize)
                    heights[idx] = heightmap[py, px];

                idx++;
            }
        }

        return heights;
    }

    private static byte[] BuildAlphaMcvt(float[] heights)
    {
        byte[] data = new byte[AlphaMcvtSize];

        // Ghidra-verified (CMapChunk::CreateVertices, client 0.5.3.3368):
        // Alpha MCVT heights are stored as ABSOLUTE world-space Z values.
        // The client reads them directly with v->z = *he, then subtracts the
        // chunk's Position (stored at MCNK 0x68/0x6C) only for rendering precision.
        // No base-height adjustment is applied during encoding or decoding.
        //
        // Layout: 81 outer vertices (9 rows × 9 cols) then 64 inner (8 rows × 8 cols).
        // heights[] is in interleaved LK order: row 0 outer(9), row 0 inner(8), ...
        // Outer row R, col C maps to heights index R*17 + C.
        // Inner row I, col C maps to heights index I*17 + 9 + C.
        int dst = 0;
        for (int outerRow = 0; outerRow < 9; outerRow++)
        {
            for (int col = 0; col < 9; col++)
            {
                int srcIdx = outerRow * 17 + col;
                float v = srcIdx < heights.Length ? heights[srcIdx] : 0f;
                BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(dst), v);
                dst += 4;
            }
        }

        for (int innerRow = 0; innerRow < 8; innerRow++)
        {
            for (int col = 0; col < 8; col++)
            {
                int srcIdx = innerRow * 17 + 9 + col;
                float v = srcIdx < heights.Length ? heights[srcIdx] : 0f;
                BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(dst), v);
                dst += 4;
            }
        }

        return data;
    }

    private static byte[] BuildAlphaMcnr(float[,,]? normalXyz, int cx, int cy)
    {
        byte[] data = new byte[AlphaMcnrSize];
        if (normalXyz == null) return data;

        int baseX = cx * 16;
        int baseY = cy * 16;

        // Alpha MCNR layout: 81 outer normals (9 rows × 9 cols) then 64 inner normals (8 rows × 8 cols)
        // followed by 13 zero-pad bytes = 448 total
        int dst = 0;

        // Outer vertices (9 rows × 9 cols)
        for (int outerRow = 0; outerRow < 9; outerRow++)
        {
            for (int col = 0; col < 9; col++)
            {
                int px = baseX + col * 2;
                int py = baseY + outerRow * 2;

                if ((uint)px < TileSize && (uint)py < TileSize)
                {
                    data[dst] = EncodeNormal(normalXyz[py, px, 0]);
                    data[dst + 1] = EncodeNormal(normalXyz[py, px, 2]);
                    data[dst + 2] = EncodeNormal(normalXyz[py, px, 1]);
                }
                dst += 3;
            }
        }

        // Inner vertices (8 rows × 8 cols)
        for (int innerRow = 0; innerRow < 8; innerRow++)
        {
            for (int col = 0; col < 8; col++)
            {
                int px = baseX + col * 2 + 1;
                int py = baseY + innerRow * 2 + 1;

                if ((uint)px < TileSize && (uint)py < TileSize)
                {
                    data[dst] = EncodeNormal(normalXyz[py, px, 0]);
                    data[dst + 1] = EncodeNormal(normalXyz[py, px, 2]);
                    data[dst + 2] = EncodeNormal(normalXyz[py, px, 1]);
                }
                dst += 3;
            }
        }

        // Remaining bytes already zero-initialized (padding)
        return data;
    }

    private static byte EncodeNormal(float value)
    {
        return unchecked((byte)(sbyte)Math.Clamp(MathF.Round(value * 127f), -128, 127));
    }

    private static byte[] SliceChunkShadowAlpha(float[,] shadowMask, int cx, int cy)
    {
        const int chunkSize = 64;
        var shadow = new byte[chunkSize * chunkSize / 8];
        int baseX = cx * chunkSize;
        int baseY = cy * chunkSize;
        int idx = 0;

        for (int y = 0; y < chunkSize; y++)
        {
            for (int byteIdx = 0; byteIdx < 8; byteIdx++)
            {
                byte bits = 0;
                for (int bit = 0; bit < 8; bit++)
                {
                    int sx = baseX + byteIdx * 8 + bit;
                    int sy = baseY + y;
                    if (sx < shadowMask.GetLength(1) && sy < shadowMask.GetLength(0) && shadowMask[sy, sx] > 0.5f)
                        bits |= (byte)(1 << bit);
                }
                shadow[idx++] = bits;
            }
        }

        return shadow;
    }

    private static byte[] BuildAlphaMcly(AlphaTileData tile, int cx, int cy, int nLayers)
    {
        byte[] data = new byte[nLayers * MclyEntrySize];
        int alphaOffset = 0;

        for (int l = 0; l < nLayers; l++)
        {
            int off = l * MclyEntrySize;
            uint texId = (uint)(cx < tile.MclyTextureIds.GetLength(0) && cy < tile.MclyTextureIds.GetLength(1)
                ? tile.MclyTextureIds[cx, cy, l] : 0);
            uint flags = 0;
            if (l > 0) flags |= 0x100;

            BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(off), texId);
            BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(off + 4), flags);
            BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(off + 8), (uint)alphaOffset);
            BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(off + 12), 0);

            if (l > 0) alphaOffset += 2048;
        }

        return data;
    }

    private static byte[] BuildAlphaMcal(AlphaTileData tile, int cx, int cy, int nLayers)
    {
        if (tile.McalAlphaPack == null || nLayers <= 1) return [];

        int totalSize = (nLayers - 1) * 2048;
        byte[] data = new byte[totalSize];
        int dstOffset = 0;

        for (int l = 1; l < nLayers; l++)
        {
            for (int y = 0; y < 64; y++)
            {
                for (int x = 0; x < 64; x += 2)
                {
                    float sampleLo = SampleAlpha(tile.McalAlphaPack, l, cx, cy, x, y);
                    float sampleHi = SampleAlpha(tile.McalAlphaPack, l, cx, cy, x + 1, y);

                    int alphaLit = (int)(sampleLo * 15f + 0.5f);
                    int alphaHi = (int)(sampleHi * 15f + 0.5f);

                    alphaLit = Math.Clamp(alphaLit, 0, 15);
                    alphaHi = Math.Clamp(alphaHi, 0, 15);

                    data[dstOffset++] = (byte)(alphaLit | (alphaHi << 4));
                }
            }
        }

        return data;
    }

    private static float SampleAlpha(float[,,] alphaPack, int layer, int chunkX, int chunkY, int localX, int localY)
    {
        int width = alphaPack.GetLength(1);
        int height = alphaPack.GetLength(0);
        if (width <= 0 || height <= 0 || layer < 0 || layer >= alphaPack.GetLength(2))
            return 0f;

        float globalX = (chunkX * AlphaChunkAlphaSize) + localX;
        float globalY = (chunkY * AlphaChunkAlphaSize) + localY;
        float scaleX = (float)width / AlphaLegacyTileAlphaSize;
        float scaleY = (float)height / AlphaLegacyTileAlphaSize;
        float sourceX = ((globalX + 0.5f) * scaleX) - 0.5f;
        float sourceY = ((globalY + 0.5f) * scaleY) - 0.5f;

        return SampleBilinear(alphaPack, layer, sourceX, sourceY);
    }

    private static float SampleBilinear(float[,,] alphaPack, int layer, float x, float y)
    {
        int width = alphaPack.GetLength(1);
        int height = alphaPack.GetLength(0);
        if (width == 0 || height == 0)
            return 0f;

        float clampedX = Math.Clamp(x, 0f, width - 1);
        float clampedY = Math.Clamp(y, 0f, height - 1);

        int x0 = (int)MathF.Floor(clampedX);
        int y0 = (int)MathF.Floor(clampedY);
        int x1 = Math.Min(x0 + 1, width - 1);
        int y1 = Math.Min(y0 + 1, height - 1);
        float tx = clampedX - x0;
        float ty = clampedY - y0;

        float v00 = alphaPack[y0, x0, layer];
        float v10 = alphaPack[y0, x1, layer];
        float v01 = alphaPack[y1, x0, layer];
        float v11 = alphaPack[y1, x1, layer];

        float top = v00 + ((v10 - v00) * tx);
        float bottom = v01 + ((v11 - v01) * tx);
        return Math.Clamp(top + ((bottom - top) * ty), 0f, 1f);
    }

    private static AlphaLiquidChunk? FindLiquidChunk(AlphaTileData tile, int cx, int cy)
    {
        foreach (AlphaLiquidChunk liquidChunk in tile.LiquidChunks)
        {
            if (liquidChunk.IndexX == cx && liquidChunk.IndexY == cy)
                return liquidChunk;
        }

        return null;
    }

    private static byte[] BuildAlphaMclq(AlphaLiquidChunk? liquidChunk)
    {
        if (liquidChunk is null)
            return [];

        // Ghidra-verified: Alpha MCLQ heights are stored as absolute world-space Z values,
        // consistent with the MCVT convention. No base-height subtraction.
        float minHeight = liquidChunk.MinHeight;
        float maxHeight = liquidChunk.MaxHeight;
        if (float.IsNaN(minHeight) || float.IsNaN(maxHeight))
            return [];

        bool hasVertexHeights = liquidChunk.Heights is { Length: >= 81 };
        bool hasTileFlags = liquidChunk.TileFlags is { Length: >= 64 };

    // Ghidra-verified (CMapChunk::Create, 0.5.3.3368): each active liquid bit
    // consumes one fixed-size 0x324-byte block. The client copies:
    // - min/max height at 0x000
    // - 81 * 8-byte vertex records at 0x008
    // - 64 tile flags at 0x290
    // - flow count at 0x2D0
    // - 20 dwords of flow data at 0x2D4
    // Even when flow data is absent, the block still needs the full size.
    byte[] payload = new byte[AlphaMclqPayloadSize];
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0), minHeight);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(4), maxHeight);

        if (hasVertexHeights)
        {
            for (int index = 0; index < 81; index++)
                BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(8 + (index * 8) + 4), liquidChunk.Heights![index]);
        }

        if (hasTileFlags)
            Buffer.BlockCopy(liquidChunk.TileFlags, 0, payload, AlphaMclqTileFlagsOffset, 64);

        BinaryPrimitives.WriteInt32LittleEndian(payload.AsSpan(AlphaMclqFlowCountOffset), 0);

        return payload;
    }

    private static uint NormalizeAlphaLiquidFlags(uint flags)
    {
        uint liquidBits = flags & 0x3Cu;
        return liquidBits switch
        {
            0x04u or 0x08u or 0x10u or 0x20u => liquidBits,
            _ when liquidBits != 0 => 0x04u,
            _ => 0u,
        };
    }

    private static bool HasAnyChunkRefs(IReadOnlyList<int>[]? refsByChunk)
    {
        if (refsByChunk is not { Length: 256 })
            return false;

        for (int chunkIndex = 0; chunkIndex < refsByChunk.Length; chunkIndex++)
        {
            if (refsByChunk[chunkIndex] is { Count: > 0 })
                return true;
        }

        return false;
    }

    private static byte[] BuildMddfData(AlphaTileData tile, Dictionary<string, int> nameIndex)
    {
        if (tile.ModelPlacements.Count == 0) return [];

        byte[] data = new byte[tile.ModelPlacements.Count * MddfEntrySize];
        for (int i = 0; i < tile.ModelPlacements.Count; i++)
        {
            var p = tile.ModelPlacements[i];
            int off = i * MddfEntrySize;
            int nameId = nameIndex.TryGetValue(p.ModelPath, out int idx) ? idx : 0;
            float filePosX = MapOrigin - p.Position.Y;
            float filePosY = p.Position.Z;
            float filePosZ = MapOrigin - p.Position.X;
            float fileRotX = p.Rotation.X;
            float fileRotY = p.Rotation.Z;
            float fileRotZ = p.Rotation.Y;
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(off + 0x00), nameId);
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(off + 0x04), p.UniqueId);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x08), filePosX);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x0C), filePosY);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x10), filePosZ);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x14), fileRotX);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x18), fileRotY);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x1C), fileRotZ);
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(off + 0x20), (ushort)MathF.Round(p.Scale * 1024f));
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(off + 0x22), 0);
        }

        return data;
    }

    private static byte[] BuildModfData(AlphaTileData tile, Dictionary<string, int> nameIndex)
    {
        if (tile.WorldModelPlacements.Count == 0) return [];

        byte[] data = new byte[tile.WorldModelPlacements.Count * ModfEntrySize];
        for (int i = 0; i < tile.WorldModelPlacements.Count; i++)
        {
            var p = tile.WorldModelPlacements[i];
            int off = i * ModfEntrySize;
            int nameId = nameIndex.TryGetValue(p.ModelPath, out int idx) ? idx : 0;
            float filePosX = MapOrigin - p.Position.Y;
            float filePosY = p.Position.Z;
            float filePosZ = MapOrigin - p.Position.X;
            float fileRotX = p.Rotation.X;
            float fileRotY = p.Rotation.Z;
            float fileRotZ = p.Rotation.Y;
            float extentsTopX = MapOrigin - p.BoundsMin.Y;
            float extentsTopY = p.BoundsMax.Z;
            float extentsTopZ = MapOrigin - p.BoundsMin.X;
            float extentsBotX = MapOrigin - p.BoundsMax.Y;
            float extentsBotY = p.BoundsMin.Z;
            float extentsBotZ = MapOrigin - p.BoundsMax.X;
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(off + 0x00), nameId);
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(off + 0x04), p.UniqueId);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x08), filePosX);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x0C), filePosY);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x10), filePosZ);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x14), fileRotX);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x18), fileRotY);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x1C), fileRotZ);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x20), extentsTopX);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x24), extentsTopY);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x28), extentsTopZ);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x2C), extentsBotX);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x30), extentsBotY);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 0x34), extentsBotZ);
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(off + 0x38), p.Flags);
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(off + 0x3A), 0);
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(off + 0x3C), 0);
        }

        return data;
    }

    private static float CalculateRadius(float[] heights)
    {
        float minH = float.MaxValue;
        float maxH = float.MinValue;
        for (int i = 0; i < heights.Length; i++)
        {
            float h = heights[i];
            if (h < minH) minH = h;
            if (h > maxH) maxH = h;
        }
        float heightRange = maxH - minH;
        float horizontalRadius = 23.57f;
        return MathF.Sqrt(horizontalRadius * horizontalRadius + (heightRange / 2) * (heightRange / 2));
    }

    private static byte[] BuildMainPayload(Dictionary<(int tileX, int tileY), AlphaTileData> tiles)
    {
        byte[] data = new byte[TilesPerAxis * TilesPerAxis * MainEntrySize];
        return data;
    }

    private static void PatchMainEntry(byte[] mainData, int index, int offset, int size)
    {
        int entryOffset = index * MainEntrySize;
        BinaryPrimitives.WriteInt32LittleEndian(mainData.AsSpan(entryOffset), offset);
        BinaryPrimitives.WriteInt32LittleEndian(mainData.AsSpan(entryOffset + 4), size);
    }

    private static void PatchMainPayload(MemoryStream ms, long mainPosition, byte[] mainData)
    {
        long pos = ms.Position;
        ms.Position = mainPosition + ChunkHeaderSize;
        ms.Write(mainData, 0, mainData.Length);
        ms.Position = pos;
    }

    private static void PatchMphd(MemoryStream ms, long mphdPosition,
        IReadOnlyList<string> mdxNames, long mdnmStart,
        IReadOnlyList<string> wmoNames, long monmStart)
    {
        long pos = ms.Position;
        ms.Position = mphdPosition + ChunkHeaderSize;

        byte[] mphdData = new byte[MphdAlphaSize];
        BinaryPrimitives.WriteInt32LittleEndian(mphdData.AsSpan(0), mdxNames.Count > 0 ? mdxNames.Count + 1 : 0);
        BinaryPrimitives.WriteInt32LittleEndian(mphdData.AsSpan(4), (int)mdnmStart);
        BinaryPrimitives.WriteInt32LittleEndian(mphdData.AsSpan(8), wmoNames.Count > 0 ? wmoNames.Count + 1 : 0);
        BinaryPrimitives.WriteInt32LittleEndian(mphdData.AsSpan(12), (int)monmStart);
        ms.Write(mphdData, 0, mphdData.Length);

        ms.Position = pos;
    }

    private static void WriteMcinOffsets(BinaryWriter bw, int[] offsets, List<byte[]> mcnkDataList, long mcinStart)
    {
        long pos = bw.Seek(0, SeekOrigin.Current);
        bw.Seek((int)mcinStart + ChunkHeaderSize, SeekOrigin.Begin);

        for (int i = 0; i < McinEntryCount; i++)
        {
            bw.Write(offsets[i]);
            bw.Write(mcnkDataList[i].Length);
            bw.Write(0);
            bw.Write(0);
        }

        bw.Seek((int)pos, SeekOrigin.Begin);
    }

    private static void WriteMhdrData(BinaryWriter bw, long mhdrStart, int mtexRelative, int mddfRelative, int modfRelative)
    {
        long pos = bw.Seek(0, SeekOrigin.Current);
        int mhdrDataStart = (int)mhdrStart + ChunkHeaderSize;

        byte[] data = new byte[64];
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(0x00), 64);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(0x04), mtexRelative);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(0x08), 0);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(0x0C), mddfRelative);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(0x10), 0);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(0x14), modfRelative);

        bw.Seek(mhdrDataStart, SeekOrigin.Begin);
        bw.Write(data, 0, 64);
        bw.Seek((int)pos, SeekOrigin.Begin);
    }

    private static IReadOnlyList<string> CollectMdxNames(Dictionary<(int tileX, int tileY), AlphaTileData> tiles)
    {
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var tile in tiles.Values)
        {
            foreach (var p in tile.ModelPlacements)
                names.Add(p.ModelPath);
        }
        return names.OrderBy(n => n, StringComparer.OrdinalIgnoreCase).ToList();
    }

    private static IReadOnlyList<string> CollectWmoNames(Dictionary<(int tileX, int tileY), AlphaTileData> tiles)
    {
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var tile in tiles.Values)
        {
            foreach (var p in tile.WorldModelPlacements)
                names.Add(p.ModelPath);
        }
        return names.OrderBy(n => n, StringComparer.OrdinalIgnoreCase).ToList();
    }

    private static byte[] BuildStringTable(IReadOnlyList<string> names)
    {
        if (names.Count == 0) return [];

        int totalSize = 0;
        foreach (var name in names)
            totalSize += Encoding.UTF8.GetByteCount(name) + 1;

        byte[] data = new byte[totalSize];
        int offset = 0;
        foreach (var name in names)
        {
            int written = Encoding.UTF8.GetBytes(name, data.AsSpan(offset));
            offset += written;
            data[offset++] = 0;
        }

        return data;
    }

    private static Dictionary<string, int> BuildNameIndex(IReadOnlyList<string> names)
    {
        var index = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < names.Count; i++)
            index[names[i]] = i;
        return index;
    }

    private static byte[] WrapChunk(string tag, byte[] payload)
    {
        byte[] result = new byte[ChunkHeaderSize + payload.Length];
        FourCC.FromString(tag).ToFileBytes().CopyTo(result, 0);
        BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(4), payload.Length);
        Buffer.BlockCopy(payload, 0, result, ChunkHeaderSize, payload.Length);
        return result;
    }

    private static void WriteChunk(BinaryWriter bw, string tag, int dataSize, Action<BinaryWriter> writePayload)
    {
        bw.Write(FourCC.FromString(tag).ToFileBytes());
        bw.Write(dataSize);
        writePayload(bw);
    }

    private static void WriteDataChunk(BinaryWriter bw, string tag, byte[] payload)
    {
        bw.Write(FourCC.FromString(tag).ToFileBytes());
        bw.Write(payload.Length);
        bw.Write(payload);
    }

    private sealed class AlphaPlacementRefs
    {
        public List<int> DoodadIndices { get; } = [];
        public List<int> MapObjIndices { get; } = [];
    }

    private sealed class AlphaPlacementWritePlan(byte[] mddfData, byte[] modfData, AlphaPlacementRefs[] refsByChunk)
    {
        public byte[] MddfData { get; } = mddfData;
        public byte[] ModfData { get; } = modfData;
        public AlphaPlacementRefs[] RefsByChunk { get; } = refsByChunk;
    }
}
