using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AlphaWdtWriter
{
    private const int ChunkHeaderSize = 8;
    private const int McnkHeaderSize = 128;
    private const int McinEntryCount = 256;
    private const int McinEntrySize = 16;
    private const int MainEntrySize = 16;
    private const int TilesPerAxis = 64;
    private const int TileSize = 257;
    private const int MddfEntrySize = 36;
    private const int ModfEntrySize = 64;
    private const int MclyEntrySize = 16;
    private const int AlphaMcvtSize = 580;
    private const int AlphaMcnrSize = 448;
    private const int AlphaTileAlphaSize = 256;
    private const int AlphaLegacyTileAlphaSize = 1024;
    private const int AlphaChunkAlphaSize = 64;
    private const int AlphaMclqTileFlagsOffset = 0x290;
    private const float MapOrigin = 17066.666f;

    public static byte[] Build(string mapName, Dictionary<(int tileX, int tileY), AlphaTileData> tiles)
    {
        ArgumentNullException.ThrowIfNull(tiles);

        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        WriteChunk(bw, "MVER", 4, w => w.Write(18));

        long mphdPosition = ms.Position;
        WriteChunk(bw, "MPHD", 16, static w => w.Write(new byte[16]));

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

        foreach (var kvp in tiles.OrderBy(t => t.Key.Item1 * TilesPerAxis + t.Key.Item2))
        {
            var (tileX, tileY) = kvp.Key;
            var tile = kvp.Value;

            int tileOffset = (int)ms.Position;
            PatchMainEntry(mainData, tileX * TilesPerAxis + tileY, tileOffset);
            WriteTileData(bw, tile, tileX, tileY, allMdxNames, allWmoNames, mdxNameIndex, wmoNameIndex);
        }

        PatchMainPayload(ms, mainPosition, mainData);

        bw.Flush();
        return ms.ToArray();
    }

    private static void WriteTileData(BinaryWriter bw, AlphaTileData tile, int tileX, int tileY,
        IReadOnlyList<string> mdxNames, IReadOnlyList<string> wmoNames,
        Dictionary<string, int> mdxIndex, Dictionary<string, int> wmoIndex)
    {
        float tileBaseHeight = ComputeTileBaseHeight(tile);

        var mcnkDataList = new List<byte[]>(256);
        for (int cy = 0; cy < 16; cy++)
        {
            for (int cx = 0; cx < 16; cx++)
            {
                mcnkDataList.Add(BuildMcnkData(tile, cx, cy, tileX, tileY, tileBaseHeight));
            }
        }

        byte[] mtexData = BuildStringTable(tile.TextureNames);
        byte[] mddfData = BuildMddfData(tile, mdxIndex);
        byte[] modfData = BuildModfData(tile, wmoIndex);

        long mhdrStart = bw.Seek(0, SeekOrigin.Current);
        WriteChunk(bw, "MHDR", 64, static w => w.Write(new byte[64]));

        byte[] mcinData = new byte[McinEntryCount * McinEntrySize];

        long mcinStart = bw.Seek(0, SeekOrigin.Current);
        WriteChunk(bw, "MCIN", mcinData.Length, w => w.Write(mcinData));

        long afterMcin = bw.Seek(0, SeekOrigin.Current);
        long mhdrDataStart = mhdrStart + ChunkHeaderSize;

        int mtexRelative = (int)(bw.Seek(0, SeekOrigin.Current) - mhdrDataStart);
        WriteDataChunk(bw, "MTEX", mtexData);

        int mddfRelative = mddfData.Length > 0 ? (int)(bw.Seek(0, SeekOrigin.Current) - mhdrDataStart) : 0;
        if (mddfData.Length > 0)
            WriteDataChunk(bw, "MDDF", mddfData);

        int modfRelative = modfData.Length > 0 ? (int)(bw.Seek(0, SeekOrigin.Current) - mhdrDataStart) : 0;
        if (modfData.Length > 0)
            WriteDataChunk(bw, "MODF", modfData);

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
    }

    private static byte[] BuildMcnkData(AlphaTileData tile, int cx, int cy, int tileX, int tileY, float tileBaseHeight)
    {
        float[] heights = ExtractChunkHeights(tile.Heightmap, cx, cy);
        float chunkBaseHeight = heights[0];

        byte[] mcvtAlpha = BuildAlphaMcvt(heights, chunkBaseHeight);
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
        byte[] mclqRaw = BuildAlphaMclq(liquidChunk, chunkBaseHeight);

        byte[] mcrfRaw = [];
        int nDoodadRefs = 0;
        int nMapObjRefs = 0;

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

        uint flags = liquidChunk is not null ? (liquidChunk.McnkFlags & 0x3Cu) : 0u;
        if (mcshRaw.Length > 0) flags |= 0x01;

        float radius = CalculateRadius(heights, tileBaseHeight);

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
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x40), (ushort)(cx < tile.HoleMask.GetLength(0) && cy < tile.HoleMask.GetLength(1) && tile.HoleMask[cx, cy] ? 1 : 0));
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x5C), chunkDataSize);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x64), offsLiquid);

        // wow-viewer currently has two Alpha readers using different offsets for the chunk base
        // height. Mirror the value into both fields so produced Alpha WDTs work across both paths.
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

    private static float ComputeTileBaseHeight(AlphaTileData tile)
    {
        float min = float.MaxValue;
        for (int y = 0; y < TileSize; y++)
        {
            for (int x = 0; x < TileSize; x++)
            {
                if (tile.Heightmap[y, x] < min && tile.Heightmap[y, x] != 0f)
                    min = tile.Heightmap[y, x];
            }
        }
        return min == float.MaxValue ? 0f : min;
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

    private static byte[] BuildAlphaMcvt(float[] heights, float chunkBaseHeight)
    {
        byte[] data = new byte[AlphaMcvtSize];

        // heights[] is in LK interleaved order: row 0 outer(9), row 0 inner(8), row 1 outer(9), ...
        // Alpha MCVT stores outer block first (81 floats), then inner block (64 floats).
        // Layout: 9 outer rows × 9 cols, then 8 inner rows × 8 cols.
        // The heights index mapping follows the same pattern.
        int dst = 0;
        for (int outerRow = 0; outerRow < 9; outerRow++)
        {
            for (int col = 0; col < 9; col++)
            {
                // Outer row N comes from LK heights index: N*17 + col  (9 outer entries for even rows)
                // Actually, LK heights layout: row 0 outer (9 entries at indices 0..8),
                //   row 0 inner (8 entries at indices 9..16), row 1 outer (9 entries at indices 17..25), ...
                // Outer row R is at heights index: R*17 (for the first entry of that outer row)
                // But wait, this is interleaved so:
                //   Even rows (0,2,...,16): 9 entries starting at index row*9 - (row/2)*8
                // That's complex. Instead, just map directly.
                // Alpha outer[R][C] = heights at grid position (C*2, R*2)
                // In LK heights[]: row r (0..16), col count 9 or 8 depending on r parity
                // For outer row R (R=0..8): LK heights position = sum of row sizes up to R*2
                // Each even row has 9 entries, each odd row has 8 entries.
                // Heights index for outer row R, col C = R*17 + C
                int srcIdx = outerRow * 17 + col;
                float v = srcIdx < heights.Length ? heights[srcIdx] - chunkBaseHeight : 0f;
                BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(dst), v);
                dst += 4;
            }
        }

        for (int innerRow = 0; innerRow < 8; innerRow++)
        {
            for (int col = 0; col < 8; col++)
            {
                // Inner row I (I=0..7) maps to LK heights row I*2+1
                // Heights index for inner row I, col C = I*17 + 9 + C
                int srcIdx = innerRow * 17 + 9 + col;
                float v = srcIdx < heights.Length ? heights[srcIdx] - chunkBaseHeight : 0f;
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

    private static byte[] BuildAlphaMclq(AlphaLiquidChunk? liquidChunk, float chunkBaseHeight)
    {
        if (liquidChunk is null)
            return [];

        float relativeMinHeight = liquidChunk.MinHeight - chunkBaseHeight;
        float relativeMaxHeight = liquidChunk.MaxHeight - chunkBaseHeight;
        if (float.IsNaN(relativeMinHeight) || float.IsNaN(relativeMaxHeight))
            return [];

        bool hasVertexHeights = liquidChunk.Heights is { Length: >= 81 };
        bool hasTileFlags = liquidChunk.TileFlags is { Length: >= 64 };
        int payloadSize = hasVertexHeights || hasTileFlags
            ? AlphaMclqTileFlagsOffset + 64
            : 8;

        byte[] payload = new byte[payloadSize];
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0), relativeMinHeight);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(4), relativeMaxHeight);

        if (hasVertexHeights)
        {
            for (int index = 0; index < 81; index++)
                BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(8 + (index * 8) + 4), liquidChunk.Heights![index] - chunkBaseHeight);
        }

        if (hasTileFlags)
            Buffer.BlockCopy(liquidChunk.TileFlags, 0, payload, AlphaMclqTileFlagsOffset, 64);

        return payload;
    }

    private static byte[] BuildMddfData(AlphaTileData tile, Dictionary<string, int> nameIndex)
    {
        if (tile.ModelPlacements.Count == 0) return [];

        byte[] data = new byte[tile.ModelPlacements.Count * MddfEntrySize];
        for (int i = 0; i < tile.ModelPlacements.Count; i++)
        {
            var p = tile.ModelPlacements[i];
            int off = i * MddfEntrySize;
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(off), nameIndex.TryGetValue(p.ModelPath, out int idx) ? idx : 0);
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(off + 4), p.UniqueId);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 8), MapOrigin - p.Position.Y);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 12), p.Position.Z);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 16), MapOrigin - p.Position.X);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 20), p.Rotation.X);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 24), p.Rotation.Z);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 28), p.Rotation.Y);
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(off + 32), (ushort)MathF.Round(p.Scale * 1024f));
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(off + 34), 0);
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
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(off), nameIndex.TryGetValue(p.ModelPath, out int idx) ? idx : 0);
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(off + 4), p.UniqueId);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 8), MapOrigin - p.Position.Y);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 12), p.Position.Z);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 16), MapOrigin - p.Position.X);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 20), p.Rotation.X);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 24), p.Rotation.Z);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 28), p.Rotation.Y);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 32), MapOrigin - p.BoundsMax.Y);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 36), p.BoundsMax.Z);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 40), MapOrigin - p.BoundsMin.X);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 44), MapOrigin - p.BoundsMin.Y);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 48), p.BoundsMin.Z);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 52), MapOrigin - p.BoundsMax.X);
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(off + 56), p.Flags);
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(off + 58), 0);
        }

        return data;
    }

    private static float CalculateRadius(float[] heights, float tileBaseHeight)
    {
        float minH = float.MaxValue;
        float maxH = float.MinValue;
        for (int i = 0; i < heights.Length; i++)
        {
            float abs = heights[i];
            if (abs < minH) minH = abs;
            if (abs > maxH) maxH = abs;
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

    private static void PatchMainEntry(byte[] mainData, int index, int offset)
    {
        int entryOffset = index * MainEntrySize;
        BinaryPrimitives.WriteInt32LittleEndian(mainData.AsSpan(entryOffset), offset);
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

        byte[] mphdData = new byte[16];
        BinaryPrimitives.WriteInt32LittleEndian(mphdData.AsSpan(0), mdxNames.Count > 0 ? mdxNames.Count + 1 : 0);
        BinaryPrimitives.WriteInt32LittleEndian(mphdData.AsSpan(4), (int)mdnmStart);
        BinaryPrimitives.WriteInt32LittleEndian(mphdData.AsSpan(8), wmoNames.Count > 0 ? wmoNames.Count + 1 : 0);
        BinaryPrimitives.WriteInt32LittleEndian(mphdData.AsSpan(12), (int)monmStart);
        ms.Write(mphdData, 0, 16);

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
        if ((payload.Length & 1) != 0)
            bw.Write((byte)0);
    }
}
