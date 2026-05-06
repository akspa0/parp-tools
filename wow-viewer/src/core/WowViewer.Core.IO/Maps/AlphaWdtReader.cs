using System.Numerics;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AlphaWdtReader
{
    private const int ChunkHeaderSize = 8;
    private const int AlphaMcvtSize = 580;
    private const int McnkHeaderSize = 128;
    private const int MainEntrySize = 16;
    private const int McinEntrySize = 16;
    private const int MddfEntrySize = 36;
    private const int ModfEntrySize = 64;
    private const int MclyEntrySize = 16;
    private const int HalfStepsPerChunk = 16;
    private const int TileHeightmapSize = 257;
    private const int MapOrigin = 17066;

    public static bool TryReadTile(string wdtFilePath, int tileX, int tileY, out AlphaTileData? data)
    {
        data = null;
        byte[] wdtData;
        try { wdtData = File.ReadAllBytes(wdtFilePath); }
        catch { return false; }
        return TryReadTile(wdtData, tileX, tileY, wdtFilePath, out data);
    }

    public static bool TryReadTile(byte[] wdtData, int tileX, int tileY, out AlphaTileData? data)
    {
        return TryReadTile(wdtData, tileX, tileY, "memory", out data);
    }

    public static bool IsAlphaWdt(byte[] wdtData)
    {
        using var ms = new MemoryStream(wdtData, writable: false);
        return TryReadAlphaTopLevelChunks(ms, out _, out _, out _, out bool hasMdnm, out bool hasMonm);
    }

    private static bool ReadMainPayload(byte[] wdtData, out byte[] mainData)
    {
        mainData = [];
        using var ms = new MemoryStream(wdtData, writable: false);
        return TryReadAlphaTopLevelChunks(ms, out _, out _, out MapChunkLocation main, out _, out _)
            && main.Size > 0
            && TryReadPayloadAt(ms, main, out mainData);
    }

    private static bool TryReadAlphaTopLevelChunks(
        MemoryStream ms,
        out MapChunkLocation mver,
        out MapChunkLocation mphd,
        out MapChunkLocation main,
        out bool hasMdnm,
        out bool hasMonm)
    {
        mver = default; mphd = default; main = default; hasMdnm = false; hasMonm = false;

        if (!TryReadChunk(ms, 0, out mver) || mver.Id != MapChunkIds.Mver) return false;
        long nextOff = mver.DataOffset + mver.Size;
        if ((mver.Size & 1) != 0) nextOff++;

        if (!TryReadChunk(ms, nextOff, out mphd) || mphd.Id != MapChunkIds.Mphd) return false;
        nextOff = mphd.DataOffset + mphd.Size;
        if ((mphd.Size & 1) != 0) nextOff++;

        if (!TryReadChunk(ms, nextOff, out main) || main.Id != MapChunkIds.Main) return false;

        if (mphd.Size >= 8)
        {
            byte[] mphdData = new byte[Math.Min(mphd.Size, 16)];
            ms.Position = mphd.DataOffset;
            ms.ReadExactly(mphdData);
            int mdnmOff = BitConverter.ToInt32(mphdData, 4);
            int monmOff = BitConverter.ToInt32(mphdData, 12);
            if (mdnmOff > 0)
            {
                int mdnmAbs = (int)(mphd.DataOffset - ChunkHeaderSize) + mdnmOff;
                if (TryReadChunk(ms, mdnmAbs, out MapChunkLocation c) && c.Id == MapChunkIds.Mdnm)
                    hasMdnm = true;
            }
            if (monmOff > 0)
            {
                int monmAbs = (int)(mphd.DataOffset - ChunkHeaderSize) + monmOff;
                if (TryReadChunk(ms, monmAbs, out MapChunkLocation c) && c.Id == MapChunkIds.Monm)
                    hasMonm = true;
            }
        }

        return true;
    }

    private static bool TryReadChunk(MemoryStream ms, long offset, out MapChunkLocation chunk)
    {
        chunk = default;
        if (offset < 0 || offset + ChunkHeaderSize > ms.Length) return false;
        ms.Position = offset;
        byte[] header = new byte[ChunkHeaderSize];
        ms.ReadExactly(header);
        FourCC id = FourCC.FromFileBytes(header);
        uint size = BitConverter.ToUInt32(header, 4);
        if (size > ms.Length - offset - ChunkHeaderSize) return false;
        chunk = new MapChunkLocation(id, size, offset, offset + ChunkHeaderSize);
        return true;
    }

    private static bool TryReadPayloadAt(MemoryStream ms, MapChunkLocation chunk, out byte[] payload)
    {
        payload = [];
        if (chunk.Size <= 0) return false;
        ms.Position = chunk.DataOffset;
        payload = new byte[chunk.Size];
        ms.ReadExactly(payload);
        return true;
    }

    private static bool TryReadTile(byte[] wdtData, int tileX, int tileY, string sourcePath, out AlphaTileData? data)
    {
        data = null;

        if (!ReadMainPayload(wdtData, out byte[] mainData))
            return false;

        int mainEntryIndex = tileX * 64 + tileY;
        int entryOffset = mainEntryIndex * MainEntrySize;
        if (entryOffset < 0 || entryOffset + sizeof(int) > mainData.Length)
            return false;

        int adtOffset = BitConverter.ToInt32(mainData, entryOffset);
        if (adtOffset <= 0)
            return false;

        return ReadAlphaTile(wdtData, adtOffset, tileX, tileY, sourcePath, out data);
    }

    private static bool ReadAlphaTile(byte[] container, int adtOffset, int tileX, int tileY, string sourcePath, out AlphaTileData? data)
    {
        data = null;

        if (!ReadMhdrField(container, adtOffset, 0x00, out int mcinRelativeOffset) || mcinRelativeOffset <= 0) return false;
        if (!ReadMhdrField(container, adtOffset, 0x04, out int mtexRelativeOffset) || mtexRelativeOffset <= 0) return false;
        if (!ReadMhdrField(container, adtOffset, 0x0C, out int mddfRelativeOffset)) return false;
        if (!ReadMhdrField(container, adtOffset, 0x14, out int modfRelativeOffset)) return false;

        int mhdrDataOffset = adtOffset + ChunkHeaderSize;

        byte[]? textureNames = ReadSubchunkPayload(container, mhdrDataOffset, mtexRelativeOffset);
        IReadOnlyList<string> textureNameList = ReadStringEntries(textureNames);
        IReadOnlyList<int> mcnkOffsets = ReadMcinOffsets(container, mhdrDataOffset, mcinRelativeOffset);

        string tilePath = $"{sourcePath}#alpha-tile({tileX},{tileY})";

        float[,] heightmap = new float[TileHeightmapSize, TileHeightmapSize];
        float[,,] alphaPack = new float[1024, 1024, 4];
        int[,,] texIds = new int[16, 16, 4];
        bool[,,] layerMask = new bool[16, 16, 4];
        bool[,] holes = new bool[16, 16];
        List<AlphaLiquidChunk> liquidChunks = [];
        List<AlphaModelPlacement> modelPlacements = [];
        List<AlphaWorldModelPlacement> worldModelPlacements = [];

        bool hasHeight = false, hasAlpha = false;

        foreach (int mcnkOffset in mcnkOffsets)
        {
            if (!TryParseMcnk(container, mcnkOffset, textureNameList,
                    heightmap, alphaPack, texIds, layerMask, holes, liquidChunks,
                    ref hasHeight, ref hasAlpha))
                continue;
        }

        byte[]? mddfData = ReadSubchunkPayload(container, mhdrDataOffset, mddfRelativeOffset);
        byte[]? modfData = ReadSubchunkPayload(container, mhdrDataOffset, modfRelativeOffset);

        if (mddfData is { Length: > 0 })
            ReadMddfPlacements(mddfData, textureNameList, modelPlacements);
        if (modfData is { Length: > 0 })
            ReadModfPlacements(modfData, textureNameList, worldModelPlacements);

        FillHeightmapGaps(heightmap);
        if (!hasHeight) return false;

        float[,,]? alphaPack256 = hasAlpha ? DownsampleAlphaPack(alphaPack) : null;

        data = new AlphaTileData(
            tilePath, heightmap,
            alphaPack256,
            texIds, layerMask, holes,
            textureNameList, modelPlacements, worldModelPlacements, liquidChunks);

        return true;
    }

    private static bool ReadMhdrField(byte[] container, int adtOffset, int fieldOffset, out int value)
    {
        value = 0;
        if (adtOffset + ChunkHeaderSize + fieldOffset + sizeof(int) > container.Length)
            return false;
        value = BitConverter.ToInt32(container, adtOffset + ChunkHeaderSize + fieldOffset);
        return true;
    }

    private static byte[]? ReadSubchunkPayload(byte[] container, int mhdrDataOffset, int relativeOffset)
    {
        if (relativeOffset <= 0) return null;
        int chunkOffset = mhdrDataOffset + relativeOffset;
        if (chunkOffset + ChunkHeaderSize > container.Length) return null;

        int chunkSize = BitConverter.ToInt32(container, chunkOffset + 4);
        if (chunkSize <= 0 || chunkOffset + ChunkHeaderSize + chunkSize > container.Length)
            return null;

        byte[] payload = new byte[chunkSize];
        Buffer.BlockCopy(container, chunkOffset + ChunkHeaderSize, payload, 0, chunkSize);
        return payload;
    }

    private static IReadOnlyList<int> ReadMcinOffsets(byte[] container, int mhdrDataOffset, int mcinRelativeOffset)
    {
        byte[]? mcinData = ReadSubchunkPayload(container, mhdrDataOffset, mcinRelativeOffset);
        if (mcinData is not { Length: >= McinEntrySize })
            return [];

        List<int> offsets = new(256);
        for (int i = 0; i < 256 && (i * McinEntrySize) + sizeof(int) <= mcinData.Length; i++)
            offsets.Add(BitConverter.ToInt32(mcinData, i * McinEntrySize));
        return offsets;
    }

    private static bool TryParseMcnk(byte[] container, int mcnkOffset,
        IReadOnlyList<string> textureNames,
        float[,] heightmap, float[,,] alphaPack, int[,,] texIds, bool[,,] layerMask, bool[,] holes,
        List<AlphaLiquidChunk> liquidChunks, ref bool hasHeight, ref bool hasAlpha)
    {
        if (mcnkOffset + ChunkHeaderSize + McnkHeaderSize > container.Length) return false;
        int chunkSize = BitConverter.ToInt32(container, mcnkOffset + 4);
        if (chunkSize <= 0) return false;

        int headerOffset = mcnkOffset + ChunkHeaderSize;
        uint flags = BitConverter.ToUInt32(container, headerOffset + 0x00);
        int indexX = BitConverter.ToInt32(container, headerOffset + 0x04);
        int indexY = BitConverter.ToInt32(container, headerOffset + 0x08);
        int layerCount = BitConverter.ToInt32(container, headerOffset + 0x10);
        ushort holeMask = (ushort)BitConverter.ToUInt32(container, headerOffset + 0x40);
        int mcvtRel = BitConverter.ToInt32(container, headerOffset + 0x18);
        int mclyRel = BitConverter.ToInt32(container, headerOffset + 0x20);
        int mcalRel = BitConverter.ToInt32(container, headerOffset + 0x28);
        int mcalSize = BitConverter.ToInt32(container, headerOffset + 0x2C);
        int mclqRel = BitConverter.ToInt32(container, headerOffset + 0x64);
        int mcnkChunksSize = BitConverter.ToInt32(container, headerOffset + 0x5C);

        if ((uint)indexX >= 16 || (uint)indexY >= 16) return true;

        int cx = indexX, cy = indexY;
        int chunkDataBase = mcnkOffset + ChunkHeaderSize + McnkHeaderSize;

        if (mcvtRel >= 0 && chunkDataBase + mcvtRel + AlphaMcvtSize <= container.Length)
        {
            float[] heights = new float[145];
            int mcvtOffset = chunkDataBase + mcvtRel;
            int dst = 0;
            for (int row = 0; row < 17; row++)
            {
                if ((row & 1) == 0)
                {
                    int outerRow = row / 2;
                    for (int col = 0; col < 9; col++)
                        heights[dst++] = BitConverter.ToSingle(container, mcvtOffset + ((outerRow * 9 + col) * 4));
                }
                else
                {
                    int innerRow = row / 2;
                    for (int col = 0; col < 8; col++)
                        heights[dst++] = BitConverter.ToSingle(container, mcvtOffset + ((81 + innerRow * 8 + col) * 4));
                }
            }

            float baseHeight = ReadAlphaBaseHeight(container, headerOffset);
            if (!float.IsNaN(baseHeight) && MathF.Abs(baseHeight) <= 50000f && baseHeight != 0f)
            {
                for (int i = 0; i < heights.Length; i++)
                    heights[i] += baseHeight;
            }

            // Coordinate mapping matches AlphaTerrainAdapter exactly:
            // worldX = MapOrigin - tileX*ChunkSize - chunkY*chunkSmall
            // worldY = MapOrigin - tileY*ChunkSize - chunkX*chunkSmall
            // chunkY drives horizontal (X), chunkX drives vertical (Y)
            int baseX = cy * HalfStepsPerChunk;
            int baseY = cx * HalfStepsPerChunk;
            for (int i = 0; i < heights.Length; i++)
            {
                GetVertexPosition(i, out int row, out int col, out bool isInner);
                int sampleX = isInner ? (col * 2) + 1 : col * 2;
                int sampleY = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;
                int px = baseX + sampleX;
                int py = baseY + sampleY;
                if ((uint)px < TileHeightmapSize && (uint)py < TileHeightmapSize)
                    heightmap[py, px] = heights[i];
            }
            hasHeight = true;
        }

        if (layerCount > 0 && mclyRel >= 0)
        {
            int mclyOffset = chunkDataBase + mclyRel;
            int maxLayers = Math.Min(layerCount, 4);

            byte[] mcalData = [];
            if (mcalRel >= 0 && mcalSize > 0 && chunkDataBase + mcalRel + mcalSize <= container.Length)
            {
                mcalData = new byte[mcalSize];
                Buffer.BlockCopy(container, chunkDataBase + mcalRel, mcalData, 0, mcalSize);
            }

            int alphaSrcOffset = 0;
            for (int l = 0; l < maxLayers; l++)
            {
                int entryOff = mclyOffset + l * MclyEntrySize;
                if (entryOff + MclyEntrySize > container.Length) break;

                uint texId = BitConverter.ToUInt32(container, entryOff);
                uint mclyFlags = BitConverter.ToUInt32(container, entryOff + 4);
                uint layerAlphaOff = BitConverter.ToUInt32(container, entryOff + 8);

                texIds[cx, cy, l] = (int)texId;
                layerMask[cx, cy, l] = true;

                if (l > 0 && alphaSrcOffset < mcalData.Length)
                {
                    bool bigAlpha = (mclyFlags & 0x200) != 0;
                    int expected = bigAlpha ? 4096 : 2048;
                    int consume = Math.Min(expected, mcalData.Length - alphaSrcOffset);

                    if (!bigAlpha)
                    {
                        int packed = Math.Min(2048, consume);
                        for (int i = 0; i < packed && alphaSrcOffset + i < mcalData.Length; i++)
                        {
                            byte b = mcalData[alphaSrcOffset + i];
                            int ax = (i * 2) % 64;
                            int ay = (i * 2) / 64;
                            alphaPack[cx * 64 + ay, cy * 64 + ax, l] = ((b & 0x0F) * 17) / 255f;
                            if (ax + 1 < 64)
                                alphaPack[cx * 64 + ay, cy * 64 + ax + 1, l] = ((b >> 4) * 17) / 255f;
                        }
                        ApplyEdgeFix(alphaPack, cx, cy, l);
                        alphaSrcOffset += packed;
                    }
                    else
                    {
                        for (int i = 0; i < 4096 && alphaSrcOffset + i < mcalData.Length; i++)
                        {
                            int ax = i % 64;
                            int ay = i / 64;
                            alphaPack[cx * 64 + ay, cy * 64 + ax, l] = mcalData[alphaSrcOffset + i] / 255f;
                        }
                        alphaSrcOffset += Math.Min(4096, consume);
                    }
                    hasAlpha = true;
                }
            }
        }

        holes[cx, cy] = holeMask != 0;

        if ((flags & 0x3Cu) != 0 && mclqRel > 0 && mcnkChunksSize > mclqRel)
        {
            int mclqPayloadOffset = chunkDataBase + mclqRel;
            int mclqPayloadSize = mcnkChunksSize - mclqRel;
            if (mclqPayloadOffset >= 0 && mclqPayloadSize >= 8 && mclqPayloadOffset + mclqPayloadSize <= container.Length)
            {
                byte[] mclqPayload = new byte[mclqPayloadSize];
                Buffer.BlockCopy(container, mclqPayloadOffset, mclqPayload, 0, mclqPayloadSize);
                mclqPayload = StripMclqHeader(mclqPayload);
                if (mclqPayload.Length >= 8)
                {
                    float minH = BitConverter.ToSingle(mclqPayload, 0);
                    float maxH = BitConverter.ToSingle(mclqPayload, 4);
                    float baseH = ReadAlphaBaseHeight(container, headerOffset);
                    if (!float.IsNaN(baseH) && MathF.Abs(baseH) <= 50000f)
                    {
                        minH += baseH;
                        maxH += baseH;
                    }

                    byte[]? tileFlags = null;
                    if (mclqPayload.Length >= 0x290 + 64)
                    {
                        tileFlags = new byte[64];
                        Buffer.BlockCopy(mclqPayload, 0x290, tileFlags, 0, 64);
                    }

                    liquidChunks.Add(new AlphaLiquidChunk(
                        cy * 16 + cx, cx, cy, minH, maxH, tileFlags, flags));
                }
            }
        }

        return true;
    }

    private static void ApplyEdgeFix(float[,,] alpha, int cx, int cy, int layer)
    {
        int baseY = cx * 64;
        int baseX = cy * 64;
        for (int row = 0; row < 64; row++)
            alpha[baseY + row, baseX + 63, layer] = alpha[baseY + row, baseX + 62, layer];
        for (int col = 0; col < 64; col++)
            alpha[baseY + 63, baseX + col, layer] = alpha[baseY + 62, baseX + col, layer];
    }

    private static float ReadAlphaBaseHeight(byte[] container, int headerOffset)
    {
        if (headerOffset < 0 || headerOffset + 0x68 + sizeof(float) > container.Length)
            return 0f;
        // Matches AlphaTerrainAdapter: Unused1 at offset 0x68 holds the base height float
        return BitConverter.Int32BitsToSingle(BitConverter.ToInt32(container, headerOffset + 0x68));
    }

    private static byte[] StripMclqHeader(byte[] payload)
    {
        if (payload.Length < 8) return payload;
        uint id = BitConverter.ToUInt32(payload, 0);
        bool isMclq = id == 0x514C434D || id == 0x4D434C51;
        if (!isMclq) return payload;

        uint size = BitConverter.ToUInt32(payload, 4);
        if (size == 0 || payload.Length < 8 + size) return payload;

        byte[] stripped = new byte[size];
        Buffer.BlockCopy(payload, 8, stripped, 0, (int)size);
        return stripped;
    }

    private static void GetVertexPosition(int index, out int row, out int col, out bool isInner)
    {
        int remaining = index;
        for (int r = 0; r < 17; r++)
        {
            int rowSize = (r & 1) == 0 ? 9 : 8;
            if (remaining < rowSize)
            {
                row = r;
                col = remaining;
                isInner = (r & 1) != 0;
                return;
            }
            remaining -= rowSize;
        }
        row = 0; col = 0; isInner = false;
    }

    private static void FillHeightmapGaps(float[,] hm)
    {
        for (int y = 0; y < TileHeightmapSize; y++)
        {
            for (int x = 0; x < TileHeightmapSize; x++)
            {
                if (hm[y, x] != 0f) continue;
                if (x > 0 && hm[y, x - 1] != 0f) hm[y, x] = hm[y, x - 1];
                else if (y > 0 && hm[y - 1, x] != 0f) hm[y, x] = hm[y - 1, x];
                else if (x < TileHeightmapSize - 1 && hm[y, x + 1] != 0f) hm[y, x] = hm[y, x + 1];
                else if (y < TileHeightmapSize - 1 && hm[y + 1, x] != 0f) hm[y, x] = hm[y + 1, x];
            }
        }
    }

    private static void ReadMddfPlacements(byte[] payload, IReadOnlyList<string> textureNames, List<AlphaModelPlacement> placements)
    {
        for (int offset = 0; offset + MddfEntrySize <= payload.Length; offset += MddfEntrySize)
        {
            int nameId = BitConverter.ToInt32(payload, offset);
            int uniqueId = BitConverter.ToInt32(payload, offset + 4);
            float rawX = BitConverter.ToSingle(payload, offset + 8);
            float rawZ = BitConverter.ToSingle(payload, offset + 12);
            float rawY = BitConverter.ToSingle(payload, offset + 16);
            float rotX = BitConverter.ToSingle(payload, offset + 20);
            float rotZ = BitConverter.ToSingle(payload, offset + 24);
            float rotY = BitConverter.ToSingle(payload, offset + 28);
            ushort scale = BitConverter.ToUInt16(payload, offset + 32);

            placements.Add(new AlphaModelPlacement(
                nameId,
                ResolveName(textureNames, nameId),
                uniqueId,
                new Vector3(MapOrigin - rawY, MapOrigin - rawX, rawZ),
                new Vector3(rotX, rotY, rotZ),
                scale / 1024f));
        }
    }

    private static void ReadModfPlacements(byte[] payload, IReadOnlyList<string> textureNames, List<AlphaWorldModelPlacement> placements)
    {
        for (int offset = 0; offset + ModfEntrySize <= payload.Length; offset += ModfEntrySize)
        {
            int nameId = BitConverter.ToInt32(payload, offset);
            int uniqueId = BitConverter.ToInt32(payload, offset + 4);
            float rawX = BitConverter.ToSingle(payload, offset + 8);
            float rawZ = BitConverter.ToSingle(payload, offset + 12);
            float rawY = BitConverter.ToSingle(payload, offset + 16);
            float rotX = BitConverter.ToSingle(payload, offset + 20);
            float rotZ = BitConverter.ToSingle(payload, offset + 24);
            float rotY = BitConverter.ToSingle(payload, offset + 28);
            float bbMinX = BitConverter.ToSingle(payload, offset + 32);
            float bbMinZ = BitConverter.ToSingle(payload, offset + 36);
            float bbMinY = BitConverter.ToSingle(payload, offset + 40);
            float bbMaxX = BitConverter.ToSingle(payload, offset + 44);
            float bbMaxZ = BitConverter.ToSingle(payload, offset + 48);
            float bbMaxY = BitConverter.ToSingle(payload, offset + 52);
            ushort flags = BitConverter.ToUInt16(payload, offset + 56);

            placements.Add(new AlphaWorldModelPlacement(
                nameId,
                ResolveName(textureNames, nameId),
                uniqueId,
                new Vector3(MapOrigin - rawY, MapOrigin - rawX, rawZ),
                new Vector3(rotX, rotY, rotZ),
                new Vector3(MapOrigin - bbMaxY, MapOrigin - bbMaxX, bbMinZ),
                new Vector3(MapOrigin - bbMinY, MapOrigin - bbMinX, bbMaxZ),
                flags));
        }
    }

    private static string ResolveName(IReadOnlyList<string> names, int index)
    {
        return index >= 0 && index < names.Count ? names[index] : $"unknown_{index}";
    }

    private static IReadOnlyList<string> ReadStringEntries(byte[]? payload)
    {
        if (payload is not { Length: > 0 }) return [];
        List<string> entries = [];
        int start = 0;
        for (int i = 0; i < payload.Length; i++)
        {
            if (payload[i] != 0) continue;
            if (i > start) entries.Add(Encoding.UTF8.GetString(payload, start, i - start));
            start = i + 1;
        }
        if (start < payload.Length)
            entries.Add(Encoding.UTF8.GetString(payload, start, payload.Length - start));
        return entries;
    }

    private static float[,,] DownsampleAlphaPack(float[,,] src)
    {
        const int srcSize = 1024;
        const int dstSize = 256;
        const int ratio = srcSize / dstSize; // 4
        const int samples = ratio * ratio; // 16
        var dst = new float[dstSize, dstSize, 4];

        for (int y = 0; y < dstSize; y++)
        {
            for (int x = 0; x < dstSize; x++)
            {
                for (int l = 0; l < 4; l++)
                {
                    float sum = 0f;
                    for (int dy = 0; dy < ratio; dy++)
                    {
                        for (int dx = 0; dx < ratio; dx++)
                        {
                            sum += src[y * ratio + dy, x * ratio + dx, l];
                        }
                    }
                    dst[y, x, l] = sum / samples;
                }
            }
        }
        return dst;
    }

    public static float[,,] SynthesizeNormals(float[,] heights)
    {
        int h = heights.GetLength(0);
        int w = heights.GetLength(1);
        var normals = new float[h, w, 3];
        float step = WorldTileSize / (TileHeightmapSize - 1);

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float dx, dy;
                if (x == 0) dx = (heights[y, x + 1] - heights[y, x]) / step;
                else if (x == w - 1) dx = (heights[y, x] - heights[y, x - 1]) / step;
                else dx = (heights[y, x + 1] - heights[y, x - 1]) / (2f * step);

                if (y == 0) dy = (heights[y + 1, x] - heights[y, x]) / step;
                else if (y == h - 1) dy = (heights[y, x] - heights[y - 1, x]) / step;
                else dy = (heights[y + 1, x] - heights[y - 1, x]) / (2f * step);

                float nx = -dx;
                float ny = dy;
                float nz = 1f;
                float len = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
                if (len > 0)
                {
                    normals[y, x, 0] = nx / len;
                    normals[y, x, 1] = ny / len;
                    normals[y, x, 2] = nz / len;
                }
                else
                {
                    normals[y, x, 0] = 0f;
                    normals[y, x, 1] = 0f;
                    normals[y, x, 2] = 1f;
                }
            }
        }

        return normals;
    }

    private const float WorldTileSize = 533.33333f;
}
