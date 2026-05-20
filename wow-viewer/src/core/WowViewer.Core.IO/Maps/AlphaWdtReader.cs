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
    private const float MapOrigin = 17066.66666f;

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
        return TryReadTileInternal(wdtData, tileX, tileY, "memory", out data);
    }

    public static bool TryReadTile(byte[] wdtData, int tileX, int tileY, string sourcePath, out AlphaTileData? data)
    {
        if (string.IsNullOrWhiteSpace(sourcePath))
            sourcePath = "memory";
        return TryReadTileInternal(wdtData, tileX, tileY, sourcePath, out data);
    }

    public static bool IsAlphaWdt(byte[] wdtData)
    {
        using var ms = new MemoryStream(wdtData, writable: false);
        if (!TryReadAlphaTopLevelChunks(ms, out _, out _, out MapChunkLocation main, out _))
            return false;
        return main.Size == 65536;
    }

    public static HashSet<(int X, int Y)> ReadExistingTiles(byte[] wdtData)
    {
        var tiles = new HashSet<(int, int)>();
        if (!ReadMainPayload(wdtData, out byte[] mainData) || mainData.Length == 0)
            return tiles;

        int cellSize = 16;
        for (int i = 0; i < 64 * 64; i++)
        {
            int off = i * cellSize;
            if (off + 4 > mainData.Length) break;
            uint val = BitConverter.ToUInt32(mainData.AsSpan(off, 4));
            if (val != 0)
            {
                int x = i % 64;
                int y = i / 64;
                tiles.Add((x, y));
            }
        }

        return tiles;
    }

    private static bool ReadMainPayload(byte[] wdtData, out byte[] mainData)
    {
        mainData = [];
        using var ms = new MemoryStream(wdtData, writable: false);
        return TryReadAlphaTopLevelChunks(ms, out _, out _, out MapChunkLocation main, out _)
            && main.Size > 0
            && TryReadPayloadAt(ms, main, out mainData);
    }

    private static (IReadOnlyList<string> mdxNames, IReadOnlyList<string> wmoNames) ReadModelNameTables(byte[] wdtData)
    {
        using var ms = new MemoryStream(wdtData, writable: false);
        if (!TryReadAlphaTopLevelChunks(ms, out _, out MapChunkLocation mphd, out _, out _))
            return ([], []);

        if (mphd.Size < 16)
            return ([], []);

        byte[] mphdData = new byte[16];
        ms.Position = mphd.DataOffset;
        ms.ReadExactly(mphdData);

        IReadOnlyList<string> mdxNames = [];
        IReadOnlyList<string> wmoNames = [];

        int mdnmOff = BitConverter.ToInt32(mphdData, 4);
        if (mdnmOff > 0)
        {
            if (TryReadChunk(ms, mdnmOff, out MapChunkLocation c) && c.Id == MapChunkIds.Mdnm && TryReadPayloadAt(ms, c, out byte[] mdnmPayload))
                mdxNames = ReadStringEntries(mdnmPayload);
        }

        int monmOff = BitConverter.ToInt32(mphdData, 12);
        if (monmOff > 0)
        {
            if (TryReadChunk(ms, monmOff, out MapChunkLocation c) && c.Id == MapChunkIds.Monm && TryReadPayloadAt(ms, c, out byte[] monmPayload))
                wmoNames = ReadStringEntries(monmPayload);
        }

        return (mdxNames, wmoNames);
    }

    private static bool TryReadAlphaTopLevelChunks(
        MemoryStream ms,
        out MapChunkLocation mver,
        out MapChunkLocation mphd,
        out MapChunkLocation main,
        out bool hasMdnm)
    {
        mver = default; mphd = default; main = default; hasMdnm = false;

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
            if (mdnmOff > 0 && TryReadChunk(ms, mdnmOff, out MapChunkLocation c) && c.Id == MapChunkIds.Mdnm)
                hasMdnm = true;
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

    private static bool TryReadTileInternal(byte[] wdtData, int tileX, int tileY, string sourcePath, out AlphaTileData? data)
    {
        data = null;

        if (!ReadMainPayload(wdtData, out byte[] mainData))
            return false;

        int mainEntryIndex = tileY * 64 + tileX;
        int entryOffset = mainEntryIndex * MainEntrySize;
        if (entryOffset < 0 || entryOffset + sizeof(int) > mainData.Length)
            return false;

        int adtOffset = BitConverter.ToInt32(mainData, entryOffset);
        if (adtOffset <= 0)
            return false;

        var (mdxNames, wmoNames) = ReadModelNameTables(wdtData);

        return ReadAlphaTile(wdtData, adtOffset, tileX, tileY, sourcePath, out data, mdxNames, wmoNames);
    }

    private static bool ReadAlphaTile(byte[] container, int adtOffset, int tileX, int tileY, string sourcePath, out AlphaTileData? data,
        IReadOnlyList<string> mdxNames, IReadOnlyList<string> wmoNames)
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
        IReadOnlyList<TerrainRawChunkBlob> rawChunks = CollectRawTileChunks(container, adtOffset, mhdrDataOffset, mcinRelativeOffset, mtexRelativeOffset, mddfRelativeOffset, modfRelativeOffset, mcnkOffsets, tilePath);

        float[,] heightmap = new float[TileHeightmapSize, TileHeightmapSize];
        float[,,] alphaPack = new float[1024, 1024, 4];
        float[,,] normalXyz = new float[TileHeightmapSize, TileHeightmapSize, 3];
        float[,] alphaPackShadow = new float[1024, 1024];
        int[,,] texIds = new int[16, 16, 4];
        bool[,,] layerMask = new bool[16, 16, 4];
        bool[,] holes = new bool[16, 16];
        List<AlphaLiquidChunk> liquidChunks = [];
        List<AlphaModelPlacement> modelPlacements = [];
        List<AlphaWorldModelPlacement> worldModelPlacements = [];

        bool hasHeight = false, hasAlpha = false, hasNormals = false, hasShadow = false;
        int totalMcshBytes = 0;
        int activeChunkCount = 0;

        foreach (int mcnkOffset in mcnkOffsets)
        {
            if (mcnkOffset <= 0) continue;
            activeChunkCount++;

            if (!TryParseMcnk(container, mcnkOffset, textureNameList,
                    heightmap, alphaPack, normalXyz, alphaPackShadow, texIds, layerMask, holes, liquidChunks,
                    ref hasHeight, ref hasAlpha, ref hasNormals, ref hasShadow, ref totalMcshBytes))
                continue;
        }

        byte[]? mddfData = ReadSubchunkPayload(container, mhdrDataOffset, mddfRelativeOffset);
        byte[]? modfData = ReadSubchunkPayload(container, mhdrDataOffset, modfRelativeOffset);

        if (mddfData is { Length: > 0 })
            ReadMddfPlacements(mddfData, mdxNames, modelPlacements);
        if (modfData is { Length: > 0 })
            ReadModfPlacements(modfData, wmoNames, worldModelPlacements);

        FillHeightmapGaps(heightmap);
        if (!hasHeight) return false;

        float[,,]? alphaPack256 = hasAlpha ? DownsampleAlphaPack(alphaPack) : null;
        float[,]? mcshShadowMask256 = hasShadow ? DownsampleShadowMask(alphaPackShadow) : null;

        float[,] mclqSurface = new float[TileHeightmapSize, TileHeightmapSize];
        int[,] mclqTypes = new int[16, 16];
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++)
                mclqTypes[y, x] = -1;
        bool hasLiquid = false;
        foreach (var lc in liquidChunks)
        {
            if ((uint)lc.IndexX < 16 && (uint)lc.IndexY < 16)
            {
                float avgHeight = (lc.MinHeight + lc.MaxHeight) * 0.5f;
                int baseX = lc.IndexX * 16;
                int baseY = lc.IndexY * 16;
                int endX = Math.Min(baseX + 17, TileHeightmapSize);
                int endY = Math.Min(baseY + 17, TileHeightmapSize);
                for (int y = baseY; y < endY; y++)
                    for (int x = baseX; x < endX; x++)
                        mclqSurface[y, x] = avgHeight;
                mclqTypes[lc.IndexY, lc.IndexX] = ClassifyLiquid(lc.McnkFlags);
                hasLiquid = true;
            }
        }

        bool hasSparseChunks = activeChunkCount < 256;
        bool mcshSunUpperRight = DetectMcshUpperRight(alphaPackShadow);

        var diagnostics = new AlphaTileDiagnostics(false, hasSparseChunks, 0, activeChunkCount, mcshSunUpperRight, totalMcshBytes);

        data = new AlphaTileData(
            tilePath, heightmap,
            alphaPack256,
            texIds, layerMask, holes,
            textureNameList, modelPlacements, worldModelPlacements, liquidChunks,
            diagnostics: diagnostics,
            mcnrNormalXyz: hasNormals ? normalXyz : null,
            mcshShadowMask256: mcshShadowMask256,
            mclqSurfaceHeight: hasLiquid ? mclqSurface : null,
            mclqTypeMask: hasLiquid ? mclqTypes : null,
            mcshShadowMask1024: hasShadow ? alphaPackShadow : null,
            rawChunks: rawChunks);

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

    private static IReadOnlyList<TerrainRawChunkBlob> CollectRawTileChunks(
        byte[] container,
        int adtOffset,
        int mhdrDataOffset,
        int mcinRelativeOffset,
        int mtexRelativeOffset,
        int mddfRelativeOffset,
        int modfRelativeOffset,
        IReadOnlyList<int> mcnkOffsets,
        string sourcePath)
    {
        List<TerrainRawChunkBlob> rawChunks = [];
        Dictionary<string, int> topCounts = new(StringComparer.OrdinalIgnoreCase);

        AddRawChunkAtAbsoluteOffset(rawChunks, topCounts, container, adtOffset, sourcePath, "alpha", "top-level");
        AddRawChunkAtRelativeOffset(rawChunks, topCounts, container, mhdrDataOffset, mcinRelativeOffset, sourcePath, "alpha", "top-level");
        AddRawChunkAtRelativeOffset(rawChunks, topCounts, container, mhdrDataOffset, mtexRelativeOffset, sourcePath, "alpha", "top-level");
        AddRawChunkAtRelativeOffset(rawChunks, topCounts, container, mhdrDataOffset, mddfRelativeOffset, sourcePath, "alpha", "top-level");
        AddRawChunkAtRelativeOffset(rawChunks, topCounts, container, mhdrDataOffset, modfRelativeOffset, sourcePath, "alpha", "top-level");

        for (int chunkIndex = 0; chunkIndex < mcnkOffsets.Count; chunkIndex++)
        {
            int offset = mcnkOffsets[chunkIndex];
            if (offset <= 0)
                continue;

            int chunkX = chunkIndex % 16;
            int chunkY = chunkIndex / 16;
            if (!TryReadEmbeddedChunkPayload(container, offset, out FourCC chunkId, out byte[] payload) || payload.Length == 0)
                continue;

            rawChunks.Add(new TerrainRawChunkBlob
            {
                EntryName = $"raw_chunks/alpha/mcnk_{chunkX:D2}_{chunkY:D2}/{chunkId}_000",
                SourceKind = "alpha",
                SourcePath = sourcePath,
                Scope = "mcnk",
                ChunkId = chunkId.ToString(),
                ChunkIndex = chunkIndex,
                ChunkX = chunkX,
                ChunkY = chunkY,
                Data = payload,
            });
        }

        return rawChunks;
    }

    private static void AddRawChunkAtRelativeOffset(
        List<TerrainRawChunkBlob> rawChunks,
        Dictionary<string, int> counts,
        byte[] container,
        int mhdrDataOffset,
        int relativeOffset,
        string sourcePath,
        string sourceKind,
        string scope)
    {
        if (relativeOffset <= 0)
            return;

        AddRawChunkAtAbsoluteOffset(rawChunks, counts, container, mhdrDataOffset + relativeOffset, sourcePath, sourceKind, scope);
    }

    private static void AddRawChunkAtAbsoluteOffset(
        List<TerrainRawChunkBlob> rawChunks,
        Dictionary<string, int> counts,
        byte[] container,
        int chunkOffset,
        string sourcePath,
        string sourceKind,
        string scope)
    {
        if (!TryReadEmbeddedChunkPayload(container, chunkOffset, out FourCC chunkId, out byte[] payload) || payload.Length == 0)
            return;

        string chunkName = chunkId.ToString();
        int occurrence = counts.TryGetValue(chunkName, out int count) ? count : 0;
        counts[chunkName] = occurrence + 1;

        rawChunks.Add(new TerrainRawChunkBlob
        {
            EntryName = $"raw_chunks/{sourceKind}/top/{chunkName}_{occurrence:D3}",
            SourceKind = sourceKind,
            SourcePath = sourcePath,
            Scope = scope,
            ChunkId = chunkName,
            Data = payload,
        });
    }

    private static bool TryReadEmbeddedChunkPayload(byte[] container, int chunkOffset, out FourCC id, out byte[] payload)
    {
        id = default;
        payload = [];
        if (chunkOffset < 0 || chunkOffset + ChunkHeaderSize > container.Length)
            return false;

        id = FourCC.FromFileBytes(container.AsSpan(chunkOffset, 4));
        int chunkSize = BitConverter.ToInt32(container, chunkOffset + 4);
        if (chunkSize <= 0 || chunkOffset + ChunkHeaderSize + chunkSize > container.Length)
            return false;

        payload = new byte[chunkSize];
        Buffer.BlockCopy(container, chunkOffset + ChunkHeaderSize, payload, 0, chunkSize);
        return true;
    }

    private static bool TryParseMcnk(byte[] container, int mcnkOffset,
        IReadOnlyList<string> textureNames,
        float[,] heightmap, float[,,] alphaPack, float[,,] normalXyz, float[,] alphaPackShadow,
        int[,,] texIds, bool[,,] layerMask, bool[,] holes,
        List<AlphaLiquidChunk> liquidChunks, ref bool hasHeight, ref bool hasAlpha,
        ref bool hasNormals, ref bool hasShadow, ref int totalMcshBytes)
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
        int mcnrRel = BitConverter.ToInt32(container, headerOffset + 0x1C);
        int mclyRel = BitConverter.ToInt32(container, headerOffset + 0x20);
        int mcalRel = BitConverter.ToInt32(container, headerOffset + 0x28);
        int mcalSize = BitConverter.ToInt32(container, headerOffset + 0x2C);
        int mcshRel = BitConverter.ToInt32(container, headerOffset + 0x30);
        int mcshSize = BitConverter.ToInt32(container, headerOffset + 0x34);
        int mclqRel = BitConverter.ToInt32(container, headerOffset + 0x64);
        int mcnkChunksSize = BitConverter.ToInt32(container, headerOffset + 0x5C);
        int mccvRel = BitConverter.ToInt32(container, headerOffset + 0x74);

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

            // Ghidra-verified (CMapChunk::CreateVertices, 0.5.3.3368):
            // Alpha MCVT heights are ABSOLUTE world-space Z values — no base height addition.
            // The MCNK header field at offset 0x80 (Unused1) stores the chunk's Position.Z
            // which the client uses for bounding-box math and vertex relativization, NOT as
            // an additive base for the heights. Adding it to absolute heights was the bug
            // that caused each chunk to float at a disconnected elevation.

            // chunkY (=indexY) drives row (Y), chunkX (=indexX) drives column (X)
            int baseX = cx * HalfStepsPerChunk;
            int baseY = cy * HalfStepsPerChunk;
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

        if (mcnrRel >= 0 && chunkDataBase + mcnrRel + 435 <= container.Length)
        {
            int mcnrOffset = chunkDataBase + mcnrRel;
            int normBaseX = cx * HalfStepsPerChunk;
            int normBaseY = cy * HalfStepsPerChunk;
            int idx = 0;
            for (int row = 0; row < 17; row++)
            {
                bool isInner = (row & 1) != 0;
                int cols = isInner ? 8 : 9;
                for (int col = 0; col < cols; col++)
                {
                    int srcIdx;
                    if (isInner)
                        srcIdx = (81 + (row / 2) * 8 + col) * 3;
                    else
                        srcIdx = ((row / 2) * 9 + col) * 3;

                    if (srcIdx + 2 < 435 && mcnrOffset + srcIdx + 2 < container.Length)
                    {
                        float nx = Math.Clamp((sbyte)container[mcnrOffset + srcIdx] / 127f, -1f, 1f);
                        float nz = Math.Clamp((sbyte)container[mcnrOffset + srcIdx + 1] / 127f, -1f, 1f);
                        float ny = Math.Clamp((sbyte)container[mcnrOffset + srcIdx + 2] / 127f, -1f, 1f);

                        int sampleX = isInner ? (col * 2) + 1 : col * 2;
                        int sampleY = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;
                        int px = normBaseX + sampleX;
                        int py = normBaseY + sampleY;
                        if ((uint)px < TileHeightmapSize && (uint)py < TileHeightmapSize)
                        {
                            normalXyz[py, px, 0] = nx;
                            normalXyz[py, px, 1] = ny;
                            normalXyz[py, px, 2] = nz;
                        }
                    }
                    idx++;
                }
            }
            hasNormals = true;
        }

        if (layerCount > 0 && mclyRel >= 0)
        {
            int mclyOffset = chunkDataBase + mclyRel;
            if (mclyOffset + 8 <= container.Length)
                mclyOffset += 8;
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
                            alphaPack[cy * 64 + ay, cx * 64 + ax, l] = ((b & 0x0F) * 17) / 255f;
                            if (ax + 1 < 64)
                                alphaPack[cy * 64 + ay, cx * 64 + ax + 1, l] = ((b >> 4) * 17) / 255f;
                        }
                        ApplyEdgeFix(alphaPack, cy, cx, l);
                        alphaSrcOffset += packed;
                    }
                    else
                    {
                        for (int i = 0; i < 4096 && alphaSrcOffset + i < mcalData.Length; i++)
                        {
                            int ax = i % 64;
                            int ay = i / 64;
                            alphaPack[cy * 64 + ay, cx * 64 + ax, l] = mcalData[alphaSrcOffset + i] / 255f;
                        }
                        alphaSrcOffset += Math.Min(4096, consume);
                    }
                    hasAlpha = true;
                }
            }
        }

        holes[cx, cy] = holeMask != 0;

        if (mcshRel >= 0 && mcshSize > 0 && chunkDataBase + mcshRel + mcshSize <= container.Length)
        {
            const int shadowChunkSize = 64;
            int mcshOffset = chunkDataBase + mcshRel;
            int mcshBytes = Math.Min(mcshSize, 512);
            if (mcshOffset + mcshBytes <= container.Length)
            {
                int shadowBaseX = cx * shadowChunkSize;
                int shadowBaseY = cy * shadowChunkSize;
                int rows = Math.Min(shadowChunkSize, mcshBytes / 8);
                for (int y = 0; y < rows; y++)
                {
                    for (int intIdx = 0; intIdx < 8; intIdx++)
                    {
                        int srcIdx = y * 8 + intIdx;
                        if (srcIdx >= mcshBytes || mcshOffset + srcIdx >= container.Length) break;
                        byte bits = container[mcshOffset + srcIdx];
                        for (int bit = 0; bit < 8; bit++)
                        {
                            int sx = shadowBaseX + intIdx * 8 + bit;
                            int sy = shadowBaseY + y;
                            if (sx < 1024 && sy < 1024)
                                alphaPackShadow[sy, sx] = ((bits >> bit) & 1) == 1 ? 1.0f : 0.0f;
                        }
                    }
                }
                hasShadow = true;
                totalMcshBytes += mcshSize;
            }
        }

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

                    byte[]? tileFlags = null;
                    if (mclqPayload.Length >= 0x290 + 64)
                    {
                        tileFlags = new byte[64];
                        Buffer.BlockCopy(mclqPayload, 0x290, tileFlags, 0, 64);
                    }

                    float[]? heights = null;
                    if (mclqPayload.Length >= 8 + (81 * 8))
                    {
                        heights = new float[81];
                        for (int index = 0; index < heights.Length; index++)
                            heights[index] = BitConverter.ToSingle(mclqPayload, 8 + (index * 8) + 4);
                    }

                    liquidChunks.Add(new AlphaLiquidChunk(
                        cy * 16 + cx, cx, cy, minH, maxH, tileFlags, flags, heights));
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
        // With absolute world-space heights, 0.0f is a valid height (sea level).
        // Use NaN as the sentinel for "unset" instead of 0.0f to avoid corrupting
        // legitimate sea-level terrain. The heightmap is pre-initialized to 0f,
        // so first mark all truly-unset positions (where no chunk wrote data) as NaN.
        // Chunks only write to GridX/GridY positions that land on their 17×9/8 grid,
        // so inner-sub-cell positions at chunk boundaries that are covered by both
        // adjacent chunks get written twice (same value), leaving no actual gaps.
        // The real gaps are at positions where NO chunk wrote data (sparse tiles).
        // Since we can't distinguish "wrote 0.0f" from "never wrote", we rely on
        // the fact that absolute heights in typical alpha terrain are never exactly
        // 0.0f at a grid vertex, and the simpler scan-fill below handles the case
        // where a heightmap position was never filled by any chunk.
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

    private static void ReadMddfPlacements(byte[] payload, IReadOnlyList<string> modelNames, List<AlphaModelPlacement> placements)
    {
        for (int offset = 0; offset + MddfEntrySize <= payload.Length; offset += MddfEntrySize)
        {
            int nameId = BitConverter.ToInt32(payload, offset);
            int uniqueId = BitConverter.ToInt32(payload, offset + 0x04);
            float filePosX = BitConverter.ToSingle(payload, offset + 0x08);
            float filePosY = BitConverter.ToSingle(payload, offset + 0x0C);
            float filePosZ = BitConverter.ToSingle(payload, offset + 0x10);
            float fileRotX = BitConverter.ToSingle(payload, offset + 0x14);
            float fileRotY = BitConverter.ToSingle(payload, offset + 0x18);
            float fileRotZ = BitConverter.ToSingle(payload, offset + 0x1C);
            ushort scale = BitConverter.ToUInt16(payload, offset + 0x20);

            float rendererX = MapOrigin - filePosZ;
            float rendererY = MapOrigin - filePosX;
            float rendererZ = filePosY;
            placements.Add(new AlphaModelPlacement(
                nameId,
                ResolveName(modelNames, nameId),
                uniqueId,
                new Vector3(rendererX, rendererY, rendererZ),
                new Vector3(fileRotX, fileRotZ, fileRotY),
                scale / 1024f));
        }
    }

    private static void ReadModfPlacements(byte[] payload, IReadOnlyList<string> modelNames, List<AlphaWorldModelPlacement> placements)
    {
        for (int offset = 0; offset + ModfEntrySize <= payload.Length; offset += ModfEntrySize)
        {
            int nameId = BitConverter.ToInt32(payload, offset);
            int uniqueId = BitConverter.ToInt32(payload, offset + 0x04);
            float filePosX = BitConverter.ToSingle(payload, offset + 0x08);
            float filePosY = BitConverter.ToSingle(payload, offset + 0x0C);
            float filePosZ = BitConverter.ToSingle(payload, offset + 0x10);
            float fileRotX = BitConverter.ToSingle(payload, offset + 0x14);
            float fileRotY = BitConverter.ToSingle(payload, offset + 0x18);
            float fileRotZ = BitConverter.ToSingle(payload, offset + 0x1C);
            float extentsTopX = BitConverter.ToSingle(payload, offset + 0x20);
            float extentsTopY = BitConverter.ToSingle(payload, offset + 0x24);
            float extentsTopZ = BitConverter.ToSingle(payload, offset + 0x28);
            float extentsBotX = BitConverter.ToSingle(payload, offset + 0x2C);
            float extentsBotY = BitConverter.ToSingle(payload, offset + 0x30);
            float extentsBotZ = BitConverter.ToSingle(payload, offset + 0x34);
            ushort flags = BitConverter.ToUInt16(payload, offset + 0x38);

            float rendererX = MapOrigin - filePosZ;
            float rendererY = MapOrigin - filePosX;
            float rendererZ = filePosY;
            float boundsMinX = MapOrigin - extentsTopZ;
            float boundsMinY = MapOrigin - extentsTopX;
            float boundsMinZ = extentsBotY;
            float boundsMaxX = MapOrigin - extentsBotZ;
            float boundsMaxY = MapOrigin - extentsBotX;
            float boundsMaxZ = extentsTopY;
            placements.Add(new AlphaWorldModelPlacement(
                nameId,
                ResolveName(modelNames, nameId),
                uniqueId,
                new Vector3(rendererX, rendererY, rendererZ),
                new Vector3(fileRotX, fileRotZ, fileRotY),
                new Vector3(boundsMinX, boundsMinY, boundsMinZ),
                new Vector3(boundsMaxX, boundsMaxY, boundsMaxZ),
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

    private static float[,] DownsampleShadowMask(float[,] src)
    {
        const int srcSize = 1024;
        const int dstSize = 256;
        const int ratio = srcSize / dstSize;
        const int samples = ratio * ratio;
        var dst = new float[dstSize, dstSize];

        for (int y = 0; y < dstSize; y++)
        {
            for (int x = 0; x < dstSize; x++)
            {
                float sum = 0f;
                for (int dy = 0; dy < ratio; dy++)
                {
                    for (int dx = 0; dx < ratio; dx++)
                    {
                        sum += src[y * ratio + dy, x * ratio + dx];
                    }
                }
                dst[y, x] = sum / samples;
            }
        }
        return dst;
    }

    private static bool DetectMcshUpperRight(float[,] shadowMask1024)
    {
        const int srcSize = 1024;
        const int chunkSize = 64;
        int topLeftCount = 0, topRightCount = 0;
        int scanned = 0;

        for (int cy = 0; cy < 16; cy++)
        {
            for (int cx = 0; cx < 16; cx++)
            {
                int baseY = cx * chunkSize;
                int baseX = cy * chunkSize;
                if (baseY + chunkSize > srcSize || baseX + chunkSize > srcSize) continue;
                scanned++;

                int leftHalf = 0, rightHalf = 0;
                int halfSize = chunkSize / 2;
                for (int y = 0; y < halfSize; y++)
                {
                    for (int x = 0; x < halfSize; x++)
                    {
                        if (shadowMask1024[baseY + y, baseX + x] > 0.5f)
                            leftHalf++;
                        if (shadowMask1024[baseY + y, baseX + halfSize + x] > 0.5f)
                            rightHalf++;
                    }
                }

                if (rightHalf > leftHalf) topRightCount++;
                else if (leftHalf > rightHalf) topLeftCount++;
            }
        }

        if (scanned == 0) return false;
        return topRightCount > topLeftCount;
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

    private static int ClassifyLiquid(uint mcnkFlags)
    {
        if ((mcnkFlags & 0x04) != 0) return 1;
        if ((mcnkFlags & 0x08) != 0) return 1;
        int bits = (int)((mcnkFlags >> 4) & 3);
        return bits switch
        {
            1 => 1,
            2 => 2,
            3 => 3,
            _ => 0
        };
    }
}
