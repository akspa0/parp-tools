namespace WoWMapConverter.Core.Formats.Liquids;

/// <summary>
/// Bidirectional converter between MCLQ (pre-WotLK) and MH2O (WotLK+) liquid formats.
/// Based on Noggit3 implementation and wowdev.wiki documentation.
/// </summary>
public static class LiquidConverter
{
    /// <summary>
    /// Convert MCLQ chunk to MH2O instance for a single MCNK.
    /// </summary>
    /// <param name="mclq">Source MCLQ data</param>
    /// <param name="liquidType">Liquid type from MCNK flags</param>
    /// <param name="chunkIndex">MCNK chunk index (0-255)</param>
    /// <returns>MH2O instance, or null if no liquid</returns>
    public static Mh2oInstance? MclqToMh2o(MclqChunk mclq, MclqLiquidType liquidType, int chunkIndex)
    {
        if (mclq == null || liquidType == MclqLiquidType.None || liquidType == MclqLiquidType.DontRender)
            return null;

        // Determine which tiles have liquid
        int minX = 8, minY = 8, maxX = -1, maxY = -1;
        for (int y = 0; y < 8; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                var tile = mclq.Tiles[y * 8 + x];
                if (!tile.DontRender)
                {
                    minX = Math.Min(minX, x);
                    minY = Math.Min(minY, y);
                    maxX = Math.Max(maxX, x);
                    maxY = Math.Max(maxY, y);
                }
            }
        }

        // No visible tiles
        if (maxX < 0) return null;

        int width = maxX - minX + 1;
        int height = maxY - minY + 1;
        int vertexCount = (width + 1) * (height + 1);

        var instance = new Mh2oInstance
        {
            ChunkIndex = chunkIndex,
            LiquidTypeId = MapMclqTypeToLiquidTypeId(liquidType),
            VertexFormat = liquidType == MclqLiquidType.Magma ? Mh2oVertexFormat.HeightUv : Mh2oVertexFormat.HeightDepth,
            MinHeightLevel = mclq.MinHeight,
            MaxHeightLevel = mclq.MaxHeight,
            XOffset = (byte)minX,
            YOffset = (byte)minY,
            Width = (byte)width,
            Height = (byte)height,
            HeightMap = new float[vertexCount],
            DepthMap = liquidType != MclqLiquidType.Magma ? new byte[vertexCount] : null,
            UvMap = liquidType == MclqLiquidType.Magma ? new ushort[vertexCount * 2] : null
        };

        // Build exists bitmap
        int bitmapSize = (width * height + 7) / 8;
        instance.ExistsBitmap = new byte[bitmapSize];
        for (int ly = 0; ly < height; ly++)
        {
            for (int lx = 0; lx < width; lx++)
            {
                int globalX = minX + lx;
                int globalY = minY + ly;
                var tile = mclq.Tiles[globalY * 8 + globalX];
                if (!tile.DontRender)
                {
                    int bitIndex = ly * width + lx;
                    instance.ExistsBitmap[bitIndex / 8] |= (byte)(1 << (bitIndex % 8));
                }
            }
        }

        // Copy vertex data
        for (int vy = 0; vy <= height; vy++)
        {
            for (int vx = 0; vx <= width; vx++)
            {
                int globalVx = minX + vx;
                int globalVy = minY + vy;
                int srcIndex = globalVy * 9 + globalVx;
                int dstIndex = vy * (width + 1) + vx;

                if (srcIndex < mclq.Vertices.Length)
                {
                    var srcVertex = mclq.Vertices[srcIndex];
                    instance.HeightMap[dstIndex] = srcVertex.Height;

                    if (liquidType == MclqLiquidType.Magma && instance.UvMap != null)
                    {
                        instance.UvMap[dstIndex * 2] = srcVertex.MagmaS;
                        instance.UvMap[dstIndex * 2 + 1] = srcVertex.MagmaT;
                    }
                    else if (instance.DepthMap != null)
                    {
                        instance.DepthMap[dstIndex] = srcVertex.Depth;
                    }
                }
            }
        }

        return instance;
    }

    /// <summary>
    /// Convert MH2O instance to MCLQ chunk for a single MCNK.
    /// </summary>
    /// <param name="instance">Source MH2O instance</param>
    /// <param name="attributes">Optional MH2O attributes for fatigue/fishable</param>
    /// <returns>MCLQ chunk data</returns>
    public static MclqChunk Mh2oToMclq(Mh2oInstance instance, Mh2oAttributes? attributes = null)
    {
        var mclq = new MclqChunk
        {
            MinHeight = instance.MinHeightLevel,
            MaxHeight = instance.MaxHeightLevel,
            FlowCount = 0
        };

        // Initialize all tiles as don't render
        for (int i = 0; i < MclqChunk.TileCount; i++)
        {
            mclq.Tiles[i] = MclqTile.Create(MclqLiquidType.DontRender, dontRender: true);
        }

        // Initialize all vertices
        for (int i = 0; i < MclqChunk.VertexCount; i++)
        {
            mclq.Vertices[i] = new MclqVertex { Height = instance.MinHeightLevel };
        }

        var mclqType = MapLiquidTypeIdToMclqType(instance.LiquidTypeId);

        // Set tiles that exist
        for (int ly = 0; ly < instance.Height; ly++)
        {
            for (int lx = 0; lx < instance.Width; lx++)
            {
                int globalX = instance.XOffset + lx;
                int globalY = instance.YOffset + ly;

                if (globalX >= 8 || globalY >= 8) continue;

                bool exists = instance.TileExists(lx, ly);
                bool fatigue = attributes?.IsDeep(globalX, globalY) ?? false;

                mclq.Tiles[globalY * 8 + globalX] = MclqTile.Create(
                    exists ? mclqType : MclqLiquidType.DontRender,
                    dontRender: !exists,
                    fatigue: fatigue
                );
            }
        }

        // Copy vertex data
        for (int vy = 0; vy <= instance.Height; vy++)
        {
            for (int vx = 0; vx <= instance.Width; vx++)
            {
                int globalVx = instance.XOffset + vx;
                int globalVy = instance.YOffset + vy;

                if (globalVx > 8 || globalVy > 8) continue;

                int srcIndex = vy * (instance.Width + 1) + vx;
                int dstIndex = globalVy * 9 + globalVx;

                if (dstIndex < mclq.Vertices.Length)
                {
                    var vertex = new MclqVertex();

                    // Height
                    if (instance.HeightMap != null && srcIndex < instance.HeightMap.Length)
                        vertex.Height = instance.HeightMap[srcIndex];
                    else
                        vertex.Height = instance.MinHeightLevel;

                    // Depth or UV based on type
                    if (mclqType == MclqLiquidType.Magma)
                    {
                        if (instance.UvMap != null && srcIndex * 2 + 1 < instance.UvMap.Length)
                        {
                            vertex.MagmaS = instance.UvMap[srcIndex * 2];
                            vertex.MagmaT = instance.UvMap[srcIndex * 2 + 1];
                        }
                    }
                    else
                    {
                        if (instance.DepthMap != null && srcIndex < instance.DepthMap.Length)
                            vertex.Depth = instance.DepthMap[srcIndex];
                    }

                    mclq.Vertices[dstIndex] = vertex;
                }
            }
        }

        return mclq;
    }

    /// <summary>
    /// Map MCLQ liquid type to LiquidType.dbc ID.
    /// </summary>
    public static ushort MapMclqTypeToLiquidTypeId(MclqLiquidType type) => type switch
    {
        MclqLiquidType.Ocean => 2,   // Ocean
        MclqLiquidType.River => 3,   // Water (river)
        MclqLiquidType.Slime => 4,   // Slime
        MclqLiquidType.Magma => 5,   // Magma
        _ => 0
    };

    /// <summary>
    /// Map LiquidType.dbc ID to MCLQ liquid type.
    /// </summary>
    public static MclqLiquidType MapLiquidTypeIdToMclqType(ushort liquidTypeId) => liquidTypeId switch
    {
        1 => MclqLiquidType.River,   // Water
        2 => MclqLiquidType.Ocean,   // Ocean
        3 => MclqLiquidType.River,   // Water (alternate)
        4 => MclqLiquidType.Slime,   // Slime
        5 => MclqLiquidType.Magma,   // Magma
        6 => MclqLiquidType.Magma,   // Lava (alternate)
        _ => MclqLiquidType.River    // Default to river/water
    };

    /// <summary>
    /// Get MCLQ liquid type from MCNK flags.
    /// </summary>
    public static MclqLiquidType GetLiquidTypeFromMcnkFlags(uint mcnkFlags)
    {
        // MCNK flags: 0x04 = river, 0x08 = ocean, 0x10 = magma, 0x20 = slime
        if ((mcnkFlags & 0x10) != 0) return MclqLiquidType.Magma;
        if ((mcnkFlags & 0x20) != 0) return MclqLiquidType.Slime;
        if ((mcnkFlags & 0x08) != 0) return MclqLiquidType.Ocean;
        if ((mcnkFlags & 0x04) != 0) return MclqLiquidType.River;
        return MclqLiquidType.None;
    }

    /// <summary>
    /// Set MCNK flags based on MCLQ liquid type.
    /// </summary>
    public static uint SetMcnkFlagsForLiquidType(uint mcnkFlags, MclqLiquidType type)
    {
        // Clear existing liquid flags
        mcnkFlags &= ~0x3CU; // Clear bits 2-5

        return type switch
        {
            MclqLiquidType.River => mcnkFlags | 0x04,
            MclqLiquidType.Ocean => mcnkFlags | 0x08,
            MclqLiquidType.Magma => mcnkFlags | 0x10,
            MclqLiquidType.Slime => mcnkFlags | 0x20,
            _ => mcnkFlags
        };
    }
}
