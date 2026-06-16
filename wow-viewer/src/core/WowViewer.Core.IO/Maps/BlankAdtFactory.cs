using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class BlankAdtFactory
{
    private const float MapOrigin = 17066.666f;
    private const float TileSize = 533.33333f;
    private const float ChunkSize = TileSize / 16f;
    private const int McvtFloatCount = 145;
    private const int McnrByteCount = 448;
    private const int ChunksPerSide = 16;
    private const int TotalChunks = ChunksPerSide * ChunksPerSide;

    public static LkAdtData CreateBlank(string mapName, int tileX, int tileY)
    {
        ArgumentNullException.ThrowIfNull(mapName);
        ArgumentOutOfRangeException.ThrowIfNegative(tileX);
        ArgumentOutOfRangeException.ThrowIfNegative(tileY);

        float baseHeight = 0.0f;
        float posY = MapOrigin - tileY * TileSize - ChunkSize;
        float posX = tileX * TileSize;

        var chunks = new LkMcnkData[TotalChunks];
        for (int cy = 0; cy < ChunksPerSide; cy++)
        {
            for (int cx = 0; cx < ChunksPerSide; cx++)
            {
                int i = cy * ChunksPerSide + cx;
                float chunkPosX = posX + cx * ChunkSize;
                float chunkPosY = posY - cy * ChunkSize;

                chunks[i] = new LkMcnkData
                {
                    IndexX = cx,
                    IndexY = cy,
                    Flags = 0,
                    AreaId = 0,
                    NLayers = 0,
                    HoleMask = 0,
                    BaseHeight = baseHeight,
                    Heights = CreateFlatHeights(baseHeight),
                    Normals = CreateUpNormals(),
                    ShadowMap = null,
                    AlphaMapData = null,
                    AlphaMapSize = 0,
                    Layers = [],
                    DoodadRefs = [],
                    WorldModelRefs = [],
                    LiquidData = null,
                    MccvColors = null,
                    MclvLighting = null,
                    PosX = chunkPosX,
                    PosY = chunkPosY,
                    PosZ = baseHeight,
                };
            }
        }

        return new LkAdtData
        {
            MapName = mapName,
            TileX = tileX,
            TileY = tileY,
            TextureNames = [],
            ModelNames = [],
            WorldModelNames = [],
            ModelPlacements = [],
            WorldModelPlacements = [],
            Chunks = chunks,
            MhdrFlags = 0,
            MfboFlightBounds = null,
        };
    }

    public static LkWdtWriteOptions CreateBlankWdtOptions() => new()
    {
        HasMccv = false,
        HasBigAlpha = false,
        HasMtxf = false,
        HasMaid = false,
        HasMclv = false,
    };

    public static WdlHeightTile CreateBlankWdlTile(int tileX, int tileY, short height = 0)
    {
        const int OuterCount = 17 * 17;
        const int InnerCount = 16 * 16;

        var outerHeights = new short[OuterCount];
        var innerHeights = new short[InnerCount];
        Array.Fill(outerHeights, height);
        Array.Fill(innerHeights, height);

        return new WdlHeightTile(tileX, tileY, outerHeights, innerHeights);
    }

    private static float[] CreateFlatHeights(float height)
    {
        var heights = new float[McvtFloatCount];
        Array.Fill(heights, height);
        return heights;
    }

    private static byte[] CreateUpNormals()
    {
        var normals = new byte[McnrByteCount];
        for (int i = 0; i < TotalChunks; i++)
        {
            int offset = i * 3;
            if (offset + 2 < normals.Length)
            {
                normals[offset] = 128;
                normals[offset + 1] = 128;
                normals[offset + 2] = 255;
            }
        }
        return normals;
    }
}