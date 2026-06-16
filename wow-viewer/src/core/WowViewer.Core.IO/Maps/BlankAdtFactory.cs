using System.Numerics;
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

    public static LkAdtData CreateBlank(string mapName, int tileX, int tileY, string l0Texture = "tileset\\ocean\\westfallseafloor.blp")
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
                    NLayers = 1,
                    HoleMask = 0,
                    BaseHeight = baseHeight,
                    Heights = CreateFlatHeights(baseHeight),
                    Normals = CreateUpNormals(),
                    ShadowMap = null,
                    AlphaMapData = null,
                    AlphaMapSize = 0,
                    Layers = [new LkMclyEntry(TextureId: 0, Flags: 0, AlphaOffset: 0, EffectId: 0)],
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
            TextureNames = [l0Texture],
            ModelNames = [],
            WorldModelNames = [],
            ModelPlacements = [],
            WorldModelPlacements = [],
            Chunks = chunks,
            MhdrFlags = 0,
            MfboFlightBounds = null,
        };
    }

    public static LkAdtData WithPlacements(LkAdtData baseAdt, AdtPlacementCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(baseAdt);
        ArgumentNullException.ThrowIfNull(catalog);

        var modelNames = baseAdt.ModelNames.Concat(catalog.ModelNames).Distinct().OrderBy(static n => n).ToList();
        var worldModelNames = baseAdt.WorldModelNames.Concat(catalog.WorldModelNames).Distinct().OrderBy(static n => n).ToList();

        var modelNameIndex = modelNames.Select((n, i) => (n, i)).ToDictionary(static p => p.n, static p => p.i);
        var wmoNameIndex = worldModelNames.Select((n, i) => (n, i)).ToDictionary(static n => n.n, static n => n.i);

        var mddfPlacements = new List<LkMddfEntry>();
        foreach (var p in catalog.ModelPlacements)
        {
            int nameId = modelNameIndex.TryGetValue(p.ModelPath, out int idx) ? idx : 0;
            mddfPlacements.Add(new LkMddfEntry(nameId, p.UniqueId, p.Position, p.Rotation, p.Scale));
        }

        var modfPlacements = new List<LkModfEntry>();
        foreach (var p in catalog.WorldModelPlacements)
        {
            int nameId = wmoNameIndex.TryGetValue(p.ModelPath, out int idx) ? idx : 0;
            modfPlacements.Add(new LkModfEntry(nameId, p.UniqueId, p.Position, p.Rotation, p.BoundsMin, p.BoundsMax, Flags: 0, DoodadSet: 0, NameSet: 0, Scale: 1.0f));
        }

        var mddfByChunk = AssignMddfToChunks(baseAdt, mddfPlacements);
        var modfByChunk = AssignModfToChunks(baseAdt, modfPlacements);

        var patchedChunks = new LkMcnkData[baseAdt.Chunks.Count];
        for (int i = 0; i < baseAdt.Chunks.Count; i++)
        {
            var c = baseAdt.Chunks[i];
            patchedChunks[i] = new LkMcnkData
            {
                IndexX = c.IndexX,
                IndexY = c.IndexY,
                Flags = c.Flags,
                AreaId = c.AreaId,
                NLayers = c.NLayers,
                HoleMask = c.HoleMask,
                BaseHeight = c.BaseHeight,
                Heights = c.Heights,
                Normals = c.Normals,
                ShadowMap = c.ShadowMap,
                AlphaMapData = c.AlphaMapData,
                AlphaMapSize = c.AlphaMapSize,
                Layers = c.Layers,
                DoodadRefs = mddfByChunk.TryGetValue(i, out var mddfRefs) ? mddfRefs : [],
                WorldModelRefs = modfByChunk.TryGetValue(i, out var modfRefs) ? modfRefs : [],
                LiquidData = c.LiquidData,
                MccvColors = c.MccvColors,
                MclvLighting = c.MclvLighting,
                PosX = c.PosX,
                PosY = c.PosY,
                PosZ = c.PosZ,
            };
        }

        return new LkAdtData
        {
            MapName = baseAdt.MapName,
            TileX = baseAdt.TileX,
            TileY = baseAdt.TileY,
            TextureNames = baseAdt.TextureNames,
            ModelNames = modelNames,
            WorldModelNames = worldModelNames,
            ModelPlacements = mddfPlacements,
            WorldModelPlacements = modfPlacements,
            Chunks = patchedChunks,
            MhdrFlags = baseAdt.MhdrFlags,
            MfboFlightBounds = baseAdt.MfboFlightBounds,
        };
    }

    private static Dictionary<int, List<int>> AssignMddfToChunks(LkAdtData adt, IReadOnlyList<LkMddfEntry> placements)
    {
        var result = new Dictionary<int, List<int>>();
        for (int p = 0; p < placements.Count; p++)
        {
            int chunkIdx = FindChunkForPosition(adt, placements[p].Position);
            if (chunkIdx >= 0)
            {
                if (!result.TryGetValue(chunkIdx, out var list))
                {
                    list = [];
                    result[chunkIdx] = list;
                }
                list.Add(p);
            }
        }
        return result;
    }

    private static Dictionary<int, List<int>> AssignModfToChunks(LkAdtData adt, IReadOnlyList<LkModfEntry> placements)
    {
        var result = new Dictionary<int, List<int>>();
        for (int p = 0; p < placements.Count; p++)
        {
            int chunkIdx = FindChunkForPosition(adt, placements[p].Position);
            if (chunkIdx >= 0)
            {
                if (!result.TryGetValue(chunkIdx, out var list))
                {
                    list = [];
                    result[chunkIdx] = list;
                }
                list.Add(p);
            }
        }
        return result;
    }

    private static int FindChunkForPosition(LkAdtData adt, Vector3 position)
    {
        for (int i = 0; i < adt.Chunks.Count; i++)
        {
            var c = adt.Chunks[i];
            float minX = c.PosX;
            float minY = c.PosY - ChunkSize;
            float maxX = c.PosX + ChunkSize;
            float maxY = c.PosY;

            if (position.X >= minX && position.X < maxX && position.Y >= minY && position.Y < maxY)
                return i;
        }
        return -1;
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

    public static AlphaTileData CreateBlankAlphaTile(int tileX, int tileY, string l0Texture = "tileset\\ocean\\westfallseafloor.blp")
    {
        const int TileSize = 257;
        const int ChunksPerSide = 16;
        const float BaseHeight = 0.0f;

        var heightmap = new float[TileSize, TileSize];
        for (int y = 0; y < TileSize; y++)
            for (int x = 0; x < TileSize; x++)
                heightmap[y, x] = BaseHeight;

        var mclyTextureIds = new int[ChunksPerSide, ChunksPerSide, 4];
        var mclyLayerMask = new bool[ChunksPerSide, ChunksPerSide, 4];
        var holeMask = new bool[ChunksPerSide, ChunksPerSide];

        for (int cy = 0; cy < ChunksPerSide; cy++)
        {
            for (int cx = 0; cx < ChunksPerSide; cx++)
            {
                mclyTextureIds[cx, cy, 0] = 0;
                mclyLayerMask[cx, cy, 0] = true;
                holeMask[cx, cy] = false;
            }
        }

        return new AlphaTileData(
            sourcePath: $"blank_alpha_{tileX}_{tileY}",
            heightmap: heightmap,
            mcalAlphaPack: null,
            mclyTextureIds: mclyTextureIds,
            mclyLayerMask: mclyLayerMask,
            holeMask: holeMask,
            textureNames: [l0Texture],
            modelPlacements: [],
            worldModelPlacements: [],
            liquidChunks: []);
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