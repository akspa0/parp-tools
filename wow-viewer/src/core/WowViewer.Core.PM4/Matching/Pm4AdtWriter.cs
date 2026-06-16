using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.PM4.Matching;

public static class Pm4AdtWriter
{
    private const float TileSize = 533.33333f;
    private const float ChunkSize = TileSize / 16f;
    private const float MapOrigin = 17066.666f;

    public static LkAdtData BuildAdt(
        int tileX, int tileY,
        IReadOnlyList<Pm4AdtM2Placement> m2Placements,
        IReadOnlyList<Pm4AdtWmoPlacement> wmoPlacements,
        LkAdtData? baseAdt = null,
        string mapName = "pm4restored")
    {
        ArgumentNullException.ThrowIfNull(m2Placements);
        ArgumentNullException.ThrowIfNull(wmoPlacements);

        List<string> modelNames;
        List<string> worldModelNames;
        List<LkMcnkData> chunks;
        List<string> textureNames;
        uint mhdrFlags;
        int[,,]? mfboFlightBounds;

        if (baseAdt != null)
        {
            textureNames = new List<string>(baseAdt.TextureNames);
            modelNames = new List<string>(baseAdt.ModelNames);
            worldModelNames = new List<string>(baseAdt.WorldModelNames);
            chunks = baseAdt.Chunks.ToList();
            mhdrFlags = baseAdt.MhdrFlags;
            mfboFlightBounds = baseAdt.MfboFlightBounds;
        }
        else
        {
            textureNames = [];
            modelNames = [];
            worldModelNames = [];
            mhdrFlags = 0;
            mfboFlightBounds = null;
            chunks = BuildEmptyChunks(tileX, tileY);
        }

        List<LkMddfEntry> mddfEntries = [];
        List<LkModfEntry> modfEntries = [];

        Dictionary<string, int> modelNameIndex = new(StringComparer.OrdinalIgnoreCase);
        Dictionary<string, int> wmoNameIndex = new(StringComparer.OrdinalIgnoreCase);

        foreach (string name in modelNames)
        {
            string normalized = NormalizeModelPath(name);
            if (!modelNameIndex.ContainsKey(normalized))
                modelNameIndex[normalized] = modelNameIndex.Count;
        }
        foreach (string name in worldModelNames)
        {
            string normalized = NormalizeModelPath(name);
            if (!wmoNameIndex.ContainsKey(normalized))
                wmoNameIndex[normalized] = wmoNameIndex.Count;
        }

        foreach (Pm4AdtM2Placement placement in m2Placements)
        {
            string normalizedPath = NormalizeModelPath(placement.ModelPath);
            if (!modelNameIndex.TryGetValue(normalizedPath, out int nameId))
            {
                nameId = modelNames.Count;
                modelNames.Add(normalizedPath);
                modelNameIndex[normalizedPath] = nameId;
            }

            mddfEntries.Add(new LkMddfEntry(
                nameId,
                placement.UniqueId,
                placement.Position,
                placement.Rotation,
                placement.Scale));
        }

        foreach (Pm4AdtWmoPlacement placement in wmoPlacements)
        {
            string normalizedPath = NormalizeModelPath(placement.ModelPath);
            if (!wmoNameIndex.TryGetValue(normalizedPath, out int nameId))
            {
                nameId = worldModelNames.Count;
                worldModelNames.Add(normalizedPath);
                wmoNameIndex[normalizedPath] = nameId;
            }

            modfEntries.Add(new LkModfEntry(
                nameId,
                placement.UniqueId,
                placement.Position,
                placement.Rotation,
                placement.BoundsMin,
                placement.BoundsMax,
                0,
                0,
                0,
                1f));
        }

        // Populate MCRF references: map each placement to its containing chunk
        List<LkMcnkData> outputChunks = new(256);
        for (int cx = 0; cx < 16; cx++)
        {
            for (int cy = 0; cy < 16; cy++)
            {
                int chunkIndex = cy * 16 + cx;
                LkMcnkData existing = chunks[chunkIndex];

                // World-space bounds of this chunk (MapOrigin-converted)
                float adtX = tileX * TileSize + cx * ChunkSize;
                float adtY = tileY * TileSize + cy * ChunkSize;
                float worldMinX = MapOrigin - adtY - ChunkSize;
                float worldMaxX = MapOrigin - adtY;
                float worldMinY = MapOrigin - adtX - ChunkSize;
                float worldMaxY = MapOrigin - adtX;

                List<int> doodadRefs = [];
                for (int i = 0; i < mddfEntries.Count; i++)
                {
                    Vector3 pos = mddfEntries[i].Position;
                    if (pos.X >= worldMinX && pos.X < worldMaxX &&
                        pos.Y >= worldMinY && pos.Y < worldMaxY)
                    {
                        doodadRefs.Add(i);
                    }
                }

                List<int> wmoRefs = [];
                for (int i = 0; i < modfEntries.Count; i++)
                {
                    Vector3 pos = modfEntries[i].Position;
                    if (pos.X >= worldMinX && pos.X < worldMaxX &&
                        pos.Y >= worldMinY && pos.Y < worldMaxY)
                    {
                        wmoRefs.Add(i);
                    }
                }

                outputChunks.Add(new LkMcnkData
                {
                    IndexX = existing.IndexX,
                    IndexY = existing.IndexY,
                    Flags = existing.Flags,
                    AreaId = existing.AreaId,
                    NLayers = existing.NLayers,
                    HoleMask = existing.HoleMask,
                    BaseHeight = existing.BaseHeight,
                    Heights = existing.Heights,
                    Normals = existing.Normals,
                    ShadowMap = existing.ShadowMap,
                    AlphaMapData = existing.AlphaMapData,
                    AlphaMapSize = existing.AlphaMapSize,
                    Layers = existing.Layers,
                    DoodadRefs = doodadRefs,
                    WorldModelRefs = wmoRefs,
                    LiquidData = existing.LiquidData,
                    MccvColors = existing.MccvColors,
                    MclvLighting = existing.MclvLighting,
                    PosX = existing.PosX,
                    PosY = existing.PosY,
                    PosZ = existing.PosZ,
                });
            }
        }

        return new LkAdtData
        {
            MapName = mapName,
            TileX = tileX,
            TileY = tileY,
            TextureNames = textureNames,
            ModelNames = modelNames,
            WorldModelNames = worldModelNames,
            ModelPlacements = mddfEntries,
            WorldModelPlacements = modfEntries,
            Chunks = outputChunks,
            MhdrFlags = mhdrFlags,
            MfboFlightBounds = mfboFlightBounds,
        };
    }

    private static List<LkMcnkData> BuildEmptyChunks(int tileX, int tileY)
    {
        List<LkMcnkData> chunks = new(256);
        for (int cy = 0; cy < 16; cy++)
        {
            for (int cx = 0; cx < 16; cx++)
            {
                float posX = tileX * TileSize + cx * ChunkSize;
                float posY = tileY * TileSize + cy * ChunkSize;
                chunks.Add(new LkMcnkData
                {
                    IndexX = cx,
                    IndexY = cy,
                    Flags = 0,
                    AreaId = 0,
                    NLayers = 0,
                    HoleMask = 0,
                    BaseHeight = 0f,
                    Heights = new float[145],
                    Normals = new byte[448],
                    PosX = posX,
                    PosY = posY,
                });
            }
        }
        return chunks;
    }

    private static string NormalizeModelPath(string modelPath)
    {
        return modelPath.Replace('\\', '/').Trim().TrimStart('/').ToLowerInvariant();
    }
}

public sealed record Pm4AdtM2Placement(
    int UniqueId,
    string ModelPath,
    Vector3 Position,
    Vector3 Rotation,
    float Scale);

public sealed record Pm4AdtWmoPlacement(
    int UniqueId,
    string ModelPath,
    Vector3 Position,
    Vector3 Rotation,
    Vector3 BoundsMin,
    Vector3 BoundsMax);
