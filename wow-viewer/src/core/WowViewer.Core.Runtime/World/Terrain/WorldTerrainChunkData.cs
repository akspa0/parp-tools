namespace WowViewer.Core.Runtime.World.Terrain;

public sealed class WorldTerrainChunkData
{
    public WorldTerrainChunkData(
        int chunkIndex,
        int indexX,
        int indexY,
        uint areaId,
        uint flags,
        int layerCount,
        bool hasHoles,
        bool hasLiquidFlags,
        bool hasVertexColors)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(chunkIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(indexX);
        ArgumentOutOfRangeException.ThrowIfNegative(indexY);
        ArgumentOutOfRangeException.ThrowIfNegative(layerCount);

        ChunkIndex = chunkIndex;
        IndexX = indexX;
        IndexY = indexY;
        AreaId = areaId;
        Flags = flags;
        LayerCount = layerCount;
        HasHoles = hasHoles;
        HasLiquidFlags = hasLiquidFlags;
        HasVertexColors = hasVertexColors;
    }

    public int ChunkIndex { get; }

    public int IndexX { get; }

    public int IndexY { get; }

    public uint AreaId { get; }

    public uint Flags { get; }

    public int LayerCount { get; }

    public bool HasHoles { get; }

    public bool HasLiquidFlags { get; }

    public bool HasVertexColors { get; }
}