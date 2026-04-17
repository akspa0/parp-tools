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
        ushort holeMask,
        bool hasLiquidFlags,
        bool hasVertexColors,
        float[]? heights)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(chunkIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(indexX);
        ArgumentOutOfRangeException.ThrowIfNegative(indexY);
        ArgumentOutOfRangeException.ThrowIfNegative(layerCount);
        if (heights is not null && heights.Length != 145)
            throw new ArgumentException("Terrain chunk height payloads must contain exactly 145 MCVT samples.", nameof(heights));

        ChunkIndex = chunkIndex;
        IndexX = indexX;
        IndexY = indexY;
        AreaId = areaId;
        Flags = flags;
        LayerCount = layerCount;
        HoleMask = holeMask;
        HasLiquidFlags = hasLiquidFlags;
        HasVertexColors = hasVertexColors;
        Heights = heights;
    }

    public int ChunkIndex { get; }

    public int IndexX { get; }

    public int IndexY { get; }

    public uint AreaId { get; }

    public uint Flags { get; }

    public int LayerCount { get; }

    public ushort HoleMask { get; }

    public bool HasHoles => HoleMask != 0;

    public bool HasLiquidFlags { get; }

    public bool HasVertexColors { get; }

    public float[]? Heights { get; }

    public bool HasHeights => Heights is { Length: 145 };
}