namespace WowViewer.Core.Runtime.World.Liquid;

public sealed class WorldLiquidChunkData
{
    public WorldLiquidChunkData(
        int chunkIndex,
        int chunkX,
        int chunkY,
        ulong? fishableMask,
        ulong? deepMask,
        IReadOnlyList<WorldLiquidLayerData> layers)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(chunkIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(chunkX);
        ArgumentOutOfRangeException.ThrowIfNegative(chunkY);
        ArgumentNullException.ThrowIfNull(layers);

        ChunkIndex = chunkIndex;
        ChunkX = chunkX;
        ChunkY = chunkY;
        FishableMask = fishableMask;
        DeepMask = deepMask;
        Layers = layers;
    }

    public int ChunkIndex { get; }

    public int ChunkX { get; }

    public int ChunkY { get; }

    public ulong? FishableMask { get; }

    public ulong? DeepMask { get; }

    public IReadOnlyList<WorldLiquidLayerData> Layers { get; }

    public int VisibleTileCount => Layers.Sum(static layer => layer.VisibleTileCount);
}