namespace WowViewer.Core.Maps;

public sealed class AdtTextureChunk
{
    public AdtTextureChunk(
        int chunkIndex,
        int chunkX,
        int chunkY,
        bool doNotFixAlphaMap,
        int alphaPayloadBytes,
        IReadOnlyList<AdtTextureChunkLayer> layers)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(chunkIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(chunkX);
        ArgumentOutOfRangeException.ThrowIfNegative(chunkY);
        ArgumentOutOfRangeException.ThrowIfNegative(alphaPayloadBytes);
        ArgumentNullException.ThrowIfNull(layers);

        ChunkIndex = chunkIndex;
        ChunkX = chunkX;
        ChunkY = chunkY;
        DoNotFixAlphaMap = doNotFixAlphaMap;
        AlphaPayloadBytes = alphaPayloadBytes;
        Layers = layers;
    }

    public int ChunkIndex { get; }

    public int ChunkX { get; }

    public int ChunkY { get; }

    public bool DoNotFixAlphaMap { get; }

    public int AlphaPayloadBytes { get; }

    public IReadOnlyList<AdtTextureChunkLayer> Layers { get; }

    public int DecodedLayerCount => Layers.Count(layer => layer.DecodedAlpha is not null);
}