namespace WowViewer.Core.Maps;

public sealed class AdtV23Summary
{
    public AdtV23Summary(
        string sourcePath,
        MapFileKind kind,
        uint headerVersion,
        int verticesX,
        int verticesY,
        int chunksX,
        int chunksY,
        int terrainChunkCount,
        int textureNameCount,
        int objectNameCount,
        bool hasVertexHeights,
        bool hasNormals,
        bool hasFlightBounds,
        bool hasVertexShading)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(verticesX);
        ArgumentOutOfRangeException.ThrowIfNegative(verticesY);
        ArgumentOutOfRangeException.ThrowIfNegative(chunksX);
        ArgumentOutOfRangeException.ThrowIfNegative(chunksY);
        ArgumentOutOfRangeException.ThrowIfNegative(terrainChunkCount);
        ArgumentOutOfRangeException.ThrowIfNegative(textureNameCount);
        ArgumentOutOfRangeException.ThrowIfNegative(objectNameCount);

        SourcePath = sourcePath;
        Kind = kind;
        HeaderVersion = headerVersion;
        VerticesX = verticesX;
        VerticesY = verticesY;
        ChunksX = chunksX;
        ChunksY = chunksY;
        TerrainChunkCount = terrainChunkCount;
        TextureNameCount = textureNameCount;
        ObjectNameCount = objectNameCount;
        HasVertexHeights = hasVertexHeights;
        HasNormals = hasNormals;
        HasFlightBounds = hasFlightBounds;
        HasVertexShading = hasVertexShading;
    }

    public string SourcePath { get; }

    public MapFileKind Kind { get; }

    public uint HeaderVersion { get; }

    public int VerticesX { get; }

    public int VerticesY { get; }

    public int ChunksX { get; }

    public int ChunksY { get; }

    public int TerrainChunkCount { get; }

    public int TextureNameCount { get; }

    public int ObjectNameCount { get; }

    public bool HasVertexHeights { get; }

    public bool HasNormals { get; }

    public bool HasFlightBounds { get; }

    public bool HasVertexShading { get; }
}