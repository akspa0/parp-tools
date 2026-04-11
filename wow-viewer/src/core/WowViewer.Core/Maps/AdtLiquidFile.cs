namespace WowViewer.Core.Maps;

public sealed class AdtLiquidFile
{
    public AdtLiquidFile(
        string sourcePath,
        MapFileKind kind,
        IReadOnlyList<AdtLiquidChunk> chunks)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentNullException.ThrowIfNull(chunks);

        SourcePath = sourcePath;
        Kind = kind;
        Chunks = chunks;
    }

    public string SourcePath { get; }

    public MapFileKind Kind { get; }

    public IReadOnlyList<AdtLiquidChunk> Chunks { get; }
}

public sealed class AdtLiquidChunk
{
    public AdtLiquidChunk(
        int chunkIndex,
        ulong? fishableMask,
        ulong? deepMask,
        IReadOnlyList<AdtLiquidLayer> layers)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(chunkIndex);
        ArgumentNullException.ThrowIfNull(layers);

        ChunkIndex = chunkIndex;
        FishableMask = fishableMask;
        DeepMask = deepMask;
        Layers = layers;
    }

    public int ChunkIndex { get; }

    public ulong? FishableMask { get; }

    public ulong? DeepMask { get; }

    public IReadOnlyList<AdtLiquidLayer> Layers { get; }
}

public sealed class AdtLiquidLayer
{
    public AdtLiquidLayer(
        ushort liquidTypeId,
        AdtLiquidBasicType basicType,
        AdtLiquidVertexFormat vertexFormat,
        float minHeight,
        float maxHeight,
        int xOffset,
        int yOffset,
        int width,
        int height,
        byte[]? existsBitmap,
        float[]? heights,
        byte[]? depths,
        ushort[]? uvs)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(xOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(yOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(width);
        ArgumentOutOfRangeException.ThrowIfNegative(height);

        LiquidTypeId = liquidTypeId;
        BasicType = basicType;
        VertexFormat = vertexFormat;
        MinHeight = minHeight;
        MaxHeight = maxHeight;
        XOffset = xOffset;
        YOffset = yOffset;
        Width = width;
        Height = height;
        ExistsBitmap = existsBitmap;
        Heights = heights;
        Depths = depths;
        Uvs = uvs;
    }

    public ushort LiquidTypeId { get; }

    public AdtLiquidBasicType BasicType { get; }

    public AdtLiquidVertexFormat VertexFormat { get; }

    public float MinHeight { get; }

    public float MaxHeight { get; }

    public int XOffset { get; }

    public int YOffset { get; }

    public int Width { get; }

    public int Height { get; }

    public byte[]? ExistsBitmap { get; }

    public float[]? Heights { get; }

    public byte[]? Depths { get; }

    public ushort[]? Uvs { get; }

    public int TileCount => Width * Height;

    public int VertexCount => (Width + 1) * (Height + 1);

    public int VisibleTileCount
    {
        get
        {
            if (Width <= 0 || Height <= 0)
                return 0;

            if (ExistsBitmap == null)
                return TileCount;

            int visible = 0;
            for (int y = 0; y < Height; y++)
            {
                for (int x = 0; x < Width; x++)
                {
                    if (TileExists(x, y))
                        visible++;
                }
            }

            return visible;
        }
    }

    public bool TileExists(int x, int y)
    {
        if ((uint)x >= (uint)Width || (uint)y >= (uint)Height)
            return false;

        if (ExistsBitmap == null)
            return true;

        int tileIndex = (y * Width) + x;
        int byteIndex = tileIndex / 8;
        if ((uint)byteIndex >= (uint)ExistsBitmap.Length)
            return false;

        int bitIndex = tileIndex % 8;
        return (ExistsBitmap[byteIndex] & (1 << bitIndex)) != 0;
    }
}

public enum AdtLiquidBasicType
{
    Water = 0,
    Ocean = 1,
    Magma = 2,
    Slime = 3,
}

public enum AdtLiquidVertexFormat
{
    HeightDepth = 0,
    HeightUv = 1,
    DepthOnly = 2,
    HeightUvDepth = 3,
}