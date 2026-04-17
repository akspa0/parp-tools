namespace WowViewer.Core.Runtime.World.Terrain;

public sealed class WorldTerrainHeightmapData
{
    public WorldTerrainHeightmapData(int width, int height, float[] heights, float minHeight, float maxHeight, int authoritativeSampleCount)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        ArgumentNullException.ThrowIfNull(heights);
        ArgumentOutOfRangeException.ThrowIfNegative(authoritativeSampleCount);
        if (heights.Length != width * height)
            throw new ArgumentException("Terrain heightmaps must match their declared dimensions.", nameof(heights));

        Width = width;
        Height = height;
        Heights = heights;
        MinHeight = minHeight;
        MaxHeight = maxHeight;
        AuthoritativeSampleCount = authoritativeSampleCount;
    }

    public int Width { get; }

    public int Height { get; }

    public float[] Heights { get; }

    public float MinHeight { get; }

    public float MaxHeight { get; }

    public int AuthoritativeSampleCount { get; }

    public float HeightRange => MaxHeight - MinHeight;

    public float CenterHeight => GetHeight((Width - 1) / 2, (Height - 1) / 2);

    public float NorthWestHeight => GetHeight(0, 0);

    public float NorthEastHeight => GetHeight(Width - 1, 0);

    public float SouthWestHeight => GetHeight(0, Height - 1);

    public float SouthEastHeight => GetHeight(Width - 1, Height - 1);

    public float GetHeight(int x, int y)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(x);
        ArgumentOutOfRangeException.ThrowIfNegative(y);
        if (x >= Width)
            throw new ArgumentOutOfRangeException(nameof(x));
        if (y >= Height)
            throw new ArgumentOutOfRangeException(nameof(y));

        return Heights[(y * Width) + x];
    }
}