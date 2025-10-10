namespace WoWRollback.WDLtoGLB;

/// <summary>
/// Low-resolution world terrain (WDL) domain model: 64x64 tiles.
/// Each present tile contains two height grids: outer 17x17 and inner 16x16,
/// and an optional 16x16 holes mask encoded as 16 rows of 16-bit values.
/// </summary>
public sealed class Wdl
{
    public string Path { get; }
    public WdlTile?[,] Tiles { get; }

    public Wdl(string path, WdlTile?[,] tiles)
    {
        Path = path;
        Tiles = tiles;
    }
}

/// <summary>
/// WDL tile heights as int16 grids (outer 17x17 fan and inner 16x16 centers)
/// and optional holes mask (16x ushort rows).
/// </summary>
public sealed class WdlTile
{
    public const int OuterGrid = 17;
    public const int InnerGrid = 16;

    public short[,] Height17 { get; }
    public short[,] Height16 { get; }

    /// <summary>
    /// Row-major 16-element array where each element is a 16-bit row mask.
    /// Bit set => hole at (y, x). If absent in source, remains all zeros.
    /// </summary>
    public ushort[] HoleMask16 { get; }

    public WdlTile(short[,] height17, short[,] height16, ushort[]? holeMask16 = null)
    {
        if (height17.GetLength(0) != OuterGrid || height17.GetLength(1) != OuterGrid)
            throw new System.ArgumentException("height17 must be 17x17", nameof(height17));
        if (height16.GetLength(0) != InnerGrid || height16.GetLength(1) != InnerGrid)
            throw new System.ArgumentException("height16 must be 16x16", nameof(height16));

        Height17 = height17;
        Height16 = height16;

        if (holeMask16 != null && holeMask16.Length == InnerGrid)
        {
            HoleMask16 = (ushort[])holeMask16.Clone();
        }
        else
        {
            HoleMask16 = new ushort[InnerGrid]; // default to no holes
        }
    }

    /// <summary>
    /// Returns true if the bit at (y,x) is set in the 16x16 hole mask.
    /// Out-of-range indices return false.
    /// </summary>
    public bool IsHole(int y, int x)
    {
        if ((uint)y >= InnerGrid || (uint)x >= InnerGrid) return false;
        return (HoleMask16[y] & (1 << x)) != 0;
    }
}
