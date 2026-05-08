using System.Text;
using WowViewer.Core.Chunks;

namespace WowViewer.Core.IO.Maps;

public sealed class WdlHeightTile
{
    public int TileX { get; }
    public int TileY { get; }
    public short[] OuterHeights { get; }
    public short[] InnerHeights { get; }

    private const int OuterHeightCount = 17 * 17;
    private const int InnerHeightCount = 16 * 16;

    public WdlHeightTile(int tileX, int tileY, short[] outerHeights, short[] innerHeights)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(tileX);
        ArgumentOutOfRangeException.ThrowIfNegative(tileY);
        ArgumentNullException.ThrowIfNull(outerHeights);
        ArgumentNullException.ThrowIfNull(innerHeights);
        if (outerHeights.Length != OuterHeightCount)
            throw new ArgumentException($"Outer heights must have exactly {OuterHeightCount} entries (17x17).", nameof(outerHeights));
        if (innerHeights.Length != InnerHeightCount)
            throw new ArgumentException($"Inner heights must have exactly {InnerHeightCount} entries (16x16).", nameof(innerHeights));

        TileX = tileX;
        TileY = tileY;
        OuterHeights = outerHeights;
        InnerHeights = innerHeights;
    }
}

public static class WdlWriter
{
    private const int TilesPerAxis = 64;
    private const int OuterDimension = 17;
    private const int InnerDimension = 16;
    private const int OuterHeightCount = OuterDimension * OuterDimension;
    private const int InnerHeightCount = InnerDimension * InnerDimension;
    private const int HeightsPerTile = OuterHeightCount + InnerHeightCount;
    private const int MareDataSize = HeightsPerTile * 2;
    private const int MahoDataSize = 32;

    public static void Write(string path, IReadOnlyList<WdlHeightTile> tiles)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(tiles);

        string? dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        File.WriteAllBytes(path, Build(tiles));
    }

    public static byte[] Build(IReadOnlyList<WdlHeightTile> tiles)
    {
        ArgumentNullException.ThrowIfNull(tiles);

        var tileMap = new bool[TilesPerAxis, TilesPerAxis];
        var heightLookup = new Dictionary<(int, int), WdlHeightTile>();

        foreach (var tile in tiles)
        {
            if (tile.TileX < TilesPerAxis && tile.TileY < TilesPerAxis)
            {
                tileMap[tile.TileY, tile.TileX] = true;
                heightLookup[(tile.TileX, tile.TileY)] = tile;
            }
        }

        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        WriteChunk(bw, "MVER", 4, static w => w.Write(18));

        WriteChunk(bw, "MWMO", 0, static _ => { });
        WriteChunk(bw, "MWID", 0, static _ => { });
        WriteChunk(bw, "MODF", 0, static _ => { });

        int mareaStartOffset = CalculateMareaStartOffset();
        uint currentTileOffset = (uint)mareaStartOffset;

        WriteChunk(bw, "MAOF", TilesPerAxis * TilesPerAxis * 4, w =>
        {
            for (int y = 0; y < TilesPerAxis; y++)
            {
                for (int x = 0; x < TilesPerAxis; x++)
                {
                    if (tileMap[y, x])
                    {
                        w.Write(currentTileOffset);
                        currentTileOffset += (uint)(8 + MareDataSize + 8 + MahoDataSize);
                    }
                    else
                    {
                        w.Write(0u);
                    }
                }
            }
        });

        for (int y = 0; y < TilesPerAxis; y++)
        {
            for (int x = 0; x < TilesPerAxis; x++)
            {
                if (!tileMap[y, x])
                    continue;

                heightLookup.TryGetValue((x, y), out var tile);
                var outer = tile?.OuterHeights ?? new short[OuterHeightCount];
                var inner = tile?.InnerHeights ?? new short[InnerHeightCount];

                WriteChunk(bw, "MARE", MareDataSize, w =>
                {
                    for (int i = 0; i < OuterHeightCount; i++)
                        w.Write(outer[i]);
                    for (int i = 0; i < InnerHeightCount; i++)
                        w.Write(inner[i]);
                });

                WriteChunk(bw, "MAHO", MahoDataSize, w =>
                {
                    w.Write(new byte[MahoDataSize]);
                });
            }
        }

        bw.Flush();
        return ms.ToArray();
    }

    public static WdlHeightTile ExtractTileHeightsFromAlpha(float[,] heightmap, int tileX, int tileY)
    {
        ArgumentNullException.ThrowIfNull(heightmap);

        const int SrcSize = 257;
        if (heightmap.GetLength(0) < SrcSize || heightmap.GetLength(1) < SrcSize)
            throw new ArgumentException($"Heightmap must be at least {SrcSize}x{SrcSize}.", nameof(heightmap));

        var outerHeights = new short[OuterHeightCount];
        var innerHeights = new short[InnerHeightCount];

        for (int row = 0; row < OuterDimension; row++)
        {
            for (int col = 0; col < OuterDimension; col++)
            {
                int srcY = tileY * 16 + row * 16;
                int srcX = tileX * 16 + col * 16;
                srcY = Math.Min(srcY, SrcSize - 1);
                srcX = Math.Min(srcX, SrcSize - 1);
                outerHeights[row * OuterDimension + col] = (short)Math.Clamp(Math.Round(heightmap[srcY, srcX]), short.MinValue, short.MaxValue);
            }
        }

        for (int row = 0; row < InnerDimension; row++)
        {
            for (int col = 0; col < InnerDimension; col++)
            {
                int srcY = tileY * 16 + row * 16 + 8;
                int srcX = tileX * 16 + col * 16 + 8;
                srcY = Math.Min(srcY, SrcSize - 1);
                srcX = Math.Min(srcX, SrcSize - 1);
                innerHeights[row * InnerDimension + col] = (short)Math.Clamp(Math.Round(heightmap[srcY, srcX]), short.MinValue, short.MaxValue);
            }
        }

        return new WdlHeightTile(tileX, tileY, outerHeights, innerHeights);
    }

    private static int CalculateMareaStartOffset()
    {
        int offset = 0;
        offset += 8 + 4;   // MVER: header(8) + data(4)
        offset += 8;       // MWMO: header(8) + data(0)
        offset += 8;       // MWID: header(8) + data(0)
        offset += 8;       // MODF: header(8) + data(0)
        offset += 8 + TilesPerAxis * TilesPerAxis * 4; // MAOF: header(8) + data(16384)
        return offset;
    }

    private static void WriteChunk(BinaryWriter bw, string tag, int dataSize, Action<BinaryWriter> writePayload)
    {
        bw.Write(FourCC.FromString(tag).ToFileBytes());
        bw.Write(dataSize);
        writePayload(bw);
    }
}