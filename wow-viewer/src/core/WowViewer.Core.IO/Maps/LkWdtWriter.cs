using System.Text;
using WowViewer.Core.Chunks;

namespace WowViewer.Core.IO.Maps;

public sealed class LkWdtWriteOptions
{
    public bool HasMccv { get; init; }
    public bool HasBigAlpha { get; init; }
    public bool HasMtxf { get; init; }
    public bool HasMaid { get; init; }
    public bool HasMclv { get; init; }
}

public static class LkWdtWriter
{
    private const int TilesPerAxis = 64;
    private const int MainEntrySize = 8;
    private const int MainDataSize = TilesPerAxis * TilesPerAxis * MainEntrySize;

    [Flags]
    public enum MphdFlags : uint
    {
        HasGlobalMapObj = 0x0001,
        AdtHasMccv = 0x0002,
        AdtHasBigAlpha = 0x0004,
        DoodadSizeSorting = 0x0008,
        HasMclv = 0x0020,
        HasMtxf = 0x0100,
        HasMaid = 0x0200,
    }

    [Flags]
    public enum MainFlags : uint
    {
        HasAdt = 0x0001,
        AllWater = 0x0002,
    }

    public static void Write(string path, HashSet<(int tileX, int tileY)> existingTiles, LkWdtWriteOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(existingTiles);

        string? dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        File.WriteAllBytes(path, Build(existingTiles, options));
    }

    public static byte[] Build(HashSet<(int tileX, int tileY)> existingTiles, LkWdtWriteOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(existingTiles);

        var mphdFlags = ComputeMphdFlags(options);

        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        WriteChunk(bw, "MVER", 4, w => w.Write(18));

        WriteChunk(bw, "MPHD", 32, w =>
        {
            w.Write((uint)mphdFlags);
            w.Write(0u);
            for (int i = 0; i < 7; i++)
                w.Write(0u);
        });

        WriteChunk(bw, "MAIN", MainDataSize, w =>
        {
            for (int y = 0; y < TilesPerAxis; y++)
            {
                for (int x = 0; x < TilesPerAxis; x++)
                {
                    if (existingTiles.Contains((x, y)))
                        w.Write((uint)MainFlags.HasAdt);
                    else
                        w.Write(0u);

                    w.Write(0u);
                }
            }
        });

        WriteChunk(bw, "MWMO", 0, static _ => { });
        WriteChunk(bw, "MODF", 0, static _ => { });

        bw.Flush();
        return ms.ToArray();
    }

    private static MphdFlags ComputeMphdFlags(LkWdtWriteOptions? options)
    {
        if (options is null)
            return 0;

        MphdFlags flags = 0;
        if (options.HasMccv) flags |= MphdFlags.AdtHasMccv;
        if (options.HasBigAlpha) flags |= MphdFlags.AdtHasBigAlpha;
        if (options.HasMtxf) flags |= MphdFlags.HasMtxf;
        if (options.HasMaid) flags |= MphdFlags.HasMaid;
        if (options.HasMclv) flags |= MphdFlags.HasMclv;
        return flags;
    }

    private static void WriteChunk(BinaryWriter bw, string tag, int dataSize, Action<BinaryWriter> writePayload)
    {
        bw.Write(FourCC.FromString(tag).ToFileBytes());
        bw.Write(dataSize);
        writePayload(bw);
    }
}