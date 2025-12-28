using System.Text;

namespace WoWMapConverter.Core.Builders;

/// <summary>
/// Builds Alpha MHDR (Map Header) chunks.
/// </summary>
public static class AlphaMhdrBuilder
{
    private const int DataSize = 64;

    /// <summary>
    /// Build MHDR for terrain tile.
    /// MCIN is placed right after MHDR, so offsInfo = 64 (relative to MHDR data start).
    /// </summary>
    public static byte[] BuildMhdr()
    {
        var data = new byte[DataSize];
        // offsInfo: offset to MCIN relative to MHDR data start
        // MCIN follows immediately after MHDR data, so offset = 64
        BitConverter.GetBytes(64).CopyTo(data, 0);
        
        // Build complete chunk with reversed FourCC
        using var ms = new MemoryStream();
        ms.Write(Encoding.ASCII.GetBytes("RDHM")); // MHDR reversed
        ms.Write(BitConverter.GetBytes(data.Length));
        ms.Write(data);
        return ms.ToArray();
    }
}
