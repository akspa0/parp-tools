namespace WoWMapConverter.Core.Builders;

/// <summary>
/// Builds Alpha MAIN chunk (4096 tile entries).
/// </summary>
public static class AlphaMainBuilder
{
    /// <summary>
    /// Build MAIN chunk data (without chunk header).
    /// Each entry is 16 bytes: [offset:4][size:4][flags:4][pad:4]
    /// </summary>
    /// <param name="mhdrOffsets">Absolute offsets to MHDR for each tile (4096 entries)</param>
    /// <param name="mhdrToFirstMcnkSizes">Size from MHDR to first MCNK (4096 entries)</param>
    public static byte[] BuildMain(int[] mhdrOffsets, int[] mhdrToFirstMcnkSizes)
    {
        if (mhdrOffsets.Length != 4096)
            throw new ArgumentException("MAIN requires 4096 MHDR offsets");
        if (mhdrToFirstMcnkSizes.Length != 4096)
            throw new ArgumentException("MAIN requires 4096 sizes");

        var data = new byte[4096 * 16];
        int pos = 0;

        for (int i = 0; i < 4096; i++)
        {
            BitConverter.GetBytes(mhdrOffsets[i]).CopyTo(data, pos);
            BitConverter.GetBytes(mhdrToFirstMcnkSizes[i]).CopyTo(data, pos + 4);
            // flags and pad remain 0
            pos += 16;
        }

        return data;
    }
}
