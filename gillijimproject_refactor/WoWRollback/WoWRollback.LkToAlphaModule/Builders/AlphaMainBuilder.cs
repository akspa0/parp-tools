using System;
using System.Collections.Generic;
using GillijimProject.WowFiles;

namespace WoWRollback.LkToAlphaModule.Builders;

public static class AlphaMainBuilder
{
    // Builds Alpha MAIN: 4096 cells * 16 bytes
    // cell layout:
    // [0..3]  = absolute offset to MHDR start (letters), or 0 if missing
    // [4..7]  = size = (first MCNK absolute offset - MHDR start)
    // [8..11] = flags (0)
    // [12..15]= pad (0)
    public static Chunk BuildMain(IReadOnlyList<int> mhdrAbsoluteOffsets, IReadOnlyList<int> mhdrToFirstMcnkSizes)
    {
        if (mhdrAbsoluteOffsets is null || mhdrAbsoluteOffsets.Count != 4096)
            throw new ArgumentException("MAIN requires 4096 MHDR offsets", nameof(mhdrAbsoluteOffsets));
        if (mhdrToFirstMcnkSizes is null || mhdrToFirstMcnkSizes.Count != 4096)
            throw new ArgumentException("MAIN requires 4096 sizes to first MCNK", nameof(mhdrToFirstMcnkSizes));

        var data = new byte[4096 * 16];
        int pos = 0;
        for (int i = 0; i < 4096; i++)
        {
            BitConverter.GetBytes(mhdrAbsoluteOffsets[i]).CopyTo(data, pos);
            BitConverter.GetBytes(mhdrToFirstMcnkSizes[i]).CopyTo(data, pos + 4);
            pos += 16;
        }
        return new Chunk("MAIN", data.Length, data);
    }
}
