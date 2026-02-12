using System;
using System.Collections.Generic;
using GillijimProject.WowFiles;

namespace WoWRollback.LkToAlphaModule.Builders;

public static class AlphaMcinBuilder
{
    // Build MCIN: 256 entries * 16 bytes; layout:
    // [0..3]  = absolute MCNK offset (0 if missing)
    // [4..7]  = MCNK size (letters+size+data+pad)
    // [8..15] = flags/pad (zeros)
    public static Chunk BuildMcin(IReadOnlyList<int> mcnkAbsoluteOffsets, IReadOnlyList<int> mcnkSizes)
    {
        if (mcnkAbsoluteOffsets is null || mcnkAbsoluteOffsets.Count != 256)
            throw new ArgumentException("MCIN requires 256 MCNK offsets", nameof(mcnkAbsoluteOffsets));
        if (mcnkSizes is null || mcnkSizes.Count != 256)
            throw new ArgumentException("MCIN requires 256 MCNK sizes", nameof(mcnkSizes));

        var data = new byte[256 * 16];
        int pos = 0;
        for (int i = 0; i < 256; i++)
        {
            BitConverter.GetBytes(mcnkAbsoluteOffsets[i]).CopyTo(data, pos);
            BitConverter.GetBytes(mcnkSizes[i]).CopyTo(data, pos + 4);
            pos += 16;
        }
        return new Chunk("MCIN", data.Length, data);
    }
}
