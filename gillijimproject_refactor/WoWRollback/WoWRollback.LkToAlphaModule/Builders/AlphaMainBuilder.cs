using System;
using System.Collections.Generic;
using GillijimProject.WowFiles;

namespace WoWRollback.LkToAlphaModule.Builders;

public static class AlphaMainBuilder
{
    // Builds Alpha MAIN: 4096 cells * 16 bytes
    // cell layout:
    // [0..3]  = absolute offset to MHDR (letters or data+0, controlled by pointToMhdrData), or 0 if missing
    // [4..7]  = size = (first MCNK absolute offset - MHDR offset base)
    // [8..11] = flags (0)
    // [12..15]= pad (0)
    // 
    // pointToMhdrData: if true, offset points to MHDR.data (letters+8); if false, points to MHDR letters
    public static Chunk BuildMain(IReadOnlyList<int> mhdrAbsoluteOffsets, IReadOnlyList<int> mhdrToFirstMcnkSizes, bool pointToMhdrData = false)
    {
        if (mhdrAbsoluteOffsets is null || mhdrAbsoluteOffsets.Count != 4096)
            throw new ArgumentException("MAIN requires 4096 MHDR offsets", nameof(mhdrAbsoluteOffsets));
        if (mhdrToFirstMcnkSizes is null || mhdrToFirstMcnkSizes.Count != 4096)
            throw new ArgumentException("MAIN requires 4096 sizes to first MCNK", nameof(mhdrToFirstMcnkSizes));

        var data = new byte[4096 * 16];
        int pos = 0;
        for (int i = 0; i < 4096; i++)
        {
            int offset = mhdrAbsoluteOffsets[i];
            int size = mhdrToFirstMcnkSizes[i];
            
            // If pointToMhdrData is true and this tile exists, adjust offset to point to MHDR.data (+8)
            // and adjust size accordingly (-8)
            if (pointToMhdrData && offset > 0)
            {
                offset += 8; // skip MHDR letters (4 bytes) + size field (4 bytes)
                size -= 8;   // size now relative to data start instead of letters
            }
            
            BitConverter.GetBytes(offset).CopyTo(data, pos);
            BitConverter.GetBytes(size).CopyTo(data, pos + 4);
            pos += 16;
        }
        return new Chunk("MAIN", data.Length, data);
    }
}
