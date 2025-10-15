using System;
using GillijimProject.WowFiles;

namespace WoWRollback.LkToAlphaModule.Builders;

public static class AlphaMhdrBuilder
{
    private const int DataSize = 64; // Alpha/LK MHDR data size

    // Terrain-only: set MCIN offset relative to MHDR start (letters). We place MCIN right after MHDR,
    // so offsInfo = 8 (MHDR header) + 64 (MHDR data) = 72.
    public static Chunk BuildMhdrForTerrain()
    {
        var data = new byte[DataSize];
        // offsInfo relative to MHDR data start (client seems to use data-relative addressing); place MCIN right after MHDR data => 64
        BitConverter.GetBytes(64).CopyTo(data, 4);
        return new Chunk("MHDR", data.Length, data);
    }
}
