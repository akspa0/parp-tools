using System.Collections.Generic;

namespace WoWRollback.LkToAlphaModule.Models;

public sealed class LkAdtData
{
    // Terrain
    public List<float[]> ChunksHeights { get; } = new(); // placeholder per-MCNK 145 heights
    public List<byte[]> McvtChunks { get; } = new(); // raw MCVT bytes per MCNK (LK order)

    // Textures
    public List<byte[]> AlphaMaps { get; } = new(); // placeholder raw alpha maps

    // Placements
    public List<string> ModelNames { get; } = new();
    public List<string> WmoNames { get; } = new();

    // Liquids
    public bool HasMh2o { get; set; }
}
