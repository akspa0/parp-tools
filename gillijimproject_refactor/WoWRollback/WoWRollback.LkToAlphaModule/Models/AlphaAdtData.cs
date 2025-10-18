using System.Collections.Generic;

namespace WoWRollback.LkToAlphaModule.Models;

public sealed class AlphaAdtData
{
    // Target single-file ADT structure placeholders
    public List<float[]> ChunksHeights { get; } = new();
    public List<byte[]> AlphaMaps { get; } = new();
    public List<string> MmdxNames { get; } = new();
    public List<int> MmidOffsets { get; } = new();
    public List<string> MwmoNames { get; } = new();
    public List<int> MwidOffsets { get; } = new();
    public List<LkMddfPlacement> MddfPlacements { get; } = new();
    public List<LkModfPlacement> ModfPlacements { get; } = new();
}
