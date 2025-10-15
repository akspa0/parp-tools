using System.Collections.Generic;

namespace WoWRollback.LkToAlphaModule.Models;

public sealed class AlphaAdtData
{
    // Target single-file ADT structure placeholders
    public List<float[]> ChunksHeights { get; } = new();
    public List<byte[]> AlphaMaps { get; } = new();
    public List<string> MmdxNames { get; } = new();
    public List<string> MwmoNames { get; } = new();
}
