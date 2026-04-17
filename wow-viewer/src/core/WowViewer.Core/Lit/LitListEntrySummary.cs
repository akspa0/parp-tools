using System.Numerics;

namespace WowViewer.Core.Lit;

public sealed class LitListEntrySummary
{
    public LitListEntrySummary(
        int index,
        int chunkX,
        int chunkY,
        int chunkRadius,
        Vector3 position,
        float lightRadius,
        float lightDropoff,
        string? name)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        Index = index;
        ChunkX = chunkX;
        ChunkY = chunkY;
        ChunkRadius = chunkRadius;
        Position = position;
        LightRadius = lightRadius;
        LightDropoff = lightDropoff;
        Name = name?.Trim() ?? string.Empty;
        IsDefaultEntry = chunkX == -1 && chunkY == -1 && chunkRadius == -1;
    }

    public int Index { get; }

    public int ChunkX { get; }

    public int ChunkY { get; }

    public int ChunkRadius { get; }

    public Vector3 Position { get; }

    public float LightRadius { get; }

    public float LightDropoff { get; }

    public string Name { get; }

    public bool IsDefaultEntry { get; }

    public bool HasName => !string.IsNullOrWhiteSpace(Name);

    public float OuterRadius => MathF.Max(LightRadius, LightRadius + MathF.Max(LightDropoff, 0f));
}