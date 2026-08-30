using System.Numerics;

namespace WowViewer.Core.Editor.Operations;

public enum TerrainStampBlendMode
{
    Additive,
    Replace,
    Maximum,
    Minimum
}

/// <summary>
/// Configuration options for applying a terrain brush paste onto terrain.
/// </summary>
public sealed class TerrainStampOptions
{
    public float CenterWorldX { get; init; }
    public float CenterWorldY { get; init; }
    public float Scale { get; init; } = 1.0f;
    public float RotationDegrees { get; init; } = 0.0f;
    public float HeightMultiplier { get; init; } = 1.0f;
    public float FeatherRadiusMeters { get; init; } = 4.0f;
    public float StampWeight { get; init; } = 1.0f;
    public TerrainStampBlendMode BlendMode { get; init; } = TerrainStampBlendMode.Additive;
}
