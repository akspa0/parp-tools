using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Lit;

/// <summary>
/// One LIT light-list entry.
///
/// COORDINATE SCALE: the file stores position and both radii in the client's fixed-point spatial
/// units, the same 1/36 scale <see cref="TerrainLightingMath.ClientFixedUnitsPerWorldUnit"/> already
/// documents ("the same 1/36 fixed scale as the outdoor-light spatial records"). Fog distances were
/// being converted through <see cref="TerrainLightingMath.ComputeClientFogRange"/> while these
/// spatial records were not, so every consumer saw positions ~36x too large -- far outside any real
/// map extent, which is why plotted lights landed nowhere sensible.
///
/// Raw file values stay available as <see cref="RawPosition"/> / <see cref="RawLightRadius"/> /
/// <see cref="RawLightDropoff"/>; decoded source data is never silently rewritten. Everything that
/// works in world space should use <see cref="Position"/>, <see cref="LightRadius"/>,
/// <see cref="LightDropoff"/>, and <see cref="OuterRadius"/>.
/// </summary>
public sealed class LitListEntrySummary
{
    /// <param name="position">Raw fixed-point position exactly as stored in the LIT file.</param>
    /// <param name="lightRadius">Raw fixed-point core radius exactly as stored in the LIT file.</param>
    /// <param name="lightDropoff">Raw fixed-point falloff distance exactly as stored in the LIT file.</param>
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
        RawPosition = position;
        RawLightRadius = lightRadius;
        RawLightDropoff = lightDropoff;
        Name = name?.Trim() ?? string.Empty;
        IsDefaultEntry = chunkX == -1 && chunkY == -1 && chunkRadius == -1;
    }

    public int Index { get; }

    public int ChunkX { get; }

    public int ChunkY { get; }

    public int ChunkRadius { get; }

    /// <summary>Position in client fixed-point units, exactly as stored in the file.</summary>
    public Vector3 RawPosition { get; }

    /// <summary>Core radius in client fixed-point units, exactly as stored in the file.</summary>
    public float RawLightRadius { get; }

    /// <summary>Falloff distance in client fixed-point units, exactly as stored in the file.</summary>
    public float RawLightDropoff { get; }

    /// <summary>
    /// Position in renderer world units. Note the file's Z is the vertical axis, which the viewer
    /// displays as its Y; consumers plotting onto a top-down minimap want X and Y.
    /// </summary>
    public Vector3 Position => RawPosition / TerrainLightingMath.ClientFixedUnitsPerWorldUnit;

    /// <summary>Core radius in renderer world units.</summary>
    public float LightRadius => RawLightRadius / TerrainLightingMath.ClientFixedUnitsPerWorldUnit;

    /// <summary>Falloff distance in renderer world units.</summary>
    public float LightDropoff => RawLightDropoff / TerrainLightingMath.ClientFixedUnitsPerWorldUnit;

    public string Name { get; }

    public bool IsDefaultEntry { get; }

    public bool HasName => !string.IsNullOrWhiteSpace(Name);

    /// <summary>Outer influence radius in renderer world units.</summary>
    public float OuterRadius => MathF.Max(LightRadius, LightRadius + MathF.Max(LightDropoff, 0f));
}
