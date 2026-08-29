using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Procedural;

/// <summary>
/// Thematic aesthetic style for procedurally generated museum landscapes.
/// </summary>
public enum ProceduralMapTheme
{
    /// <summary>Lush Elwynn garden grass, cobblestone paths, and ornamental foliage.</summary>
    Garden,
    /// <summary>Polished marble plazas, grand stone promenades, and monumental arches.</summary>
    Marble,
    /// <summary>Autumnal golden leaves, timber boardwalks, and rustic stone.</summary>
    Autumn,
    /// <summary>Sun-baked sandstone, terracotta tiles, and palm courtyard fountains.</summary>
    Desert
}

/// <summary>
/// Unified parameters for procedural map generation across all supported client eras.
/// </summary>
public sealed record ProceduralMapGenerationParams(
    string MapName,
    string DisplayName,
    ProceduralMapTheme Theme = ProceduralMapTheme.Garden,
    DensityPreset Density = DensityPreset.Balanced,
    float GlobalM2Scale = 2.5f,
    float TerrainRoughness = 0.3f,
    float MaxTerrainSlopeDegrees = 25.0f,
    float PedestalHeight = 3.0f,
    int Seed = 1337);

/// <summary>
/// General interface for procedural world generation engines.
/// </summary>
public interface IGenerativeMapSurface
{
    /// <summary>The name and profile of the generative engine.</summary>
    string EngineName { get; }

    /// <summary>
    /// Generates a complete procedural world map plan from a collection of source assets and generation parameters.
    /// </summary>
    List<AdaptiveExhibitPlacement> GenerateMap(
        IReadOnlyList<RosettaAssetEntry> assets,
        ProceduralMapGenerationParams parameters);
}
