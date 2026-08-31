using System.Numerics;

namespace WowViewer.Core.Runtime.World.Terrain.Stratigraphy;

/// <summary>
/// Classification of terrain chunks and tiles into historical temporal development strata.
/// </summary>
public enum TemporalStratum
{
    /// <summary>Active, full-scale (1x) authored terrain.</summary>
    Active_1x = 0,

    /// <summary>Moderate compression (4x to 8x) from late editing passes or smoothing passes.</summary>
    LateRevision_4x_8x = 1,

    /// <summary>Classic WoWEdit compression (~30x to 33.334x = 1/0.03) from terrain erasure/wiping.</summary>
    ClassicErasure_33x = 2,

    /// <summary>Sub-millimeter early prototype relief or extreme scale down (64x to 512x).</summary>
    DeepProto_64x_512x = 3,

    /// <summary>Full-fidelity (1x) geometry preserved behind MCNK HoleMask flags (dev caves, subterranean paths, Outland blockouts).</summary>
    Holed_DevMesh_1x = 4,

    /// <summary>Underwater terrain bathymetry obscured by ocean or lake liquid planes.</summary>
    Submerged_OceanFloor = 5,

    /// <summary>Map perimeter developer asset staging area with untextured or prototype geometry.</summary>
    StagingArea_Untextured = 6,

    /// <summary>Bit-exact flat terrain (&lt;= 1 unique height level, no authored relief).</summary>
    BitExact_Flat = 7,
}

/// <summary>
/// Analysis result for a single 33.333m MCNK chunk within a tile.
/// </summary>
public sealed class ChunkStratumRecord
{
    public int ChunkX { get; init; }
    public int ChunkY { get; init; }
    public int ChunkIndex => ChunkY * 16 + ChunkX;

    public float MinHeight { get; init; }
    public float MaxHeight { get; init; }
    public float HeightRange => Math.Max(0f, MaxHeight - MinHeight);

    public int SurvivingLevels { get; init; }
    public float RelativeEntropy { get; init; }
    public ushort HoleMask { get; init; }
    public bool HasHoles => HoleMask != 0;
    public int HoledQuadCount { get; init; }

    public TemporalStratum Stratum { get; init; }
    public float EstimatedAmplificationFactor { get; init; } = 1f;
    public string Description { get; init; } = string.Empty;
}

/// <summary>
/// Analysis result for an entire 533.333m ADT / WDT tile (16x16 chunks, 257x257 height lattice).
/// </summary>
public sealed class StratigraphyTileAnalysis
{
    public string TileName { get; init; } = string.Empty;
    public int TileX { get; init; }
    public int TileY { get; init; }

    public float MinHeight { get; init; }
    public float MaxHeight { get; init; }
    public float HeightRange => Math.Max(0f, MaxHeight - MinHeight);

    public int TotalSurvivingLevels { get; init; }
    public TemporalStratum DominantStratum { get; init; }
    public float SuggestedAmplificationFactor { get; init; } = 1f;

    public int ActiveChunkCount { get; init; }
    public int SqueezedChunkCount { get; init; }
    public int HoledChunkCount { get; init; }
    public int FlatChunkCount { get; init; }

    public SeamDiscontinuityResult? SeamProfile { get; init; }
    public ChunkStratumRecord[] Chunks { get; init; } = [];

    public bool IsWeakSignalCandidate => SqueezedChunkCount > 0 || (DominantStratum != TemporalStratum.Active_1x && DominantStratum != TemporalStratum.BitExact_Flat);
    public bool HasDevMeshes => HoledChunkCount > 0;
}

/// <summary>
/// Discontinuity profile across internal MCNK boundaries (1..15) within a tile.
/// </summary>
public sealed class SeamDiscontinuityResult
{
    public float[] HorizontalBoundaryStepC0 { get; init; } = new float[15];
    public float[] HorizontalBoundarySlopeC1 { get; init; } = new float[15];
    public float[] VerticalBoundaryStepC0 { get; init; } = new float[15];
    public float[] VerticalBoundarySlopeC1 { get; init; } = new float[15];

    public bool Has2x2MergeSpikeAt8 { get; init; }
    public bool Has4x4MergeSpikesAt4_8_12 { get; init; }
    public bool IsElevatedAcrossAllBoundaries { get; init; }
    public string InferredMergeOrigin { get; init; } = "Unified";
}

/// <summary>
/// Configuration options for stratigraphy scanning and mesh restoration.
/// </summary>
public sealed class TemporalStratigraphyOptions
{
    public const float DefaultClassicFactor = 33.334f;
    public const float DefaultMaxFactor = 512f;

    public float AmplificationFactor { get; set; } = DefaultClassicFactor;
    public bool PolarityInverted { get; set; } = false;
    public StratigraphyAnchorMode AnchorMode { get; set; } = StratigraphyAnchorMode.LowestZ_Floor;
    public float? CustomAnchorHeight { get; set; }
    public float VerticalOffsetZ { get; set; } = 0f;

    public bool UseNeighborAutoFit { get; set; } = false;
    public int NeighborSearchRadiusChunks { get; set; } = 3;

    public bool UseWdlMagnetization { get; set; } = false;
    public float WdlMagnetizationStrength { get; set; } = 1.0f;

    public bool UseAutoStratumFactor { get; set; } = true;
    public bool UnhideDevMeshes { get; set; } = true;
    public bool RestoreSubmergedBathymetry { get; set; } = true;
    public bool StitchAdjacentBoundaries { get; set; } = true;
    public bool PreserveNegativeFloor { get; set; } = true;
    public float BoundaryBlendWidthMeters { get; set; } = 16.667f;
}
