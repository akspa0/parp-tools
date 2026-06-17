using System.Numerics;

namespace WowViewer.Core.PM4.Models;

public enum Pm4FingerprintMatchStatus
{
    Matched,
    Ambiguous,
    Unresolved,
    Ineligible,
}

public sealed record HullPoint2(float X, float Y)
{
    public Vector2 AsVector2() => new(X, Y);

    public static HullPoint2 FromVector2(Vector2 v) => new(v.X, v.Y);
}

public sealed record Bounds3Serial(float MinX, float MinY, float MinZ, float MaxX, float MaxY, float MaxZ)
{
    public Vector3 Min => new(MinX, MinY, MinZ);
    public Vector3 Max => new(MaxX, MaxY, MaxZ);
    public float SpanX => MaxX - MinX;
    public float SpanY => MaxY - MinY;
    public float SpanZ => MaxZ - MinZ;

    public static Bounds3Serial FromBounds(Vector3 min, Vector3 max) =>
        new(min.X, min.Y, min.Z, max.X, max.Y, max.Z);
}

public sealed record Pm4FingerprintRecord(
    string AssetId,
    string AssetPath,
    string AssetKind,
    byte Ck24Type,
    int SurfaceCount,
    int VertexCount,
    int IndexCount,
    int GroupCount,
    float SortedDim0,
    float SortedDim1,
    float SortedDim2,
    Bounds3Serial NormalizedBounds,
    HullPoint2 NormalizedCenter,
    IReadOnlyList<HullPoint2> NormalizedFootprintHull,
    float FootprintArea,
    IReadOnlyDictionary<byte, int> TypeFlagsProfile,
    string SourceLabel)
{
    public IReadOnlyList<Vector2> FootprintHullAsVectors =>
        NormalizedFootprintHull.Select(static p => p.AsVector2()).ToList();
}

public sealed record Pm4FingerprintDatabase(
    string ArchiveRoot,
    string BuildDate,
    int WmoCount,
    IReadOnlyList<Pm4FingerprintRecord> Records)
{
    public IReadOnlyList<Pm4FingerprintRecord> WmoRecords =>
        Records.Where(static r => string.Equals(r.AssetKind, "wmo", StringComparison.OrdinalIgnoreCase)).ToList();

    public IReadOnlyList<Pm4FingerprintRecord> M2Records =>
        Records.Where(static r => string.Equals(r.AssetKind, "m2", StringComparison.OrdinalIgnoreCase)).ToList();
}

public sealed record Pm4FingerprintMatchCandidate(
    Pm4FingerprintRecord Candidate,
    int Rank,
    Pm4FingerprintMatchStatus Status,
    float FootprintOverlapRatio,
    float FootprintAreaRatio,
    float FootprintDistance,
    float PlanarGap,
    float VerticalGap,
    float CenterDistance,
    float PlanarOverlapRatio,
    float VolumeOverlapRatio,
    double OverallScore,
    IReadOnlyList<string> Rationale);

public sealed record Pm4FingerprintMatchResult(
    string Pm4FingerprintId,
    string Pm4AssetPath,
    byte Ck24Type,
    uint Ck24,
    int SurfaceCount,
    int VertexCount,
    int IndexCount,
    float SortedDim0,
    float SortedDim1,
    float SortedDim2,
    Pm4FingerprintMatchStatus Status,
    bool ReviewRequired,
    IReadOnlyList<string> Rationale,
    IReadOnlyList<Pm4FingerprintMatchCandidate> Candidates);

public sealed record Pm4FingerprintMatchOptions(
    double MinScore = 0.45,
    double AmbiguousWindow = 0.03,
    float DimPrefilterTolerance = 0.25f,
    int MaxCandidates = 10)
{
    public static Pm4FingerprintMatchOptions Default => new();
}
