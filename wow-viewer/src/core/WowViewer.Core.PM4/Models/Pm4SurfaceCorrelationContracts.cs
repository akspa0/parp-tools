using System.Numerics;

namespace WowViewer.Core.PM4.Models;

public readonly record struct TriangleFeature(
    float Edge0,
    float Edge1,
    float Edge2,
    float Area)
{
    public static TriangleFeature FromTriangle(Vector3 a, Vector3 b, Vector3 c)
    {
        float e0 = Vector3.Distance(a, b);
        float e1 = Vector3.Distance(b, c);
        float e2 = Vector3.Distance(c, a);

        float area = Vector3.Cross(b - a, c - a).Length() * 0.5f;

        float[] sorted = [e0, e1, e2];
        Array.Sort(sorted);

        return new TriangleFeature(sorted[0], sorted[1], sorted[2], area);
    }

    public TriangleKey Key(float binSize = 1.0f)
    {
        return new TriangleKey(
            (int)MathF.Round(Edge0 / binSize),
            (int)MathF.Round(Edge1 / binSize),
            (int)MathF.Round(Edge2 / binSize));
    }
}

public readonly record struct TriangleKey(int Edge0Bin, int Edge1Bin, int Edge2Bin)
{
    public string HistogramKey => $"{Edge0Bin}_{Edge1Bin}_{Edge2Bin}";
}

public sealed record SurfaceCorrelationFingerprint(
    string AssetId,
    string AssetPath,
    string AssetKind,
    byte Ck24Type,
    int SurfaceCount,
    int TriangleCount,
    int VertexCount,
    IReadOnlyDictionary<string, int> TriangleHistogram,
    string SourceLabel)
{
    public float MeanTriangleArea { get; set; }
    public float TotalTriangleArea { get; set; }
}

public sealed record SurfaceCorrelationDatabase(
    string ArchiveRoot,
    string BuildDate,
    int WmoCount,
    IReadOnlyList<SurfaceCorrelationFingerprint> Records);

public sealed record SurfaceMatchCandidate(
    SurfaceCorrelationFingerprint Candidate,
    int Rank,
    int Pm4TrianglesMatched,
    int Pm4TriangleTotal,
    int WmoTrianglesMatched,
    int WmoTriangleTotal,
    double Pm4Coverage,
    double WmoCoverage,
    double SymmetricScore,
    string Status,
    IReadOnlyList<string> Rationale);

public sealed record SurfaceMatchResult(
    string Pm4FingerprintId,
    string Pm4AssetPath,
    byte Ck24Type,
    int SurfaceCount,
    int TriangleCount,
    int VertexCount,
    string Status,
    bool ReviewRequired,
    IReadOnlyList<string> Rationale,
    IReadOnlyList<SurfaceMatchCandidate> Candidates);

public sealed record SurfaceMatchOptions(
    double MinScore = 0.50,
    double AmbiguousWindow = 0.03,
    float EdgeBinSize = 1.0f,
    int MaxCandidates = 10)
{
    public static SurfaceMatchOptions Default => new();
}
