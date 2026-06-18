using System.Numerics;

namespace WowViewer.Core.PM4.Models;

public readonly record struct TriangleFeature(
    float Edge0,
    float Edge1,
    float Edge2,
    float Area,
    Vector3 Normal,
    Vector3 Centroid)
{
    public static TriangleFeature FromTriangle(Vector3 a, Vector3 b, Vector3 c)
    {
        float e0 = Vector3.Distance(a, b);
        float e1 = Vector3.Distance(b, c);
        float e2 = Vector3.Distance(c, a);

        float area = Vector3.Cross(b - a, c - a).Length() * 0.5f;

        float[] sorted = [e0, e1, e2];
        Array.Sort(sorted);

        Vector3 normal = Vector3.Normalize(Vector3.Cross(b - a, c - a));
        Vector3 centroid = (a + b + c) / 3f;

        return new TriangleFeature(sorted[0], sorted[1], sorted[2], area, normal, centroid);
    }

    public TriangleKey Key(
        Vector3 groupDominantNormal,
        Vector3 groupCentroid,
        float edgeBinSize = 1.0f,
        float areaBinSize = 1.0f,
        float normalAlignmentBinSize = 0.0f,
        float planarOffsetBinSize = 0.0f)
    {
        int normalAlignmentBin = 0;
        if (normalAlignmentBinSize > 0.0f)
        {
            float normalAlignment = Vector3.Dot(Normal, groupDominantNormal);
            normalAlignmentBin = (int)MathF.Round(normalAlignment / normalAlignmentBinSize);
        }

        int planarOffsetBin = 0;
        if (planarOffsetBinSize > 0.0f)
        {
            float planarOffset = Vector3.Dot(Centroid - groupCentroid, groupDominantNormal);
            planarOffsetBin = (int)MathF.Round(planarOffset / planarOffsetBinSize);
        }

        return new TriangleKey(
            (int)MathF.Round(Edge0 / edgeBinSize),
            (int)MathF.Round(Edge1 / edgeBinSize),
            (int)MathF.Round(Edge2 / edgeBinSize),
            (int)MathF.Round(Area / areaBinSize),
            normalAlignmentBin,
            planarOffsetBin);
    }

    public string AbsoluteKey(
        float edgeBinSize = 1.0f,
        float areaBinSize = 1.0f,
        float normalBinSize = 0.1f,
        float heightBinSize = 1.0f)
    {
        float height = -Vector3.Dot(Normal, Centroid);
        return $"{(int)MathF.Round(Edge0 / edgeBinSize)}_{(int)MathF.Round(Edge1 / edgeBinSize)}_{(int)MathF.Round(Edge2 / edgeBinSize)}_{(int)MathF.Round(Area / areaBinSize)}_{(int)MathF.Round(Normal.X / normalBinSize)}_{(int)MathF.Round(Normal.Y / normalBinSize)}_{(int)MathF.Round(Normal.Z / normalBinSize)}_{(int)MathF.Round(height / heightBinSize)}";
    }
}

public readonly record struct TriangleKey(
    int Edge0Bin,
    int Edge1Bin,
    int Edge2Bin,
    int AreaBin,
    int NormalAlignmentBin,
    int PlanarOffsetBin)
{
    public string HistogramKey => $"{Edge0Bin}_{Edge1Bin}_{Edge2Bin}_{AreaBin}_{NormalAlignmentBin}_{PlanarOffsetBin}";
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
    float EdgeBinSize,
    float AreaBinSize,
    float NormalAlignmentBinSize,
    float PlanarOffsetBinSize,
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
    float AreaBinSize = 1.0f,
    float NormalAlignmentBinSize = 0.0f,
    float PlanarOffsetBinSize = 0.0f,
    int MaxCandidates = 10)
{
    public static SurfaceMatchOptions Default => new();
}
