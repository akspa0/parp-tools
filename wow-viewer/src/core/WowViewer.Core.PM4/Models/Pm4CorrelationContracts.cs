using System.Numerics;

namespace WowViewer.Core.PM4.Models;

public readonly record struct Pm4CorrelationMetrics(
    float PlanarGap,
    float VerticalGap,
    float CenterDistance,
    float PlanarOverlapRatio,
    float VolumeOverlapRatio,
    float FootprintOverlapRatio,
    float FootprintAreaRatio,
    float FootprintDistance);

public readonly record struct Pm4CorrelationCandidateScore(
    bool SameTile,
    Pm4CorrelationMetrics Metrics,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3 Center);

public sealed record Pm4CorrelationObjectDescriptor(
    uint Ck24,
    byte Ck24Type,
    int ObjectPartId,
    uint LinkGroupObjectId,
    int SurfaceCount,
    int LinkedPositionRefCount,
    byte DominantGroupKey,
    byte DominantAttributeMask,
    uint DominantMdosIndex,
    float AverageSurfaceHeight)
{
    public ushort Ck24ObjectId => (ushort)(Ck24 & 0xFFFF);
}

public sealed record Pm4CorrelationObjectInput(
    int TileX,
    int TileY,
    Pm4ObjectGroupKey GroupKey,
    Pm4CorrelationObjectDescriptor Object,
    IReadOnlyList<Vector3> WorldGeometryPoints,
    Vector3 EmptyGeometryCenter);

public readonly record struct Pm4GeometryLineSegment(Vector3 From, Vector3 To);

public readonly record struct Pm4GeometryTriangle(Vector3 A, Vector3 B, Vector3 C);

public sealed record Pm4CorrelationGeometryInput(
    int TileX,
    int TileY,
    Pm4ObjectGroupKey GroupKey,
    Pm4CorrelationObjectDescriptor Object,
    IReadOnlyList<Pm4GeometryLineSegment> Lines,
    IReadOnlyList<Pm4GeometryTriangle> Triangles,
    Matrix4x4 GeometryTransform);

public sealed record Pm4CorrelationObjectState(
    int TileX,
    int TileY,
    Pm4ObjectGroupKey GroupKey,
    Pm4CorrelationObjectDescriptor Object,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3 Center,
    IReadOnlyList<Vector2> FootprintHull,
    float FootprintArea);