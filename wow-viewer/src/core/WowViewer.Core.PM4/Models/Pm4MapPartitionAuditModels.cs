using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Research;

namespace WowViewer.Core.PM4.Models;

public sealed record Pm4MapAuditInput(string FileId, Pm4ResearchDocument Document);
public sealed record Pm4MapPlacementInput(string FileId, AdtPlacementCatalog Catalog);

public sealed record Pm4MapSurfaceNode(
    int Id, string FileId, int LocalIndex, uint? HeaderField04, Pm4MsurEntry Raw,
    bool WindowValid, IReadOnlyList<int> OwnedLinkIds, int InvalidVertexIndices);

public sealed record Pm4MapLinkNode(
    int Id, string FileId, int LocalIndex, Pm4MslkEntry Raw,
    IReadOnlyList<int> WindowOwnerIds, int? SurfaceReferenceId,
    bool FitsMprlIndexDomain, bool PathWindowValid,
    string? AddressedFileId = null, int? AddressedSurfaceId = null,
    string AddressStatus = "NotEvaluated");

public sealed record Pm4MapGeometryIncidence(int X, int Y, int Z, IReadOnlyList<int> SurfaceIds);
public sealed record Pm4MapAuditEdge(int OwnerSurfaceId, int LinkId, int ReferencedSurfaceId);
public sealed record Pm4MapSurfaceFieldValue(string Field, uint RawValue, IReadOnlyList<int> SurfaceIds);
public sealed record Pm4MapWindowControl(string LengthHypothesis, int InvalidWindows, int UnownedLinks, int MultiplyOwnedLinks, int OwnerEqualsReference, int ValidSurfaceReferences);

public sealed record Pm4MapFieldValue(
    string Field, uint RawValue, IReadOnlyList<int> LinkIds,
    IReadOnlyList<int> WindowOwnerIds, IReadOnlyList<int> ReferencedSurfaceIds, int Files);

public sealed record Pm4MapShape(
    int ValidTriangles, int InvalidTriangles, Vector3 Centroid, float RmsRadius,
    string? NormalizedSignature, IReadOnlyDictionary<string, int> NormalizedHistogram);

public sealed record Pm4MapPartitionComponent(
    int Id, IReadOnlyList<int> SurfaceIds, int ZeroCk24Surfaces, int RawWordZeroSurfaces,
    IReadOnlyList<string> Files, Pm4MapShape Shape);

public sealed record Pm4MapScaleFamily(
    string Signature, IReadOnlyList<int> ComponentIds, float MinimumRadius, float MaximumRadius,
    float ScaleRatio, int Files);

public sealed record Pm4MapPartitionHypothesis(
    string Name, string Assumption, IReadOnlyList<Pm4MapPartitionComponent> Components,
    IReadOnlyList<Pm4MapScaleFamily> CandidateScaleFamilies);

public sealed record Pm4MapPlacementReference(
    string FileId, string Kind, string ModelPath, int UniqueId,
    Vector3 Position, Vector3 Rotation, float Scale);

public sealed record Pm4MapAuthoredFamily(
    string ModelPath, IReadOnlyList<int> PlacementIndices, IReadOnlyList<float> Scales,
    IReadOnlyList<string> Files);

public sealed record Pm4MapPartitionAuditReport(
    string Warning, int FileCount, int SurfaceCount, int LinkCount,
    int ZeroCk24SurfaceCount, int RawWordZeroSurfaceCount,
    int InvalidSurfaceWindows, int UnownedLinks, int InvalidSurfaceReferences,
    int AnchorOnlyLinks, int InvalidPathWindows,
    IReadOnlyList<Pm4MapSurfaceNode> Surfaces, IReadOnlyList<Pm4MapLinkNode> Links,
    IReadOnlyList<Pm4MapAuditEdge> WindowReferenceEdges,
    IReadOnlyList<Pm4MapAuditEdge> AddressedWindowReferenceEdges,
    IReadOnlyList<Pm4MapGeometryIncidence> CrossFileGeometry,
    IReadOnlyList<Pm4MapFieldValue> FieldValues,
    IReadOnlyList<Pm4MapSurfaceFieldValue> SurfaceFieldValues,
    IReadOnlyList<Pm4MapWindowControl> WindowControls,
    IReadOnlyList<Pm4MapPartitionHypothesis> Hypotheses,
    IReadOnlyList<Pm4MapPlacementReference> AuthoredPlacements,
    IReadOnlyList<Pm4MapAuthoredFamily> AuthoredM2Families);
