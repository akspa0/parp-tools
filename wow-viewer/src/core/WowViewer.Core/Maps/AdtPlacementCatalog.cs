using System.Numerics;

namespace WowViewer.Core.Maps;

public sealed record AdtPlacementCatalog(
    string SourcePath,
    MapFileKind Kind,
    IReadOnlyList<string> ModelNames,
    IReadOnlyList<string> WorldModelNames,
    IReadOnlyList<AdtModelPlacement> ModelPlacements,
    IReadOnlyList<AdtWorldModelPlacement> WorldModelPlacements);

public sealed record AdtModelPlacement(
    int NameId,
    string ModelPath,
    int UniqueId,
    Vector3 Position,
    Vector3 Rotation,
    float Scale);

public sealed record AdtWorldModelPlacement(
    int NameId,
    string ModelPath,
    int UniqueId,
    Vector3 Position,
    Vector3 Rotation,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    ushort Flags);