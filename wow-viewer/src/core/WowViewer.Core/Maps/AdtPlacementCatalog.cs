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

/// <param name="DoodadSet">
/// Index of the WMO's own doodad set to draw with this placement - the rocks and props that come
/// draped on the model rather than as separate MDDF rows. Read but not yet acted on: nothing in the
/// world render path loads WMO doodads, so a placement whose set is non-zero still shows bare.
/// </param>
public sealed record AdtWorldModelPlacement(
    int NameId,
    string ModelPath,
    int UniqueId,
    Vector3 Position,
    Vector3 Rotation,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    ushort Flags,
    ushort DoodadSet = 0,
    ushort NameSet = 0,
    ushort Scale = 0);