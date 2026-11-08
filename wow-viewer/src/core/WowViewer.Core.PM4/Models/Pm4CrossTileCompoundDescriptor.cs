using System.Collections.Generic;
using System.Numerics;

namespace WowViewer.Core.PM4.Models;

/// <summary>
/// Descriptor representing a compound object candidate for cross-tile spatial and MSCN clustering (Spec 301).
/// </summary>
public sealed record Pm4CrossTileCompoundDescriptor(
    int CompoundId,
    int TileX,
    int TileY,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    IReadOnlyList<Vector3> MscnWorldPoints,
    IReadOnlyCollection<Pm4ConnectorKey>? ConnectorKeys = null);
