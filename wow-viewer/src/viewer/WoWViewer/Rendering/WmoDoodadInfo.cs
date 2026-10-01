using System.Numerics;

namespace WoWViewer.Rendering;

/// <summary>
/// One placed doodad inside a WMO, as described by its MODD record.
/// </summary>
/// <remarks>
/// MODD carries <c>NameIndex, Position, Orientation, Scale, Color</c> and <b>no uniqueId</b>.
/// uniqueId is an MDDF/MODF concept — it identifies a placement in an ADT, not a doodad inside a
/// WMO. <see cref="DoodadDefIndex"/> is an index into this WMO's own MODD table and is meaningful
/// only relative to this WMO. Do not present it as a uniqueId and do not feed it to anything that
/// keys on uniqueId; two different WMOs both have a doodad 65.
/// </remarks>
public readonly record struct WmoDoodadInfo(
    int Index,
    string ModelPath,
    int DoodadDefIndex,
    Vector3 LocalPosition,
    bool Visible,
    bool IsLoaded,
    Quaternion Orientation = default,
    float Scale = 1f,
    uint NameIndex = 0);
