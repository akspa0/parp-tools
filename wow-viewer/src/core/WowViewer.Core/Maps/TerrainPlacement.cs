using System.Numerics;

namespace WowViewer.Core.Maps;

public readonly record struct MddfPlacement(
    int NameId,
    string ModelPath,
    int UniqueId,
    Vector3 Position,
    Vector3 Rotation,
    float Scale);

public readonly record struct ModfPlacement(
    int NameId,
    string ModelPath,
    int UniqueId,
    Vector3 Position,
    Vector3 Rotation,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    ushort Flags);