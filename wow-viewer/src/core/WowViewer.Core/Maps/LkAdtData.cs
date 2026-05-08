using System.Numerics;

namespace WowViewer.Core.Maps;

public sealed class LkAdtData
{
    public string MapName { get; init; } = "";
    public int TileX { get; init; }
    public int TileY { get; init; }

    public IReadOnlyList<string> TextureNames { get; init; } = [];
    public IReadOnlyList<string> ModelNames { get; init; } = [];
    public IReadOnlyList<string> WorldModelNames { get; init; } = [];

    public IReadOnlyList<LkMddfEntry> ModelPlacements { get; init; } = [];
    public IReadOnlyList<LkModfEntry> WorldModelPlacements { get; init; } = [];

    public IReadOnlyList<LkMcnkData> Chunks { get; init; } = [];

    public uint MhdrFlags { get; init; }
}

public sealed class LkMcnkData
{
    public int IndexX { get; init; }
    public int IndexY { get; init; }
    public int Flags { get; init; }
    public int AreaId { get; init; }
    public int NLayers { get; init; }
    public int HoleMask { get; init; }
    public float BaseHeight { get; init; }
    public float[] Heights { get; init; } = [];
    public byte[] Normals { get; init; } = [];
    public byte[]? ShadowMap { get; init; }
    public byte[]? AlphaMapData { get; init; }
    public int AlphaMapSize { get; init; }
    public IReadOnlyList<LkMclyEntry> Layers { get; init; } = [];
    public IReadOnlyList<int> DoodadRefs { get; init; } = [];
    public IReadOnlyList<int> WorldModelRefs { get; init; } = [];
    public float PosX { get; init; }
    public float PosY { get; init; }
    public float PosZ { get; init; }
}

public sealed record LkMclyEntry(
    uint TextureId,
    uint Flags,
    uint AlphaOffset,
    uint EffectId);

public sealed record LkMddfEntry(
    int NameId,
    int UniqueId,
    Vector3 Position,
    Vector3 Rotation,
    float Scale);

public sealed record LkModfEntry(
    int NameId,
    int UniqueId,
    Vector3 Position,
    Vector3 Rotation,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    ushort Flags,
    ushort DoodadSet,
    ushort NameSet,
    float Scale);