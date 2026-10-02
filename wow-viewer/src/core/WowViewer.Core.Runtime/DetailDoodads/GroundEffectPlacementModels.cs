using System.Numerics;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Runtime.DetailDoodads;

/// <summary>
/// A single evaluated and placed detail doodad instance ready for instanced GPU rendering.
/// </summary>
public readonly record struct DetailDoodadInstance(
    Vector3 Position,
    Quaternion Orientation,
    float Scale,
    uint ColorBgra,
    uint DoodadId,
    uint? FileDataId,
    string? ModelPath,
    GroundEffectDoodadFlags Flags)
{
    public bool AlignToNormal => (Flags & GroundEffectDoodadFlags.AlignToNormal) != 0;
    public bool IgnoreMccv => (Flags & GroundEffectDoodadFlags.IgnoreMCCV) != 0;
}

/// <summary>
/// A texture layer input for ground effect generation.
/// </summary>
public sealed class TerrainChunkLayerInput
{
    public uint EffectId { get; init; }
    public int TextureIndex { get; init; }
    public byte[]? AlphaMap { get; init; }
}

/// <summary>
/// Input data extracted from a terrain chunk (MCNK) needed to evaluate ground effect doodads.
/// </summary>
public sealed class TerrainChunkPlacementInput
{
    public Vector3 WorldPosition { get; init; }
    public float[] Heights { get; init; } = Array.Empty<float>();
    public Vector3[] Normals { get; init; } = Array.Empty<Vector3>();
    public byte[]? MccvColors { get; init; }
    public byte[]? ShadowMap { get; init; }
    public ulong HoleMask64 { get; init; }
    public int HoleMask { get; init; }
    public IReadOnlyList<TerrainChunkLayerInput> Layers { get; init; } = Array.Empty<TerrainChunkLayerInput>();
    public int ChunkX { get; init; }
    public int ChunkY { get; init; }
    public int TileX { get; init; }
    public int TileY { get; init; }

    public bool IsCellHoled(int cellX, int cellY)
    {
        if (HoleMask64 != 0UL)
            return Maps.TerrainHoleMath.IsCellHoled64(HoleMask64, cellX, cellY);
        if (HoleMask != 0)
            return Maps.TerrainHoleMath.IsCellHoled16((ushort)HoleMask, cellX, cellY);
        return false;
    }
}

/// <summary>
/// Input data from a WMO group for evaluating WMO detail doodad placements.
/// </summary>
public sealed class WmoGroupPlacementInput
{
    public int GroupIndex { get; init; }
    public Vector3[] Vertices { get; init; } = Array.Empty<Vector3>();
    public Vector3[] Normals { get; init; } = Array.Empty<Vector3>();
    public ushort[] Indices { get; init; } = Array.Empty<ushort>();
    public Matrix4x4 WorldTransform { get; init; } = Matrix4x4.Identity;
    public IReadOnlyList<WmoDetailDoodadLayer> Layers { get; init; } = Array.Empty<WmoDetailDoodadLayer>();
    public IReadOnlyList<WmoDetailDoodadDecodedCommand> Commands { get; init; } = Array.Empty<WmoDetailDoodadDecodedCommand>();
}
