using System.Numerics;

namespace WowViewer.Core.Maps;

public sealed class TerrainChunkData
{
    public int McinIndex { get; init; } = -1;
    public int TileX { get; init; }
    public int TileY { get; init; }
    public int ChunkX { get; init; }
    public int ChunkY { get; init; }
    public float[] Heights { get; init; } = [];
    public Vector3[] Normals { get; init; } = [];
    public int HoleMask { get; init; }
    public TerrainLayer[] Layers { get; init; } = [];
    public Dictionary<int, byte[]> AlphaMaps { get; init; } = new();
    public byte[]? ShadowMap { get; init; }
    public byte[]? MccvColors { get; init; }
    public LiquidChunkData? Liquid { get; set; }
    public Vector3 WorldPosition { get; init; }
    public int AreaId { get; init; }
    public int McnkFlags { get; init; }
    public int AlphaSourceFlags { get; init; }
}

public readonly struct TerrainLayer
{
    public int TextureIndex { get; init; }
    public uint Flags { get; init; }
    public uint AlphaOffset { get; init; }
    public uint EffectId { get; init; }
}

public sealed class LiquidChunkData
{
    public int LiquidType { get; init; }
    public float MinHeight { get; init; }
    public float MaxHeight { get; init; }
    public byte[]? TileFlags { get; init; }
}