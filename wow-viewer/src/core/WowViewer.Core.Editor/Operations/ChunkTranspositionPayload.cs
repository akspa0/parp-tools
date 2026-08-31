using System.Numerics;

namespace WowViewer.Core.Editor.Operations;

/// <summary>
/// Serializable clipboard / payload representation of transposed chunk data.
/// </summary>
public sealed class ChunkTranspositionPayload
{
    public GlobalChunkCoordinate Origin { get; set; }
    public int WidthInChunks { get; set; }
    public int HeightInChunks { get; set; }
    public List<TransposedChunkRecord> Chunks { get; } = new();
    public List<TransposedObjectPlacement> Placements { get; } = new();
}

public sealed class TransposedChunkRecord
{
    public int RelativeGx { get; set; }
    public int RelativeGy { get; set; }
    public float[]? Heights { get; set; }
    public Vector3[]? Normals { get; set; }
    public int HoleMask { get; set; }
    public int AreaId { get; set; }
    public int McnkFlags { get; set; }
    public List<TransposedLayerRecord> Layers { get; } = new();
    public byte[]? ShadowMap { get; set; }
    public byte[]? MccvColors { get; set; }
    public byte[]? Liquid { get; set; }
}

public sealed class TransposedLayerRecord
{
    public int TextureIndex { get; set; }
    public string TexturePath { get; set; } = string.Empty;
    public byte[]? AlphaMap { get; set; }
    public int Flags { get; set; }
    public int EffectId { get; set; }
}

public sealed class TransposedObjectPlacement
{
    public bool IsWmo { get; set; }
    public string AssetPath { get; set; } = string.Empty;
    public Vector3 RelativePosition { get; set; }
    public Vector3 Rotation { get; set; }
    public float Scale { get; set; } = 1.0f;
    public int UniqueId { get; set; }
}
