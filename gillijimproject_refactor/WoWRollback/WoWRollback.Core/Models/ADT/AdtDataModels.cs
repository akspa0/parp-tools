using System.Numerics;

namespace WoWRollback.Core.Models.ADT;

public record AdtData
{
    public string MapName { get; init; } = "";
    public int TileX { get; init; }
    public int TileY { get; init; }
    public List<string> Textures { get; init; } = new();
    public List<AdtChunk> Chunks { get; init; } = new();
    public List<AdtM2Placement> M2Objects { get; init; } = new();
    public List<AdtWmoPlacement> WmoObjects { get; init; } = new();
}

public record AdtChunk
{
    public int IndexX { get; init; }
    public int IndexY { get; init; }
    public float PositionX { get; init; }
    public float PositionY { get; init; }
    public float PositionZ { get; init; }
    public float[] Heights { get; init; } = Array.Empty<float>();
    public int StartIndex { get; init; } // For mesh generation
    public int Holes { get; init; }
    public bool HasWater { get; init; }
    public List<AdtTextureLayer> Layers { get; init; } = new();
    public byte[]? AlphaMap { get; init; } // Raw MCAL data
    public byte[]? ShadowMap { get; init; } // Raw MCSH data
}

public record AdtTextureLayer
{
    public string TextureName { get; init; } = "";
    public int TextureId { get; init; }
    public uint Flags { get; init; }
    public int AlphaOffset { get; init; }
    public int EffectId { get; init; }
}

public record AdtM2Placement
{
    public string ModelName { get; init; } = "";
    public uint NameId { get; init; }      // Raw MDDF nameId for fallback
    public uint UniqueId { get; init; }
    public Vector3 Position { get; init; }
    public Vector3 Rotation { get; init; } // Euler angles in radians
    public float Scale { get; init; }
}

public record AdtWmoPlacement
{
    public string WmoName { get; init; } = "";
    public uint NameId { get; init; }      // Raw MODF nameId for fallback
    public uint UniqueId { get; init; }
    public Vector3 Position { get; init; }
    public Vector3 Rotation { get; init; }
    public Vector3 ExtentsMin { get; init; }
    public Vector3 ExtentsMax { get; init; }
    public float Scale { get; init; }
    public ushort DoodadSet { get; init; }
    public ushort NameSet { get; init; }
}
