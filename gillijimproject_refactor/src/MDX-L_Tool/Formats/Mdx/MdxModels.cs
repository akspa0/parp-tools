namespace MdxLTool.Formats.Mdx;

/// <summary>
/// MDX model data classes for in-memory representation.
/// Based on MDLDATA structure from Ghidra analysis.
/// </summary>

/// <summary>Model section (MODL chunk)</summary>
public class MdlModel
{
    public string Name { get; set; } = "";
    public string AnimationFile { get; set; } = "";
    public CMdlBounds Bounds { get; set; }
    public uint BlendTime { get; set; }
    public byte Flags { get; set; }
}

/// <summary>Animation sequence (SEQS entry)</summary>
public class MdlSequence
{
    public string Name { get; set; } = "";
    public CiRange Time { get; set; }
    public float MoveSpeed { get; set; }
    public uint Flags { get; set; }
    public CMdlBounds Bounds { get; set; }
    public float Frequency { get; set; }
    public CiRange Replay { get; set; }
    public uint BlendTime { get; set; }
}

/// <summary>Texture reference (TEXS entry)</summary>
public class MdlTexture
{
    public uint ReplaceableId { get; set; }
    public string Path { get; set; } = "";
    public uint Flags { get; set; }
}

/// <summary>Texture layer within material</summary>
public class MdlTexLayer
{
    public MdlTexOp BlendMode { get; set; }
    public MdlGeoFlags Flags { get; set; }
    public int TextureId { get; set; } = -1;
    public int TransformId { get; set; } = -1;
    public int CoordId { get; set; } = -1;
    public float StaticAlpha { get; set; } = 1.0f;
}

/// <summary>Material (MTLS entry)</summary>
public class MdlMaterial
{
    public int PriorityPlane { get; set; }
    public List<MdlTexLayer> Layers { get; } = new();
}

/// <summary>Geoset - mesh geometry (GEOS entry)</summary>
public class MdlGeoset
{
    public List<C3Vector> Vertices { get; } = new();
    public List<C3Vector> Normals { get; } = new();
    public List<C2Vector> TexCoords { get; } = new();
    public List<ushort> Indices { get; } = new();
    public List<byte> VertexGroups { get; } = new();
    public List<uint> MatrixGroups { get; } = new();
    public List<uint> MatrixIndices { get; } = new();
    public CMdlBounds Bounds { get; set; }
    public int MaterialId { get; set; }
    public uint SelectionGroup { get; set; }
    public uint Flags { get; set; }

    // Extent per animation
    public List<CMdlBounds> AnimExtents { get; } = new();
}

/// <summary>Bone (BONE entry)</summary>
public class MdlBone
{
    public string Name { get; set; } = "";
    public int ObjectId { get; set; }
    public int ParentId { get; set; } = -1;
    public uint Flags { get; set; }
    public int GeosetId { get; set; } = -1;
    public int GeosetAnimId { get; set; } = -1;

    // Pivot from PIVT chunk (by index)
    public C3Vector Pivot { get; set; }

    // Animation tracks (simplified for now)
    public List<(int time, C3Vector value)> Translation { get; } = new();
    public List<(int time, C4Quaternion value)> Rotation { get; } = new();
    public List<(int time, C3Vector value)> Scaling { get; } = new();
}

/// <summary>Attachment point (ATCH entry)</summary>  
public class MdlAttachment
{
    public string Name { get; set; } = "";
    public int ObjectId { get; set; }
    public int ParentId { get; set; } = -1;
    public uint AttachmentId { get; set; }
    public string Path { get; set; } = "";
}

/// <summary>Camera target (part of CAMS)</summary>
public class MdlCameraTarget
{
    public string Name { get; set; } = "";
    public C3Vector Position { get; set; }
}

/// <summary>Camera (CAMS entry)</summary>
public class MdlCamera
{
    public string Name { get; set; } = "";
    public C3Vector Position { get; set; }
    public MdlCameraTarget Target { get; set; } = new();
    public float FieldOfView { get; set; }
    public float FarClip { get; set; }
    public float NearClip { get; set; }
}

/// <summary>Light (LITE entry)</summary>
public class MdlLight : MdlBone
{
    public int Type { get; set; }
    public float AttenuationStart { get; set; }
    public float AttenuationEnd { get; set; }
    public C3Vector Color { get; set; }
    public float Intensity { get; set; }
    public C3Vector AmbientColor { get; set; }
    public float AmbientIntensity { get; set; }
}
