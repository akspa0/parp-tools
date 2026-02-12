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

/// <summary>Animation track with keyframes for a specific transform channel</summary>
public class MdlAnimTrack<T>
{
    public MdlTrackType InterpolationType { get; set; } = MdlTrackType.Linear;
    public int GlobalSeqId { get; set; } = -1;
    public List<MdlTrackKey<T>> Keys { get; } = new();
}

/// <summary>Single keyframe in an animation track</summary>
public class MdlTrackKey<T>
{
    public int Frame { get; set; }
    public T Value { get; set; } = default!;
    public T InTan { get; set; } = default!;
    public T OutTan { get; set; } = default!;
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

    // Animation tracks with full keyframe + tangent data
    public MdlAnimTrack<C3Vector>? TranslationTrack { get; set; }
    public MdlAnimTrack<C4Quaternion>? RotationTrack { get; set; }
    public MdlAnimTrack<C3Vector>? ScalingTrack { get; set; }
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

/// <summary>Particle Emitter 2 filter modes (PRE2 chunk)</summary>
public enum ParticleFilterMode
{
    Blend = 0,
    Additive = 1,
    Modulate = 2,
    Modulate2x = 3,
    AlphaKey = 4,
}

/// <summary>Particle Emitter 2 (PRE2 chunk) — the main particle system in WC3 MDX</summary>
public class MdlParticleEmitter2
{
    // Node fields
    public string Name { get; set; } = "";
    public int ObjectId { get; set; }
    public int ParentId { get; set; } = -1;
    public uint Flags { get; set; }
    public C3Vector Position { get; set; } = new();

    // Emitter properties
    public float Speed { get; set; }
    public float Variation { get; set; }
    public float Latitude { get; set; }
    public float Gravity { get; set; }
    public float Lifespan { get; set; }
    public float EmissionRate { get; set; }
    public float Length { get; set; }
    public float Width { get; set; }

    public ParticleFilterMode FilterMode { get; set; }
    public int Rows { get; set; } = 1;
    public int Columns { get; set; } = 1;
    public int HeadOrTail { get; set; } // 0=Head, 1=Tail, 2=Both

    public float TailLength { get; set; }
    public float Time { get; set; }

    // Segment colors (3 segments: birth, mid, death)
    public C3Vector[] SegmentColor { get; set; } = new C3Vector[3] { new(), new(), new() };
    public byte[] SegmentAlpha { get; set; } = new byte[3];
    public float[] SegmentScaling { get; set; } = new float[3];

    // Head/Tail interval data (12 DWORDs)
    public uint[] Intervals { get; set; } = new uint[12];

    public int TextureId { get; set; } = -1;
    public int Squirt { get; set; }
    public int PriorityPlane { get; set; }
    public uint ReplaceableId { get; set; }
}

/// <summary>Ribbon Emitter (RIBB chunk)</summary>
public class MdlRibbonEmitter
{
    // Node fields
    public string Name { get; set; } = "";
    public int ObjectId { get; set; }
    public int ParentId { get; set; } = -1;
    public uint Flags { get; set; }
    public C3Vector Position { get; set; } = new();

    public float HeightAbove { get; set; }
    public float HeightBelow { get; set; }
    public float Alpha { get; set; }
    public C3Vector Color { get; set; } = new();
    public float Lifespan { get; set; }
    public int TextureSlot { get; set; }
    public int EmissionRate { get; set; }
    public int Rows { get; set; } = 1;
    public int Columns { get; set; } = 1;
    public int MaterialId { get; set; }
    public float Gravity { get; set; }
}

/// <summary>Geoset animation data for alpha/color animations (ATSQ chunk)</summary>
public class MdlGeosetAnimation
{
    public uint GeosetId { get; set; }
    public float DefaultAlpha { get; set; } = 1.0f;
    public C3Color DefaultColor { get; set; } = new C3Color(1.0f, 1.0f, 1.0f);
    public uint Unknown { get; set; }
    
    // Alpha animation keys
    public List<MdlAnimKey<float>> AlphaKeys { get; } = new();
    public MdlAnimInterpolation AlphaInterpolation { get; set; }
    public int AlphaGlobalSeqId { get; set; }
    
    // Color animation keys
    public List<MdlAnimKey<C3Color>> ColorKeys { get; } = new();
    public MdlAnimInterpolation ColorInterpolation { get; set; }
    public int ColorGlobalSeqId { get; set; }
}

/// <summary>Animation keyframe for generic values</summary>
public class MdlAnimKey<T>
{
    public int Time { get; set; }
    public T Value { get; set; } = default!;
    public float TangentIn { get; set; }
    public float TangentOut { get; set; }
    public C3Color ColorTangentIn { get; set; } = new C3Color();
    public C3Color ColorTangentOut { get; set; } = new C3Color();
}

/// <summary>Interpolation type for animation tracks</summary>
public enum MdlAnimInterpolation
{
    None = 0,
    Linear = 1,
    Hermite = 2,
    Bezier = 3,
    Bezier2 = 4,
}

/// <summary>RGB color (0-1 range)</summary>
public class C3Color
{
    public float R { get; set; }
    public float G { get; set; }
    public float B { get; set; }

    public C3Color() { }

    public C3Color(float r, float g, float b)
    {
        R = r;
        G = g;
        B = b;
    }
}
