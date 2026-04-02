using System.Numerics;
using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2StaticRenderModel
{
    public M2StaticRenderModel(
        M2ModelDocument model,
        IReadOnlyList<M2StaticRenderSection> sections,
        bool usesCompatibilityFallback)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(sections);

        Model = model;
        Sections = sections;
        UsesCompatibilityFallback = usesCompatibilityFallback;
    }

    public M2ModelDocument Model { get; }

    public IReadOnlyList<M2StaticRenderSection> Sections { get; }

    public bool UsesCompatibilityFallback { get; }

    public Vector3 BoundsMin => Model.BoundsMin;

    public Vector3 BoundsMax => Model.BoundsMax;
}

public sealed class M2StaticRenderSection
{
    public M2StaticRenderSection(
        int sectionIndex,
        ushort skinSectionId,
        IReadOnlyList<M2StaticRenderVertex> vertices,
        IReadOnlyList<uint> indices,
        M2StaticRenderMaterial material)
    {
        ArgumentNullException.ThrowIfNull(vertices);
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(material);

        SectionIndex = sectionIndex;
        SkinSectionId = skinSectionId;
        Vertices = vertices;
        Indices = indices;
        Material = material;
    }

    public int SectionIndex { get; }

    public ushort SkinSectionId { get; }

    public IReadOnlyList<M2StaticRenderVertex> Vertices { get; }

    public IReadOnlyList<uint> Indices { get; }

    public M2StaticRenderMaterial Material { get; }
}

public readonly record struct M2StaticRenderVertex(
    Vector3 Position,
    Vector3 Normal,
    Vector2 TextureCoords,
    Vector4 BoneIndices,
    Vector4 BoneWeights);

public sealed class M2StaticRenderMaterial
{
    public M2StaticRenderMaterial(
        int batchIndex,
        byte batchFlags,
        byte priorityPlane,
        short colorIndex,
        ushort materialIndex,
        ushort textureComboIndex,
        ushort textureCoordComboIndex,
        ushort transparencyComboIndex,
        ushort textureAnimationLookupIndex,
        ushort renderFlags,
        ushort rawBlendMode,
        M2BlendMode blendMode,
        string? texturePath,
        uint replaceableId,
        uint textureFlags)
    {
        BatchIndex = batchIndex;
        BatchFlags = batchFlags;
        PriorityPlane = priorityPlane;
        ColorIndex = colorIndex;
        MaterialIndex = materialIndex;
        TextureComboIndex = textureComboIndex;
        TextureCoordComboIndex = textureCoordComboIndex;
        TransparencyComboIndex = transparencyComboIndex;
        TextureAnimationLookupIndex = textureAnimationLookupIndex;
        RenderFlags = renderFlags;
        RawBlendMode = rawBlendMode;
        BlendMode = blendMode;
        TexturePath = texturePath;
        ReplaceableId = replaceableId;
        TextureFlags = textureFlags;
    }

    public int BatchIndex { get; }

    public byte BatchFlags { get; }

    public byte PriorityPlane { get; }

    public short ColorIndex { get; }

    public ushort MaterialIndex { get; }

    public ushort TextureComboIndex { get; }

    public ushort TextureCoordComboIndex { get; }

    public ushort TransparencyComboIndex { get; }

    public ushort TextureAnimationLookupIndex { get; }

    public ushort RenderFlags { get; }

    public ushort RawBlendMode { get; }

    public M2BlendMode BlendMode { get; }

    public string? TexturePath { get; }

    public uint ReplaceableId { get; }

    public uint TextureFlags { get; }

    public bool IsTransparent => BlendMode != M2BlendMode.Opaque;

    public bool IsUnshaded => (RenderFlags & 0x1) != 0;

    public bool IsTwoSided => (RenderFlags & 0x4) != 0;
}