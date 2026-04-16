using System.Numerics;
using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2StaticRenderModel
{
    public M2StaticRenderModel(
        M2ModelDocument model,
        IReadOnlyList<M2StaticRenderSection> sections,
        IReadOnlyList<M2StructuredRenderSection> structuredSections,
        IReadOnlyList<ushort> boneLookup,
        bool usesCompatibilityFallback)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(sections);
        ArgumentNullException.ThrowIfNull(structuredSections);
        ArgumentNullException.ThrowIfNull(boneLookup);

        Model = model;
        Sections = sections;
        StructuredSections = structuredSections;
        BoneLookup = boneLookup;
        UsesCompatibilityFallback = usesCompatibilityFallback;
    }

    public M2ModelDocument Model { get; }

    public IReadOnlyList<M2StaticRenderSection> Sections { get; }

    public IReadOnlyList<M2StructuredRenderSection> StructuredSections { get; }

    public IReadOnlyList<ushort> BoneLookup { get; }

    public bool UsesCompatibilityFallback { get; }

    public Vector3 BoundsMin => Model.BoundsMin;

    public Vector3 BoundsMax => Model.BoundsMax;
}

public sealed class M2StaticRenderSection
{
    public M2StaticRenderSection(
        int sectionIndex,
        ushort skinSectionId,
        ushort boneComboIndex,
        ushort boneCount,
        ushort boneInfluences,
        ushort centerBoneIndex,
        IReadOnlyList<M2StaticRenderVertex> vertices,
        IReadOnlyList<uint> indices,
        M2StaticRenderMaterial material)
    {
        ArgumentNullException.ThrowIfNull(vertices);
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(material);

        SectionIndex = sectionIndex;
        SkinSectionId = skinSectionId;
        BoneComboIndex = boneComboIndex;
        BoneCount = boneCount;
        BoneInfluences = boneInfluences;
        CenterBoneIndex = centerBoneIndex;
        Vertices = vertices;
        Indices = indices;
        Material = material;
    }

    public int SectionIndex { get; }

    public ushort SkinSectionId { get; }

    public ushort BoneComboIndex { get; }

    public ushort BoneCount { get; }

    public ushort BoneInfluences { get; }

    public ushort CenterBoneIndex { get; }

    public IReadOnlyList<M2StaticRenderVertex> Vertices { get; }

    public IReadOnlyList<uint> Indices { get; }

    public M2StaticRenderMaterial Material { get; }
}

public sealed class M2StructuredRenderSection
{
    public M2StructuredRenderSection(
        int sectionIndex,
        ushort skinSectionId,
        ushort boneComboIndex,
        ushort boneCount,
        ushort boneInfluences,
        ushort centerBoneIndex,
        IReadOnlyList<M2StaticRenderVertex> vertices,
        IReadOnlyList<uint> indices,
        IReadOnlyList<M2StructuredRenderPass> passes)
    {
        ArgumentNullException.ThrowIfNull(vertices);
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(passes);

        SectionIndex = sectionIndex;
        SkinSectionId = skinSectionId;
        BoneComboIndex = boneComboIndex;
        BoneCount = boneCount;
        BoneInfluences = boneInfluences;
        CenterBoneIndex = centerBoneIndex;
        Vertices = vertices;
        Indices = indices;
        Passes = passes;
    }

    public int SectionIndex { get; }

    public ushort SkinSectionId { get; }

    public ushort BoneComboIndex { get; }

    public ushort BoneCount { get; }

    public ushort BoneInfluences { get; }

    public ushort CenterBoneIndex { get; }

    public IReadOnlyList<M2StaticRenderVertex> Vertices { get; }

    public IReadOnlyList<uint> Indices { get; }

    public IReadOnlyList<M2StructuredRenderPass> Passes { get; }

    public int PassCount => Passes.Count;
}

public sealed class M2StructuredRenderPass
{
    public M2StructuredRenderPass(int passIndex, M2StaticRenderMaterial material)
    {
        ArgumentNullException.ThrowIfNull(material);

        PassIndex = passIndex;
        Material = material;
    }

    public int PassIndex { get; }

    public M2StaticRenderMaterial Material { get; }
}

public readonly record struct M2StaticRenderVertex(
    Vector3 Position,
    Vector3 Normal,
    Vector2 TextureCoords0,
    Vector2 TextureCoords1,
    Vector4 BoneIndices,
    Vector4 BoneWeights);

public sealed class M2StaticRenderMaterial
{
    public M2StaticRenderMaterial(
        int batchIndex,
        byte batchFlags,
        byte priorityPlane,
        ushort shaderId,
        ushort geosetIndex,
        short colorIndex,
        ushort renderFlagsIndex,
        ushort materialLayer,
        ushort textureCount,
        ushort textureComboIndex,
        ushort textureCoordComboIndex,
        ushort transparencyComboIndex,
        ushort textureAnimationLookupIndex,
        ushort renderFlags,
        ushort rawBlendMode,
        M2BlendMode blendMode,
        string? texturePath,
        uint replaceableId,
        uint textureFlags,
        IReadOnlyList<M2StaticRenderTextureBinding> textureBindings,
        M2EffectRecipe effectRecipe)
    {
        ArgumentNullException.ThrowIfNull(textureBindings);
        ArgumentNullException.ThrowIfNull(effectRecipe);

        BatchIndex = batchIndex;
        BatchFlags = batchFlags;
        PriorityPlane = priorityPlane;
        ShaderId = shaderId;
        GeosetIndex = geosetIndex;
        ColorIndex = colorIndex;
        RenderFlagsIndex = renderFlagsIndex;
        MaterialLayer = materialLayer;
        TextureCount = textureCount;
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
        TextureBindings = textureBindings;
        EffectRecipe = effectRecipe;
    }

    public int BatchIndex { get; }

    public byte BatchFlags { get; }

    public byte PriorityPlane { get; }

    public ushort ShaderId { get; }

    public ushort GeosetIndex { get; }

    public short ColorIndex { get; }

    public ushort RenderFlagsIndex { get; }

    public ushort MaterialIndex => RenderFlagsIndex;

    public ushort MaterialLayer { get; }

    public ushort TextureCount { get; }

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

    public IReadOnlyList<M2StaticRenderTextureBinding> TextureBindings { get; }

    public M2EffectRecipe EffectRecipe { get; }

    public bool IsTransparent => BlendMode != M2BlendMode.Opaque;

    public bool IsUnshaded => (RenderFlags & 0x1) != 0;

    public bool IsTwoSided => (RenderFlags & 0x4) != 0;
}

public sealed class M2StaticRenderTextureBinding
{
    public M2StaticRenderTextureBinding(
        int stageIndex,
        int? textureLookupIndex,
        ushort? textureId,
        string? texturePath,
        uint replaceableId,
        uint textureFlags,
        int? textureCoordLookupIndex,
        ushort? textureCoordLookupValue,
        int? transparencyLookupIndex,
        ushort? transparencyLookupValue,
        int? textureAnimationLookupIndex,
        ushort? textureAnimationLookupValue)
    {
        StageIndex = stageIndex;
        TextureLookupIndex = textureLookupIndex;
        TextureId = textureId;
        TexturePath = texturePath;
        ReplaceableId = replaceableId;
        TextureFlags = textureFlags;
        TextureCoordLookupIndex = textureCoordLookupIndex;
        TextureCoordLookupValue = textureCoordLookupValue;
        TransparencyLookupIndex = transparencyLookupIndex;
        TransparencyLookupValue = transparencyLookupValue;
        TextureAnimationLookupIndex = textureAnimationLookupIndex;
        TextureAnimationLookupValue = textureAnimationLookupValue;
    }

    public int StageIndex { get; }

    public int? TextureLookupIndex { get; }

    public ushort? TextureId { get; }

    public string? TexturePath { get; }

    public uint ReplaceableId { get; }

    public uint TextureFlags { get; }

    public int? TextureCoordLookupIndex { get; }

    public ushort? TextureCoordLookupValue { get; }

    public int? TransparencyLookupIndex { get; }

    public ushort? TransparencyLookupValue { get; }

    public int? TextureAnimationLookupIndex { get; }

    public ushort? TextureAnimationLookupValue { get; }
}
