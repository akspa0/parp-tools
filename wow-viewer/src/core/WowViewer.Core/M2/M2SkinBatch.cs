namespace WowViewer.Core.M2;

public sealed class M2SkinBatch
{
    public M2SkinBatch(
        byte flags,
        byte priorityPlane,
        ushort shaderId,
        ushort skinSectionIndex,
        ushort geosetIndex,
        short colorIndex,
        ushort renderFlagsIndex,
        ushort materialLayer,
        ushort textureCount,
        ushort textureComboIndex,
        ushort textureCoordComboIndex,
        ushort transparencyComboIndex,
        ushort textureAnimationLookupIndex)
    {
        Flags = flags;
        PriorityPlane = priorityPlane;
        ShaderId = shaderId;
        SkinSectionIndex = skinSectionIndex;
        GeosetIndex = geosetIndex;
        ColorIndex = colorIndex;
        RenderFlagsIndex = renderFlagsIndex;
        MaterialLayer = materialLayer;
        TextureCount = textureCount;
        TextureComboIndex = textureComboIndex;
        TextureCoordComboIndex = textureCoordComboIndex;
        TransparencyComboIndex = transparencyComboIndex;
        TextureAnimationLookupIndex = textureAnimationLookupIndex;
    }

    public byte Flags { get; }

    public byte PriorityPlane { get; }

    public ushort ShaderId { get; }

    public ushort SkinSectionIndex { get; }

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
}