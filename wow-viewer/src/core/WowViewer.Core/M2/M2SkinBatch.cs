namespace WowViewer.Core.M2;

public sealed class M2SkinBatch
{
    public M2SkinBatch(
        byte flags,
        byte priorityPlane,
        ushort skinSectionIndex,
        short colorIndex,
        ushort materialIndex,
        ushort textureComboIndex,
        ushort textureCoordComboIndex,
        ushort transparencyComboIndex,
        ushort textureAnimationLookupIndex)
    {
        Flags = flags;
        PriorityPlane = priorityPlane;
        SkinSectionIndex = skinSectionIndex;
        ColorIndex = colorIndex;
        MaterialIndex = materialIndex;
        TextureComboIndex = textureComboIndex;
        TextureCoordComboIndex = textureCoordComboIndex;
        TransparencyComboIndex = transparencyComboIndex;
        TextureAnimationLookupIndex = textureAnimationLookupIndex;
    }

    public byte Flags { get; }

    public byte PriorityPlane { get; }

    public ushort SkinSectionIndex { get; }

    public short ColorIndex { get; }

    public ushort MaterialIndex { get; }

    public ushort TextureComboIndex { get; }

    public ushort TextureCoordComboIndex { get; }

    public ushort TransparencyComboIndex { get; }

    public ushort TextureAnimationLookupIndex { get; }
}