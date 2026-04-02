using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2ActiveSkinBatch
{
    public M2ActiveSkinBatch(int batchIndex, M2SkinBatch batch)
    {
        ArgumentNullException.ThrowIfNull(batch);

        BatchIndex = batchIndex;
        Flags = batch.Flags;
        PriorityPlane = batch.PriorityPlane;
        SkinSectionIndex = batch.SkinSectionIndex;
        ColorIndex = batch.ColorIndex;
        MaterialIndex = batch.MaterialIndex;
        TextureComboIndex = batch.TextureComboIndex;
        TextureCoordComboIndex = batch.TextureCoordComboIndex;
        TransparencyComboIndex = batch.TransparencyComboIndex;
        TextureAnimationLookupIndex = batch.TextureAnimationLookupIndex;
    }

    public int BatchIndex { get; }

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