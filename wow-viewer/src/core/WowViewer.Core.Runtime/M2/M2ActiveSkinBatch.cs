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
        ShaderId = batch.ShaderId;
        SkinSectionIndex = batch.SkinSectionIndex;
        GeosetIndex = batch.GeosetIndex;
        ColorIndex = batch.ColorIndex;
        RenderFlagsIndex = batch.RenderFlagsIndex;
        MaterialLayer = batch.MaterialLayer;
        TextureCount = batch.TextureCount;
        TextureComboIndex = batch.TextureComboIndex;
        TextureCoordComboIndex = batch.TextureCoordComboIndex;
        TransparencyComboIndex = batch.TransparencyComboIndex;
        TextureAnimationLookupIndex = batch.TextureAnimationLookupIndex;
    }

    public int BatchIndex { get; }

    public byte Flags { get; }

    public byte PriorityPlane { get; }

    public ushort ShaderId { get; }

    public ushort SkinSectionIndex { get; }

    public ushort GeosetIndex { get; }

    public short ColorIndex { get; }

    public ushort RenderFlagsIndex { get; }

    public ushort MaterialLayer { get; }

    public ushort TextureCount { get; }

    public ushort MaterialIndex => RenderFlagsIndex;

    public ushort TextureComboIndex { get; }

    public ushort TextureCoordComboIndex { get; }

    public ushort TransparencyComboIndex { get; }

    public ushort TextureAnimationLookupIndex { get; }
}