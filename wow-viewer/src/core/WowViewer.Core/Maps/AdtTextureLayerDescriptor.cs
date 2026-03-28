namespace WowViewer.Core.Maps;

public sealed class AdtTextureLayerDescriptor
{
    public AdtTextureLayerDescriptor(int index, uint textureId, uint flags, uint alphaOffset, uint effectId)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        Index = index;
        TextureId = textureId;
        Flags = flags;
        AlphaOffset = alphaOffset;
        EffectId = effectId;
    }

    public int Index { get; }

    public uint TextureId { get; }

    public uint Flags { get; }

    public uint AlphaOffset { get; }

    public uint EffectId { get; }
}