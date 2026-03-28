namespace WowViewer.Core.Maps;

public sealed class AdtTextureChunkLayer
{
    public AdtTextureChunkLayer(
        int index,
        uint textureId,
        string? texturePath,
        uint flags,
        uint alphaOffset,
        uint effectId,
        AdtMcalDecodedLayer? decodedAlpha)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        Index = index;
        TextureId = textureId;
        TexturePath = texturePath;
        Flags = flags;
        AlphaOffset = alphaOffset;
        EffectId = effectId;
        DecodedAlpha = decodedAlpha;
    }

    public int Index { get; }

    public uint TextureId { get; }

    public string? TexturePath { get; }

    public uint Flags { get; }

    public uint AlphaOffset { get; }

    public uint EffectId { get; }

    public AdtMcalDecodedLayer? DecodedAlpha { get; }
}