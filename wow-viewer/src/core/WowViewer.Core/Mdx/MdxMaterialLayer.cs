namespace WowViewer.Core.Mdx;

public sealed class MdxMaterialLayer
{
    public MdxMaterialLayer(
        int index,
        uint blendMode,
        uint flags,
        int textureId,
        int transformId,
        int coordId,
        float staticAlpha,
        float staticEmissiveGain,
        MdxScalarTrack? emissiveTrack,
        MdxScalarTrack? alphaTrack,
        MdxIntTrack? textureLayerTrack)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        Index = index;
        BlendMode = blendMode;
        Flags = flags;
        TextureId = textureId;
        TransformId = transformId;
        CoordId = coordId;
        StaticAlpha = staticAlpha;
        StaticEmissiveGain = staticEmissiveGain;
        EmissiveTrack = emissiveTrack;
        AlphaTrack = alphaTrack;
        TextureLayerTrack = textureLayerTrack;
    }

    public int Index { get; }

    public uint BlendMode { get; }

    public uint Flags { get; }

    public int TextureId { get; }

    public int TransformId { get; }

    public int CoordId { get; }

    public float StaticAlpha { get; }

    public float StaticEmissiveGain { get; }

    public MdxScalarTrack? EmissiveTrack { get; }

    public MdxScalarTrack? AlphaTrack { get; }

    public MdxIntTrack? TextureLayerTrack { get; }
}