namespace WowViewer.Core.Mdx;

public sealed class MdxMaterialLayerSummary
{
    public MdxMaterialLayerSummary(int index, uint blendMode, uint flags, int textureId, int transformId, int coordId, float staticAlpha)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        Index = index;
        BlendMode = blendMode;
        Flags = flags;
        TextureId = textureId;
        TransformId = transformId;
        CoordId = coordId;
        StaticAlpha = staticAlpha;
    }

    public int Index { get; }

    public uint BlendMode { get; }

    public uint Flags { get; }

    public int TextureId { get; }

    public int TransformId { get; }

    public int CoordId { get; }

    public float StaticAlpha { get; }
}