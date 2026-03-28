namespace WowViewer.Core.Maps;

public sealed class AdtMcalDecodedLayer
{
    public AdtMcalDecodedLayer(
        int layerIndex,
        uint textureId,
        uint flags,
        int alphaOffset,
        int sourceBytesConsumed,
        AdtMcalAlphaEncoding encoding,
        bool appliedFixup,
        byte[] alphaMap)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(layerIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(alphaOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(sourceBytesConsumed);
        ArgumentNullException.ThrowIfNull(alphaMap);

        LayerIndex = layerIndex;
        TextureId = textureId;
        Flags = flags;
        AlphaOffset = alphaOffset;
        SourceBytesConsumed = sourceBytesConsumed;
        Encoding = encoding;
        AppliedFixup = appliedFixup;
        AlphaMap = alphaMap;
    }

    public int LayerIndex { get; }

    public uint TextureId { get; }

    public uint Flags { get; }

    public int AlphaOffset { get; }

    public int SourceBytesConsumed { get; }

    public AdtMcalAlphaEncoding Encoding { get; }

    public bool AppliedFixup { get; }

    public byte[] AlphaMap { get; }
}