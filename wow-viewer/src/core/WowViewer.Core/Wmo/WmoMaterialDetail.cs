namespace WowViewer.Core.Wmo;

public sealed class WmoMaterialDetail
{
    public WmoMaterialDetail(
        int materialIndex,
        int payloadOffset,
        int entrySizeBytes,
        uint flags,
        uint shader,
        uint blendMode,
        uint texture1Offset,
        string texture1Name,
        uint texture2Offset,
        string texture2Name,
        uint texture3Offset,
        string texture3Name)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(materialIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(entrySizeBytes);
        ArgumentNullException.ThrowIfNull(texture1Name);
        ArgumentNullException.ThrowIfNull(texture2Name);
        ArgumentNullException.ThrowIfNull(texture3Name);

        MaterialIndex = materialIndex;
        PayloadOffset = payloadOffset;
        EntrySizeBytes = entrySizeBytes;
        Flags = flags;
        Shader = shader;
        BlendMode = blendMode;
        Texture1Offset = texture1Offset;
        Texture1Name = texture1Name;
        Texture2Offset = texture2Offset;
        Texture2Name = texture2Name;
        Texture3Offset = texture3Offset;
        Texture3Name = texture3Name;
    }

    public int MaterialIndex { get; }

    public int PayloadOffset { get; }

    public int EntrySizeBytes { get; }

    public uint Flags { get; }

    public uint Shader { get; }

    public uint BlendMode { get; }

    public uint Texture1Offset { get; }

    public string Texture1Name { get; }

    public uint Texture2Offset { get; }

    public string Texture2Name { get; }

    public uint Texture3Offset { get; }

    public string Texture3Name { get; }
}