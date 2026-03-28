using System.Numerics;

namespace WowViewer.Core.Wmo;

public sealed class WmoLightDetail
{
    public WmoLightDetail(
        int lightIndex,
        int payloadOffset,
        int entrySizeBytes,
        byte lightType,
        bool usesAttenuation,
        uint colorBgra,
        Vector3 position,
        float intensity,
        float attenStart,
        float attenEnd,
        ushort? headerFlagsWord,
        Quaternion? rotation,
        float? rotationLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(lightIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(entrySizeBytes);

        LightIndex = lightIndex;
        PayloadOffset = payloadOffset;
        EntrySizeBytes = entrySizeBytes;
        LightType = lightType;
        UsesAttenuation = usesAttenuation;
        ColorBgra = colorBgra;
        Position = position;
        Intensity = intensity;
        AttenStart = attenStart;
        AttenEnd = attenEnd;
        HeaderFlagsWord = headerFlagsWord;
        Rotation = rotation;
        RotationLength = rotationLength;
    }

    public int LightIndex { get; }

    public int PayloadOffset { get; }

    public int EntrySizeBytes { get; }

    public byte LightType { get; }

    public bool UsesAttenuation { get; }

    public uint ColorBgra { get; }

    public Vector3 Position { get; }

    public float Intensity { get; }

    public float AttenStart { get; }

    public float AttenEnd { get; }

    public ushort? HeaderFlagsWord { get; }

    public Quaternion? Rotation { get; }

    public float? RotationLength { get; }
}