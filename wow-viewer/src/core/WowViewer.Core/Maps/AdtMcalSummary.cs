namespace WowViewer.Core.Maps;

public sealed class AdtMcalSummary
{
    public AdtMcalSummary(
        string sourcePath,
        MapFileKind kind,
        AdtMcalDecodeProfile decodeProfile,
        int mcnkWithLayerTableCount,
        int overlayLayerCount,
        int decodedLayerCount,
        int missingPayloadLayerCount,
        int decodeFailureCount,
        int compressedLayerCount,
        int bigAlphaLayerCount,
        int bigAlphaFixedLayerCount,
        int packedLayerCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(mcnkWithLayerTableCount);
        ArgumentOutOfRangeException.ThrowIfNegative(overlayLayerCount);
        ArgumentOutOfRangeException.ThrowIfNegative(decodedLayerCount);
        ArgumentOutOfRangeException.ThrowIfNegative(missingPayloadLayerCount);
        ArgumentOutOfRangeException.ThrowIfNegative(decodeFailureCount);
        ArgumentOutOfRangeException.ThrowIfNegative(compressedLayerCount);
        ArgumentOutOfRangeException.ThrowIfNegative(bigAlphaLayerCount);
        ArgumentOutOfRangeException.ThrowIfNegative(bigAlphaFixedLayerCount);
        ArgumentOutOfRangeException.ThrowIfNegative(packedLayerCount);

        SourcePath = sourcePath;
        Kind = kind;
        DecodeProfile = decodeProfile;
        McnkWithLayerTableCount = mcnkWithLayerTableCount;
        OverlayLayerCount = overlayLayerCount;
        DecodedLayerCount = decodedLayerCount;
        MissingPayloadLayerCount = missingPayloadLayerCount;
        DecodeFailureCount = decodeFailureCount;
        CompressedLayerCount = compressedLayerCount;
        BigAlphaLayerCount = bigAlphaLayerCount;
        BigAlphaFixedLayerCount = bigAlphaFixedLayerCount;
        PackedLayerCount = packedLayerCount;
    }

    public string SourcePath { get; }

    public MapFileKind Kind { get; }

    public AdtMcalDecodeProfile DecodeProfile { get; }

    public int McnkWithLayerTableCount { get; }

    public int OverlayLayerCount { get; }

    public int DecodedLayerCount { get; }

    public int MissingPayloadLayerCount { get; }

    public int DecodeFailureCount { get; }

    public int CompressedLayerCount { get; }

    public int BigAlphaLayerCount { get; }

    public int BigAlphaFixedLayerCount { get; }

    public int PackedLayerCount { get; }
}