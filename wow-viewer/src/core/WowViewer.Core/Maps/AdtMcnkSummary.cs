namespace WowViewer.Core.Maps;

public sealed class AdtMcnkSummary
{
    public AdtMcnkSummary(
        string sourcePath,
        MapFileKind kind,
        int mcnkCount,
        int zeroLengthMcnkCount,
        int headerLikeMcnkCount,
        int distinctIndexCount,
        int duplicateIndexCount,
        int distinctAreaIdCount,
        int chunksWithHoles,
        int chunksWithLiquidFlags,
        int chunksWithMccvFlag,
        int chunksWithMcvt,
        int chunksWithMcnr,
        int chunksWithMcly,
        int chunksWithMcal,
        int chunksWithMcsh,
        int chunksWithMccv,
        int chunksWithMclq,
        int chunksWithMcrd,
        int chunksWithMcrw,
        int totalLayerCount,
        int maxLayerCount,
        int chunksWithMultipleLayers,
        int mccvFlagWithoutPayloadCount,
        int liquidFlagWithoutPayloadCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(mcnkCount);
        ArgumentOutOfRangeException.ThrowIfNegative(zeroLengthMcnkCount);
        ArgumentOutOfRangeException.ThrowIfNegative(headerLikeMcnkCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctIndexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(duplicateIndexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctAreaIdCount);
        ArgumentOutOfRangeException.ThrowIfNegative(chunksWithHoles);
        ArgumentOutOfRangeException.ThrowIfNegative(chunksWithLiquidFlags);
        ArgumentOutOfRangeException.ThrowIfNegative(chunksWithMccvFlag);
        ArgumentOutOfRangeException.ThrowIfNegative(chunksWithMcvt);
        ArgumentOutOfRangeException.ThrowIfNegative(chunksWithMcnr);
        ArgumentOutOfRangeException.ThrowIfNegative(chunksWithMcly);
        ArgumentOutOfRangeException.ThrowIfNegative(chunksWithMcal);
        ArgumentOutOfRangeException.ThrowIfNegative(chunksWithMcsh);
        ArgumentOutOfRangeException.ThrowIfNegative(chunksWithMccv);
        ArgumentOutOfRangeException.ThrowIfNegative(chunksWithMclq);
        ArgumentOutOfRangeException.ThrowIfNegative(chunksWithMcrd);
        ArgumentOutOfRangeException.ThrowIfNegative(chunksWithMcrw);
        ArgumentOutOfRangeException.ThrowIfNegative(totalLayerCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxLayerCount);
        ArgumentOutOfRangeException.ThrowIfNegative(chunksWithMultipleLayers);
        ArgumentOutOfRangeException.ThrowIfNegative(mccvFlagWithoutPayloadCount);
        ArgumentOutOfRangeException.ThrowIfNegative(liquidFlagWithoutPayloadCount);

        SourcePath = sourcePath;
        Kind = kind;
        McnkCount = mcnkCount;
        ZeroLengthMcnkCount = zeroLengthMcnkCount;
        HeaderLikeMcnkCount = headerLikeMcnkCount;
        DistinctIndexCount = distinctIndexCount;
        DuplicateIndexCount = duplicateIndexCount;
        DistinctAreaIdCount = distinctAreaIdCount;
        ChunksWithHoles = chunksWithHoles;
        ChunksWithLiquidFlags = chunksWithLiquidFlags;
        ChunksWithMccvFlag = chunksWithMccvFlag;
        ChunksWithMcvt = chunksWithMcvt;
        ChunksWithMcnr = chunksWithMcnr;
        ChunksWithMcly = chunksWithMcly;
        ChunksWithMcal = chunksWithMcal;
        ChunksWithMcsh = chunksWithMcsh;
        ChunksWithMccv = chunksWithMccv;
        ChunksWithMclq = chunksWithMclq;
        ChunksWithMcrd = chunksWithMcrd;
        ChunksWithMcrw = chunksWithMcrw;
        TotalLayerCount = totalLayerCount;
        MaxLayerCount = maxLayerCount;
        ChunksWithMultipleLayers = chunksWithMultipleLayers;
        MccvFlagWithoutPayloadCount = mccvFlagWithoutPayloadCount;
        LiquidFlagWithoutPayloadCount = liquidFlagWithoutPayloadCount;
    }

    public string SourcePath { get; }

    public MapFileKind Kind { get; }

    public int McnkCount { get; }

    public int ZeroLengthMcnkCount { get; }

    public int HeaderLikeMcnkCount { get; }

    public int DistinctIndexCount { get; }

    public int DuplicateIndexCount { get; }

    public int DistinctAreaIdCount { get; }

    public int ChunksWithHoles { get; }

    public int ChunksWithLiquidFlags { get; }

    public int ChunksWithMccvFlag { get; }

    public int ChunksWithMcvt { get; }

    public int ChunksWithMcnr { get; }

    public int ChunksWithMcly { get; }

    public int ChunksWithMcal { get; }

    public int ChunksWithMcsh { get; }

    public int ChunksWithMccv { get; }

    public int ChunksWithMclq { get; }

    public int ChunksWithMcrd { get; }

    public int ChunksWithMcrw { get; }

    public int TotalLayerCount { get; }

    public int MaxLayerCount { get; }

    public int ChunksWithMultipleLayers { get; }

    public int MccvFlagWithoutPayloadCount { get; }

    public int LiquidFlagWithoutPayloadCount { get; }
}
