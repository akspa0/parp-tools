namespace WowViewer.Core.Wmo;

public sealed class WmoGroupBatchDetail
{
    public WmoGroupBatchDetail(
        int batchIndex,
        int payloadOffset,
        byte materialIdRaw,
        bool hasMaterialId,
        ushort firstIndex,
        ushort indexCount,
        byte flags,
        byte[] rawEntryBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(batchIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadOffset);
        ArgumentNullException.ThrowIfNull(rawEntryBytes);

        BatchIndex = batchIndex;
        PayloadOffset = payloadOffset;
        MaterialIdRaw = materialIdRaw;
        HasMaterialId = hasMaterialId;
        FirstIndex = firstIndex;
        IndexCount = indexCount;
        Flags = flags;
        RawEntryBytes = rawEntryBytes;
    }

    public int BatchIndex { get; }

    public int PayloadOffset { get; }

    public byte MaterialIdRaw { get; }

    public bool HasMaterialId { get; }

    public int? MaterialId => HasMaterialId ? MaterialIdRaw : null;

    public ushort FirstIndex { get; }

    public ushort IndexCount { get; }

    public byte Flags { get; }

    public byte[] RawEntryBytes { get; }
}