namespace WowViewer.Core.Wmo;

public sealed class WmoGroupFaceMaterialDetail
{
    public WmoGroupFaceMaterialDetail(int faceIndex, byte flags, byte materialId, ushort? legacyExtraValue)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(faceIndex);

        FaceIndex = faceIndex;
        Flags = flags;
        MaterialId = materialId;
        LegacyExtraValue = legacyExtraValue;
    }

    public int FaceIndex { get; }

    public byte Flags { get; }

    public byte MaterialId { get; }

    public bool IsHidden => MaterialId == byte.MaxValue;

    public ushort? LegacyExtraValue { get; }
}