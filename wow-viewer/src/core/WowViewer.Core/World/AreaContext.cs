namespace WowViewer.Core.World;

/// <summary>
/// Alpha AreaNumber storage is a 32-bit value whose identity is two 16-bit
/// values: the zone in the high word and the subzone in the low word.
/// </summary>
public readonly record struct AreaNumberParts(ushort Zone, ushort Subzone)
{
    public uint Raw => ((uint)Zone << 16) | Subzone;

    public uint ZoneBase => (uint)Zone << 16;

    public int SignedRaw => unchecked((int)Raw);

    public int SignedZoneBase => unchecked((int)ZoneBase);

    public static AreaNumberParts FromRaw(int raw)
    {
        uint value = unchecked((uint)raw);
        return new AreaNumberParts(
            (ushort)(value >> 16),
            (ushort)(value & ushort.MaxValue));
    }
}

public enum AreaResolutionReason
{
    Resolved,
    NoTerrainChunk,
    MissingAreaId,
    AreaRowMissing,
    MapMismatch,
    MissingLocalizedName
}

public enum AreaContextSource
{
    DirectAreaId,
    PackedAreaNumber,
    Unresolved
}

public sealed record AreaContextEntry(
    int Id,
    string Name,
    int ParentAreaId,
    int ParentAreaNumber,
    int MapId,
    int Flags,
    int AreaNumber);

public sealed record AreaDisplayText(
    string? ZoneText,
    string? SubzoneText,
    AreaContextSource Source,
    AreaResolutionReason Reason)
{
    public string? PrimaryText => SubzoneText ?? ZoneText;
}

public sealed record AreaLookupResult(
    int RawAreaId,
    int MapId,
    int? CanonicalAreaId,
    int? ParentAreaId,
    int? AreaNumber,
    string? AreaName,
    string? ZoneText,
    string? SubzoneText,
    AreaContextSource Source,
    AreaResolutionReason Reason,
    bool MapMatched)
{
    public bool IsResolved => Reason == AreaResolutionReason.Resolved;
    public string? PrimaryText => SubzoneText ?? ZoneText;

    public static AreaLookupResult Unresolved(int rawAreaId, int mapId, AreaResolutionReason reason)
    {
        return new AreaLookupResult(
            rawAreaId,
            mapId,
            CanonicalAreaId: null,
            ParentAreaId: null,
            AreaNumber: null,
            AreaName: null,
            ZoneText: null,
            SubzoneText: null,
            Source: AreaContextSource.Unresolved,
            Reason: reason,
            MapMatched: false);
    }
}

public static class AreaDisplayTextResolver
{
    public static AreaDisplayText Resolve(
        AreaContextEntry entry,
        AreaContextEntry? parent,
        AreaContextSource source,
        AreaResolutionReason reason)
    {
        ArgumentNullException.ThrowIfNull(entry);

        string entryName = entry.Name.Trim();
        string parentName = parent?.Name.Trim() ?? string.Empty;
        string zoneText = string.IsNullOrWhiteSpace(parentName) ? entryName : parentName;
        string subzoneText = string.IsNullOrWhiteSpace(entryName) ? zoneText : entryName;

        if (string.IsNullOrWhiteSpace(entryName))
            reason = AreaResolutionReason.MissingLocalizedName;

        return new AreaDisplayText(
            string.IsNullOrWhiteSpace(zoneText) ? null : zoneText,
            string.IsNullOrWhiteSpace(subzoneText) ? null : subzoneText,
            source,
            reason);
    }
}
