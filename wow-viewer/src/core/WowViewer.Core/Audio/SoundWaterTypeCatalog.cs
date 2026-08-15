namespace WowViewer.Core.Audio;

/// <summary>
/// One build-scoped SoundWaterType row. The client stores the liquid/sound
/// relationship here; sound IDs are never inferred from a liquid family.
/// </summary>
public sealed record SoundWaterTypeEntry(
    int Id,
    int SoundType,
    int SoundSubtype,
    int SoundId);

/// <summary>
/// Build-scoped SoundWaterType metadata used by legacy MCNK liquid triggers.
/// </summary>
public sealed class SoundWaterTypeCatalog
{
    private readonly Dictionary<(int SoundType, int SoundSubtype), SoundWaterTypeEntry> _byTypeAndSubtype;

    public SoundWaterTypeCatalog(IEnumerable<SoundWaterTypeEntry> entries)
    {
        ArgumentNullException.ThrowIfNull(entries);

        Entries = entries
            .Where(static entry => entry.Id > 0 && entry.SoundId > 0)
            .OrderBy(static entry => entry.Id)
            .ToArray();
        _byTypeAndSubtype = Entries
            .GroupBy(static entry => (entry.SoundType, entry.SoundSubtype))
            .ToDictionary(static group => group.Key, static group => group.First());
    }

    public IReadOnlyList<SoundWaterTypeEntry> Entries { get; }

    public bool TryResolve(int soundType, int soundSubtype, out SoundWaterTypeEntry entry)
        => _byTypeAndSubtype.TryGetValue((soundType, soundSubtype), out entry!);
}
