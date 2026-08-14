namespace WowViewer.Core.Audio;

public sealed record AlphaAreaMidiAmbience(
    int Id,
    string DaySequence,
    string NightSequence,
    string DlsFile,
    float Volume);

public sealed record AlphaAreaRecord(
    int Id,
    int ContinentId,
    int ParentAreaId,
    string AreaName,
    int MidiAmbienceId,
    int MidiAmbienceUnderwaterId,
    int ZoneMusicId,
    int IntroSoundId,
    int IntroPriority);

public sealed record AlphaAreaAudioBinding(
    AlphaAreaRecord Area,
    AlphaAreaMidiAmbience? MidiAmbience,
    AlphaAreaMidiAmbience? UnderwaterMidiAmbience);

public sealed class AlphaAreaAudioCatalog
{
    public AlphaAreaAudioCatalog(
        IReadOnlyDictionary<int, AlphaAreaRecord> areas,
        IReadOnlyDictionary<int, AlphaAreaMidiAmbience> midiAmbiences)
    {
        Areas = areas ?? throw new ArgumentNullException(nameof(areas));
        MidiAmbiences = midiAmbiences ?? throw new ArgumentNullException(nameof(midiAmbiences));
    }

    public IReadOnlyDictionary<int, AlphaAreaRecord> Areas { get; }

    public IReadOnlyDictionary<int, AlphaAreaMidiAmbience> MidiAmbiences { get; }

    public AlphaAreaAudioBinding? TryResolve(int areaId)
    {
        if (!Areas.TryGetValue(areaId, out AlphaAreaRecord? area))
        {
            return null;
        }

        MidiAmbiences.TryGetValue(area.MidiAmbienceId, out AlphaAreaMidiAmbience? midiAmbience);
        MidiAmbiences.TryGetValue(area.MidiAmbienceUnderwaterId, out AlphaAreaMidiAmbience? underwaterMidiAmbience);
        return new AlphaAreaAudioBinding(area, midiAmbience, underwaterMidiAmbience);
    }

    /// <summary>
    /// Resolve the most specific area with audio metadata, walking the active
    /// area's parent chain when the child row has no usable music assignment.
    /// This mirrors the game's area inheritance instead of treating an
    /// MCNK/WMO area as an isolated sound zone.
    /// </summary>
    public AlphaAreaAudioBinding? TryResolveWithParents(int areaId)
    {
        HashSet<int> visited = [];
        int currentId = areaId;

        while (currentId > 0 && visited.Add(currentId) && Areas.TryGetValue(currentId, out AlphaAreaRecord? area))
        {
            AlphaAreaAudioBinding binding = TryResolve(currentId)!;
            if (area.ZoneMusicId > 0
                || binding.MidiAmbience is not null
                || binding.UnderwaterMidiAmbience is not null)
            {
                return binding;
            }

            currentId = area.ParentAreaId;
        }

        return null;
    }

    public IEnumerable<AlphaAreaAudioBinding> EnumerateBindings()
    {
        foreach (AlphaAreaRecord area in Areas.Values.OrderBy(static area => area.Id))
        {
            MidiAmbiences.TryGetValue(area.MidiAmbienceId, out AlphaAreaMidiAmbience? midiAmbience);
            MidiAmbiences.TryGetValue(area.MidiAmbienceUnderwaterId, out AlphaAreaMidiAmbience? underwaterMidiAmbience);
            yield return new AlphaAreaAudioBinding(area, midiAmbience, underwaterMidiAmbience);
        }
    }
}
