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