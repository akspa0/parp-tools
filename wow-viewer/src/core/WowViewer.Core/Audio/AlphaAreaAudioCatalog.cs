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
    int IntroPriority,
    int AreaNumber = 0,
    int ParentAreaNumber = 0);

public sealed record AlphaAreaAudioBinding(
    AlphaAreaRecord Area,
    AlphaAreaMidiAmbience? MidiAmbience,
    AlphaAreaMidiAmbience? UnderwaterMidiAmbience);

public sealed class AlphaAreaAudioCatalog
{
    private readonly Dictionary<(int ContinentId, int AreaNumber), AlphaAreaRecord> _areasByContinentAreaNumber = [];
    private readonly Dictionary<int, List<AlphaAreaRecord>> _areasByAreaNumber = [];

    public AlphaAreaAudioCatalog(
        IReadOnlyDictionary<int, AlphaAreaRecord> areas,
        IReadOnlyDictionary<int, AlphaAreaMidiAmbience> midiAmbiences)
    {
        Areas = areas ?? throw new ArgumentNullException(nameof(areas));
        MidiAmbiences = midiAmbiences ?? throw new ArgumentNullException(nameof(midiAmbiences));

        foreach (AlphaAreaRecord area in Areas.Values)
        {
            if (area.AreaNumber == 0)
                continue;

            _areasByContinentAreaNumber[(area.ContinentId, area.AreaNumber)] = area;
            if (!_areasByAreaNumber.TryGetValue(area.AreaNumber, out List<AlphaAreaRecord>? matches))
            {
                matches = [];
                _areasByAreaNumber[area.AreaNumber] = matches;
            }

            if (matches.All(existing => existing.Id != area.Id))
                matches.Add(area);
        }
    }

    public IReadOnlyDictionary<int, AlphaAreaRecord> Areas { get; }

    public IReadOnlyDictionary<int, AlphaAreaMidiAmbience> MidiAmbiences { get; }

    public AlphaAreaAudioBinding? TryResolve(int areaId, int? continentId = null)
    {
        if (areaId <= 0)
            return null;

        // DBCTool defines Alpha MCNK.Unknown3 as the packed AreaNumber
        // (zone << 16) | subzone. The packed form is preferred for values
        // outside the 16-bit ID range. For ordinary direct IDs, keep the
        // canonical ID row authoritative and use AreaNumber only as a
        // fallback so a modern ID cannot be hijacked by an alias.
        bool packedAreaNumber = areaId > ushort.MaxValue;
        if (packedAreaNumber)
        {
            if (continentId is int qualifiedContinent
                && _areasByContinentAreaNumber.TryGetValue((qualifiedContinent, areaId), out AlphaAreaRecord? qualifiedArea))
            {
                return CreateBinding(qualifiedArea);
            }

            if (_areasByAreaNumber.TryGetValue(areaId, out List<AlphaAreaRecord>? areaNumberMatches)
                && areaNumberMatches.Count == 1)
            {
                return CreateBinding(areaNumberMatches[0]);
            }
        }

        if (Areas.TryGetValue(areaId, out AlphaAreaRecord? area))
        {
            return CreateBinding(area);
        }

        if (!packedAreaNumber
            && continentId is int fallbackContinent
            && _areasByContinentAreaNumber.TryGetValue((fallbackContinent, areaId), out AlphaAreaRecord? fallbackQualifiedArea))
        {
            return CreateBinding(fallbackQualifiedArea);
        }

        if (!packedAreaNumber
            && _areasByAreaNumber.TryGetValue(areaId, out List<AlphaAreaRecord>? fallbackAreaNumberMatches)
            && fallbackAreaNumberMatches.Count == 1)
        {
            return CreateBinding(fallbackAreaNumberMatches[0]);
        }

        return null;
    }

    public AlphaAreaAudioBinding? TryResolveWithParents(int areaId, int? continentId = null)
    {
        HashSet<(int Id, int AreaNumber)> visited = [];
        AlphaAreaAudioBinding? binding = TryResolve(areaId, continentId);

        while (binding is not null && visited.Add((binding.Area.Id, binding.Area.AreaNumber)))
        {
            if (HasAudioAssignment(binding))
                return binding;

            AlphaAreaRecord area = binding.Area;
            AlphaAreaAudioBinding? parent = null;
            if (area.ParentAreaNumber != 0)
                parent = TryResolve(area.ParentAreaNumber, continentId);

            if (parent is null && area.ParentAreaId != 0 && area.ParentAreaId != area.ParentAreaNumber)
                parent = TryResolve(area.ParentAreaId, continentId);

            binding = parent;
        }

        return null;
    }

    private AlphaAreaAudioBinding CreateBinding(AlphaAreaRecord area)
    {
        MidiAmbiences.TryGetValue(area.MidiAmbienceId, out AlphaAreaMidiAmbience? midiAmbience);
        MidiAmbiences.TryGetValue(area.MidiAmbienceUnderwaterId, out AlphaAreaMidiAmbience? underwaterMidiAmbience);
        return new AlphaAreaAudioBinding(area, midiAmbience, underwaterMidiAmbience);
    }

    private static bool HasAudioAssignment(AlphaAreaAudioBinding binding)
    {
        return binding.Area.ZoneMusicId > 0
            || binding.MidiAmbience is not null
            || binding.UnderwaterMidiAmbience is not null;
    }

    public IEnumerable<AlphaAreaAudioBinding> EnumerateBindings()
    {
        foreach (AlphaAreaRecord area in Areas.Values.OrderBy(static area => area.Id))
        {
            yield return CreateBinding(area);
        }
    }
}
