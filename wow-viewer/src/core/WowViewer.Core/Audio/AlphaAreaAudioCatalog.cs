using WowViewer.Core.World;

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
    private readonly Dictionary<(int ContinentId, AreaNumberParts AreaNumber), AlphaAreaRecord> _areasByContinentAreaNumber = [];
    private readonly Dictionary<AreaNumberParts, List<AlphaAreaRecord>> _areasByAreaNumber = [];
    private readonly AreaIdentityLayout _identityLayout;

    public AlphaAreaAudioCatalog(
        IReadOnlyDictionary<int, AlphaAreaRecord> areas,
        IReadOnlyDictionary<int, AlphaAreaMidiAmbience> midiAmbiences,
        AreaIdentityLayout identityLayout = AreaIdentityLayout.PackedAreaNumber)
    {
        Areas = areas ?? throw new ArgumentNullException(nameof(areas));
        MidiAmbiences = midiAmbiences ?? throw new ArgumentNullException(nameof(midiAmbiences));
        _identityLayout = identityLayout;

        foreach (AlphaAreaRecord area in Areas.Values)
        {
            if (area.AreaNumber == 0)
                continue;

            AreaNumberParts areaNumber = AreaNumberParts.FromRaw(area.AreaNumber);
            _areasByContinentAreaNumber[(area.ContinentId, areaNumber)] = area;
            if (!_areasByAreaNumber.TryGetValue(areaNumber, out List<AlphaAreaRecord>? matches))
            {
                matches = [];
                _areasByAreaNumber[areaNumber] = matches;
            }

            if (matches.All(existing => existing.Id != area.Id))
                matches.Add(area);
        }
    }

    public IReadOnlyDictionary<int, AlphaAreaRecord> Areas { get; }

    public IReadOnlyDictionary<int, AlphaAreaMidiAmbience> MidiAmbiences { get; }

    public AreaIdentityLayout IdentityLayout => _identityLayout;

    public AlphaAreaAudioBinding? TryResolve(int areaId, int? continentId = null)
    {
        if (areaId == 0)
            return null;

        if (_identityLayout == AreaIdentityLayout.DirectAreaId)
            return Areas.TryGetValue(areaId, out AlphaAreaRecord? directArea)
                ? CreateBinding(directArea)
                : null;

        // DBCTool defines Alpha MCNK.Unknown3 as AreaNumber with two 16-bit
        // components: high16=zone and low16=subzone. Keep the two halves as
        // the lookup key instead of treating the storage field as one ID.
        AreaNumberParts areaNumber = AreaNumberParts.FromRaw(areaId);
        bool packedAreaNumber = areaNumber.Zone != 0;
        if (packedAreaNumber)
        {
            if (continentId is int qualifiedContinent
                && _areasByContinentAreaNumber.TryGetValue((qualifiedContinent, areaNumber), out AlphaAreaRecord? qualifiedArea))
            {
                return CreateBinding(qualifiedArea);
            }

            if (_areasByAreaNumber.TryGetValue(areaNumber, out List<AlphaAreaRecord>? areaNumberMatches)
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
            && _areasByContinentAreaNumber.TryGetValue((fallbackContinent, areaNumber), out AlphaAreaRecord? fallbackQualifiedArea))
        {
            return CreateBinding(fallbackQualifiedArea);
        }

        if (!packedAreaNumber
            && _areasByAreaNumber.TryGetValue(areaNumber, out List<AlphaAreaRecord>? fallbackAreaNumberMatches)
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
            if (_identityLayout == AreaIdentityLayout.PackedAreaNumber
                && area.ParentAreaNumber != 0)
                parent = TryResolve(area.ParentAreaNumber, continentId);

            if (parent is null && area.ParentAreaId != 0)
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
