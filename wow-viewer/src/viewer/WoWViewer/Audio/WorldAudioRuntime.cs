using System.Diagnostics;
using System.Numerics;
using Silk.NET.OpenAL;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using WowViewer.Core.Audio;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.World;

namespace WoWViewer.Audio;

/// <summary>
/// Viewer-owned resident world audio. MCSE candidates are admitted and
/// released with terrain tiles; SoundEntries and audio files come from the
/// active client source.
/// </summary>
public sealed class WorldAudioRuntime : IDisposable
{
    private readonly IDataSource _dataSource;
    private readonly Dictionary<(int TileX, int TileY), IReadOnlyList<TerrainSoundEmitter>> _tileEmitters = [];
    private readonly object _tileEmittersLock = new();
    private readonly Dictionary<string, uint> _buffers = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<EmitterKey, ActiveEmitter> _active = [];
    private readonly HashSet<EmitterKey> _heardInRange = [];
    private readonly HashSet<string> _diagnosticKeys = new(StringComparer.OrdinalIgnoreCase);
    private IReadOnlyList<TerrainSoundEmitter> _residentEmitterSnapshot = Array.Empty<TerrainSoundEmitter>();
    private IReadOnlyList<AudioTriggerDiagnostic> _emitterDiagnostics = Array.Empty<AudioTriggerDiagnostic>();
    private long _nextDiagnosticRefreshTimestamp;

    private static long DiagnosticRefreshIntervalTicks => Math.Max(1L, Stopwatch.Frequency / 4L);

    private AudioContext? _context;
    private AL? _al;
    private AlphaSoundEntriesCatalog? _soundEntries;
    private SoundWaterTypeCatalog? _soundWaterTypes;
    private AlphaAreaAudioCatalog? _areaAudioCatalog;
    private AreaIdentityLayout _areaIdentityLayout = AreaIdentityLayout.DirectAreaId;
    private uint? _previewSource;
    private string? _previewPath;
    private float _previewBaseGain = 1f;
    private uint? _areaMusicSource;
    private int _areaMusicSoundEntryId;
    private bool _areaMusicNight;
    private Vector3 _listenerPosition;
    private bool _disposed;
    private bool _loggedNoBackend;
    private bool _worldTriggersEnabled;

    public WorldAudioRuntime(IDataSource dataSource)
    {
        _dataSource = dataSource ?? throw new ArgumentNullException(nameof(dataSource));
    }

    public string Status { get; private set; } = "Audio runtime not configured.";

    public string LastDiagnostic { get; private set; } = "No audio diagnostic yet.";

    public string AreaMusicStatus { get; private set; } = "Area music metadata not loaded.";

    public AreaIdentityLayout AreaIdentityLayout => _areaIdentityLayout;

    public bool BackendReady => _al is not null;

    public string? PreviewPath => _previewPath;

    public float MasterGain { get; private set; } = 1f;

    public float EmitterGain { get; private set; } = 1f;

    public bool IsMuted { get; private set; }

    public int ResidentEmitterCount => _residentEmitterSnapshot.Count;

    /// <summary>
    /// Snapshot of the normalized resident emitter records for non-audio
    /// visualization. This is rebuilt only when tile residency changes so the
    /// renderer does not enumerate all emitter lists every frame.
    /// </summary>
    public IReadOnlyList<TerrainSoundEmitter> ResidentEmitters => _residentEmitterSnapshot;

    public int ActiveEmitterCount => _active.Count;

    public IReadOnlyList<AudioTriggerDiagnostic> EmitterDiagnostics => _emitterDiagnostics;

    public int ResolvedSoundEntryCount => _soundEntries?.Entries.Count ?? 0;

    /// <summary>
    /// Spatial MCSE/MCNK world triggers are opt-in so a looping client sample
    /// cannot interrupt inspection merely because a tile became resident.
    /// </summary>
    public bool WorldTriggersEnabled => _worldTriggersEnabled;

    public bool AreaMusicPlaybackEnabled => WorldAudioPlaybackPolicy.AutomaticZoneMusicPlaybackEnabled;

    public int ResolvedSoundWaterTypeCount => _soundWaterTypes?.Entries.Count ?? 0;

    public IReadOnlyList<int> ResidentSoundEntryIds
        => _residentEmitterSnapshot
            .Select(ResolveResidentSoundEntryId)
            .Where(static id => id > 0 && id <= int.MaxValue)
            .Select(static id => (int)id)
            .Distinct()
            .OrderBy(static id => id)
            .ToArray();

    public void Configure(DBCD.Providers.IDBCProvider dbcProvider, string definitionsDirectory, string buildVersion)
    {
        ThrowIfDisposed();
        StopAll();
        _worldTriggersEnabled = false;
        _soundEntries = null;
        _soundWaterTypes = null;
        _areaAudioCatalog = null;
        _areaIdentityLayout = AreaIdentityLayoutResolver.FromBuild(buildVersion);
        AreaMusicStatus = $"Area music metadata not loaded for {buildVersion} ({_areaIdentityLayout}).";
        _loggedNoBackend = false;

        try
        {
            _soundEntries = new AlphaSoundEntriesCatalogReader()
                .Load(dbcProvider, definitionsDirectory, buildVersion);

            try
            {
                _soundWaterTypes = new AlphaSoundWaterTypeCatalogReader()
                    .Load(dbcProvider, definitionsDirectory, buildVersion);
            }
            catch (Exception ex)
            {
                ViewerLog.Info(ViewerLog.Category.General,
                    $"[Audio] SoundWaterType unavailable for {buildVersion}: {ex.Message}");
            }

            try
            {
                _areaAudioCatalog = new AlphaAreaAudioCatalogReader()
                    .Load(dbcProvider, definitionsDirectory, buildVersion);
                AreaMusicStatus = $"AreaTable/AreaMIDIAmbiences loaded: {_areaAudioCatalog.Areas.Count} areas ({_areaIdentityLayout}).";
            }
            catch (Exception ex)
            {
                AreaMusicStatus = $"Area music metadata unavailable for {buildVersion}: {ex.Message}";
                ViewerLog.Info(ViewerLog.Category.General, $"[Audio] {AreaMusicStatus}");
            }
        }
        catch (Exception ex)
        {
            Status = $"SoundEntries unavailable for {buildVersion}: {ex.Message}";
            LastDiagnostic = Status;
            ViewerLog.Info(ViewerLog.Category.General, $"[Audio] {Status}");
            return;
        }

        if (!OpenAlNativeLibraryProbe.TryFind(out string? libraryName))
        {
            Status = "Audio backend unavailable: OpenAL native library was not found; audio playback is disabled.";
            LastDiagnostic = Status;
            ViewerLog.Info(ViewerLog.Category.General, $"[Audio] {Status}");
            return;
        }

        AudioContext? context = null;
        try
        {
            context = new AudioContext();
            context.MakeCurrent();
            AL al = AL.GetApi();
            _context = context;
            _al = al;
            Status = $"OpenAL ready ({libraryName}); SoundEntries={_soundEntries.Entries.Count}; SoundWaterType={_soundWaterTypes?.Entries.Count ?? 0}; WAV/OGG/MP3 decoders available; MIDI+DLS pair renderer pending.";
            LastDiagnostic = Status;
            ViewerLog.Important(ViewerLog.Category.General, $"[Audio] {Status}");
        }
        catch (Exception ex)
        {
            try { context?.Dispose(); } catch { }
            _context = null;
            _al = null;
            Status = $"Audio backend unavailable: {ex.Message}";
            LastDiagnostic = Status;
            ViewerLog.Info(ViewerLog.Category.General, $"[Audio] {Status}");
        }
    }

    public void SetMasterGain(float gain)
    {
        MasterGain = Math.Clamp(gain, 0f, 2f);
        ApplyActiveGains();
    }

    public void SetEmitterGain(float gain)
    {
        EmitterGain = Math.Clamp(gain, 0f, 2f);
        ApplyActiveGains();
    }

    public void SetWorldTriggersEnabled(bool enabled)
    {
        if (_worldTriggersEnabled == enabled)
            return;

        _worldTriggersEnabled = enabled;
        if (!enabled)
        {
            foreach (EmitterKey key in _active.Keys.ToArray())
                StopEmitter(key);
            _heardInRange.Clear();
        }

        LastDiagnostic = enabled
            ? "MCSE/MCNK world triggers enabled."
            : "MCSE/MCNK world triggers disabled; resident rows remain inspectable.";
        InvalidateEmitterDiagnostics();
    }

    public void SetMuted(bool muted)
    {
        IsMuted = muted;
        ApplyActiveGains();
        LastDiagnostic = muted ? "Audio output muted." : "Audio output unmuted.";
    }

    public bool TryPlaySoundEntry(uint soundEntryId, bool loop, out string reason)
    {
        reason = string.Empty;
        if (_al is null || _soundEntries is null)
        {
            reason = "OpenAL and SoundEntries must be ready before previewing a sound.";
            LastDiagnostic = reason;
            return false;
        }

        if (!_soundEntries.TryResolve(soundEntryId, out AlphaSoundEntry? soundEntry))
        {
            reason = $"SoundEntries {soundEntryId} is not present in the loaded client schema.";
            LastDiagnostic = reason;
            return false;
        }

        string? virtualPath = soundEntry.EnumerateVirtualPaths()
            .FirstOrDefault(path => _dataSource.FileExists(path));
        if (virtualPath is null)
        {
            reason = $"SoundEntries {soundEntryId} has no resolvable client audio file.";
            LastDiagnostic = reason;
            LogOnce($"preview-missing:{soundEntryId}", $"[Audio] {reason}");
            return false;
        }

        try
        {
            StopPreview();
            uint buffer = GetOrCreateBuffer(virtualPath);
            if (buffer == 0)
            {
                reason = LastDiagnostic;
                return false;
            }

            uint source = _al.GenSource();
            _al.SetSourceProperty(source, SourceInteger.Buffer, buffer);
            _al.SetSourceProperty(source, SourceVector3.Position, _listenerPosition);
            _al.SetSourceProperty(source, SourceFloat.Gain, soundEntry.Volume * EffectiveMasterGain);
            _al.SetSourceProperty(source, SourceFloat.ReferenceDistance, MathF.Max(1f, soundEntry.MinDistance));
            _al.SetSourceProperty(source, SourceFloat.MaxDistance, MathF.Max(1f, soundEntry.MaxDistance));
            _al.SetSourceProperty(source, SourceBoolean.Looping, loop);
            _al.SourcePlay(source);
            _previewSource = source;
            _previewPath = virtualPath;
            _previewBaseGain = soundEntry.Volume;
            reason = $"Playing SoundEntries {soundEntryId}: {virtualPath}";
            LastDiagnostic = reason;
            return true;
        }
        catch (Exception ex)
        {
            reason = $"SoundEntries {soundEntryId} preview failed: {ex.Message}";
            LastDiagnostic = reason;
            DisableBackend(reason);
            return false;
        }
    }

    public void StopPreview()
    {
        if (_previewSource is uint source && _al is not null)
        {
            try
            {
                _al.SourceStop(source);
                _al.DeleteSource(source);
            }
            catch (Exception ex)
            {
                LastDiagnostic = $"Audio preview stop failed: {ex.Message}";
            }
        }

        _previewSource = null;
        _previewPath = null;
        _previewBaseGain = 1f;
    }

    public void AddTile(int tileX, int tileY, IReadOnlyList<TerrainSoundEmitter> emitters)
    {
        ThrowIfDisposed();
        lock (_tileEmittersLock)
        {
            _tileEmitters[(tileX, tileY)] = emitters ?? Array.Empty<TerrainSoundEmitter>();
            _residentEmitterSnapshot = BuildResidentEmitterSnapshotLocked();
        }
        InvalidateEmitterDiagnostics();
    }

    public void RemoveTile(int tileX, int tileY)
    {
        if (_disposed)
            return;

        lock (_tileEmittersLock)
        {
            _tileEmitters.Remove((tileX, tileY));
            _residentEmitterSnapshot = BuildResidentEmitterSnapshotLocked();
        }
        foreach (EmitterKey key in _active.Keys.Where(key => key.TileX == tileX && key.TileY == tileY).ToArray())
            StopEmitter(key);
        _heardInRange.RemoveWhere(key => key.TileX == tileX && key.TileY == tileY);
        InvalidateEmitterDiagnostics();
        RefreshEmitterDiagnostics();
    }

    public void Update(
        Vector3 listenerPosition,
        Vector3 listenerForward,
        int areaId = 0,
        float gameTime = 0.5f,
        int continentId = -1,
        AreaLookupResult? areaLookup = null)
    {
        if (_disposed)
            return;

        _listenerPosition = listenerPosition;
        if (_al is null || _soundEntries is null)
        {
            RefreshEmitterDiagnosticsIfDue();
            return;
        }

        try
        {
            _al.SetListenerProperty(ListenerVector3.Position, listenerPosition);
            unsafe
            {
                float* orientation = stackalloc float[6]
                {
                    listenerForward.X,
                    listenerForward.Y,
                    listenerForward.Z,
                    0f,
                    0f,
                    1f
                };
                _al.SetListenerProperty(ListenerFloatArray.Orientation, orientation);
            }
        }
        catch (Exception ex)
        {
            DisableBackend($"listener update failed: {ex.Message}");
            return;
        }

        if (_worldTriggersEnabled)
            UpdateAreaMusic(areaId, gameTime, continentId, areaLookup);
        else
        {
            StopAreaMusic();
            AreaMusicStatus = "World triggers disabled; area music is not started.";
        }

        HashSet<EmitterKey> inRange = [];
        KeyValuePair<(int TileX, int TileY), IReadOnlyList<TerrainSoundEmitter>>[] tileSnapshot;
        lock (_tileEmittersLock)
            tileSnapshot = _tileEmitters.ToArray();

        if (_worldTriggersEnabled)
        {
            foreach (KeyValuePair<(int TileX, int TileY), IReadOnlyList<TerrainSoundEmitter>> pair in tileSnapshot)
            {
                int tileX = pair.Key.TileX;
                int tileY = pair.Key.TileY;
                IReadOnlyList<TerrainSoundEmitter> emitters = pair.Value;
                for (int index = 0; index < emitters.Count; index++)
                {
                    TerrainSoundEmitter emitter = emitters[index];
                    EmitterKey key = new(tileX, tileY, index);
                    uint soundEntryId = ResolveResidentSoundEntryId(emitter);
                    if (soundEntryId == 0 || !_soundEntries.TryResolve(soundEntryId, out AlphaSoundEntry? soundEntry))
                        continue;

                    float maxDistance = FirstPositive(emitter.CutoffDistance, emitter.MaxDistance, soundEntry.DistanceCutoff, soundEntry.MaxDistance, 100f);
                    float minDistance = Math.Clamp(FirstPositive(emitter.MinDistance, soundEntry.MinDistance, 0f), 0f, maxDistance);
                    float distance = Vector3.Distance(listenerPosition, emitter.Position);
                    if (distance > maxDistance)
                        continue;

                    inRange.Add(key);
                    float gain = soundEntry.Volume * Attenuation(distance, minDistance, maxDistance);
                    if (!_heardInRange.Contains(key))
                    {
                        TryStartEmitter(key, emitter, soundEntry, minDistance, maxDistance, gain);
                        _heardInRange.Add(key);
                    }

                    if (_active.TryGetValue(key, out ActiveEmitter? active))
                    {
                        try
                        {
                            active.BaseGain = gain;
                            _al.SetSourceProperty(active.Source, SourceVector3.Position, emitter.Position);
                            _al.SetSourceProperty(active.Source, SourceFloat.Gain, gain * EffectiveMasterGain * EmitterGain);
                        }
                        catch (Exception ex)
                        {
                            DisableBackend($"emitter update failed: {ex.Message}");
                            return;
                        }
                    }
                }
            }
        }

        if (_previewSource is uint previewSource)
        {
            try
            {
                _al.SetSourceProperty(previewSource, SourceVector3.Position, listenerPosition);
            }
            catch (Exception ex)
            {
                DisableBackend($"preview update failed: {ex.Message}");
                return;
            }
        }

        if (_areaMusicSource is uint areaMusicSource)
        {
            try
            {
                _al.SetSourceProperty(areaMusicSource, SourceVector3.Position, listenerPosition);
            }
            catch (Exception ex)
            {
                DisableBackend($"area music update failed: {ex.Message}");
                return;
            }
        }

        foreach (EmitterKey key in _active.Keys.Where(key => !inRange.Contains(key)).ToArray())
            StopEmitter(key);

        _heardInRange.RemoveWhere(key => !inRange.Contains(key));
        RefreshEmitterDiagnosticsIfDue();
    }

    /// <summary>
    /// Rebuilds the inspectable MCSE decision list without starting an OpenAL source. File reads and
    /// decodes are opt-in because probing every resident emitter every frame would reintroduce the
    /// performance problem this surface is intended to diagnose.
    /// </summary>
    public void RefreshEmitterDiagnostics(bool probeFiles = false)
    {
        if (_disposed)
            return;

        List<AudioTriggerDiagnostic> diagnostics = [];
        KeyValuePair<(int TileX, int TileY), IReadOnlyList<TerrainSoundEmitter>>[] tileSnapshot;
        lock (_tileEmittersLock)
            tileSnapshot = _tileEmitters.ToArray();

        foreach (KeyValuePair<(int TileX, int TileY), IReadOnlyList<TerrainSoundEmitter>> pair in tileSnapshot)
        {
            int tileX = pair.Key.TileX;
            int tileY = pair.Key.TileY;
            IReadOnlyList<TerrainSoundEmitter> emitters = pair.Value;
            for (int index = 0; index < emitters.Count; index++)
            {
                TerrainSoundEmitter emitter = emitters[index];
                diagnostics.Add(BuildEmitterDiagnostic(tileX, tileY, index, emitter, probeFiles));
            }
        }

        _emitterDiagnostics = diagnostics;
        Interlocked.Exchange(
            ref _nextDiagnosticRefreshTimestamp,
            Stopwatch.GetTimestamp() + DiagnosticRefreshIntervalTicks);
    }

    private void InvalidateEmitterDiagnostics()
    {
        Interlocked.Exchange(ref _nextDiagnosticRefreshTimestamp, 0L);
    }

    private IReadOnlyList<TerrainSoundEmitter> BuildResidentEmitterSnapshotLocked()
        => _tileEmitters
            .OrderBy(static pair => pair.Key.TileY)
            .ThenBy(static pair => pair.Key.TileX)
            .SelectMany(static pair => pair.Value)
            .ToArray();

    private void RefreshEmitterDiagnosticsIfDue()
    {
        long now = Stopwatch.GetTimestamp();
        if (now < Interlocked.Read(ref _nextDiagnosticRefreshTimestamp))
            return;

        RefreshEmitterDiagnostics();
    }

    private AudioTriggerDiagnostic BuildEmitterDiagnostic(
        int tileX,
        int tileY,
        int index,
        TerrainSoundEmitter emitter,
        bool probeFiles)
    {
        string backendStatus = _al is null ? Status : "OpenAL ready";
        EmitterKey key = new(tileX, tileY, index);
        uint soundEntryId = ResolveResidentSoundEntryId(emitter);
        if (_soundEntries is null || soundEntryId == 0 || !_soundEntries.TryResolve(soundEntryId, out AlphaSoundEntry? soundEntry))
        {
            return new AudioTriggerDiagnostic(
                emitter.TriggerKind,
                tileX,
                tileY,
                emitter.ChunkX,
                emitter.ChunkY,
                emitter.SoundPointId,
                soundEntryId,
                emitter.RawPosition,
                emitter.Position,
                emitter.CoordinateProfile,
                emitter.MinDistance,
                emitter.MaxDistance,
                emitter.CutoffDistance,
                Vector3.Distance(_listenerPosition, emitter.Position),
                false,
                false,
                Array.Empty<string>(),
                null,
                "Not resolved",
                false,
                false,
                "Not attempted",
                backendStatus,
                AudioTriggerTerminalState.UnresolvedSoundEntry,
                emitter.TriggerKind == AudioTriggerKind.McnkLiquid
                    ? $"MCNK liquid family={emitter.LiquidFamily} flags=0x{emitter.McnkFlags:X8} subtype={emitter.SoundWaterSubtype} has no SoundWaterType mapping in the active build."
                    : $"SoundEntries {soundEntryId} is not present in the active schema.",
                emitter.McnkFlags,
                emitter.LiquidFamily,
                emitter.SoundWaterSubtype);
        }

        string[] candidatePaths = soundEntry.EnumerateVirtualPaths().Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
        string? selectedPath = null;
        foreach (string candidate in candidatePaths)
        {
            try
            {
                if (_dataSource.FileExists(candidate))
                {
                    selectedPath = candidate;
                    break;
                }
            }
            catch (Exception ex)
            {
                return CreateEmitterDiagnostic(
                    emitter,
                    tileX,
                    tileY,
                    emitter.ChunkX,
                    emitter.ChunkY,
                    soundEntry.Id,
                    candidatePaths,
                    candidate,
                    $"{_dataSource.Name} (probe failed)",
                    false,
                    false,
                    "Not attempted",
                    backendStatus,
                    AudioTriggerTerminalState.ReadFailed,
                    ex.Message);
            }
        }

        float maxDistance = FirstPositive(emitter.CutoffDistance, emitter.MaxDistance, soundEntry.DistanceCutoff, soundEntry.MaxDistance, 100f);
        float minDistance = Math.Clamp(FirstPositive(emitter.MinDistance, soundEntry.MinDistance, 0f), 0f, maxDistance);
        float distance = Vector3.Distance(_listenerPosition, emitter.Position);
        bool inRange = distance <= maxDistance;
        string source = selectedPath is null
            ? "Missing"
            : DescribeResourceSource(selectedPath);

        if (selectedPath is null)
        {
            return CreateEmitterDiagnostic(
                emitter,
                tileX,
                tileY,
                emitter.ChunkX,
                emitter.ChunkY,
                soundEntry.Id,
                candidatePaths,
                null,
                source,
                false,
                false,
                "Not attempted",
                backendStatus,
                AudioTriggerTerminalState.MissingResource,
                $"SoundEntries {soundEntry.Id} has no readable candidate path.",
                minDistance,
                maxDistance,
                distance,
                inRange);
        }

        bool bytesRead = false;
        string decodeStatus = probeFiles ? "Not decoded" : "Not probed";
        if (probeFiles)
        {
            byte[]? bytes = _dataSource.ReadFile(selectedPath);
            if (bytes is null)
            {
                return CreateEmitterDiagnostic(
                    emitter,
                    tileX,
                    tileY,
                    emitter.ChunkX,
                    emitter.ChunkY,
                    soundEntry.Id,
                    candidatePaths,
                    selectedPath,
                    source,
                    true,
                    false,
                    "Read failed",
                    backendStatus,
                    AudioTriggerTerminalState.ReadFailed,
                    $"{selectedPath} was catalog-visible but returned no bytes.",
                    minDistance,
                    maxDistance,
                    distance,
                    inRange);
            }

            bytesRead = true;
            if (!ClientAudioDecoder.TryDecode(bytes, selectedPath, out _, out string reason))
            {
                return CreateEmitterDiagnostic(
                    emitter,
                    tileX,
                    tileY,
                    emitter.ChunkX,
                    emitter.ChunkY,
                    soundEntry.Id,
                    candidatePaths,
                    selectedPath,
                    source,
                    true,
                    true,
                    reason,
                    backendStatus,
                    AudioTriggerTerminalState.DecodeFailed,
                    $"Decoder rejected {selectedPath}: {reason}",
                    minDistance,
                    maxDistance,
                    distance,
                    inRange);
            }

            decodeStatus = "Decoded";
        }

        AudioTriggerTerminalState terminalState;
        string detail;
        if (!inRange)
        {
            terminalState = AudioTriggerTerminalState.OutOfRange;
            detail = $"Distance {distance:F1} exceeds max {maxDistance:F1}.";
        }
        else if (!_worldTriggersEnabled)
        {
            terminalState = AudioTriggerTerminalState.Disabled;
            detail = "World triggers are disabled; resident metadata is inspectable but playback is blocked.";
        }
        else if (IsMuted)
        {
            terminalState = AudioTriggerTerminalState.Muted;
            detail = "Emitter is in range but master audio is muted.";
        }
        else if (_al is null)
        {
            terminalState = AudioTriggerTerminalState.BackendUnavailable;
            detail = backendStatus;
        }
        else if (_active.ContainsKey(key))
        {
            terminalState = AudioTriggerTerminalState.Active;
            detail = "OpenAL source is active.";
        }
        else if (!probeFiles)
        {
            terminalState = AudioTriggerTerminalState.DecodePending;
            detail = "Path resolved; press Probe current emitters to read and decode it.";
        }
        else
        {
            terminalState = AudioTriggerTerminalState.Ready;
            detail = "Resource read and decoded; no active source is currently associated.";
        }

        return CreateEmitterDiagnostic(
            emitter,
            tileX,
            tileY,
            emitter.ChunkX,
            emitter.ChunkY,
            soundEntry.Id,
            candidatePaths,
            selectedPath,
            source,
            true,
            bytesRead,
            decodeStatus,
            backendStatus,
            terminalState,
            detail,
            minDistance,
            maxDistance,
            distance,
            inRange);
    }

    private static AudioTriggerDiagnostic CreateEmitterDiagnostic(
        TerrainSoundEmitter emitter,
        int tileX,
        int tileY,
        int chunkX,
        int chunkY,
        int soundEntryId,
        IReadOnlyList<string> candidatePaths,
        string? selectedPath,
        string source,
        bool resourceExists,
        bool bytesRead,
        string decodeStatus,
        string backendStatus,
        AudioTriggerTerminalState terminalState,
        string detail,
        float? minDistance = null,
        float? maxDistance = null,
        float? distance = null,
        bool? inRange = null)
    {
        return new AudioTriggerDiagnostic(
            emitter.TriggerKind,
            tileX,
            tileY,
            chunkX,
            chunkY,
            emitter.SoundPointId,
            soundEntryId > 0 ? (uint)soundEntryId : emitter.SoundNameId,
            emitter.RawPosition,
            emitter.Position,
            emitter.CoordinateProfile,
            minDistance ?? emitter.MinDistance,
            maxDistance ?? emitter.MaxDistance,
            emitter.CutoffDistance,
            distance ?? 0f,
            inRange ?? false,
            soundEntryId > 0,
            candidatePaths,
            selectedPath,
            source,
            resourceExists,
            bytesRead,
            decodeStatus,
            backendStatus,
            terminalState,
            detail,
            emitter.McnkFlags,
            emitter.LiquidFamily,
            emitter.SoundWaterSubtype);
    }

    private string DescribeResourceSource(string virtualPath)
    {
        if (_dataSource.TryResolveWritablePath(virtualPath, out string? loosePath) && loosePath is not null)
            return $"Loose file: {loosePath}";

        if (_dataSource is MpqDataSource mpqDataSource &&
            mpqDataSource.TryGetFileSource(virtualPath, out string sourcePath))
        {
            return $"Archive: {sourcePath}";
        }

        return $"{_dataSource.Name} (archive source unknown)";
    }

    public void StopAll()
    {
        foreach (EmitterKey key in _active.Keys.ToArray())
            StopEmitter(key);
        StopAreaMusic();
        StopPreview();
        _heardInRange.Clear();
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        StopAll();
        if (_al is not null)
        {
            foreach (uint buffer in _buffers.Values)
            {
                try { _al.DeleteBuffer(buffer); } catch { }
            }
        }

        _buffers.Clear();
        StopPreview();
        try { _al?.Dispose(); } catch { }
        _al = null;
        try { _context?.Dispose(); } catch { }
        _context = null;
        lock (_tileEmittersLock)
        {
            _tileEmitters.Clear();
            _residentEmitterSnapshot = Array.Empty<TerrainSoundEmitter>();
        }
        _disposed = true;
    }

    private void TryStartEmitter(
        EmitterKey key,
        TerrainSoundEmitter emitter,
        AlphaSoundEntry soundEntry,
        float minDistance,
        float maxDistance,
        float gain)
    {
        if (_al is null)
            return;

        string? virtualPath = soundEntry.EnumerateVirtualPaths()
            .FirstOrDefault(path => _dataSource.FileExists(path));
        if (virtualPath is null)
        {
            LogOnce($"missing:{soundEntry.Id}", $"[Audio] SoundEntries {soundEntry.Id} has no resolvable client file.");
            return;
        }

        try
        {
            uint buffer = GetOrCreateBuffer(virtualPath);
            if (buffer == 0)
                return;

            uint source = _al.GenSource();
            _al.SetSourceProperty(source, SourceInteger.Buffer, buffer);
            _al.SetSourceProperty(source, SourceVector3.Position, emitter.Position);
            _al.SetSourceProperty(source, SourceFloat.Gain, gain * EffectiveMasterGain * EmitterGain);
            _al.SetSourceProperty(source, SourceFloat.ReferenceDistance, MathF.Max(1f, minDistance));
            _al.SetSourceProperty(source, SourceFloat.MaxDistance, MathF.Max(1f, maxDistance));
            _al.SetSourceProperty(source, SourceBoolean.Looping, true);
            _al.SourcePlay(source);
            _active[key] = new ActiveEmitter(source, virtualPath, gain);
        }
        catch (Exception ex)
        {
            string message = $"[Audio] Failed to start SoundEntries {soundEntry.Id} ({virtualPath}): {ex.Message}";
            LastDiagnostic = message;
            LogOnce($"start:{soundEntry.Id}", message);
        }
    }

    private void UpdateAreaMusic(
        int areaId,
        float gameTime,
        int continentId,
        AreaLookupResult? areaLookup)
    {
        if (_soundEntries is null || _areaAudioCatalog is null)
        {
            AreaMusicStatus = "Area music metadata is unavailable for the active build.";
            return;
        }

        bool packedAreaNumber = _areaIdentityLayout == AreaIdentityLayout.PackedAreaNumber;
        int areaKey = packedAreaNumber
            ? areaLookup is { Source: AreaContextSource.PackedAreaNumber, AreaNumber: int resolvedAreaNumber }
                ? resolvedAreaNumber
                : areaId
            : areaLookup?.CanonicalAreaId ?? areaId;
        AlphaAreaAudioBinding? binding = _areaAudioCatalog.TryResolveWithParents(
            areaKey,
            continentId >= 0 ? continentId : null);
        if (binding is null)
        {
            StopAreaMusic();
            string areaIdentity = packedAreaNumber
                ? areaKey != 0
                    ? $"AreaNumber 0x{AreaNumberParts.FromRaw(areaKey).Raw:X8} (zone={AreaNumberParts.FromRaw(areaKey).Zone}, subzone={AreaNumberParts.FromRaw(areaKey).Subzone})"
                    : "No terrain/WMO area"
                : areaKey != 0 ? $"AreaID {areaKey}" : "No terrain/WMO area";
            AreaMusicStatus = areaKey != 0
                ? $"{areaIdentity} has no DBC music assignment (continent={continentId})."
                : "No terrain/WMO area is active for music resolution.";
            return;
        }

        string areaLabel = areaLookup?.PrimaryText is { Length: > 0 } primaryText
            ? areaLookup.ZoneText is { Length: > 0 } zoneText
                && !string.Equals(zoneText, primaryText, StringComparison.Ordinal)
                ? $"{primaryText} [{zoneText}]"
                : primaryText
            : $"Area {binding.Area.Id}";
        if (packedAreaNumber)
        {
            AreaNumberParts areaNumberParts = AreaNumberParts.FromRaw(areaKey);
            areaLabel += $" (AreaNumber=0x{areaNumberParts.Raw:X8}, zone={areaNumberParts.Zone}, subzone={areaNumberParts.Subzone})";
        }
        else
        {
            areaLabel += $" (AreaID={areaKey})";
        }
        bool night = gameTime < 0.25f || gameTime >= 0.75f;
        int soundEntryId = binding.Area.ZoneMusicId;
        if (!WorldAudioPlaybackPolicy.AutomaticZoneMusicPlaybackEnabled)
        {
            StopAreaMusic();
            AreaMusicStatus = soundEntryId > 0
                ? $"{areaLabel} selects ZoneMusic {soundEntryId}; automatic ZoneMusic playback is muted."
                : $"{areaLabel} has area music metadata; automatic ZoneMusic playback is muted.";
            return;
        }

        if (soundEntryId <= 0)
        {
            StopAreaMusic();
            string midi = night
                ? binding.MidiAmbience?.NightSequence ?? binding.MidiAmbience?.DaySequence ?? string.Empty
                : binding.MidiAmbience?.DaySequence ?? string.Empty;
            string dls = binding.MidiAmbience?.DlsFile ?? string.Empty;
            AreaMusicStatus = string.IsNullOrWhiteSpace(midi)
                ? $"{areaLabel} has no ZoneMusic or MIDI sequence."
                : $"{areaLabel} selects MIDI '{midi}'" +
                  (string.IsNullOrWhiteSpace(dls) ? ". MIDI+DLS playback is unavailable." : $" with DLS '{dls}'. MIDI+DLS playback is unavailable.");
            return;
        }

        if (_al is null)
        {
            bool soundEntryResolved = _soundEntries.TryResolve((uint)soundEntryId, out AlphaSoundEntry? unavailableBackendEntry);
            string? backendPath = soundEntryResolved
                ? unavailableBackendEntry!.EnumerateVirtualPaths().FirstOrDefault(path => _dataSource.FileExists(path))
                : null;
            AreaMusicStatus = soundEntryResolved
                ? backendPath is null
                    ? $"{areaLabel} selects ZoneMusic {soundEntryId}, but no DBC-declared file is present; {Status}"
                    : $"{areaLabel} selects ZoneMusic {soundEntryId}: {backendPath}; {Status}"
                : $"{areaLabel} selects ZoneMusic {soundEntryId}, missing from SoundEntries; {Status}";
            return;
        }

        if (_areaMusicSoundEntryId == soundEntryId && _areaMusicNight == night && _areaMusicSource is not null)
        {
            AreaMusicStatus = $"Playing DBC ZoneMusic {soundEntryId} for {areaLabel}.";
            return;
        }

        if (!_soundEntries.TryResolve((uint)soundEntryId, out AlphaSoundEntry? soundEntry))
        {
            StopAreaMusic();
            AreaMusicStatus = $"{areaLabel} selects ZoneMusic {soundEntryId}, missing from SoundEntries.";
            return;
        }

        string? virtualPath = soundEntry.EnumerateVirtualPaths()
            .FirstOrDefault(path => _dataSource.FileExists(path));
        if (virtualPath is null)
        {
            StopAreaMusic();
            AreaMusicStatus = $"{areaLabel} selects ZoneMusic {soundEntryId}, but its DBC file is not present.";
            LogOnce($"area-music-missing:{soundEntryId}", $"[Audio] {AreaMusicStatus}");
            return;
        }

        try
        {
            StopAreaMusic();
            uint buffer = GetOrCreateBuffer(virtualPath);
            if (buffer == 0)
            {
                AreaMusicStatus = $"ZoneMusic {soundEntryId} could not be decoded: {virtualPath}.";
                return;
            }

            uint source = _al.GenSource();
            _al.SetSourceProperty(source, SourceInteger.Buffer, buffer);
            _al.SetSourceProperty(source, SourceVector3.Position, _listenerPosition);
            _al.SetSourceProperty(source, SourceFloat.Gain, soundEntry.Volume * EffectiveMasterGain);
            _al.SetSourceProperty(source, SourceBoolean.Looping, true);
            _al.SourcePlay(source);
            _areaMusicSource = source;
            _areaMusicSoundEntryId = soundEntryId;
            _areaMusicNight = night;
            AreaMusicStatus = $"Playing DBC ZoneMusic {soundEntryId} for {areaLabel}: {virtualPath}.";
        }
        catch (Exception ex)
        {
            StopAreaMusic();
            AreaMusicStatus = $"ZoneMusic {soundEntryId} failed to start: {ex.Message}";
            LogOnce($"area-music-start:{soundEntryId}", $"[Audio] {AreaMusicStatus}");
        }
    }

    private void StopAreaMusic()
    {
        if (_areaMusicSource is uint source && _al is not null)
        {
            try
            {
                _al.SourceStop(source);
                _al.DeleteSource(source);
            }
            catch { }
        }

        _areaMusicSource = null;
        _areaMusicSoundEntryId = 0;
        _areaMusicNight = false;
    }

    private uint GetOrCreateBuffer(string virtualPath)
    {
        if (_al is null)
            return 0;
        if (_buffers.TryGetValue(virtualPath, out uint existing))
            return existing;

        byte[]? bytes = _dataSource.ReadFile(virtualPath);
        string reason = bytes is null ? "client file could not be read" : "decoder rejected the file";
        if (bytes is null || !ClientAudioDecoder.TryDecode(bytes, virtualPath, out PcmAudioData? audio, out reason) || audio is null)
        {
            LogOnce($"decode:{virtualPath}", $"[Audio] Skipping '{virtualPath}': {reason}");
            return 0;
        }

        BufferFormat format = (audio.Channels, audio.BitsPerSample) switch
        {
            (1, 8) => BufferFormat.Mono8,
            (1, 16) => BufferFormat.Mono16,
            (2, 8) => BufferFormat.Stereo8,
            (2, 16) => BufferFormat.Stereo16,
            _ => throw new InvalidDataException("Unsupported PCM WAV format.")
        };

        uint buffer = _al.GenBuffer();
        _al.BufferData(buffer, format, audio.PcmBytes, audio.SampleRate);
        _buffers.Add(virtualPath, buffer);
        return buffer;
    }

    private void StopEmitter(EmitterKey key)
    {
        if (_al is null || !_active.Remove(key, out ActiveEmitter? active))
            return;

        try
        {
            _al.SourceStop(active.Source);
            _al.DeleteSource(active.Source);
        }
        catch { }
    }

    private void DisableBackend(string reason)
    {
        if (_loggedNoBackend)
            return;
        _loggedNoBackend = true;
        Status = $"Audio backend disabled: {reason}";
        LastDiagnostic = Status;
        AL? failedAl = _al;
        _active.Clear();
        if (_areaMusicSource is uint areaMusicSource && failedAl is not null)
        {
            try
            {
                failedAl.SourceStop(areaMusicSource);
                failedAl.DeleteSource(areaMusicSource);
            }
            catch { }
        }

        _areaMusicSource = null;
        _areaMusicSoundEntryId = 0;
        if (_previewSource is uint previewSource && failedAl is not null)
        {
            try
            {
                failedAl.SourceStop(previewSource);
                failedAl.DeleteSource(previewSource);
            }
            catch { }
        }

        _al = null;
        _previewSource = null;
        _previewPath = null;
        _previewBaseGain = 1f;
        try { failedAl?.Dispose(); } catch { }
        try { _context?.Dispose(); } catch { }
        _context = null;
        ViewerLog.Info(ViewerLog.Category.General, $"[Audio] {Status}");
    }

    private void LogOnce(string key, string message)
    {
        if (_diagnosticKeys.Add(key))
        {
            LastDiagnostic = message;
            ViewerLog.Info(ViewerLog.Category.General, message);
        }
    }

    private uint ResolveResidentSoundEntryId(TerrainSoundEmitter emitter)
    {
        if (emitter.TriggerKind != AudioTriggerKind.McnkLiquid)
            return emitter.SoundNameId;

        return _soundWaterTypes is not null
            && emitter.LiquidFamily >= 0
            && _soundWaterTypes.TryResolve(emitter.LiquidFamily, emitter.SoundWaterSubtype, out SoundWaterTypeEntry? waterEntry)
            ? (uint)waterEntry.SoundId
            : 0u;
    }

    private void ApplyActiveGains()
    {
        if (_al is null)
            return;

        try
        {
            foreach (ActiveEmitter active in _active.Values)
                _al.SetSourceProperty(active.Source, SourceFloat.Gain, active.BaseGain * EffectiveMasterGain * EmitterGain);

            if (_previewSource is uint previewSource)
                _al.SetSourceProperty(previewSource, SourceFloat.Gain, _previewBaseGain * EffectiveMasterGain);

            if (_areaMusicSource is uint areaMusicSource && _soundEntries is not null &&
                _soundEntries.TryResolve((uint)_areaMusicSoundEntryId, out AlphaSoundEntry? areaMusicEntry))
            {
                _al.SetSourceProperty(areaMusicSource, SourceFloat.Gain, areaMusicEntry.Volume * EffectiveMasterGain);
            }
        }
        catch (Exception ex)
        {
            DisableBackend($"gain update failed: {ex.Message}");
        }
    }

    private float EffectiveMasterGain => IsMuted ? 0f : MasterGain;

    private static float FirstPositive(params float[] values)
        => values.FirstOrDefault(value => value > 0f);

    private static float Attenuation(float distance, float minDistance, float maxDistance)
    {
        if (distance <= minDistance)
            return 1f;
        float span = MathF.Max(0.001f, maxDistance - minDistance);
        return Math.Clamp(1f - ((distance - minDistance) / span), 0f, 1f);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(WorldAudioRuntime));
    }

    private readonly record struct EmitterKey(int TileX, int TileY, int Index);
    private sealed class ActiveEmitter(uint source, string virtualPath, float baseGain)
    {
        public uint Source { get; } = source;
        public string VirtualPath { get; } = virtualPath;
        public float BaseGain { get; set; } = baseGain;
    }
}
