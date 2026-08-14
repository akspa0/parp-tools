using System.Numerics;
using Silk.NET.OpenAL;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using WowViewer.Core.Audio;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Maps;

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
    private readonly Dictionary<string, uint> _buffers = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<EmitterKey, ActiveEmitter> _active = [];
    private readonly HashSet<EmitterKey> _heardInRange = [];
    private readonly HashSet<string> _diagnosticKeys = new(StringComparer.OrdinalIgnoreCase);

    private AudioContext? _context;
    private AL? _al;
    private AlphaSoundEntriesCatalog? _soundEntries;
    private bool _disposed;
    private bool _loggedNoBackend;

    public WorldAudioRuntime(IDataSource dataSource)
    {
        _dataSource = dataSource ?? throw new ArgumentNullException(nameof(dataSource));
    }

    public string Status { get; private set; } = "Audio runtime not configured.";

    public int ResidentEmitterCount => _tileEmitters.Values.Sum(static emitters => emitters.Count);

    public int ActiveEmitterCount => _active.Count;

    public void Configure(DBCD.Providers.IDBCProvider dbcProvider, string definitionsDirectory, string buildVersion)
    {
        ThrowIfDisposed();
        StopAll();
        _soundEntries = null;

        try
        {
            _soundEntries = new AlphaSoundEntriesCatalogReader()
                .Load(dbcProvider, definitionsDirectory, buildVersion);
        }
        catch (Exception ex)
        {
            Status = $"SoundEntries unavailable for {buildVersion}: {ex.Message}";
            ViewerLog.Info(ViewerLog.Category.General, $"[Audio] {Status}");
            return;
        }

        if (!OpenAlNativeLibraryProbe.TryFind(out string? libraryName))
        {
            Status = "Audio backend unavailable: OpenAL native library was not found; audio playback is disabled.";
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
            Status = $"OpenAL ready ({libraryName}); SoundEntries={_soundEntries.Entries.Count}; WAV/OGG/MP3 playback enabled; MIDI+DLS pair renderer pending.";
            ViewerLog.Important(ViewerLog.Category.General, $"[Audio] {Status}");
        }
        catch (Exception ex)
        {
            try { context?.Dispose(); } catch { }
            _context = null;
            _al = null;
            Status = $"Audio backend unavailable: {ex.Message}";
            ViewerLog.Info(ViewerLog.Category.General, $"[Audio] {Status}");
        }
    }

    public void AddTile(int tileX, int tileY, IReadOnlyList<TerrainSoundEmitter> emitters)
    {
        ThrowIfDisposed();
        _tileEmitters[(tileX, tileY)] = emitters ?? Array.Empty<TerrainSoundEmitter>();
    }

    public void RemoveTile(int tileX, int tileY)
    {
        if (_disposed)
            return;

        _tileEmitters.Remove((tileX, tileY));
        foreach (EmitterKey key in _active.Keys.Where(key => key.TileX == tileX && key.TileY == tileY).ToArray())
            StopEmitter(key);
        _heardInRange.RemoveWhere(key => key.TileX == tileX && key.TileY == tileY);
    }

    public void Update(Vector3 listenerPosition, Vector3 listenerForward)
    {
        if (_disposed || _al is null || _soundEntries is null)
            return;

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

        HashSet<EmitterKey> inRange = [];
        foreach ((int tileX, int tileY, IReadOnlyList<TerrainSoundEmitter> emitters) in _tileEmitters.Select(static pair => (pair.Key.TileX, pair.Key.TileY, pair.Value)))
        {
            for (int index = 0; index < emitters.Count; index++)
            {
                TerrainSoundEmitter emitter = emitters[index];
                EmitterKey key = new(tileX, tileY, index);
                if (!_soundEntries.TryResolve(emitter.SoundNameId, out AlphaSoundEntry? soundEntry))
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
                    _al.SetSourceProperty(active.Source, SourceVector3.Position, emitter.Position);
                    _al.SetSourceProperty(active.Source, SourceFloat.Gain, gain);
                }
            }
        }

        foreach (EmitterKey key in _active.Keys.Where(key => !inRange.Contains(key)).ToArray())
            StopEmitter(key);

        _heardInRange.RemoveWhere(key => !inRange.Contains(key));
    }

    public void StopAll()
    {
        foreach (EmitterKey key in _active.Keys.ToArray())
            StopEmitter(key);
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
        try { _al?.Dispose(); } catch { }
        _al = null;
        try { _context?.Dispose(); } catch { }
        _context = null;
        _tileEmitters.Clear();
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
            _al.SetSourceProperty(source, SourceFloat.Gain, gain);
            _al.SetSourceProperty(source, SourceFloat.ReferenceDistance, MathF.Max(1f, minDistance));
            _al.SetSourceProperty(source, SourceFloat.MaxDistance, MathF.Max(1f, maxDistance));
            _al.SetSourceProperty(source, SourceBoolean.Looping, true);
            _al.SourcePlay(source);
            _active[key] = new ActiveEmitter(source, virtualPath);
        }
        catch (Exception ex)
        {
            LogOnce($"start:{soundEntry.Id}", $"[Audio] Failed to start SoundEntries {soundEntry.Id} ({virtualPath}): {ex.Message}");
        }
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
        ViewerLog.Info(ViewerLog.Category.General, $"[Audio] {Status}");
    }

    private void LogOnce(string key, string message)
    {
        if (_diagnosticKeys.Add(key))
            ViewerLog.Info(ViewerLog.Category.General, message);
    }

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
    private sealed record ActiveEmitter(uint Source, string VirtualPath);
}
