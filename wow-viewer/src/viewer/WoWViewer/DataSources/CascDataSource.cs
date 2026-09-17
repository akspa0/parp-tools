using System.Collections.Concurrent;
using System.Text;
using System.Threading.Channels;
using WoWViewer.Logging;
using WowViewer.Core.IO.Casc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.M2;

namespace WoWViewer.DataSources;

/// <summary>
/// Spec 238: data source over one or more products of a local CASC install. Virtual paths resolve
/// through the community listfile to FileDataIDs; each product is tried in order and the first
/// that returns bytes wins. This matters when a product lists a file in its root but does not have
/// the data on disk (measured: wow_classic_beta 1.60.1 lists the DAT v26 tileset BLPs without local
/// data, while wow_classic_era 1.15.9 has most of them).
/// <para>
/// Loading speed (measured 2026-09-16 on the beta development map): first-time CDN downloads of the
/// 460 WMO material textures took 87.7 s one at a time versus 1.0 s warm. <see cref="PrefetchFile"/>
/// therefore reads in the background on a worker pool, fans out to the files a model needs (M2 SFID
/// skin + TXID textures; WMO GFID groups, MOMT texture FileDataIDs, MODI doodads) and keeps the bytes
/// in a bounded cache the render-thread loaders then hit.
/// </para>
/// </summary>
public sealed class CascDataSource : IDataSource
{
    private const long MaxCachedBytes = 768L * 1024 * 1024;

    private readonly IReadOnlyList<CascStorage> _storages;
    private readonly CommunityListfile _listfile;
    private readonly Func<uint, string?> _resolver;
    private readonly ConcurrentDictionary<string, uint> _aliases = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, IReadOnlyList<string>> _fileListsByExtension = new(StringComparer.OrdinalIgnoreCase);
    private readonly Lazy<List<string>> _fileList;

    private readonly ConcurrentDictionary<uint, byte[]> _bytes = new();
    private readonly ConcurrentQueue<uint> _cacheOrder = new();
    private long _cachedBytes;
    private readonly ConcurrentDictionary<uint, byte> _prefetchSeen = new();
    private readonly Channel<uint> _prefetchQueue = Channel.CreateUnbounded<uint>(new UnboundedChannelOptions { SingleReader = false });
    private readonly CancellationTokenSource _prefetchCancellation = new();
    private long _prefetchCompleted;
    private long _cacheHits;

    public CascDataSource(IReadOnlyList<CascStorage> storages, CommunityListfile listfile)
    {
        if (storages.Count == 0)
            throw new ArgumentException("At least one CASC product is required.", nameof(storages));

        _storages = storages;
        _listfile = listfile;
        _resolver = listfile.GetPath;
        FileDataIdPaths.Resolver = _resolver;
        _fileList = new Lazy<List<string>>(() => _listfile.Entries
            .Where(entry => FileExists(entry.Key))
            .Select(static entry => entry.Value.Replace('/', '\\'))
            .ToList());

        int workers = Math.Clamp(Environment.ProcessorCount / 2, 4, 12);
        for (int i = 0; i < workers; i++)
            _ = Task.Run(() => PrefetchWorkerAsync(_prefetchCancellation.Token));
    }

    public string Name => "CASC: " + string.Join(" + ", _storages.Select(static s => $"{s.Product.Product} {s.Product.Version}"));

    public bool IsLoaded => true;

    public IReadOnlyList<CascStorage> Storages => _storages;

    /// <summary>Background prefetch reads completed, cache hits served to loaders, and bytes held.</summary>
    public (long PrefetchCompleted, long CacheHits, long CachedBytes) Stats => (Interlocked.Read(ref _prefetchCompleted), Interlocked.Read(ref _cacheHits), Interlocked.Read(ref _cachedBytes));

    public bool FileExists(string virtualPath) =>
        TryGetFileDataId(virtualPath, out uint fileDataId) && FileExists(fileDataId);

    public bool FileExists(uint fileDataId) => _storages.Any(s => s.FileExists(fileDataId));

    public byte[]? ReadFile(string virtualPath) =>
        TryGetFileDataId(virtualPath, out uint fileDataId) ? ReadFile(fileDataId) : null;

    public byte[]? ReadFile(uint fileDataId)
    {
        if (_bytes.TryGetValue(fileDataId, out byte[]? cached))
        {
            Interlocked.Increment(ref _cacheHits);
            return cached;
        }

        return ReadFromStorages(fileDataId);
    }

    public void PrefetchFile(string virtualPath)
    {
        if (TryGetFileDataId(virtualPath, out uint fileDataId))
            EnqueuePrefetch(fileDataId);
    }

    public void RegisterFileDataIdAlias(string virtualPath, uint fileDataId)
    {
        if (fileDataId != 0)
            _aliases[NormalizeAlias(virtualPath)] = fileDataId;
    }

    public bool TryResolveWritablePath(string virtualPath, out string? fullPath)
    {
        fullPath = null;
        return false;
    }

    public IReadOnlyList<string> GetFileList(string? extensionFilter = null)
    {
        if (extensionFilter is null)
            return _fileList.Value;

        // Measured: one filter pass over the ~1.4M present entries costs ~23 ms; callers invoke this per
        // model (skin lookup) and per missing texture, so cache each extension's list once.
        return _fileListsByExtension.GetOrAdd(extensionFilter, ext =>
            _fileList.Value.Where(f => f.EndsWith(ext, StringComparison.OrdinalIgnoreCase)).ToList());
    }

    public void Dispose()
    {
        _prefetchCancellation.Cancel();
        _prefetchQueue.Writer.TryComplete();
        if (FileDataIdPaths.Resolver == _resolver)
            FileDataIdPaths.Resolver = null;
    }

    /// <summary>Accepts <c>fdid:&lt;id&gt;</c> paths, registered aliases, then listfile paths.</summary>
    private bool TryGetFileDataId(string virtualPath, out uint fileDataId) =>
        FileDataIdPaths.TryParse(virtualPath, out fileDataId)
        || _aliases.TryGetValue(NormalizeAlias(virtualPath), out fileDataId)
        || _listfile.TryGetFileDataId(virtualPath, out fileDataId);

    private static string NormalizeAlias(string path) => path.Replace('/', '\\').TrimStart('\\').ToLowerInvariant();

    private byte[]? ReadFromStorages(uint fileDataId)
    {
        foreach (CascStorage storage in _storages)
        {
            if (storage.TryReadFile(fileDataId, out byte[]? data) == CascReadStatus.Ok && data is not null)
                return data;
        }

        return null;
    }

    private void EnqueuePrefetch(uint fileDataId)
    {
        if (fileDataId == 0 || _bytes.ContainsKey(fileDataId) || !_prefetchSeen.TryAdd(fileDataId, 0))
            return;

        _prefetchQueue.Writer.TryWrite(fileDataId);
    }

    private async Task PrefetchWorkerAsync(CancellationToken cancellationToken)
    {
        try
        {
            await foreach (uint fileDataId in _prefetchQueue.Reader.ReadAllAsync(cancellationToken))
            {
                try
                {
                    byte[]? data = ReadFromStorages(fileDataId);
                    if (data is null)
                        continue;

                    AddToCache(fileDataId, data);
                    EnqueueDependencies(data);
                    Interlocked.Increment(ref _prefetchCompleted);
                }
                catch (Exception ex)
                {
                    ViewerLog.Debug(ViewerLog.Category.MpqData, $"[CASC] prefetch {fileDataId} failed: {ex.Message}");
                }
            }
        }
        catch (OperationCanceledException)
        {
        }
    }

    private void AddToCache(uint fileDataId, byte[] data)
    {
        if (!_bytes.TryAdd(fileDataId, data))
            return;

        _cacheOrder.Enqueue(fileDataId);
        Interlocked.Add(ref _cachedBytes, data.Length);
        while (Interlocked.Read(ref _cachedBytes) > MaxCachedBytes && _cacheOrder.TryDequeue(out uint evict))
        {
            if (_bytes.TryRemove(evict, out byte[]? removed))
                Interlocked.Add(ref _cachedBytes, -removed.Length);
        }
    }

    /// <summary>Queues the files a just-read model will ask for next.</summary>
    private void EnqueueDependencies(byte[] data)
    {
        if (M2ChunkedFileIds.TryRead(data, out M2ChunkedFileIds ids))
        {
            if (ids.SkinFileDataIds.Length > 0)
                EnqueuePrefetch(ids.SkinFileDataIds[0]);
            foreach (uint texture in ids.TextureFileDataIds)
                EnqueuePrefetch(texture);
            return;
        }

        if (!IsWmoRoot(data))
            return;

        bool hasMotx = false;
        (int Offset, int Size)? momt = null;
        int position = 0;
        while (position + 8 <= data.Length)
        {
            string id = ChunkId(data, position);
            int size = BitConverter.ToInt32(data, position + 4);
            if (size < 0 || position + 8L + size > data.Length)
                break;

            int payload = position + 8;
            switch (id)
            {
                case "MOTX":
                    hasMotx = true;
                    break;
                case "MOMT":
                    momt = (payload, size);
                    break;
                case "GFID":
                case "MODI":
                    for (int i = 0; i + 4 <= size; i += 4)
                        EnqueuePrefetch(BitConverter.ToUInt32(data, payload + i));
                    break;
            }

            position += 8 + size;
        }

        // v17 MOMT (64 bytes): texture1 +0x0C, texture2 +0x18, texture3 +0x24 are FileDataIDs when MOTX is absent.
        if (!hasMotx && momt is { } materials)
        {
            for (int record = materials.Offset; record + 64 <= materials.Offset + materials.Size; record += 64)
            {
                EnqueuePrefetch(BitConverter.ToUInt32(data, record + 0x0C));
                EnqueuePrefetch(BitConverter.ToUInt32(data, record + 0x18));
                EnqueuePrefetch(BitConverter.ToUInt32(data, record + 0x24));
            }
        }
    }

    private static bool IsWmoRoot(byte[] data) =>
        data.Length >= 16 && ChunkId(data, 0) == "MVER" && ChunkId(data, 8 + BitConverter.ToInt32(data, 4)) == "MOHD";

    private static string ChunkId(byte[] data, int offset) =>
        offset + 4 <= data.Length ? new string(Encoding.ASCII.GetString(data, offset, 4).Reverse().ToArray()) : string.Empty;
}
