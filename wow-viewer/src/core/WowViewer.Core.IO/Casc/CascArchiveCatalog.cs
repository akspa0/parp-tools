using System.Collections.Concurrent;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.IO.Casc;

/// <summary>
/// Spec 238: exposes a local CASC install through <see cref="IArchiveCatalog"/>, so path-based tools
/// written against MPQ clients (harvest, synthetic minimaps) read modern clients unchanged.
/// Virtual paths resolve as <c>fdid:&lt;id&gt;</c>, then registered aliases (WDT <c>MAID</c> tile
/// files), then the community listfile. Products are tried newest version first; the first that
/// returns bytes wins.
/// </summary>
public sealed class CascArchiveCatalog : IArchiveCatalog
{
    private readonly IReadOnlyList<CascStorage> _storages;
    private readonly CommunityListfile _listfile;
    private readonly Func<uint, string?> _resolver;
    private readonly ConcurrentDictionary<string, uint> _aliases = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, byte> _maidRegisteredMaps = new(StringComparer.OrdinalIgnoreCase);
    private readonly Lazy<IReadOnlyList<string>> _knownFiles;

    public CascArchiveCatalog(IReadOnlyList<CascStorage> storages, CommunityListfile listfile)
    {
        if (storages.Count == 0)
            throw new ArgumentException("At least one CASC product is required.", nameof(storages));

        _storages = storages;
        _listfile = listfile;
        _resolver = listfile.GetPath;
        FileDataIdPaths.Resolver = _resolver;
        _knownFiles = new Lazy<IReadOnlyList<string>>(() => _listfile.Entries
            .Where(entry => FileExists(entry.Key))
            .Select(static entry => entry.Value.Replace('/', '\\'))
            .ToArray());
    }

    public IReadOnlyList<CascStorage> Storages => _storages;

    /// <summary>The newest opened product's version, e.g. <c>1.60.1.69876</c>.</summary>
    public string BuildVersion => _storages[0].Product.Version;

    public string Name => "CASC: " + string.Join(" + ", _storages.Select(static s => $"{s.Product.Product} {s.Product.Version}"));

    /// <summary>True when <paramref name="directory"/> is a CASC install root (has <c>.build.info</c>).</summary>
    public static bool IsCascInstall(string directory) =>
        File.Exists(Path.Combine(directory, ".build.info"));

    /// <summary>
    /// Opens the requested products (all listed products when <paramref name="products"/> is empty),
    /// newest version first. Products that fail to open are reported through <paramref name="log"/> and skipped.
    /// </summary>
    public static CascArchiveCatalog Open(
        string installDir,
        IReadOnlyCollection<string> products,
        IEnumerable<string> listfilePaths,
        string cacheDir,
        bool allowCdnFill,
        Action<string>? log = null)
    {
        IEnumerable<CascProductInfo> listed = CascStorage.ListProducts(installDir);
        if (products.Count > 0)
            listed = listed.Where(p => products.Contains(p.Product, StringComparer.OrdinalIgnoreCase));

        var storages = new List<CascStorage>();
        foreach (CascProductInfo product in listed.OrderByDescending(static p => Version.TryParse(p.Version, out Version? v) ? v : new Version()))
        {
            try
            {
                CascStorage storage = CascStorage.OpenLocal(installDir, product.Product, cacheDir, allowCdnFill);
                if (storage.ManifestsFetchedFromCdn)
                    log?.Invoke($"CASC {product.Product} {product.Version}: local manifests missing; fetched them from the CDN for this build.");
                storages.Add(storage);
            }
            catch (Exception ex)
            {
                log?.Invoke($"CASC product {product.Product} {product.Version} failed to open: {ex.Message}");
            }
        }

        if (storages.Count == 0)
            throw new InvalidOperationException($"No CASC product could be opened from {installDir}.");

        return new CascArchiveCatalog(storages, CommunityListfile.Load(listfilePaths));
    }

    public bool FileExists(string virtualPath) =>
        TryGetFileDataId(virtualPath, out uint fileDataId) && FileExists(fileDataId);

    public bool FileExists(uint fileDataId) => _storages.Any(s => s.FileExists(fileDataId));

    public byte[]? ReadFile(string virtualPath)
    {
        if (!TryGetFileDataId(virtualPath, out uint fileDataId))
            return null;

        byte[]? data = ReadFile(fileDataId);

        // FileDataID-era WDTs name their tile files only by MAID ids; register them the first time a
        // map's WDT is read so the path-based ADT and minimap lookups that follow resolve.
        if (data is not null
            && virtualPath.EndsWith(".wdt", StringComparison.OrdinalIgnoreCase)
            && Path.GetFileNameWithoutExtension(virtualPath.Replace('\\', '/')) is { Length: > 0 } mapName
            && _maidRegisteredMaps.TryAdd(mapName, 0))
        {
            RegisterMaidAliases(mapName, data);
        }

        return data;
    }

    public byte[]? ReadFile(uint fileDataId)
    {
        foreach (CascStorage storage in _storages)
        {
            if (storage.TryReadFile(fileDataId, out byte[]? data) == CascReadStatus.Ok && data is not null)
                return data;
        }

        return null;
    }

    public bool TryGetFileDataId(string virtualPath, out uint fileDataId) =>
        FileDataIdPaths.TryParse(virtualPath, out fileDataId)
        || _aliases.TryGetValue(NormalizeAlias(virtualPath), out fileDataId)
        || _listfile.TryGetFileDataId(virtualPath, out fileDataId);

    /// <summary>Maps a virtual path the listfile does not name (e.g. an unnamed tile file) to its FileDataID.</summary>
    public void RegisterFileDataIdAlias(string virtualPath, uint fileDataId)
    {
        if (fileDataId != 0)
            _aliases[NormalizeAlias(virtualPath)] = fileDataId;
    }

    /// <summary>
    /// Registers the WDT <c>MAID</c> tile files of <paramref name="mapName"/> as aliases under
    /// <c>World\Maps\&lt;map&gt;\&lt;map&gt;_X_Y*.adt</c> and <c>World\Minimaps\&lt;map&gt;\mapXX_YY.blp</c>.
    /// Returns the number of aliases registered (0 when the WDT has no MAID).
    /// </summary>
    public int RegisterMaidAliases(string mapName, byte[] wdtBytes)
    {
        if (!TryFindChunk(wdtBytes, "DIAM", out int payload, out int size))
            return 0;

        // MAID slot layout (8 uint32 per tile): root, obj0, obj1, tex0, lod, mapTexture, mapTextureN, minimap.
        string[] suffixes = [".adt", "_obj0.adt", "_obj1.adt", "_tex0.adt", "_lod.adt"];
        const int slotSize = 32;
        int registered = 0;
        for (int slot = 0; slot < 4096 && (slot + 1) * slotSize <= size; slot++)
        {
            int row = slot / 64;
            int column = slot % 64;
            int slotOffset = payload + slot * slotSize;
            string basePath = $"World\\Maps\\{mapName}\\{mapName}_{column}_{row}";
            for (int field = 0; field < suffixes.Length; field++)
            {
                uint fileDataId = BitConverter.ToUInt32(wdtBytes, slotOffset + field * 4);
                if (fileDataId == 0)
                    continue;

                RegisterFileDataIdAlias(basePath + suffixes[field], fileDataId);
                registered++;
            }

            uint minimapFileDataId = BitConverter.ToUInt32(wdtBytes, slotOffset + 7 * 4);
            if (minimapFileDataId != 0)
            {
                RegisterFileDataIdAlias($"World\\Minimaps\\{mapName}\\map{column:00}_{row:00}.blp", minimapFileDataId);
                registered++;
            }
        }

        return registered;
    }

    public void LoadArchives(IEnumerable<string> searchPaths)
    {
        // Storages are opened at construction; CASC has no per-archive search paths.
    }

    public void LoadListfile(string path)
    {
        // CASC resolves names through the community listfile given at construction. Plain path
        // listfiles (MPQ era) carry no FileDataIDs and cannot add anything here.
    }

    public void LoadListfileEntries(IEnumerable<string> entries)
    {
    }

    public IReadOnlyList<string> ExtractInternalListfiles() => [];

    /// <summary>Listfile paths whose FileDataID is present in at least one opened product's root.</summary>
    public IReadOnlyList<string> GetAllKnownFiles() => _knownFiles.Value;

    public void Dispose()
    {
        if (FileDataIdPaths.Resolver == _resolver)
            FileDataIdPaths.Resolver = null;
    }

    private static string NormalizeAlias(string path) => path.Replace('/', '\\').TrimStart('\\').ToLowerInvariant();

    private static bool TryFindChunk(byte[] bytes, string onDiskId, out int payloadOffset, out int size)
    {
        for (int position = 0; position + 8 <= bytes.Length;)
        {
            int chunkSize = BitConverter.ToInt32(bytes, position + 4);
            if (chunkSize < 0 || position + 8 + chunkSize > bytes.Length)
                break;

            if (bytes[position] == onDiskId[0] && bytes[position + 1] == onDiskId[1]
                && bytes[position + 2] == onDiskId[2] && bytes[position + 3] == onDiskId[3])
            {
                payloadOffset = position + 8;
                size = chunkSize;
                return true;
            }

            position += 8 + chunkSize;
        }

        payloadOffset = 0;
        size = 0;
        return false;
    }
}
