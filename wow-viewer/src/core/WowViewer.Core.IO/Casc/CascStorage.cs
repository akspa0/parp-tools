using System.Diagnostics;
using TACTSharp;

namespace WowViewer.Core.IO.Casc;

/// <summary>One product entry from a local install's <c>.build.info</c>.</summary>
public sealed record CascProductInfo(string Product, string Version, string BuildConfig, string CdnConfig, string CdnPath);

/// <summary>Why a CASC read produced no bytes.</summary>
public enum CascReadStatus
{
    Ok,
    /// <summary>The build's root manifest has no entry for the FileDataID.</summary>
    NotPresent,
    /// <summary>The root lists the file, but its data is not in the local install (and CDN access is off).</summary>
    NotLocal,
    KeyUnavailable,
    Failed,
}

/// <summary>
/// Spec 238: read-only access to a local CASC installation through TACTSharp. Local mode never
/// touches the network (<see cref="Settings.TryCDN"/> is off); a file whose data is not on disk
/// reports <see cref="CascReadStatus.NotPresent"/>.
/// </summary>
public sealed class CascStorage : IDisposable
{
    /// <summary>Fast local reads are compared with the stock reader for this many files, then trusted.</summary>
    private const int ShadowVerifyReads = 64;

    private readonly BuildInstance _build;
    private readonly Lock _readLock = new();

    // Spec 275 CASC-1: cached-handle local reads. Null when the install has no readable Data/data.
    private LocalCascArchiveReader? _local;
    private volatile bool _fastEnabled;
    private int _verifyBudget = ShadowVerifyReads;
    private string? _fastDisabledReason;
    private long _fastReads, _fastTicks, _stockReads, _stockTicks, _shadowChecked, _shadowMismatches;

    private CascStorage(BuildInstance build, CascProductInfo product, string installDir)
    {
        _build = build;
        Product = product;
        InstallDir = installDir;
    }

    /// <summary>
    /// Receives a message when the fast local read path switches itself off (for example because a
    /// shadow comparison with the stock reader found different bytes). Set by the host application.
    /// </summary>
    public Action<string>? Log { get; set; }

    /// <summary>True when the cached-handle local reader was opened for this install.</summary>
    public bool HasFastLocalReader => _local is not null;

    /// <summary>
    /// When true (default whenever <see cref="HasFastLocalReader"/>), reads are served from the local
    /// archives through cached handles and fall back to the stock TACTSharp path on any miss. Set the
    /// environment variable <c>PARP_CASC_FAST_READS=0</c> to start with it off.
    /// </summary>
    public bool FastLocalReads
    {
        get => _fastEnabled;
        set => _fastEnabled = value && _local is not null;
    }

    /// <summary>Read counters for the fast and stock paths (cumulative since the storage was opened).</summary>
    public CascReadStats ReadStats => new(
        Interlocked.Read(ref _fastReads), TicksToMs(Interlocked.Read(ref _fastTicks)),
        Interlocked.Read(ref _stockReads), TicksToMs(Interlocked.Read(ref _stockTicks)),
        Interlocked.Read(ref _shadowChecked), Interlocked.Read(ref _shadowMismatches),
        _fastEnabled, _fastDisabledReason);

    private static double TicksToMs(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

    public void Dispose()
    {
        _fastEnabled = false;
        _local?.Dispose();
    }

    public CascProductInfo Product { get; }

    /// <summary>
    /// When true, reads go through one lock. Default off: measured 2026-09-16 with <c>inspect casc bench</c>
    /// on wow_classic_beta 1.60.1 (1,487 distinct files incl. CDN-cached textures, each read twice by 12
    /// threads without the lock): 0 SHA-256 mismatches vs sequential locked reads, ~4x throughput.
    /// </summary>
    public bool SerializeReads { get; set; }

    public string InstallDir { get; }

    /// <summary>Lists the products recorded in <c>&lt;installDir&gt;/.build.info</c>.</summary>
    public static IReadOnlyList<CascProductInfo> ListProducts(string installDir)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(installDir);

        string buildInfoPath = Path.Combine(installDir, ".build.info");
        if (!File.Exists(buildInfoPath))
            throw new FileNotFoundException("No .build.info found; not a CASC installation root.", buildInfoPath);

        var settings = new Settings { BaseDir = installDir, TryCDN = false };
        var buildInfo = new BuildInfo(buildInfoPath, settings, new CDN(settings));
        return buildInfo.Entries
            .Select(static e => new CascProductInfo(e.Product, e.Version, e.BuildConfig, e.CDNConfig, e.CDNPath))
            .ToArray();
    }

    /// <summary>
    /// Opens one product of a local install. <paramref name="cacheDir"/> receives decoded
    /// manifests TACTSharp writes while loading (encoding/root); it is a local read-side cache.
    /// <para>
    /// <paramref name="allowCdnFill"/> (Spec 238 hybrid mode, off by default): when the root lists a
    /// file whose data is not on disk, fetch it from Blizzard's CDN for the same build into
    /// <paramref name="cacheDir"/>. Measured need: wow_classic_beta 1.60.1 lists all 33 DAT v26
    /// tileset BLPs without local data.
    /// </para>
    /// </summary>
    public static CascStorage OpenLocal(string installDir, string product, string cacheDir, bool allowCdnFill = false)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(installDir);
        ArgumentException.ThrowIfNullOrWhiteSpace(product);
        ArgumentException.ThrowIfNullOrWhiteSpace(cacheDir);

        CascProductInfo info = ListProducts(installDir).FirstOrDefault(p => string.Equals(p.Product, product, StringComparison.OrdinalIgnoreCase))
            ?? throw new InvalidOperationException($"Product '{product}' is not listed in {Path.Combine(installDir, ".build.info")}.");

        Directory.CreateDirectory(cacheDir);
        var build = new BuildInstance();
        build.Settings.BaseDir = installDir;
        build.Settings.Product = info.Product;
        build.Settings.TryCDN = false; // manifests come from the install when present
        build.Settings.CacheDir = cacheDir;
        build.cdn.ProductDirectory = info.CdnPath;

        bool manifestsFromCdn = false;
        try
        {
            build.LoadConfigs(info.BuildConfig, info.CdnConfig);
            build.Load();
        }
        catch (FileNotFoundException)
        {
            // Measured 2026-09-16: the launcher rewrote .build.info with a new CDN config
            // (272d201d… → 504e831a…) for the same build before putting that config on disk.
            // Fetch only the missing manifests for this exact build from the CDN (cached in cacheDir).
            build = new BuildInstance();
            build.Settings.BaseDir = installDir;
            build.Settings.Product = info.Product;
            build.Settings.TryCDN = true;
            build.Settings.CacheDir = cacheDir;
            build.cdn.ProductDirectory = info.CdnPath;
            build.LoadConfigs(info.BuildConfig, info.CdnConfig);
            build.Load();
            manifestsFromCdn = true;
        }

        build.Settings.TryCDN = allowCdnFill;
        var storage = new CascStorage(build, info, installDir) { AllowsCdnFill = allowCdnFill, ManifestsFetchedFromCdn = manifestsFromCdn };
        storage._local = LocalCascArchiveReader.TryOpen(installDir);
        storage._fastEnabled = storage._local is not null
            && !string.Equals(Environment.GetEnvironmentVariable("PARP_CASC_FAST_READS"), "0", StringComparison.Ordinal);
        return storage;
    }

    /// <summary>True when reads may fetch missing local data from the CDN for this build.</summary>
    public bool AllowsCdnFill { get; private init; }

    /// <summary>True when the install lacked this build's manifests locally and they were fetched from the CDN.</summary>
    public bool ManifestsFetchedFromCdn { get; private init; }

    /// <summary>Adds a TACT decryption key (hex key name → key bytes).</summary>
    public static void AddKey(ulong keyName, byte[] key) => KeyService.SetKey(keyName, key);

    public bool FileExists(uint fileDataId) => _build.Root?.FileExists(fileDataId) == true;

    public IReadOnlyCollection<uint> GetAvailableFileDataIds() => _build.Root?.GetAvailableFDIDs() ?? [];

    public CascReadStatus TryReadFile(uint fileDataId, out byte[]? data)
    {
        data = null;
        if (!FileExists(fileDataId))
            return CascReadStatus.NotPresent;

        if (_fastEnabled && TryReadFastLocal(fileDataId, out data))
            return CascReadStatus.Ok;

        long started = Stopwatch.GetTimestamp();
        try
        {
            if (SerializeReads)
            {
                lock (_readLock)
                    data = _build.OpenFileByFDID(fileDataId);
            }
            else
            {
                data = _build.OpenFileByFDID(fileDataId);
            }

            Interlocked.Increment(ref _stockReads);
            Interlocked.Add(ref _stockTicks, Stopwatch.GetTimestamp() - started);
            return CascReadStatus.Ok;
        }
        catch (FileNotFoundException)
        {
            return CascReadStatus.NotLocal;
        }
        catch (Exception ex) when (ex.Message.Contains("key", StringComparison.OrdinalIgnoreCase))
        {
            return CascReadStatus.KeyUnavailable;
        }
        catch (Exception)
        {
            return CascReadStatus.Failed;
        }
    }

    /// <summary>
    /// Same lookup chain as <see cref="BuildInstance.OpenFileByFDID"/> (root -&gt; content key -&gt; encoding
    /// key -&gt; BLTE decode) with only the byte fetch replaced. Returns false on any miss or error so the
    /// caller falls through to the stock path, which also owns CDN fill and the failure mapping.
    /// </summary>
    private bool TryReadFastLocal(uint fileDataId, out byte[]? data)
    {
        data = null;
        LocalCascArchiveReader? local = _local;
        if (local is null)
            return false;

        long started = Stopwatch.GetTimestamp();
        byte[] decoded;
        try
        {
            List<RootInstance.RootEntry>? entries = _build.Root?.GetEntriesByFDID(fileDataId);
            EncodingInstance? encoding = _build.Encoding;
            if (entries is null || entries.Count == 0 || encoding is null)
                return false;

            RootInstance.RootEntry entry = entries[0];
            EncodingInstance.EncodingResult result = encoding.FindContentKey(entry.md5.AsSpan());
            if (result.Length == 0)
                return false;

            if (!local.TryReadEncoded(result[0], out byte[]? encoded) || encoded is null)
                return false;

            decoded = BLTE.Decode(encoded, result.DecodedFileSize);
        }
        catch (Exception)
        {
            return false;
        }

        Interlocked.Increment(ref _fastReads);
        Interlocked.Add(ref _fastTicks, Stopwatch.GetTimestamp() - started);

        // The first reads are compared byte-for-byte with the stock reader. A single difference turns
        // the fast path off for this storage and returns the stock bytes, so a defect here can cost
        // speed but not correctness.
        if (Interlocked.Decrement(ref _verifyBudget) >= 0)
        {
            byte[]? stock = null;
            try { stock = _build.OpenFileByFDID(fileDataId); }
            catch (Exception) { /* cannot verify this file; do not count it */ }

            if (stock is not null)
            {
                Interlocked.Increment(ref _shadowChecked);
                if (!decoded.AsSpan().SequenceEqual(stock))
                {
                    Interlocked.Increment(ref _shadowMismatches);
                    _fastDisabledReason = $"fast read of FileDataID {fileDataId} differed from the stock reader ({decoded.Length} vs {stock.Length} bytes)";
                    _fastEnabled = false;
                    Log?.Invoke($"[CASC] fast local reads disabled for {Product.Product} {Product.Version}: {_fastDisabledReason}");
                    data = stock;
                    return true;
                }
            }
        }

        data = decoded;
        return true;
    }
}

/// <summary>Cumulative CASC read counters (Spec 275 CASC-1).</summary>
/// <param name="FastReads">Files served by the cached-handle local reader.</param>
/// <param name="FastTotalMs">Total time spent in those reads (lookup, I/O, BLTE decode).</param>
/// <param name="StockReads">Files served by the stock TACTSharp path (fallbacks, CDN fill, fast path off).</param>
/// <param name="StockTotalMs">Total time spent in those reads.</param>
/// <param name="ShadowChecked">Fast reads compared byte-for-byte with the stock reader.</param>
/// <param name="ShadowMismatches">Comparisons that differed (the fast path disables itself on the first).</param>
/// <param name="FastEnabled">Whether the fast path is currently active.</param>
/// <param name="DisabledReason">Why the fast path switched itself off, when it did.</param>
public readonly record struct CascReadStats(
    long FastReads,
    double FastTotalMs,
    long StockReads,
    double StockTotalMs,
    long ShadowChecked,
    long ShadowMismatches,
    bool FastEnabled,
    string? DisabledReason)
{
    public double FastAverageMs => FastReads == 0 ? 0 : FastTotalMs / FastReads;

    public double StockAverageMs => StockReads == 0 ? 0 : StockTotalMs / StockReads;

    /// <summary>Sums counters across several opened products.</summary>
    public static CascReadStats Combine(IEnumerable<CascReadStats> parts)
    {
        long fast = 0, stock = 0, checkedCount = 0, mismatches = 0;
        double fastMs = 0, stockMs = 0;
        bool enabled = false;
        string? reason = null;
        foreach (CascReadStats part in parts)
        {
            fast += part.FastReads;
            fastMs += part.FastTotalMs;
            stock += part.StockReads;
            stockMs += part.StockTotalMs;
            checkedCount += part.ShadowChecked;
            mismatches += part.ShadowMismatches;
            enabled |= part.FastEnabled;
            reason ??= part.DisabledReason;
        }

        return new CascReadStats(fast, fastMs, stock, stockMs, checkedCount, mismatches, enabled, reason);
    }
}
