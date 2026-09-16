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
public sealed class CascStorage
{
    private readonly BuildInstance _build;
    private readonly Lock _readLock = new();

    private CascStorage(BuildInstance build, CascProductInfo product, string installDir)
    {
        _build = build;
        Product = product;
        InstallDir = installDir;
    }

    public CascProductInfo Product { get; }

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
        build.Settings.TryCDN = false; // manifests must come from the install itself
        build.Settings.CacheDir = cacheDir;
        build.cdn.ProductDirectory = info.CdnPath;

        build.LoadConfigs(info.BuildConfig, info.CdnConfig);
        build.Load();
        build.Settings.TryCDN = allowCdnFill;
        return new CascStorage(build, info, installDir) { AllowsCdnFill = allowCdnFill };
    }

    /// <summary>True when reads may fetch missing local data from the CDN for this build.</summary>
    public bool AllowsCdnFill { get; private init; }

    /// <summary>Adds a TACT decryption key (hex key name → key bytes).</summary>
    public static void AddKey(ulong keyName, byte[] key) => KeyService.SetKey(keyName, key);

    public bool FileExists(uint fileDataId) => _build.Root?.FileExists(fileDataId) == true;

    public IReadOnlyCollection<uint> GetAvailableFileDataIds() => _build.Root?.GetAvailableFDIDs() ?? [];

    public CascReadStatus TryReadFile(uint fileDataId, out byte[]? data)
    {
        data = null;
        if (!FileExists(fileDataId))
            return CascReadStatus.NotPresent;

        try
        {
            // TACTSharp's shared decode buffers are not documented as thread-safe; serialize reads.
            lock (_readLock)
                data = _build.OpenFileByFDID(fileDataId);
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
}
