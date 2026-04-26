using WowViewer.Core.IO.Files;

namespace WowViewer.App;

internal readonly record struct ViewerIoSourceKey(string ClientRoot, string BuildLabel, string LooseOverlayRoot)
{
    public static ViewerIoSourceKey Create(string? clientRoot, string? buildLabel, string? looseOverlayRoot)
    {
        string normalizedClientRoot = NormalizeRoot(clientRoot);
        string normalizedLooseOverlayRoot = NormalizeRoot(looseOverlayRoot);
        string normalizedBuildLabel = buildLabel?.Trim() ?? string.Empty;
        return new ViewerIoSourceKey(normalizedClientRoot, normalizedBuildLabel, normalizedLooseOverlayRoot);
    }

    public bool HasClientRoot => !string.IsNullOrWhiteSpace(ClientRoot) && Directory.Exists(ClientRoot);

    public string Signature => string.Join('|', ClientRoot, BuildLabel, LooseOverlayRoot);

    private static string NormalizeRoot(string? root)
    {
        string normalized = root?.Trim() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(normalized))
            return string.Empty;

        try
        {
            return Path.GetFullPath(normalized).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        }
        catch (Exception) when (root is not null)
        {
            return normalized.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        }
    }
}

internal sealed class ViewerIoCatalogLease
{
    internal ViewerIoCatalogLease(ViewerIoSourceKey sourceKey, IArchiveCatalog archiveCatalog, int bootstrapCount)
    {
        SourceKey = sourceKey;
        ArchiveCatalog = archiveCatalog;
        BootstrapCount = bootstrapCount;
    }

    public ViewerIoSourceKey SourceKey { get; }

    public IArchiveCatalog ArchiveCatalog { get; }

    public int BootstrapCount { get; }
}

internal interface IViewerIoService : IDisposable
{
    ViewerIoCatalogLease GetCatalog(ViewerIoSourceKey sourceKey);

    void Invalidate(ViewerIoSourceKey sourceKey);

    void InvalidateAll();

    bool TryReadVirtualFile(ViewerIoSourceKey sourceKey, string virtualPath, out byte[]? data, out string sourcePath);

}

internal sealed class ViewerIoService : IViewerIoService
{
    private sealed class CatalogEntry : IDisposable
    {
        public CatalogEntry(IArchiveCatalog catalog, int bootstrapCount)
        {
            Catalog = catalog;
            BootstrapCount = bootstrapCount;
        }

        public IArchiveCatalog Catalog { get; }

        public int BootstrapCount { get; }

        public void Dispose() => Catalog.Dispose();
    }

    private readonly object _syncRoot = new();
    private readonly Dictionary<string, CatalogEntry> _catalogsBySignature = new(StringComparer.OrdinalIgnoreCase);
    private int _bootstrapCount;
    private bool _disposed;

    public ViewerIoCatalogLease GetCatalog(ViewerIoSourceKey sourceKey)
    {
        lock (_syncRoot)
        {
            ThrowIfDisposed();
            if (!sourceKey.HasClientRoot)
                throw new DirectoryNotFoundException($"Viewer I/O source does not have a valid client root: {sourceKey.ClientRoot}");

            if (_catalogsBySignature.TryGetValue(sourceKey.Signature, out CatalogEntry? existing))
                return new ViewerIoCatalogLease(sourceKey, existing.Catalog, existing.BootstrapCount);

            IArchiveCatalog catalog = new MpqArchiveCatalogFactory().Create();
            ArchiveCatalogBootstrapper.Bootstrap(
                catalog,
                BuildLegacySearchRoots(sourceKey.ClientRoot),
                WowViewerArchiveBootstrap.CreateBootstrapOptions(sourceKey.BuildLabel, sourceKey.ClientRoot));

            if (catalog is MpqArchiveCatalog mpqArchiveCatalog)
                mpqArchiveCatalog.ScanMapMpqArchives(sourceKey.ClientRoot);

            int bootstrapCount = ++_bootstrapCount;
            CatalogEntry entry = new(catalog, bootstrapCount);
            _catalogsBySignature[sourceKey.Signature] = entry;
            return new ViewerIoCatalogLease(sourceKey, catalog, bootstrapCount);
        }
    }

    public void Invalidate(ViewerIoSourceKey sourceKey)
    {
        CatalogEntry? entry = null;
        lock (_syncRoot)
        {
            if (_catalogsBySignature.Remove(sourceKey.Signature, out CatalogEntry? existing))
                entry = existing;
        }

        entry?.Dispose();
    }

    public void InvalidateAll()
    {
        List<CatalogEntry> entries;
        lock (_syncRoot)
        {
            entries = _catalogsBySignature.Values.ToList();
            _catalogsBySignature.Clear();
        }

        foreach (CatalogEntry entry in entries)
            entry.Dispose();
    }

    public bool TryReadVirtualFile(ViewerIoSourceKey sourceKey, string virtualPath, out byte[]? data, out string sourcePath)
    {
        if (VirtualAssetOverlayResolver.TryReadLooseVirtualFile(virtualPath, sourceKey.LooseOverlayRoot, out data, out sourcePath))
            return true;

        ViewerIoCatalogLease lease = GetCatalog(sourceKey);
        if (AlphaEmbeddedAdtReader.TryReadVirtualOrLooseFile(sourceKey.ClientRoot, virtualPath, lease.ArchiveCatalog, out data, out sourcePath))
            return true;

        data = null;
        sourcePath = string.Empty;
        return false;
    }

    public void Dispose()
    {
        lock (_syncRoot)
        {
            if (_disposed)
                return;

            _disposed = true;
        }

        InvalidateAll();
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(ViewerIoService));
    }

    private static IReadOnlyList<string> BuildLegacySearchRoots(string clientRoot)
    {
        List<string> roots = [];
        string dataRoot = Path.Combine(clientRoot, "Data");
        if (Directory.Exists(dataRoot))
            roots.Add(dataRoot);

        if (!string.Equals(clientRoot, dataRoot, StringComparison.OrdinalIgnoreCase))
            roots.Add(clientRoot);

        return roots.Count > 0 ? roots : [clientRoot];
    }
}
