using WowViewer.Core.IO.Casc;
using WowViewer.Core.IO.Files;

namespace WoWViewer.DataSources;

/// <summary>
/// Spec 238: data source over one or more products of a local CASC install. Virtual paths resolve
/// through the community listfile to FileDataIDs; each product is tried in order and the first
/// that returns bytes wins. This matters when a product lists a file in its root but does not have
/// the data on disk (measured: wow_classic_beta 1.60.1 lists the DAT v26 tileset BLPs without local
/// data, while wow_classic_era 1.15.9 has most of them).
/// </summary>
public sealed class CascDataSource : IDataSource
{
    private readonly IReadOnlyList<CascStorage> _storages;
    private readonly CommunityListfile _listfile;
    private List<string>? _fileList;

    public CascDataSource(IReadOnlyList<CascStorage> storages, CommunityListfile listfile)
    {
        if (storages.Count == 0)
            throw new ArgumentException("At least one CASC product is required.", nameof(storages));

        _storages = storages;
        _listfile = listfile;
        _resolver = listfile.GetPath;
        FileDataIdPaths.Resolver = _resolver;
    }

    private readonly Func<uint, string?> _resolver;

    public string Name => "CASC: " + string.Join(" + ", _storages.Select(static s => $"{s.Product.Product} {s.Product.Version}"));

    public bool IsLoaded => true;

    public IReadOnlyList<CascStorage> Storages => _storages;

    public bool FileExists(string virtualPath) =>
        TryGetFileDataId(virtualPath, out uint fileDataId) && FileExists(fileDataId);

    /// <summary>Accepts listfile paths and <c>fdid:&lt;id&gt;</c> virtual paths.</summary>
    private bool TryGetFileDataId(string virtualPath, out uint fileDataId) =>
        FileDataIdPaths.TryParse(virtualPath, out fileDataId) || _listfile.TryGetFileDataId(virtualPath, out fileDataId);

    public bool FileExists(uint fileDataId) => _storages.Any(s => s.FileExists(fileDataId));

    public byte[]? ReadFile(string virtualPath) =>
        TryGetFileDataId(virtualPath, out uint fileDataId) ? ReadFile(fileDataId) : null;

    public byte[]? ReadFile(uint fileDataId)
    {
        foreach (CascStorage storage in _storages)
        {
            if (storage.TryReadFile(fileDataId, out byte[]? data) == CascReadStatus.Ok && data is not null)
                return data;
        }

        return null;
    }

    public bool TryResolveWritablePath(string virtualPath, out string? fullPath)
    {
        fullPath = null;
        return false;
    }

    public IReadOnlyList<string> GetFileList(string? extensionFilter = null)
    {
        _fileList ??= _listfile.Entries
            .Where(entry => FileExists(entry.Key))
            .Select(static entry => entry.Value.Replace('/', '\\'))
            .ToList();

        return extensionFilter is null
            ? _fileList
            : _fileList.Where(f => f.EndsWith(extensionFilter, StringComparison.OrdinalIgnoreCase)).ToList();
    }

    public void Dispose()
    {
        if (FileDataIdPaths.Resolver == _resolver)
            FileDataIdPaths.Resolver = null;
    }
}
