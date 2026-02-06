namespace MdxViewer.DataSources;

/// <summary>
/// Data source for loose files on disk (extracted game data or individual files).
/// </summary>
public class LooseFileDataSource : IDataSource
{
    private readonly string _rootPath;
    private List<string>? _fileList;

    public string Name => $"Loose: {Path.GetFileName(_rootPath)}";
    public bool IsLoaded => Directory.Exists(_rootPath);

    public LooseFileDataSource(string rootPath)
    {
        _rootPath = rootPath;
    }

    public bool FileExists(string virtualPath)
    {
        var fullPath = ResolvePath(virtualPath);
        return File.Exists(fullPath);
    }

    public byte[]? ReadFile(string virtualPath)
    {
        var fullPath = ResolvePath(virtualPath);
        return File.Exists(fullPath) ? File.ReadAllBytes(fullPath) : null;
    }

    public IReadOnlyList<string> GetFileList(string? extensionFilter = null)
    {
        if (_fileList == null)
        {
            _fileList = Directory.EnumerateFiles(_rootPath, "*.*", SearchOption.AllDirectories)
                .Select(f => Path.GetRelativePath(_rootPath, f).Replace('/', '\\'))
                .ToList();
        }

        if (extensionFilter != null)
        {
            return _fileList
                .Where(f => f.EndsWith(extensionFilter, StringComparison.OrdinalIgnoreCase))
                .ToList();
        }

        return _fileList;
    }

    private string ResolvePath(string virtualPath)
    {
        var normalized = virtualPath.Replace('/', '\\').TrimStart('\\');
        return Path.Combine(_rootPath, normalized);
    }

    public void Dispose() { }
}
