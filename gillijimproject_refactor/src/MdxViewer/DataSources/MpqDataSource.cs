using WoWMapConverter.Core.Services;

namespace MdxViewer.DataSources;

/// <summary>
/// Data source backed by MPQ archives + loose files on disk.
/// Uses WoWMapConverter.Core's NativeMpqService for MPQ reading.
/// Builds file list from MPQ-internal (listfile) entries — accurate for any client version.
/// Supports Alpha 0.5.3, Classic, TBC, and WotLK 3.3.5 game folders.
/// </summary>
public class MpqDataSource : IDataSource
{
    private readonly NativeMpqService _mpq = new();
    private readonly string _gamePath;

    /// <summary>Exposes the underlying MPQ service for DBC provider access.</summary>
    public NativeMpqService MpqService => _mpq;
    private readonly HashSet<string> _fileSet = new(StringComparer.OrdinalIgnoreCase);
    private List<string> _fileList = new();
    private bool _loaded;

    // Loose file roots to check (game folder structure)
    private readonly List<string> _looseRoots = new();

    public string Name => $"Game: {Path.GetFileName(_gamePath)}";
    public bool IsLoaded => _loaded;

    public MpqDataSource(string gamePath, string? listfilePath = null)
    {
        _gamePath = gamePath;

        Console.WriteLine($"[MpqDataSource] Loading game folder: {gamePath}");

        // 1. Load MPQ archives
        _mpq.LoadArchives(new[] { gamePath });

        // 2. Extract internal (listfile) from each MPQ — version-accurate
        var internalFiles = _mpq.ExtractInternalListfiles();
        foreach (var path in internalFiles)
            _fileSet.Add(path);
        Console.WriteLine($"[MpqDataSource] {internalFiles.Count} files from MPQ internal listfiles.");

        // 3. Feed extracted entries back to MPQ service for hash resolution
        if (internalFiles.Count > 0)
        {
            var cacheDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output", "cache");
            Directory.CreateDirectory(cacheDir);
            var tempListfile = Path.Combine(cacheDir, "_internal_listfile.txt");
            File.WriteAllLines(tempListfile, internalFiles);
            _mpq.LoadListfile(tempListfile);
        }

        // 4. Optionally load user-provided external listfile (supplements MPQ listfiles)
        if (!string.IsNullOrEmpty(listfilePath) && File.Exists(listfilePath))
        {
            Console.WriteLine($"[MpqDataSource] Loading supplemental listfile: {listfilePath}");
            _mpq.LoadListfile(listfilePath);
            AddExternalListfileEntries(listfilePath);
        }

        // 4. Scan loose files on disk
        ScanLooseFiles(gamePath);

        _fileList = _fileSet.OrderBy(f => f, StringComparer.OrdinalIgnoreCase).ToList();
        _loaded = true;

        Console.WriteLine($"[MpqDataSource] Ready. {_fileList.Count} known files ({_looseRoots.Count} loose roots scanned).");
    }

    private void AddExternalListfileEntries(string listfilePath)
    {
        int count = 0;
        foreach (var line in File.ReadLines(listfilePath))
        {
            var name = line.Trim();
            if (string.IsNullOrEmpty(name)) continue;

            if (name.Contains(';'))
            {
                var parts = name.Split(';', 2);
                if (parts.Length > 1) name = parts[1].Trim();
            }

            if (!string.IsNullOrEmpty(name))
            {
                _fileSet.Add(name);
                count++;
            }
        }
        Console.WriteLine($"[MpqDataSource] Added {count} supplemental entries.");
    }

    private void ScanLooseFiles(string gamePath)
    {
        // Scan for loose files in the game directory structure
        // Alpha 0.5.3 has files directly in Data/ subfolders
        string[] scanRoots = new[]
        {
            gamePath,
            Path.Combine(gamePath, "Data"),
        };

        foreach (var root in scanRoots)
        {
            if (!Directory.Exists(root)) continue;

            // Look for common WoW data subdirectories
            string[] dataDirs = { "World", "Creature", "Character", "Item", "Textures",
                                  "Interface", "Spells", "Environments", "Dungeons" };

            bool foundAny = false;
            foreach (var subDir in dataDirs)
            {
                var fullDir = Path.Combine(root, subDir);
                if (!Directory.Exists(fullDir)) continue;

                if (!_looseRoots.Contains(root))
                    _looseRoots.Add(root);

                Console.WriteLine($"[MpqDataSource] Scanning loose files: {fullDir}");
                int before = _fileSet.Count;

                try
                {
                    foreach (var file in Directory.EnumerateFiles(fullDir, "*.*", SearchOption.AllDirectories))
                    {
                        var ext = Path.GetExtension(file).ToLowerInvariant();
                        if (ext is ".mdx" or ".wmo" or ".m2" or ".blp" or ".skin" or ".anim" or ".dbc")
                        {
                            var virtualPath = Path.GetRelativePath(root, file).Replace('/', '\\');
                            _fileSet.Add(virtualPath);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[MpqDataSource] Scan error in {fullDir}: {ex.Message}");
                }

                Console.WriteLine($"[MpqDataSource]   Found {_fileSet.Count - before} files in {subDir}/");
                foundAny = true;
            }

            if (foundAny) break; // Found valid root, don't double-scan gamePath and gamePath/Data
        }
    }

    public bool FileExists(string virtualPath)
    {
        // Check loose files first (faster), then MPQ
        if (TryResolveLoosePath(virtualPath) != null)
            return true;
        return _mpq.FileExists(virtualPath);
    }

    public byte[]? ReadFile(string virtualPath)
    {
        // Try loose file first
        var loosePath = TryResolveLoosePath(virtualPath);
        if (loosePath != null)
            return File.ReadAllBytes(loosePath);

        // Fall back to MPQ
        return _mpq.ReadFile(virtualPath);
    }

    private string? TryResolveLoosePath(string virtualPath)
    {
        var normalized = virtualPath.Replace('/', '\\').TrimStart('\\');

        // Check each loose root
        foreach (var root in _looseRoots)
        {
            var fullPath = Path.Combine(root, normalized);
            if (File.Exists(fullPath))
                return fullPath;
        }

        // Also check game path directly
        var directPath = Path.Combine(_gamePath, normalized);
        if (File.Exists(directPath))
            return directPath;

        var dataPath = Path.Combine(_gamePath, "Data", normalized);
        if (File.Exists(dataPath))
            return dataPath;

        return null;
    }

    public IReadOnlyList<string> GetFileList(string? extensionFilter = null)
    {
        if (extensionFilter != null)
        {
            return _fileList
                .Where(f => f.EndsWith(extensionFilter, StringComparison.OrdinalIgnoreCase))
                .ToList();
        }
        return _fileList;
    }

    public void Dispose()
    {
        _mpq.Dispose();
    }
}
