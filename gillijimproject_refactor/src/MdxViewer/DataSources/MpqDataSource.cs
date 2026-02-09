using MdxViewer.Logging;
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

    // Alpha 0.5.3: virtual path → disk path for listfile-less .ext.MPQ files (WMO, WDT, WDL)
    private readonly Dictionary<string, string> _alphaMpqCache = new(StringComparer.OrdinalIgnoreCase);

    public string Name => $"Game: {Path.GetFileName(_gamePath)}";
    public bool IsLoaded => _loaded;

    public MpqDataSource(string gamePath, string? listfilePath = null)
    {
        _gamePath = gamePath;

        ViewerLog.Important(ViewerLog.Category.MpqData, $"Loading game folder: {gamePath}");

        // 1. Load MPQ archives (large MPQs with listfiles)
        _mpq.LoadArchives(new[] { gamePath });

        // 2. Extract files from MPQ internal (listfile) entries
        var internalFiles = _mpq.ExtractInternalListfiles();
        foreach (var file in internalFiles)
            _fileSet.Add(file);
        ViewerLog.Info(ViewerLog.Category.MpqData, $"Added {internalFiles.Count} files from MPQ internal listfiles.");

        // Also add any previously known files (from hash table / scanned)
        var knownFiles = _mpq.GetAllKnownFiles();
        foreach (var file in knownFiles)
            _fileSet.Add(file);
        if (knownFiles.Count > 0)
            ViewerLog.Info(ViewerLog.Category.MpqData, $"Added {knownFiles.Count} previously known files.");

        // 3. Optionally load user-provided external listfile
        if (!string.IsNullOrEmpty(listfilePath) && File.Exists(listfilePath))
        {
            ViewerLog.Info(ViewerLog.Category.MpqData, $"Loading listfile: {listfilePath}");
            _mpq.LoadListfile(listfilePath);
            AddExternalListfileEntries(listfilePath);
        }

        // 4. Scan loose files on disk
        ScanLooseFiles(gamePath);
        
        // 5. Scan for Alpha 0.5.3 listfile-less .ext.MPQ archives (WMO, WDT, WDL)
        ScanAlphaNestedMpqArchives(gamePath);

        _fileList = _fileSet.OrderBy(f => f, StringComparer.OrdinalIgnoreCase).ToList();
        _loaded = true;

        // Debug: show file counts by extension
        var extCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var f in _fileList)
        {
            var ext = Path.GetExtension(f).ToLowerInvariant();
            if (!extCounts.ContainsKey(ext)) extCounts[ext] = 0;
            extCounts[ext]++;
        }
        ViewerLog.Important(ViewerLog.Category.MpqData, $"Ready. {_fileList.Count} known files:");
        foreach (var ec in extCounts.OrderByDescending(x => x.Value).Take(10))
        {
            ViewerLog.Important(ViewerLog.Category.MpqData, $"  {ec.Key}: {ec.Value} files");
        }
    }

    /// <summary>
    /// Scans for Alpha 0.5.3 listfile-less .ext.MPQ archives (WMO, WDT, WDL).
    /// These files wrap a single data file as file 0 inside an individual MPQ archive.
    /// Builds a virtual path → disk path cache for fast reads.
    /// </summary>
    private void ScanAlphaNestedMpqArchives(string gamePath)
    {
        string[] validScanRoots = new[]
        {
            Path.Combine(gamePath, "Data"),
            Path.Combine(gamePath, "Data", "World"),
            Path.Combine(gamePath, "World"),
        };

        HashSet<string> excludeFolders = new(StringComparer.OrdinalIgnoreCase)
        {
            "wmos", "addons", "interface", "backup", "cache", "logs"
        };

        // Extensions that use listfile-less individual .ext.MPQ wrapping in Alpha 0.5.3
        string[] nestedExts = { ".wmo.MPQ", ".wmo.mpq", ".wdt.MPQ", ".wdt.mpq", ".wdl.MPQ", ".wdl.mpq" };

        var countByExt = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var root in validScanRoots)
        {
            if (!Directory.Exists(root)) continue;

            IEnumerable<string> allMpqs;
            try
            {
                allMpqs = Directory.EnumerateFiles(root, "*.MPQ", SearchOption.AllDirectories)
                    .Concat(Directory.EnumerateFiles(root, "*.mpq", SearchOption.AllDirectories))
                    .Distinct(StringComparer.OrdinalIgnoreCase);
            }
            catch { continue; }

            foreach (var mpqFile in allMpqs)
            {
                var fileName = Path.GetFileName(mpqFile);

                // Check if this matches any nested extension pattern
                string? matchedSuffix = null;
                foreach (var suffix in nestedExts)
                {
                    if (fileName.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
                    {
                        matchedSuffix = suffix;
                        break;
                    }
                }
                if (matchedSuffix == null) continue;

                // Check exclude folders
                var dir = Path.GetDirectoryName(mpqFile);
                bool shouldExclude = false;
                while (!string.IsNullOrEmpty(dir) && dir.Length > root.Length)
                {
                    var dirName = Path.GetFileName(dir);
                    if (excludeFolders.Contains(dirName))
                    {
                        shouldExclude = true;
                        break;
                    }
                    dir = Path.GetDirectoryName(dir);
                }
                if (shouldExclude) continue;

                // Generate virtual path: strip the trailing .MPQ/.mpq
                // e.g., "World\wmo\Azeroth\test.wmo.MPQ" → "World\wmo\Azeroth\test.wmo"
                var relativePath = Path.GetRelativePath(root, mpqFile);
                var virtualPath = relativePath;
                if (virtualPath.EndsWith(".MPQ", StringComparison.OrdinalIgnoreCase))
                    virtualPath = virtualPath[..^4];
                virtualPath = virtualPath.Replace('/', '\\');

                _fileSet.Add(virtualPath);
                _alphaMpqCache[virtualPath] = mpqFile;

                var ext = Path.GetExtension(virtualPath).ToLowerInvariant();
                if (!countByExt.ContainsKey(ext)) countByExt[ext] = 0;
                countByExt[ext]++;
            }
        }

        ViewerLog.Info(ViewerLog.Category.MpqData, $"Alpha nested MPQ scan: {_alphaMpqCache.Count} files found");
        foreach (var kvp in countByExt.OrderByDescending(x => x.Value))
            ViewerLog.Info(ViewerLog.Category.MpqData, $"  {kvp.Key}: {kvp.Value} files");
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
        ViewerLog.Info(ViewerLog.Category.MpqData, $"Added {count} listfile entries.");
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

            ViewerLog.Info(ViewerLog.Category.MpqData, $"Scanning root: {root}");

            // Look for common WoW data subdirectories
            string[] dataDirs = { "World", "Creature", "Character", "Item", "Textures",
                                  "Interface", "Spells", "Environments", "Dungeons" };

            // Also scan for WMO files in World/wmo subdirectories
            string[] wmoScanDirs = { @"World\wmo", @"World\WMO" };

            bool foundAny = false;
            foreach (var subDir in dataDirs)
            {
                var fullDir = Path.Combine(root, subDir);
                if (!Directory.Exists(fullDir)) continue;

                if (!_looseRoots.Contains(root))
                    _looseRoots.Add(root);

                ViewerLog.Debug(ViewerLog.Category.MpqData, $"Scanning loose files: {fullDir}");
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
                    ViewerLog.Error(ViewerLog.Category.MpqData, $"Scan error in {fullDir}: {ex.Message}");
                }

                int found = _fileSet.Count - before;
                ViewerLog.Debug(ViewerLog.Category.MpqData, $"  Found {found} files in {subDir}/");
                foundAny = found > 0;
            }

            // Additional scan for WMO files in nested wmo directories
            foreach (var wmoDir in wmoScanDirs)
            {
                var fullWmoDir = Path.Combine(root, wmoDir);
                if (!Directory.Exists(fullWmoDir)) continue;

                ViewerLog.Debug(ViewerLog.Category.MpqData, $"Scanning WMO files: {fullWmoDir}");
                int before = _fileSet.Count;

                try
                {
                    foreach (var file in Directory.EnumerateFiles(fullWmoDir, "*.wmo", SearchOption.AllDirectories))
                    {
                        var virtualPath = Path.GetRelativePath(root, file).Replace('/', '\\');
                        _fileSet.Add(virtualPath);
                    }
                }
                catch (Exception ex)
                {
                    ViewerLog.Error(ViewerLog.Category.MpqData, $"WMO scan error in {fullWmoDir}: {ex.Message}");
                }

                int found = _fileSet.Count - before;
                ViewerLog.Debug(ViewerLog.Category.MpqData, $"  Found {found} WMO files");
            }

            if (foundAny) break;
        }
    }

    public bool FileExists(string virtualPath)
    {
        // Check loose files first (faster), then MPQ
        if (TryResolveLoosePath(virtualPath) != null)
            return true;
        return _mpq.FileExists(virtualPath);
    }

    /// <summary>Find the actual path from the file set (case-insensitive, returns correctly-cased path).</summary>
    public string? FindInFileSet(string virtualPath)
    {
        string normalized = virtualPath.Replace('/', '\\').ToLowerInvariant();
        
        // Try exact match first
        if (_fileSet.Contains(normalized))
            return normalized;
        
        // Try normalized match
        foreach (var file in _fileSet)
        {
            if (file.Equals(normalized, StringComparison.OrdinalIgnoreCase))
                return file;
        }
        
        return null;
    }

    public byte[]? ReadFile(string virtualPath)
    {
        // Try loose file first
        var loosePath = TryResolveLoosePath(virtualPath);
        if (loosePath != null)
            return File.ReadAllBytes(loosePath);

        // Try Alpha nested .ext.MPQ cache (WMO, WDT, WDL — file 0 inside individual MPQ)
        var normalized = virtualPath.Replace('/', '\\');
        if (_alphaMpqCache.TryGetValue(normalized, out var alphaMpqPath))
        {
            var data = ReadFromAlphaMpq(alphaMpqPath, normalized);
            if (data != null) return data;
        }
        // Also try original path if different
        if (!normalized.Equals(virtualPath, StringComparison.OrdinalIgnoreCase) &&
            _alphaMpqCache.TryGetValue(virtualPath, out var alphaMpqPath2))
        {
            var data = ReadFromAlphaMpq(alphaMpqPath2, virtualPath);
            if (data != null) return data;
        }

        // Try standard MPQ archives (large MPQs with listfiles — MDX, BLP, etc.)
        var mpqData = _mpq.ReadFile(virtualPath);
        if (mpqData != null)
            return mpqData;
            
        return null;
    }

    /// <summary>
    /// Reads the primary data file from an Alpha listfile-less .ext.MPQ archive.
    /// Uses AlphaMpqReader which has smart block selection (name hash lookup, largest block fallback,
    /// magic byte checking) — critical for WMO MPQs that may contain multiple files.
    /// </summary>
    private byte[]? ReadFromAlphaMpq(string mpqDiskPath, string virtualPath)
    {
        // Build internal name candidates from the virtual path for hash-based lookup
        var candidates = AlphaMpqReader.BuildInternalNameCandidates(virtualPath).ToList();
        // Also add just the filename
        var fileName = Path.GetFileName(virtualPath);
        if (!string.IsNullOrEmpty(fileName) && !candidates.Contains(fileName, StringComparer.OrdinalIgnoreCase))
            candidates.Insert(0, fileName);
        return AlphaMpqReader.ReadFromMpq(mpqDiskPath, candidates);
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
