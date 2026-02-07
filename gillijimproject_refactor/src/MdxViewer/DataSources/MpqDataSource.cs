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

        // 2. Load files from MPQ archives internal listfiles
        var mpqFiles = _mpq.GetAllKnownFiles();
        foreach (var file in mpqFiles)
        {
            if (!_fileSet.Contains(file))
                _fileSet.Add(file);
        }
        Console.WriteLine($"[MpqDataSource] Added {mpqFiles.Count} MPQ internal files.");

        // 3. Optionally load user-provided external listfile
        if (!string.IsNullOrEmpty(listfilePath) && File.Exists(listfilePath))
        {
            Console.WriteLine($"[MpqDataSource] Loading listfile: {listfilePath}");
            _mpq.LoadListfile(listfilePath);
            AddExternalListfileEntries(listfilePath);
        }

        // 4. Scan loose files on disk
        ScanLooseFiles(gamePath);
        
        // 5. Scan for WMO MPQ archives manually (nested MPQs with embedded WMO data)
        ScanWmoMpqArchives(gamePath);

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
        Console.WriteLine($"[MpqDataSource] Ready. {_fileList.Count} known files:");
        foreach (var ec in extCounts.OrderByDescending(x => x.Value).Take(10))
        {
            Console.WriteLine($"  {ec.Key}: {ec.Value} files");
        }
    }

    /// <summary>
    /// Scans for .wmo.mpq/.wmo.MPQ files and generates virtual paths for their embedded WMO data.
    /// WMO files in Alpha 0.5.3 are stored as file 0 inside .wmo.MPQ archives.
    /// Only scans valid game directories, excludes user folders like "wmos", "addons", etc.
    /// </summary>
    private void ScanWmoMpqArchives(string gamePath)
    {
        // Only scan in valid game data directories - exclude user-created folders
        string[] validScanRoots = new[]
        {
            Path.Combine(gamePath, "Data"),
            Path.Combine(gamePath, "Data", "World"),
            Path.Combine(gamePath, "World"),
        };

        // User-created folders to exclude
        HashSet<string> excludeFolders = new(StringComparer.OrdinalIgnoreCase)
        {
            "wmos", "addons", "interface", "addons", "backup", "cache", "logs"
        };

        foreach (var root in validScanRoots)
        {
            if (!Directory.Exists(root)) continue;

            // Scan for both .wmo.mpq and .wmo.MPQ
            foreach (var ext in new[] { "*.wmo.mpq", "*.wmo.MPQ" })
            {
                foreach (var wmoMpqFile in Directory.EnumerateFiles(root, ext, SearchOption.AllDirectories))
                {
                    // Check if any parent folder is in the exclude list
                    var dir = Path.GetDirectoryName(wmoMpqFile);
                    bool shouldExclude = false;
                    while (!string.IsNullOrEmpty(dir))
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

                    // Generate virtual path like "World\wmo\test.wmo" from "test.wmo.MPQ"
                    var relativePath = Path.GetRelativePath(root, wmoMpqFile);
                    var virtualPath = Path.Combine(
                        Path.GetDirectoryName(relativePath) ?? "",
                        Path.GetFileNameWithoutExtension(relativePath)
                    ).Replace('/', '\\');

                    if (!_fileSet.Contains(virtualPath))
                    {
                        _fileSet.Add(virtualPath);
                        Console.WriteLine($"[MpqDataSource] Added WMO MPQ: {virtualPath}");
                    }
                }
            }
        }
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
        Console.WriteLine($"[MpqDataSource] Added {count} listfile entries.");
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

            Console.WriteLine($"[MpqDataSource] Scanning root: {root}");

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

                int found = _fileSet.Count - before;
                Console.WriteLine($"[MpqDataSource]   Found {found} files in {subDir}/");
                foundAny = found > 0;
            }

            // Additional scan for WMO files in nested wmo directories
            foreach (var wmoDir in wmoScanDirs)
            {
                var fullWmoDir = Path.Combine(root, wmoDir);
                if (!Directory.Exists(fullWmoDir)) continue;

                Console.WriteLine($"[MpqDataSource] Scanning WMO files: {fullWmoDir}");
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
                    Console.WriteLine($"[MpqDataSource] WMO scan error in {fullWmoDir}: {ex.Message}");
                }

                int found = _fileSet.Count - before;
                Console.WriteLine($"[MpqDataSource]   Found {found} WMO files");
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

        // Try reading from .wmo.mpq archives (file 0 contains the actual WMO)
        var wmoMpqData = ReadWmoMpqFile(virtualPath);
        if (wmoMpqData != null)
            return wmoMpqData;

        // Try MPQ
        var mpqData = _mpq.ReadFile(virtualPath);
        if (mpqData != null)
            return mpqData;
            
        return null;
    }

    /// <summary>
    /// Reads WMO data from .wmo.mpq/.wmo.MPQ archives.
    /// The actual WMO data is stored as file 0 inside these nested MPQ archives.
    /// </summary>
    private byte[]? ReadWmoMpqFile(string virtualPath)
    {
        // Convert virtual path like "World\wmo\test.wmo" to find "test.wmo.mpq"
        var fileName = Path.GetFileName(virtualPath);
        var baseFileName = fileName + ".mpq";

        // Only search in valid game data directories - exclude user folders
        string[] searchRoots = new[]
        {
            Path.Combine(_gamePath, "Data"),
            Path.Combine(_gamePath, "Data", "World"),
            Path.Combine(_gamePath, "World"),
        };

        // User-created folders to exclude
        HashSet<string> excludeFolders = new(StringComparer.OrdinalIgnoreCase)
        {
            "wmos", "addons", "interface", "backup", "cache", "logs"
        };

        foreach (var root in searchRoots)
        {
            if (!Directory.Exists(root)) continue;

            // Search for both .wmo.mpq and .wmo.MPQ
            foreach (var ext in new[] { "*.wmo.mpq", "*.wmo.MPQ" })
            {
                foreach (var wmoMpqPath in Directory.EnumerateFiles(root, ext, SearchOption.AllDirectories))
                {
                    // Check if any parent folder is excluded
                    var dir = Path.GetDirectoryName(wmoMpqPath);
                    bool shouldExclude = false;
                    while (!string.IsNullOrEmpty(dir))
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

                    if (Path.GetFileName(wmoMpqPath).Equals(baseFileName, StringComparison.OrdinalIgnoreCase))
                    {
                        // Found the archive, try to read file 0
                        Console.WriteLine($"[MpqDataSource] Reading WMO from: {wmoMpqPath}");
                        return ReadFile0FromMpq(wmoMpqPath);
                    }
                }
            }
        }

        Console.WriteLine($"[MpqDataSource] WMO MPQ not found for: {virtualPath}");
        return null;
    }

    /// <summary>
    /// Reads file 0 from an MPQ archive (where the actual WMO data is stored).
    /// </summary>
    private byte[]? ReadFile0FromMpq(string mpqPath)
    {
        try
        {
            using var fs = new FileStream(mpqPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var reader = new BinaryReader(fs);

            // Read MPQ header to find hash table and block table
            fs.Position = 0;
            var signature = reader.ReadUInt32();
            
            // Check for MPQ signature (may have BOM or be at different offset)
            long headerOffset = 0;
            if (signature == 0x1A51504D) // 'MPQ\x1A'
            {
                headerOffset = 0;
            }
            else
            {
                // Search for MPQ signature
                fs.Position = 0;
                bool found = false;
                while (fs.Position < fs.Length - 4)
                {
                    var pos = fs.Position;
                    var sig = reader.ReadUInt32();
                    if (sig == 0x1A51504D)
                    {
                        headerOffset = pos;
                        found = true;
                        break;
                    }
                }
                if (!found) return null;
            }

            // Read MPQ header
            fs.Position = headerOffset;
            var headerMagic = reader.ReadUInt32();
            var headerSize = reader.ReadUInt32();
            var archiveSize = reader.ReadUInt32();
            var formatVersion = reader.ReadUInt16();
            var blockSizePower = reader.ReadUInt16();
            var hashTableOffset = reader.ReadUInt32();
            var blockTableOffset = reader.ReadUInt32();
            var hashTableEntries = reader.ReadUInt32();
            var blockTableEntries = reader.ReadUInt32();

            var sectorSize = 512u << blockSizePower;

            // Read hash table
            fs.Position = headerOffset + hashTableOffset;
            var hashTable = new (uint BlockIndex, uint NameHash1, uint NameHash2, uint LocaleFlags)[hashTableEntries];
            for (uint i = 0; i < hashTableEntries; i++)
            {
                hashTable[i] = (reader.ReadUInt32(), reader.ReadUInt32(), reader.ReadUInt32(), reader.ReadUInt32());
            }

            // Read block table
            fs.Position = headerOffset + blockTableOffset;
            var blockTable = new (uint BlockOffset, uint BlockSize, uint FileSize, uint Flags)[blockTableEntries];
            for (uint i = 0; i < blockTableEntries; i++)
            {
                blockTable[i] = (reader.ReadUInt32(), reader.ReadUInt32(), reader.ReadUInt32(), reader.ReadUInt32());
            }

            // Find file 0 (first valid file in archive)
            int? file0Index = null;
            foreach (var (blockIndex, _, _, _) in hashTable)
            {
                if (blockIndex != 0xFFFFFFFF && blockIndex < blockTableEntries)
                {
                    var (_, _, _, flags) = blockTable[blockIndex];
                    if ((flags & 0x80000000) != 0) // FLAG_EXISTS
                    {
                        file0Index = (int)blockIndex;
                        break;
                    }
                }
            }

            if (file0Index == null) return null;

            // Read file 0
            var (fileOffset, fileBlockSize, fileSize, fileFlags) = blockTable[file0Index.Value];
            fs.Position = headerOffset + fileOffset;

            byte[] fileData;
            if ((fileFlags & 0x00000200) != 0) // FLAG_COMPRESSED
            {
                // Read compressed data
                var compressedData = reader.ReadBytes((int)fileBlockSize);
                fileData = DecompressMpqFile(compressedData, (int)fileSize);
            }
            else
            {
                fileData = reader.ReadBytes((int)fileSize);
            }

            return fileData;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MpqDataSource] Error reading WMO MPQ {mpqPath}: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Simple decompression for MPQ compressed files.
    /// Handles both uncompressed and basic compression types.
    /// </summary>
    private byte[] DecompressMpqFile(byte[] compressed, int expectedSize)
    {
        if (compressed.Length == expectedSize)
            return compressed;

        // Try inflate for zlib compression (common in MPQ)
        try
        {
            using var compressedStream = new MemoryStream(compressed);
            using var deflateStream = new System.IO.Compression.DeflateStream(compressedStream, System.IO.Compression.CompressionMode.Decompress);
            using var resultStream = new MemoryStream();
            deflateStream.CopyTo(resultStream);
            return resultStream.ToArray();
        }
        catch
        {
            // If deflate fails, return compressed data (might be imploded or another format)
            return compressed;
        }
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
