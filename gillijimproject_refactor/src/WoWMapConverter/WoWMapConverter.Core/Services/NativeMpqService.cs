using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text;
using System.Linq;
using WoWMapConverter.Core.Diagnostics;

namespace WoWMapConverter.Core.Services;

/// <summary>
/// Pure C# MPQ archive service for LK/Cata multi-file archives.
/// Replaces StormLib which has issues with patch chains.
/// Based on AlphaMpqReader primitives.
/// </summary>
public class NativeMpqService : IDisposable
{
    private readonly List<MpqArchive> _archives = new();
    
    // Listfile support
    private readonly HashSet<ulong> _knownFileHashes = new();
    private readonly Dictionary<ulong, string> _hashToName = new();
    
    // Scanned file mappings for listfile-less MPQs (placeholder path -> archive path + block)
    private readonly Dictionary<string, (string ArchivePath, uint BlockOffset)> _scannedFiles = new(StringComparer.OrdinalIgnoreCase);

    public void LoadListfile(string path)
    {
        if (!File.Exists(path))
        {
            Console.WriteLine($"[NativeMpqService] Listfile not found: {path}");
            return;
        }

        Console.WriteLine($"[NativeMpqService] Loading listfile: {path}");
        int count = 0;
        foreach (var line in File.ReadLines(path))
        {
            var name = line;
            if (line.Contains(';'))
            {
                var parts = line.Split(';');
                if (parts.Length > 1) name = parts[1]; // ID;Name format
            }
            
            name = name.Trim();
            if (string.IsNullOrEmpty(name)) continue;

            // Normalize for hashing (MPQ standard)
            string normalized = name.Replace('/', '\\');
            ulong hash = ((ulong)HashString(normalized, HASH_NAME_A) << 32) | HashString(normalized, HASH_NAME_B);
            
            if (!_knownFileHashes.Contains(hash))
            {
                _knownFileHashes.Add(hash);
                _hashToName[hash] = name; 
            }
            count++;
        }
        Console.WriteLine($"[NativeMpqService] Loaded {count} listfile entries.");
    }

    public bool HasFile(string filename)
    {
         var normalized = filename.Replace('/', '\\');
         foreach (var archive in _archives)
         {
             if (FindFileInArchive(archive, normalized) != null) return true;
         }
         return false;
    }

    /// <summary>
    /// Extracts internal listfile entries from loaded MPQ archives.
    /// Looks for the standard "(listfile)" entry in each archive.
    /// </summary>
    /// <returns>List of file paths found in MPQ internal listfiles.</returns>
    public List<string> ExtractInternalListfiles()
    {
        var allFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        
        foreach (var archive in _archives)
        {
            // Try to read the listfile entry
            var listfileBlock = FindFileInArchive(archive, "(listfile)");
            if (listfileBlock != null && listfileBlock.FileSize > 0)
            {
                var listfileData = ReadFileFromArchive(archive, listfileBlock, "(listfile)");
                if (listfileData != null && listfileData.Length > 0)
                {
                    var content = Encoding.UTF8.GetString(listfileData);
                    foreach (var line in content.Split('\n', '\r'))
                    {
                        var trimmed = line.Trim();
                        if (!string.IsNullOrEmpty(trimmed))
                        {
                            // Normalize path separators
                            var normalizedPath = trimmed.Replace('/', '\\').TrimStart('\\');
                            allFiles.Add(normalizedPath);
                        }
                    }
                }
            }
        }
        
        Console.WriteLine($"[NativeMpqService] Extracted {allFiles.Count} files from MPQ internal listfiles.");
        return allFiles.ToList();
    }

    /// <summary>
    /// Scans for WMO MPQ archives and their embedded WMO files.
    /// Alpha 0.5.3 uses .wmo.mpq files where file 0 contains the actual WMO data.
    /// </summary>
    public List<string> ScanWmoMpqArchives(string gamePath)
    {
        var foundWmos = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        
        // Search for .wmo.mpq files in common locations
        string[] searchPaths = {
            Path.Combine(gamePath, "Data"),
            Path.Combine(gamePath, "Data", "World"),
            Path.Combine(gamePath, "Data", "World", "wmo"),
            Path.Combine(gamePath, "Data", "World", "WMO"),
            Path.Combine(gamePath, "World"),
            Path.Combine(gamePath, "World", "wmo"),
            Path.Combine(gamePath, "World", "WMO")
        };
        
        foreach (var searchPath in searchPaths)
        {
            if (!Directory.Exists(searchPath)) continue;
            
            Console.WriteLine($"[NativeMpqService] Scanning for WMO MPQs in: {searchPath}");
            
            try
            {
                // Find all .wmo.mpq files
                var wmoMpqFiles = Directory.GetFiles(searchPath, "*.wmo.mpq", SearchOption.AllDirectories)
                    .Concat(Directory.GetFiles(searchPath, "*.WMO.MPQ", SearchOption.AllDirectories))
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToList();
                
                foreach (var mpqPath in wmoMpqFiles)
                {
                    Console.WriteLine($"[NativeMpqService]   Found WMO MPQ: {Path.GetFileName(mpqPath)}");
                    
                    // Generate virtual path from MPQ filename
                    // e.g., "Dungeons\\test.wmo.mpq" -> "World\\wmo\\test.wmo"
                    var mpqFileName = Path.GetFileNameWithoutExtension(mpqPath); // removes .mpq
                    var mpqFileNameWithoutWmo = mpqFileName;
                    
                    // Handle case where filename ends with .wmo
                    if (mpqFileNameWithoutWmo.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
                    {
                        mpqFileNameWithoutWmo = mpqFileNameWithoutWmo[0..^4]; // remove .wmo suffix
                    }
                    
                    // Build virtual path
                    var relativePath = Path.GetRelativePath(gamePath, mpqPath);
                    var virtualPath = $"World\\wmo\\{mpqFileNameWithoutWmo}.wmo";
                    
                    // Normalize path separators
                    virtualPath = virtualPath.Replace('/', '\\');
                    
                    // Store for reading
                    if (!_scannedFiles.ContainsKey(virtualPath))
                    {
                        _scannedFiles[virtualPath] = (mpqPath, 0); // 0 = read from file 0 in MPQ
                    }
                    
                    foundWmos.Add(virtualPath);
                    Console.WriteLine($"[NativeMpqService]     Added: {virtualPath}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[NativeMpqService]   Error scanning {searchPath}: {ex.Message}");
            }
        }
        
        Console.WriteLine($"[NativeMpqService] Found {foundWmos.Count} WMO MPQ archives.");
        return foundWmos.ToList();
    }
    
    /// <summary>
    /// Reads file 0 from a WMO MPQ archive.
    /// </summary>
    public byte[]? ReadWmoMpqFile(string virtualPath)
    {
        if (!_scannedFiles.TryGetValue(virtualPath, out var mpqInfo))
            return null;
        
        // Check if it's a nested MPQ read
        if (mpqInfo.BlockOffset == 0 && mpqInfo.ArchivePath.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                // This is a .wmo.mpq file, read file 0 from it
                using var fs = new FileStream(mpqInfo.ArchivePath, FileMode.Open, FileAccess.Read, FileShare.Read);
                using var reader = new BinaryReader(fs);
                
                // Parse nested MPQ header
                long headerOffset = FindMpqHeader(reader);
                if (headerOffset < 0) return null;
                
                fs.Position = headerOffset;
                var header = ReadMpqHeader(reader);
                if (header == null) return null;
                
                // Read hash table to find file 0
                fs.Position = headerOffset + header.HashTableOffset;
                var hashTable = ReadHashTable(reader, header.HashTableEntries);
                
                // Read block table
                fs.Position = headerOffset + header.BlockTableOffset;
                var blockTable = ReadBlockTable(reader, header.BlockTableEntries);
                
                // Find file 0 (usually the first file or one with hash matching "")
                BlockEntry? file0Block = null;
                
                // Try to find by looking for first valid file in hash table
                foreach (var entry in hashTable)
                {
                    if (entry.BlockIndex != 0xFFFFFFFF && entry.BlockIndex < blockTable.Length)
                    {
                        var block = blockTable[entry.BlockIndex];
                        if ((block.Flags & 0x80000000) != 0) // FLAG_EXISTS
                        {
                            file0Block = block;
                            break;
                        }
                    }
                }
                
                if (file0Block == null) return null;
                
                // Read file 0 data
                fs.Position = headerOffset + file0Block.BlockOffset;
                return ReadFileData(reader, file0Block, header.SectorSize, "file_0", fs.Position);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[NativeMpqService] Error reading WMO MPQ {mpqInfo.ArchivePath}: {ex.Message}");
                return null;
            }
        }
        
        return null;
    }
    
    /// <summary>
    /// Reads file 0 from any listfile-less .ext.MPQ archive given its disk path.
    /// Used by MdxViewer for Alpha 0.5.3 WMO, WDT, and WDL files.
    /// </summary>
    public byte[]? ReadFile0FromPath(string mpqDiskPath)
    {
        try
        {
            using var fs = new FileStream(mpqDiskPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var reader = new BinaryReader(fs);
            
            long headerOffset = FindMpqHeader(reader);
            if (headerOffset < 0) return null;
            
            fs.Position = headerOffset;
            var header = ReadMpqHeader(reader);
            if (header == null) return null;
            
            fs.Position = headerOffset + header.HashTableOffset;
            var hashTable = ReadHashTable(reader, header.HashTableEntries);
            
            fs.Position = headerOffset + header.BlockTableOffset;
            var blockTable = ReadBlockTable(reader, header.BlockTableEntries);
            
            // Find first valid file in hash table (file 0)
            BlockEntry? file0Block = null;
            foreach (var entry in hashTable)
            {
                if (entry.BlockIndex != 0xFFFFFFFF && entry.BlockIndex < blockTable.Length)
                {
                    var block = blockTable[entry.BlockIndex];
                    if ((block.Flags & 0x80000000) != 0)
                    {
                        file0Block = block;
                        break;
                    }
                }
            }
            
            if (file0Block == null) return null;
            
            fs.Position = headerOffset + file0Block.BlockOffset;
            return ReadFileData(reader, file0Block, header.SectorSize, "file_0", fs.Position);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[NativeMpqService] Error reading file 0 from {Path.GetFileName(mpqDiskPath)}: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Gets all known file paths including scanned files.
    /// </summary>
    public IReadOnlyList<string> GetAllKnownFiles()
    {
        var allFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        
        // Add files from hash table
        foreach (var kvp in _hashToName)
        {
            allFiles.Add(kvp.Value);
        }
        
        // Add scanned file paths
        foreach (var path in _scannedFiles.Keys)
        {
            allFiles.Add(path);
        }
        
        return allFiles.ToList();
    }

    private bool _disposed;

    // MPQ hash type constants
    private const uint HASH_TABLE_INDEX = 0;
    private const uint HASH_NAME_A = 1;
    private const uint HASH_NAME_B = 2;
    private const uint HASH_FILE_KEY = 3;
    
    private const uint HASH_ENTRY_DELETED = 0xFFFFFFFE;
    private const uint HASH_ENTRY_EMPTY = 0xFFFFFFFF;
    
    // Block flags
    private const uint FLAG_COMPRESSED = 0x00000200;
    private const uint FLAG_ENCRYPTED = 0x00010000;
    private const uint FLAG_FIX_KEY = 0x00020000;
    private const uint FLAG_IMPLODED = 0x00000100;
    private const uint FLAG_SINGLE_UNIT = 0x01000000;
    private const uint FLAG_EXISTS = 0x80000000;
    
    // Encryption table for MPQ hash/decrypt
    private static readonly uint[] CryptTable = BuildCryptTable();

    public void LoadArchives(IEnumerable<string> searchPaths)
    {
        var pathsToSearch = new HashSet<string>();
        
        foreach (var path in searchPaths)
        {
            if (Directory.Exists(path))
                pathsToSearch.Add(path);
            
            var dataSubdir = Path.Combine(path, "Data");
            if (Directory.Exists(dataSubdir))
            {
                pathsToSearch.Add(dataSubdir);
                
                // Check for locale folders (enUS, deDE, etc.)
                foreach (var localeDir in Directory.GetDirectories(dataSubdir))
                {
                    var localeName = Path.GetFileName(localeDir);
                    if (localeName.Length == 4 && char.IsLetter(localeName[0]) && char.IsLetter(localeName[1]) &&
                        char.IsUpper(localeName[2]) && char.IsUpper(localeName[3]))
                    {
                        pathsToSearch.Add(localeDir);
                    }
                }
            }
        }
        
        Console.WriteLine($"[NativeMpqService] Searching {pathsToSearch.Count} paths for MPQs:");
        
        // Collect all MPQ files
        var allMpqFiles = new List<string>();
        foreach (var path in pathsToSearch)
        {
            var mpqFiles = Directory.GetFiles(path, "*.mpq", SearchOption.TopDirectoryOnly);
            allMpqFiles.AddRange(mpqFiles);
            
            var mpqFilesUpper = Directory.GetFiles(path, "*.MPQ", SearchOption.TopDirectoryOnly);
            allMpqFiles.AddRange(mpqFilesUpper);
        }
        
        allMpqFiles = allMpqFiles.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        
        // Sort: base archives first, patches last (patches override base)
        allMpqFiles = allMpqFiles
            .OrderBy(f => GetMpqPriority(Path.GetFileName(f)))
            .ToList();
        
        Console.WriteLine($"[NativeMpqService] Loading {allMpqFiles.Count} archives:");
        foreach (var mpqPath in allMpqFiles)
        {
            try
            {
                var archive = LoadArchive(mpqPath);
                if (archive != null)
                {
                    _archives.Add(archive);
                    Console.WriteLine($"  [{_archives.Count:D2}] {Path.GetFileName(mpqPath)} ({archive.BlockTable.Length} files)");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [XX] {Path.GetFileName(mpqPath)} - Error: {ex.Message}");
            }
        }
        
        Console.WriteLine($"[NativeMpqService] Loaded {_archives.Count} archives.");
    }
    
    private static int GetMpqPriority(string filename)
    {
        var lower = filename.ToLowerInvariant();
        
        // Patches get highest priority (loaded last = searched first)
        if (lower.StartsWith("patch"))
        {
            if (lower == "patch.mpq") return 1000;
            if (lower.StartsWith("patch-"))
            {
                var numPart = lower.Replace("patch-", "").Replace(".mpq", "");
                if (int.TryParse(numPart, out int num))
                    return 1000 + num;
            }
            return 1099;
        }
        
        if (lower.Contains("enus") || lower.Contains("engb") || lower.Contains("dede") || lower.Contains("locale"))
            return 500;
        
        if (lower.StartsWith("expansion") || lower.StartsWith("lichking"))
            return 300;
        
        if (lower == "common.mpq" || lower == "common-2.mpq")
            return 100;
        
        return 200;
    }
    
    public bool FileExists(string virtualPath)
    {
        var normalized = virtualPath.Replace('/', '\\');
        
        // Search archives in reverse order (patches first)
        for (int i = _archives.Count - 1; i >= 0; i--)
        {
            if (FindFileInArchive(_archives[i], normalized) != null)
            {
                if (normalized.Contains("md5translate", StringComparison.OrdinalIgnoreCase))
                    Console.WriteLine($"[DEBUG] Found md5translate: {normalized} in {Path.GetFileName(_archives[i].Path)}");
                return true;
            }
        }
        
        if (normalized.Contains("md5translate", StringComparison.OrdinalIgnoreCase))
            Console.WriteLine($"[DEBUG] md5translate NOT found: {normalized}");
            
        return false;
    }
    
    public byte[]? ReadFile(string virtualPath)
    {
        var normalized = virtualPath.Replace('/', '\\');
        
        // First check if it's a scanned loose file
        if (_scannedFiles.TryGetValue(normalized, out var scannedInfo))
        {
            if (scannedInfo.BlockOffset == 0 && !string.IsNullOrEmpty(scannedInfo.ArchivePath))
            {
                // Loose file - read directly from disk
                if (File.Exists(scannedInfo.ArchivePath))
                {
                    return File.ReadAllBytes(scannedInfo.ArchivePath);
                }
            }
        }
        
        // Search archives in reverse order (patches first)
        for (int i = _archives.Count - 1; i >= 0; i--)
        {
            var archive = _archives[i];
            var block = FindFileInArchive(archive, normalized);
            
            if (block != null)
            {
                // Check if file is deleted (0 size in patches)
                if (block.FileSize == 0)
                {
                    Build335Diagnostics.Increment("MpqPatchedDeleteHitCount");
                    Console.WriteLine($"[NativeMpqService] ReadFile '{normalized}' → found in {Path.GetFileName(archive.Path)} but FileSize=0 (patched out)");
                    continue; // File was "patched out", try base archives
                }
                
                Console.WriteLine($"[NativeMpqService] ReadFile '{normalized}' → found in {Path.GetFileName(archive.Path)}, block: offset={block.BlockOffset}, size={block.BlockSize}, fileSize={block.FileSize}, flags=0x{block.Flags:X8}");
                var data = ReadFileFromArchive(archive, block, Path.GetFileName(normalized));
                if (data != null && data.Length > 0)
                {
                    Console.WriteLine($"[NativeMpqService] ReadFile '{normalized}' → extracted {data.Length} bytes");
                    return data;
                }
                Console.WriteLine($"[NativeMpqService] ReadFile '{normalized}' → ReadFileFromArchive returned null/empty");
            }
        }
        
        Console.WriteLine($"[NativeMpqService] ReadFile '{normalized}' → not found in any of {_archives.Count} archives");
        return null;
    }
    
    /// <summary>
    /// Reads a scanned file by its placeholder path (e.g., "WMO_12345678.wmo").
    /// Used for files found in listfile-less MPQs.
    /// </summary>
    public byte[]? ReadScannedFile(string placeholderPath)
    {
        if (!_scannedFiles.TryGetValue(placeholderPath, out var info))
            return null;
        
        try
        {
            using var fs = new FileStream(info.ArchivePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var reader = new BinaryReader(fs);
            
            // Find the archive
            var archive = _archives.FirstOrDefault(a => a.Path == info.ArchivePath);
            if (archive == null) return null;
            
            fs.Position = archive.HeaderOffset + info.BlockOffset;
            return ReadFileData(reader, new BlockEntry { BlockOffset = info.BlockOffset, BlockSize = 0, FileSize = 0, Flags = 0 }, archive.Header.SectorSize, placeholderPath, archive.HeaderOffset + info.BlockOffset);
        }
        catch
        {
            return null;
        }
    }
    
    private MpqArchive? LoadArchive(string mpqPath)
    {
        using var fs = new FileStream(mpqPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var reader = new BinaryReader(fs);
        
        long headerOffset = FindMpqHeader(reader);
        if (headerOffset < 0)
            return null;
        
        fs.Position = headerOffset;
        
        var header = ReadMpqHeader(reader);
        if (header == null)
            return null;
        
        // Read and decrypt hash table
        fs.Position = headerOffset + header.HashTableOffset;
        var hashTable = ReadHashTable(reader, header.HashTableEntries);
        
        // Read and decrypt block table
        fs.Position = headerOffset + header.BlockTableOffset;
        var blockTable = ReadBlockTable(reader, header.BlockTableEntries);
        
        return new MpqArchive
        {
            Path = mpqPath,
            HeaderOffset = headerOffset,
            Header = header,
            HashTable = hashTable,
            BlockTable = blockTable
        };
    }
    
    private BlockEntry? FindFileInArchive(MpqArchive archive, string filename)
    {
        // Standard MPQ hashing uses backslashes and is case-insensitive
        var normalized = filename.Replace('/', '\\');
        
        var hashIndex = HashString(normalized, HASH_TABLE_INDEX) % (uint)archive.HashTable.Length;
        var nameA = HashString(normalized, HASH_NAME_A);
        var nameB = HashString(normalized, HASH_NAME_B);
        
        for (uint i = 0; i < archive.HashTable.Length; i++)
        {
            var entry = archive.HashTable[(hashIndex + i) % archive.HashTable.Length];
            
            if (entry.BlockIndex == HASH_ENTRY_EMPTY)
                break;
            
            if (entry.BlockIndex == HASH_ENTRY_DELETED)
                continue;
            
            if (entry.Name1 == nameA && entry.Name2 == nameB && entry.BlockIndex < archive.BlockTable.Length)
            {
                var block = archive.BlockTable[entry.BlockIndex];
                if ((block.Flags & FLAG_EXISTS) != 0)
                    return block;
            }
        }
        
        return null;
    }
    
    private byte[]? ReadFileFromArchive(MpqArchive archive, BlockEntry block, string filename)
    {
        try
        {
            using var fs = new FileStream(archive.Path, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var reader = new BinaryReader(fs);
            
            var baseOffset = archive.HeaderOffset + block.BlockOffset;
            fs.Position = baseOffset;
            return ReadFileData(reader, block, archive.Header.SectorSize, filename, baseOffset);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[NativeMpqService] ReadFileFromArchive exception for '{filename}': {ex.Message}");
            return null;
        }
    }
    
    #region MPQ Parsing (from AlphaMpqReader)
    
    private static long FindMpqHeader(BinaryReader reader)
    {
        if (reader.BaseStream.Length >= 4)
        {
            reader.BaseStream.Position = 0;
            uint magic = reader.ReadUInt32();
            if (magic == 0x1A51504D) // "MPQ\x1A"
                return 0;
        }
        
        for (long offset = 0x200; offset < reader.BaseStream.Length && offset < 0x100000; offset += 0x200)
        {
            reader.BaseStream.Position = offset;
            if (reader.BaseStream.Length - offset < 4)
                break;
            
            uint magic = reader.ReadUInt32();
            if (magic == 0x1A51504D)
                return offset;
        }
        
        return -1;
    }
    
    private static MpqHeader? ReadMpqHeader(BinaryReader reader)
    {
        uint magic = reader.ReadUInt32();
        if (magic != 0x1A51504D)
            return null;
        
        var header = new MpqHeader
        {
            HeaderSize = reader.ReadUInt32(),
            ArchiveSize = reader.ReadUInt32(),
            FormatVersion = reader.ReadUInt16(),
            SectorSizeShift = reader.ReadUInt16(),
            HashTableOffset = reader.ReadUInt32(),
            BlockTableOffset = reader.ReadUInt32(),
            HashTableEntries = reader.ReadUInt32(),
            BlockTableEntries = reader.ReadUInt32()
        };
        
        header.SectorSize = 512u << header.SectorSizeShift;
        
        return header;
    }
    
    private static HashEntry[] ReadHashTable(BinaryReader reader, uint entryCount)
    {
        var encryptedData = new uint[entryCount * 4];
        for (int i = 0; i < encryptedData.Length; i++)
            encryptedData[i] = reader.ReadUInt32();
        
        uint key = HashString("(hash table)", HASH_FILE_KEY);
        DecryptBlock(encryptedData, key);
        
        var entries = new HashEntry[entryCount];
        for (uint i = 0; i < entryCount; i++)
        {
            entries[i] = new HashEntry
            {
                Name1 = encryptedData[i * 4 + 0],
                Name2 = encryptedData[i * 4 + 1],
                Locale = (ushort)(encryptedData[i * 4 + 2] & 0xFFFF),
                Platform = (ushort)((encryptedData[i * 4 + 2] >> 16) & 0xFFFF),
                BlockIndex = encryptedData[i * 4 + 3]
            };
        }
        
        return entries;
    }
    
    private static BlockEntry[] ReadBlockTable(BinaryReader reader, uint entryCount)
    {
        var encryptedData = new uint[entryCount * 4];
        for (int i = 0; i < encryptedData.Length; i++)
            encryptedData[i] = reader.ReadUInt32();
        
        uint key = HashString("(block table)", HASH_FILE_KEY);
        DecryptBlock(encryptedData, key);
        
        var entries = new BlockEntry[entryCount];
        for (uint i = 0; i < entryCount; i++)
        {
            entries[i] = new BlockEntry
            {
                BlockOffset = encryptedData[i * 4 + 0],
                BlockSize = encryptedData[i * 4 + 1],
                FileSize = encryptedData[i * 4 + 2],
                Flags = encryptedData[i * 4 + 3]
            };
        }
        
        return entries;
    }
    
    private static byte[]? ReadFileData(BinaryReader reader, BlockEntry block, uint sectorSize, string filename, long fileBaseOffset)
    {
        // Calculate encryption key if needed
        uint key = 0;
        if ((block.Flags & FLAG_ENCRYPTED) != 0)
        {
            key = HashString(Path.GetFileName(filename), HASH_FILE_KEY);
            if ((block.Flags & FLAG_FIX_KEY) != 0)
                key = (key + block.BlockOffset) ^ block.FileSize;
        }

        // Single unit files: no sector offset table, read directly
        if ((block.Flags & FLAG_SINGLE_UNIT) != 0)
        {
            var data = reader.ReadBytes((int)block.BlockSize);
            
            if ((block.Flags & FLAG_ENCRYPTED) != 0)
                DecryptData(data, key);
            
            if ((block.Flags & FLAG_COMPRESSED) != 0 && block.BlockSize < block.FileSize)
            {
                return DecompressData(data, block.FileSize);
            }
            
            return data;
        }
        
        // Multi-sector file
        uint sectorCount = (block.FileSize + sectorSize - 1) / sectorSize;
        
        // Read sector offsets
        // If encrypted, sector offsets are encrypted with key-1
        var offsetBytes = reader.ReadBytes(4 * ((int)sectorCount + 1));
        if ((block.Flags & FLAG_ENCRYPTED) != 0)
            DecryptData(offsetBytes, key - 1);
            
        var sectorOffsets = new uint[sectorCount + 1];
        Buffer.BlockCopy(offsetBytes, 0, sectorOffsets, 0, offsetBytes.Length);

        // Sanity checks for malformed sector tables.
        if (sectorOffsets.Length == 0)
        {
            Build335Diagnostics.Increment("MpqSectorTableInvalidCount");
            return null;
        }

        uint previous = sectorOffsets[0];
        if (previous > block.BlockSize)
        {
            Build335Diagnostics.Increment("MpqSectorTableInvalidCount");
            return null;
        }

        for (int si = 1; si < sectorOffsets.Length; si++)
        {
            uint current = sectorOffsets[si];
            if (current < previous || current > block.BlockSize)
            {
                Build335Diagnostics.Increment("MpqSectorTableInvalidCount");
                return null;
            }
            previous = current;
        }
        
        using var output = new MemoryStream();
        
        for (uint i = 0; i < sectorCount; i++)
        {
            uint sectorStart = sectorOffsets[i];
            uint sectorEnd = sectorOffsets[i + 1];
            if (sectorEnd < sectorStart)
            {
                Build335Diagnostics.Increment("MpqSectorTableInvalidCount");
                return null;
            }
            uint compressedSize = sectorEnd - sectorStart;
            uint uncompressedSize = Math.Min(sectorSize, block.FileSize - (i * sectorSize));

            if (fileBaseOffset + sectorStart > reader.BaseStream.Length ||
                fileBaseOffset + sectorEnd > reader.BaseStream.Length)
            {
                Build335Diagnostics.Increment("MpqSectorTableInvalidCount");
                return null;
            }
            
            // Seek to exact sector location (skipping potential CRC tables or gaps)
            reader.BaseStream.Position = fileBaseOffset + sectorStart;
            
            var sectorData = reader.ReadBytes((int)compressedSize);
            
            if ((block.Flags & FLAG_ENCRYPTED) != 0)
                DecryptData(sectorData, key + i); // Sector key is key + index
            
            if ((block.Flags & FLAG_COMPRESSED) != 0 && compressedSize < uncompressedSize)
            {
                // Decompression
                var decompressed = DecompressData(sectorData, uncompressedSize);
                if (decompressed != null)
                    output.Write(decompressed, 0, decompressed.Length);
                else
                    output.Write(sectorData, 0, sectorData.Length); // Fallback
            }
            else
            {
                // Stored
                output.Write(sectorData, 0, sectorData.Length);
            }
        }
        
        return output.ToArray();
    }
    
    private static void DecryptData(byte[] data, uint key)
    {
        uint seed = 0xEEEEEEEE;
        int numBlocks = data.Length / 4;
        
        // We need to operate on uints.
        // Unsafe or BitConverter?
        // Let's use BitConverter for safety, or unsafe for speed.
        // Given existing project style, unsafe might not be enabled.
        // Use BinaryReader/Writer on MemoryStream over the array? Overhead.
        // Use manual byte manipulation.
        
        for (int i = 0; i < numBlocks; i++)
        {
            int offset = i * 4;
            // Read uint
            uint v = (uint)(data[offset] | (data[offset+1] << 8) | (data[offset+2] << 16) | (data[offset+3] << 24));
            
            seed += CryptTable[0x400 + (key & 0xFF)];
            uint temp = v ^ (key + seed);
            
            key = ((~key << 0x15) + 0x11111111) | (key >> 0x0B);
            seed = temp + seed + (seed << 5) + 3;
            
            // Write uint
            data[offset] = (byte)(temp & 0xFF);
            data[offset+1] = (byte)((temp >> 8) & 0xFF);
            data[offset+2] = (byte)((temp >> 16) & 0xFF);
            data[offset+3] = (byte)((temp >> 24) & 0xFF);
        }
    }
    
    private static byte[]? DecompressData(byte[] data, uint expectedSize)
    {
        if (data.Length == 0)
            return null;
        
        byte compressionType = data[0];
        var compressedData = new byte[data.Length - 1];
        Array.Copy(data, 1, compressedData, 0, compressedData.Length);
        
        switch (compressionType)
        {
            case 0x02: // ZLIB
                return DecompressZlib(compressedData);
            
            case 0x08: // PKWARE DCL (implode)
                Console.WriteLine($"[NativeMpqService] PKWARE: first bytes after type: {(compressedData.Length >= 8 ? BitConverter.ToString(compressedData, 0, Math.Min(16, compressedData.Length)) : "too short")} ({compressedData.Length} bytes → {expectedSize})");
                return PkwareExplode.Decompress(compressedData, expectedSize);
            
            case 0x10: // BZip2
                Console.WriteLine($"[NativeMpqService] BZip2 compression not implemented ({data.Length} bytes)");
                return null;
            
            case 0x12: // ZLIB + PKWARE combo (seen in some MPQs)
                Console.WriteLine($"[NativeMpqService] ZLIB+PKWARE combo compression ({data.Length} bytes)");
                return DecompressZlib(compressedData);
            
            default:
                Console.WriteLine($"[NativeMpqService] Unknown compression type 0x{compressionType:X2} ({data.Length} bytes)");
                return data;
        }
    }

    
    private static byte[]? DecompressZlib(byte[] data)
    {
        try
        {
            using var input = new MemoryStream(data);
            input.ReadByte(); // Skip zlib header
            input.ReadByte();
            
            using var deflate = new DeflateStream(input, CompressionMode.Decompress);
            using var output = new MemoryStream();
            deflate.CopyTo(output);
            return output.ToArray();
        }
        catch
        {
            try
            {
                using var input = new MemoryStream(data);
                using var deflate = new DeflateStream(input, CompressionMode.Decompress);
                using var output = new MemoryStream();
                deflate.CopyTo(output);
                return output.ToArray();
            }
            catch
            {
                return null;
            }
        }
    }
    
    private static uint HashString(string str, uint hashType)
    {
        uint seed1 = 0x7FED7FED;
        uint seed2 = 0xEEEEEEEE;
        
        foreach (char c in str.ToUpperInvariant())
        {
            uint ch = (byte)c;
            seed1 = CryptTable[hashType * 0x100 + ch] ^ (seed1 + seed2);
            seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
        }
        
        return seed1;
    }
    
    private static void DecryptBlock(uint[] data, uint key)
    {
        uint seed = 0xEEEEEEEE;
        
        for (int i = 0; i < data.Length; i++)
        {
            seed += CryptTable[0x400 + (key & 0xFF)];
            uint temp = data[i] ^ (key + seed);
            
            key = ((~key << 0x15) + 0x11111111) | (key >> 0x0B);
            seed = temp + seed + (seed << 5) + 3;
            
            data[i] = temp;
        }
    }
    
    private static uint[] BuildCryptTable()
    {
        var table = new uint[0x500];
        uint seed = 0x00100001;
        
        for (uint i = 0; i < 0x100; i++)
        {
            uint index = i;
            for (int j = 0; j < 5; j++, index += 0x100)
            {
                seed = (seed * 125 + 3) % 0x2AAAAB;
                uint temp1 = (seed & 0xFFFF) << 0x10;
                
                seed = (seed * 125 + 3) % 0x2AAAAB;
                uint temp2 = seed & 0xFFFF;
                
                table[index] = temp1 | temp2;
            }
        }
        
        return table;
    }
    
    #endregion
    
    public void Dispose()
    {
        if (_disposed) return;
        _archives.Clear();
        _disposed = true;
    }
    
    #region Data Structures
    
    private class MpqArchive
    {
        public string Path = "";
        public long HeaderOffset;
        public MpqHeader Header = new();
        public HashEntry[] HashTable = Array.Empty<HashEntry>();
        public BlockEntry[] BlockTable = Array.Empty<BlockEntry>();
    }
    
    private class MpqHeader
    {
        public uint HeaderSize;
        public uint ArchiveSize;
        public ushort FormatVersion;
        public ushort SectorSizeShift;
        public uint HashTableOffset;
        public uint BlockTableOffset;
        public uint HashTableEntries;
        public uint BlockTableEntries;
        public uint SectorSize;
    }
    
    private class BlockEntry
    {
        public uint BlockOffset;
        public uint BlockSize;
        public uint FileSize;
        public uint Flags;
    }
    
    private class HashEntry
    {
        public uint Name1;
        public uint Name2;
        public ushort Locale;
        public ushort Platform;
        public uint BlockIndex;
    }
    
    #endregion
}
