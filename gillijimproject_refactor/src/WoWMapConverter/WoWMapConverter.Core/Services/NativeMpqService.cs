using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text;
using System.Linq;

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
    /// Scans all blocks in loaded MPQ archives for WMO files (identified by 'MOHD' magic).
    /// Useful for listfile-less MPQs containing Alpha WMO files.
    /// </summary>
    /// <returns>List of virtual paths for found WMO files.</returns>
    public List<string> ScanForWmoFiles()
    {
        var foundWmos = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        
        foreach (var archive in _archives)
        {
            foreach (var block in archive.BlockTable)
            {
                if (block.FileSize < 16) continue; // Too small for WMO header
                
                // Read the magic number from this block
                try
                {
                    using var fs = new FileStream(archive.Path, FileMode.Open, FileAccess.Read, FileShare.Read);
                    fs.Position = archive.HeaderOffset + block.BlockOffset;
                    var magicBytes = new byte[4];
                    if (fs.Read(magicBytes, 0, 4) != 4) continue;
                    
                    // Check for 'MOHD' (WMO root) magic
                    if (magicBytes[0] == 'M' && magicBytes[1] == 'O' && magicBytes[2] == 'H' && magicBytes[3] == 'D')
                    {
                        // Found a WMO file - generate a placeholder path
                        var wmoPath = $"WMO_{block.BlockOffset:X8}.wmo";
                        foundWmos.Add(wmoPath);
                        
                        // Store the mapping for later reading
                        if (!_scannedFiles.ContainsKey(wmoPath))
                        {
                            _scannedFiles[wmoPath] = (archive.Path, block.BlockOffset);
                        }
                    }
                }
                catch { continue; }
            }
        }
        
        Console.WriteLine($"[NativeMpqService] Scanned {foundWmos.Count} WMO files in listfile-less MPQs.");
        return foundWmos.ToList();
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
        
        // Search archives in reverse order (patches first)
        for (int i = _archives.Count - 1; i >= 0; i--)
        {
            var archive = _archives[i];
            var block = FindFileInArchive(archive, normalized);
            
            if (block != null)
            {
                // Check if file is deleted (0 size in patches)
                if (block.FileSize == 0)
                    continue; // File was "patched out", try base archives
                
                var data = ReadFileFromArchive(archive, block, Path.GetFileName(normalized));
                if (data != null && data.Length > 0)
                    return data;
            }
        }
        
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
        catch
        {
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

        // Single unit (no sectors)
        if ((block.Flags & FLAG_SINGLE_UNIT) != 0 || block.FileSize <= sectorSize)
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
        
        using var output = new MemoryStream();
        
        for (uint i = 0; i < sectorCount; i++)
        {
            uint sectorStart = sectorOffsets[i];
            uint sectorEnd = sectorOffsets[i + 1];
            uint compressedSize = sectorEnd - sectorStart;
            uint uncompressedSize = Math.Min(sectorSize, block.FileSize - (i * sectorSize));
            
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
            
            case 0x08: // PKWARE DCL
                // TODO: Implement DCL decompression
                return null;
            
            case 0x10: // BZip2
                // TODO: Implement BZip2
                return null;
            
            default:
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
