using System;
using System.IO;
using System.IO.Compression;
using System.Text;
using System.Linq;

namespace WoWMapConverter.Core.Services;

/// <summary>
/// Pure C# implementation for reading Alpha 0.5.3 per-asset MPQ archives.
/// These are simple MPQ v1 archives with 2 files: data and MD5 checksum.
/// StormLib has issues with these archives, so we implement our own reader.
/// </summary>
public static class AlphaMpqReader
{
    // MPQ hash type constants
    private const uint HASH_TABLE_INDEX = 0;
    private const uint HASH_NAME_A = 1;
    private const uint HASH_NAME_B = 2;
    private const uint HASH_FILE_KEY = 3;
    
    // Encryption table for MPQ hash/decrypt
    private static readonly uint[] CryptTable = BuildCryptTable();
    
    /// <summary>
    /// Read the main data file from an Alpha per-asset MPQ archive.
    /// Returns null if the archive can't be read.
    /// </summary>
    public static byte[]? ReadFromMpq(string mpqPath)
    {
        return ReadFromMpq(mpqPath, Enumerable.Empty<string>());
    }

    public static byte[]? ReadFromMpq(string mpqPath, IEnumerable<string> internalNames)
    {
        if (!File.Exists(mpqPath))
        {
            Console.WriteLine($"[WARN] MPQ file not found: {mpqPath}");
            return null;
        }
        
        try
        {
            using var fs = new FileStream(mpqPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var reader = new BinaryReader(fs);
            
            // Find MPQ header (can be at offset 0 or at 512-byte boundary)
            long headerOffset = FindMpqHeader(reader);
            if (headerOffset < 0)
            {
                Console.WriteLine($"[WARN] No MPQ header found in: {mpqPath}");
                return null;
            }
            
            fs.Position = headerOffset;
            
            // Read MPQ header (v1 = 32 bytes)
            var header = ReadMpqHeader(reader);
            if (header == null)
            {
                Console.WriteLine($"[WARN] Invalid MPQ header in: {mpqPath}");
                return null;
            }
            
            // Read and decrypt block table
            fs.Position = headerOffset + header.BlockTableOffset;
            var blockTable = ReadBlockTable(reader, header.BlockTableEntries);

            // Read and decrypt hash table (listfile-less MPQs still use it)
            fs.Position = headerOffset + header.HashTableOffset;
            var hashTable = ReadHashTable(reader, header.HashTableEntries);

            // Prefer hash lookup when internal name is known, fallback to first valid/large block
            var primaryBlock = TryGetPrimaryBlock(blockTable, hashTable, internalNames);

            if (primaryBlock == null)
            {
                Console.WriteLine($"[WARN] No valid blocks in MPQ: {mpqPath}");
                return null;
            }

            // Read and decompress the file data
            fs.Position = headerOffset + primaryBlock.BlockOffset;
            var primaryData = ReadFileData(reader, primaryBlock, header.SectorSize);
            if (primaryData != null && primaryData.Length >= 8)
            {
                string magic4 = primaryData.Length >= 4 
                    ? $"{(char)primaryData[0]}{(char)primaryData[1]}{(char)primaryData[2]}{(char)primaryData[3]}" 
                    : "???";
                Console.WriteLine($"[AlphaMpqReader] Primary block: {primaryBlock.FileSize} bytes, magic='{magic4}', valid={IsLikelyWdtOrWdl(primaryData)}");
            }

            if (IsLikelyWdtOrWdl(primaryData))
                return primaryData;

            // Primary block wasn't the data file — scan all blocks for one with valid magic
            foreach (var block in blockTable)
            {
                if (block == primaryBlock || block.FileSize == 0) continue;
                fs.Position = headerOffset + block.BlockOffset;
                var blockData = ReadFileData(reader, block, header.SectorSize);
                if (IsLikelyWdtOrWdl(blockData))
                {
                    Console.WriteLine($"[AlphaMpqReader] Found valid data in alternate block: {block.FileSize} bytes");
                    return blockData;
                }
            }

            // No block had valid magic — return the largest block as best guess
            BlockEntry? largestBlock = null;
            foreach (var block in blockTable)
            {
                if (block.FileSize > 0 && (largestBlock == null || block.FileSize > largestBlock.FileSize))
                    largestBlock = block;
            }
            if (largestBlock != null && largestBlock != primaryBlock)
            {
                fs.Position = headerOffset + largestBlock.BlockOffset;
                var largestData = ReadFileData(reader, largestBlock, header.SectorSize);
                Console.WriteLine($"[AlphaMpqReader] Falling back to largest block: {largestBlock.FileSize} bytes");
                return largestData;
            }

            return primaryData;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WARN] Error reading MPQ {mpqPath}: {ex.Message}");
            return null;
        }
    }
    
    /// <summary>
    /// Try to read a file, falling back to MPQ version if not found.
    /// </summary>
    public static byte[]? ReadWithMpqFallback(string filePath)
    {
        // If path is already an MPQ archive, extract from it
        if (filePath.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
            return ReadFromMpq(filePath);
        
        // Try direct file first (non-MPQ)
        if (File.Exists(filePath))
            return File.ReadAllBytes(filePath);
        
        // Try MPQ version (e.g., castle01.wmo -> castle01.wmo.MPQ)
        var mpqPath = filePath + ".MPQ";
        if (File.Exists(mpqPath))
        {
            Console.WriteLine($"[AlphaMpqReader] Found MPQ: {mpqPath}");
            var candidates = BuildInternalNameCandidates(filePath).ToList();
            Console.WriteLine($"[AlphaMpqReader] Internal name candidates: {string.Join(", ", candidates)}");
            var data = ReadFromMpq(mpqPath, candidates);
            Console.WriteLine($"[AlphaMpqReader] ReadFromMpq result: {(data != null ? $"{data.Length} bytes" : "null")}");
            return data;
        }
        
        // Try lowercase mpq extension
        mpqPath = filePath + ".mpq";
        if (File.Exists(mpqPath))
        {
            Console.WriteLine($"[AlphaMpqReader] Found MPQ (lowercase): {mpqPath}");
            var candidates = BuildInternalNameCandidates(filePath).ToList();
            Console.WriteLine($"[AlphaMpqReader] Internal name candidates: {string.Join(", ", candidates)}");
            var data = ReadFromMpq(mpqPath, candidates);
            Console.WriteLine($"[AlphaMpqReader] ReadFromMpq result: {(data != null ? $"{data.Length} bytes" : "null")}");
            return data;
        }
        
        return null;
    }
    
    private static long FindMpqHeader(BinaryReader reader)
    {
        // Check offset 0
        if (reader.BaseStream.Length >= 4)
        {
            reader.BaseStream.Position = 0;
            uint magic = reader.ReadUInt32();
            if (magic == 0x1A51504D) // "MPQ\x1A" little-endian
                return 0;
        }
        
        // Search at 512-byte boundaries
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
    
    private static BlockEntry[] ReadBlockTable(BinaryReader reader, uint entryCount)
    {
        // Read encrypted block table
        var encryptedData = new uint[entryCount * 4];
        for (int i = 0; i < encryptedData.Length; i++)
            encryptedData[i] = reader.ReadUInt32();
        
        // Decrypt block table using key = Hash("(block table)", HASH_FILE_KEY)
        uint key = HashString("(block table)", HASH_FILE_KEY);
        DecryptBlock(encryptedData, key);
        
        // Parse entries
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

    private static BlockEntry? TryGetPrimaryBlock(BlockEntry[] blockTable, HashEntry[] hashTable, IEnumerable<string> internalNames)
    {
        const uint HashEntryDeleted = 0xFFFFFFFE;
        const uint HashEntryEmpty = 0xFFFFFFFF;

        foreach (var candidate in internalNames)
        {
            if (string.IsNullOrWhiteSpace(candidate)) continue;
            var normalized = candidate.Replace('/', '\\');
            var hashIndex = HashString(normalized, HASH_TABLE_INDEX) % (uint)hashTable.Length;
            var nameA = HashString(normalized, HASH_NAME_A);
            var nameB = HashString(normalized, HASH_NAME_B);

            for (uint i = 0; i < hashTable.Length; i++)
            {
                var entry = hashTable[(hashIndex + i) % hashTable.Length];
                if (entry.BlockIndex == HashEntryEmpty)
                    break;

                if (entry.BlockIndex == HashEntryDeleted)
                    continue;

                if (entry.Name1 == nameA && entry.Name2 == nameB && entry.BlockIndex < blockTable.Length)
                {
                    var block = blockTable[entry.BlockIndex];
                    if (block.FileSize > 0)
                        return block;
                }
            }
        }

        // No name match — fall back to the largest block (actual data, not MD5 checksum)
        BlockEntry? largestBlock = null;
        foreach (var block in blockTable)
        {
            if (block.FileSize > 0 && (largestBlock == null || block.FileSize > largestBlock.FileSize))
                largestBlock = block;
        }

        return largestBlock;
    }

    private static bool IsLikelyWdtOrWdl(byte[]? data)
    {
        if (data == null || data.Length < 8) return false;

        // WDT/WDL start with MVER chunk (forward: "MVER")
        if (data[0] == (byte)'M' && data[1] == (byte)'V' && data[2] == (byte)'E' && data[3] == (byte)'R')
            return true;

        // WMO v14 starts with reversed MVER ("REVM")
        if (data[0] == (byte)'R' && data[1] == (byte)'E' && data[2] == (byte)'V' && data[3] == (byte)'M')
            return true;

        return false;
    }

    public static IEnumerable<string> BuildInternalNameCandidates(string filePath)
    {
        var fileName = Path.GetFileName(filePath);
        if (string.IsNullOrEmpty(fileName))
            yield break;

        yield return fileName;

        var markers = new[] { "World", "Maps" };
        var parts = filePath.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var idx = Array.FindIndex(parts, part => string.Equals(part, markers[0], StringComparison.OrdinalIgnoreCase));
        if (idx >= 0 && idx + 2 < parts.Length && string.Equals(parts[idx + 1], markers[1], StringComparison.OrdinalIgnoreCase))
        {
            var relative = string.Join('\\', parts.Skip(idx));
            yield return relative;

            var mapName = parts.ElementAtOrDefault(idx + 2);
            if (!string.IsNullOrWhiteSpace(mapName))
            {
                yield return $"{mapName}\\{fileName}";
                yield return $"World\\Maps\\{mapName}\\{fileName}";
            }
        }
        else
        {
            var mapName = Path.GetFileNameWithoutExtension(fileName);
            if (!string.IsNullOrWhiteSpace(mapName))
            {
                yield return $"{mapName}\\{fileName}";
                yield return $"World\\Maps\\{mapName}\\{fileName}";
            }
        }
    }
    
    private static byte[]? ReadFileData(BinaryReader reader, BlockEntry block, uint sectorSize)
    {
        const uint FLAG_COMPRESSED = 0x00000200;
        const uint FLAG_IMPLODED = 0x00000100;
        const uint FLAG_SINGLE_UNIT = 0x01000000;
        
        // Single unit files: no sector offset table, read directly
        // NOTE: Do NOT use `fileSize <= sectorSize` shortcut here — without FLAG_SINGLE_UNIT,
        // even small files have a sector offset table that must be read first.
        if ((block.Flags & FLAG_SINGLE_UNIT) != 0)
        {
            var data = reader.ReadBytes((int)block.BlockSize);
            
            if ((block.Flags & FLAG_COMPRESSED) != 0 && block.BlockSize < block.FileSize)
            {
                return DecompressData(data, block.FileSize);
            }
            
            return data;
        }
        
        // Multi-sector file - read sector offset table and decompress each sector
        uint sectorCount = (block.FileSize + sectorSize - 1) / sectorSize;
        var sectorOffsets = new uint[sectorCount + 1];
        
        for (uint i = 0; i <= sectorCount; i++)
            sectorOffsets[i] = reader.ReadUInt32();
        
        using var output = new MemoryStream();
        
        for (uint i = 0; i < sectorCount; i++)
        {
            uint sectorStart = sectorOffsets[i];
            uint sectorEnd = sectorOffsets[i + 1];
            uint compressedSize = sectorEnd - sectorStart;
            uint uncompressedSize = Math.Min(sectorSize, block.FileSize - (i * sectorSize));
            
            var sectorData = reader.ReadBytes((int)compressedSize);
            
            if ((block.Flags & FLAG_COMPRESSED) != 0 && compressedSize < uncompressedSize)
            {
                var decompressed = DecompressData(sectorData, uncompressedSize);
                if (decompressed != null)
                    output.Write(decompressed, 0, decompressed.Length);
            }
            else
            {
                output.Write(sectorData, 0, sectorData.Length);
            }
        }
        
        return output.ToArray();
    }
    
    private static byte[]? DecompressData(byte[] data, uint expectedSize)
    {
        if (data.Length == 0)
            return null;
        
        // First byte indicates compression type
        byte compressionType = data[0];
        var compressedData = new byte[data.Length - 1];
        Array.Copy(data, 1, compressedData, 0, compressedData.Length);
        
        switch (compressionType)
        {
            case 0x02: // ZLIB
                return DecompressZlib(compressedData, expectedSize);
            
            case 0x08: // PKWARE DCL (implode)
                return PkwareExplode.Decompress(compressedData, expectedSize);
            
            default:
                // Unknown compression, return raw
                return data;
        }
    }
    
    private static byte[]? DecompressZlib(byte[] data, uint expectedSize)
    {
        try
        {
            using var input = new MemoryStream(data);
            // Skip zlib header (2 bytes)
            input.ReadByte();
            input.ReadByte();
            
            using var deflate = new DeflateStream(input, CompressionMode.Decompress);
            using var output = new MemoryStream();
            deflate.CopyTo(output);
            return output.ToArray();
        }
        catch
        {
            // Try without skipping header
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
}
