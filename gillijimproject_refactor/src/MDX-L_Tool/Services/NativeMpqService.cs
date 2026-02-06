using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text;
using System.Linq;

namespace MdxLTool.Services;

/// <summary>
/// Pure C# MPQ archive service (ported from WoWMapConverter.Core/NativeMpqService.cs).
/// Handles multi-file archives and archive patching logic.
/// </summary>
public class NativeMpqService : IDisposable
{
    private readonly List<MpqArchive> _archives = new();
    
    // Listfile support
    private readonly HashSet<ulong> _knownFileHashes = new();
    private readonly Dictionary<ulong, string> _hashToName = new();

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
        
        var allMpqFiles = new List<string>();
        foreach (var path in pathsToSearch)
        {
            var mpqFiles = Directory.GetFiles(path, "*.mpq", SearchOption.TopDirectoryOnly);
            allMpqFiles.AddRange(mpqFiles);
            
            var mpqFilesUpper = Directory.GetFiles(path, "*.MPQ", SearchOption.TopDirectoryOnly);
            allMpqFiles.AddRange(mpqFilesUpper);
        }
        
        allMpqFiles = allMpqFiles.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        
        allMpqFiles = allMpqFiles
            .OrderBy(f => GetMpqPriority(Path.GetFileName(f)))
            .ToList();
        
        foreach (var mpqPath in allMpqFiles)
        {
            try
            {
                var archive = LoadArchive(mpqPath);
                if (archive != null)
                {
                    _archives.Add(archive);
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[ERROR] Failed to load archive {mpqPath}: {ex.Message}");
            }
        }
    }
    
    private static int GetMpqPriority(string filename)
    {
        var lower = filename.ToLowerInvariant();
        
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
        for (int i = _archives.Count - 1; i >= 0; i--)
        {
            if (FindFileInArchive(_archives[i], normalized) != null)
                return true;
        }
        return false;
    }
    
    public byte[]? ReadFile(string virtualPath)
    {
        var normalized = virtualPath.Replace('/', '\\');
        for (int i = _archives.Count - 1; i >= 0; i--)
        {
            var archive = _archives[i];
            var block = FindFileInArchive(archive, normalized);
            
            if (block != null)
            {
                if (block.FileSize == 0)
                    continue; 
                
                var data = ReadFileFromArchive(archive, block, Path.GetFileName(normalized));
                if (data != null && data.Length > 0)
                    return data;
            }
        }
        return null;
    }
    
    private MpqArchive? LoadArchive(string mpqPath)
    {
        using var fs = new FileStream(mpqPath, FileMode.Open, FileAccess.Read, FileShare.Read);
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
        var normalized = filename.Replace('/', '\\');
        var hashIndex = HashString(normalized, HASH_TABLE_INDEX) % (uint)archive.HashTable.Length;
        var nameA = HashString(normalized, HASH_NAME_A);
        var nameB = HashString(normalized, HASH_NAME_B);
        
        for (uint i = 0; i < archive.HashTable.Length; i++)
        {
            var entry = archive.HashTable[(hashIndex + i) % archive.HashTable.Length];
            if (entry.BlockIndex == HASH_ENTRY_EMPTY) break;
            if (entry.BlockIndex == HASH_ENTRY_DELETED) continue;
            
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
        catch { return null; }
    }
    
    private static long FindMpqHeader(BinaryReader reader)
    {
        if (reader.BaseStream.Length >= 4)
        {
            reader.BaseStream.Position = 0;
            if (reader.ReadUInt32() == 0x1A51504D) return 0;
        }
        for (long offset = 0x200; offset < reader.BaseStream.Length && offset < 0x100000; offset += 0x200)
        {
            reader.BaseStream.Position = offset;
            if (reader.BaseStream.Length - offset < 4) break;
            if (reader.ReadUInt32() == 0x1A51504D) return offset;
        }
        return -1;
    }
    
    private static MpqHeader? ReadMpqHeader(BinaryReader reader)
    {
        uint magic = reader.ReadUInt32();
        if (magic != 0x1A51504D) return null;
        
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
        for (int i = 0; i < encryptedData.Length; i++) encryptedData[i] = reader.ReadUInt32();
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
        for (int i = 0; i < encryptedData.Length; i++) encryptedData[i] = reader.ReadUInt32();
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
        uint key = 0;
        if ((block.Flags & FLAG_ENCRYPTED) != 0)
        {
            key = HashString(Path.GetFileName(filename), HASH_FILE_KEY);
            if ((block.Flags & FLAG_FIX_KEY) != 0) key = (key + block.BlockOffset) ^ block.FileSize;
        }

        if ((block.Flags & FLAG_SINGLE_UNIT) != 0 || block.FileSize <= sectorSize)
        {
            var data = reader.ReadBytes((int)block.BlockSize);
            if ((block.Flags & FLAG_ENCRYPTED) != 0) DecryptData(data, key);
            if ((block.Flags & FLAG_COMPRESSED) != 0 && block.BlockSize < block.FileSize) return DecompressData(data, block.FileSize);
            return data;
        }
        
        uint sectorCount = (block.FileSize + sectorSize - 1) / sectorSize;
        var offsetBytes = reader.ReadBytes(4 * ((int)sectorCount + 1));
        if ((block.Flags & FLAG_ENCRYPTED) != 0) DecryptData(offsetBytes, key - 1);
        var sectorOffsets = new uint[sectorCount + 1];
        Buffer.BlockCopy(offsetBytes, 0, sectorOffsets, 0, offsetBytes.Length);
        
        using var output = new MemoryStream();
        for (uint i = 0; i < sectorCount; i++)
        {
            uint sectorStart = sectorOffsets[i];
            uint sectorEnd = sectorOffsets[i + 1];
            uint compressedSize = sectorEnd - sectorStart;
            uint uncompressedSize = Math.Min(sectorSize, block.FileSize - (i * sectorSize));
            reader.BaseStream.Position = fileBaseOffset + sectorStart;
            var sectorData = reader.ReadBytes((int)compressedSize);
            if ((block.Flags & FLAG_ENCRYPTED) != 0) DecryptData(sectorData, key + i);
            if ((block.Flags & FLAG_COMPRESSED) != 0 && compressedSize < uncompressedSize)
            {
                var decompressed = DecompressData(sectorData, uncompressedSize);
                output.Write(decompressed ?? sectorData, 0, (decompressed ?? sectorData).Length);
            }
            else output.Write(sectorData, 0, sectorData.Length);
        }
        return output.ToArray();
    }
    
    private static void DecryptData(byte[] data, uint key)
    {
        uint seed = 0xEEEEEEEE;
        int numBlocks = data.Length / 4;
        for (int i = 0; i < numBlocks; i++)
        {
            int offset = i * 4;
            uint v = (uint)(data[offset] | (data[offset+1] << 8) | (data[offset+2] << 16) | (data[offset+3] << 24));
            seed += CryptTable[0x400 + (key & 0xFF)];
            uint temp = v ^ (key + seed);
            key = ((~key << 0x15) + 0x11111111) | (key >> 0x0B);
            seed = temp + seed + (seed << 5) + 3;
            data[offset] = (byte)(temp & 0xFF);
            data[offset+1] = (byte)((temp >> 8) & 0xFF);
            data[offset+2] = (byte)((temp >> 16) & 0xFF);
            data[offset+3] = (byte)((temp >> 24) & 0xFF);
        }
    }
    
    private static byte[]? DecompressData(byte[] data, uint expectedSize)
    {
        if (data.Length == 0) return null;
        byte compressionType = data[0];
        var compressedData = new byte[data.Length - 1];
        Array.Copy(data, 1, compressedData, 0, compressedData.Length);
        switch (compressionType)
        {
            case 0x02: return DecompressZlib(compressedData);
            default: return data;
        }
    }
    
    private static byte[]? DecompressZlib(byte[] data)
    {
        try
        {
            using var input = new MemoryStream(data);
            input.ReadByte(); input.ReadByte();
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
            catch { return null; }
        }
    }
    
    private static uint HashString(string str, uint hashType)
    {
        uint seed1 = 0x7FED7FED, seed2 = 0xEEEEEEEE;
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
    
    public void Dispose()
    {
        if (_disposed) return;
        _archives.Clear();
        _disposed = true;
    }
    
    private class MpqArchive
    {
        public string Path = "";
        public long HeaderOffset;
        public MpqHeader Header = new();
        public HashEntry[] HashTable = Array.Empty<HashEntry>();
        public BlockEntry[] BlockTable = Array.Empty<BlockEntry>();
    }
    
    private class MpqHeader { public uint HeaderSize, ArchiveSize, HashTableOffset, BlockTableOffset, HashTableEntries, BlockTableEntries, SectorSize; public ushort FormatVersion, SectorSizeShift; }
    private class BlockEntry { public uint BlockOffset, BlockSize, FileSize, Flags; }
    private class HashEntry { public uint Name1, Name2, BlockIndex; public ushort Locale, Platform; }
}
