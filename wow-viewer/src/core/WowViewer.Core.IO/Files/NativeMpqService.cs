using System.Collections.Concurrent;
using System.IO;
using System.IO.Compression;
using System.Text;

namespace WowViewer.Core.IO.Files;

public sealed class NativeMpqServiceFactory : IArchiveCatalogFactory
{
    public IArchiveCatalog Create() => new NativeMpqService();
}

public sealed class NativeMpqService : IArchiveCatalog
{
    private readonly List<MpqArchive> _archives = [];
    private readonly HashSet<ulong> _knownFileHashes = [];
    private readonly Dictionary<ulong, string> _hashToName = [];
    private readonly Dictionary<string, string> _scannedArchives = new(StringComparer.OrdinalIgnoreCase);
    private bool _disposed;

    private const uint HashTableIndex = 0;
    private const uint HashNameA = 1;
    private const uint HashNameB = 2;
    private const uint HashFileKey = 3;

    private const uint HashEntryDeleted = 0xFFFFFFFE;
    private const uint HashEntryEmpty = 0xFFFFFFFF;

    private const uint FlagCompressed = 0x00000200;
    private const uint FlagEncrypted = 0x00010000;
    private const uint FlagFixKey = 0x00020000;
    private const uint FlagImploded = 0x00000100;
    private const uint FlagSingleUnit = 0x01000000;
    private const uint FlagExists = 0x80000000;

    private static readonly uint[] CryptTable = BuildCryptTable();

    public void LoadArchives(IEnumerable<string> searchPaths)
    {
        var pathsToSearch = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

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
            allMpqFiles.AddRange(Directory.GetFiles(path, "*.mpq", SearchOption.TopDirectoryOnly));
            allMpqFiles.AddRange(Directory.GetFiles(path, "*.MPQ", SearchOption.TopDirectoryOnly));
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
                    _archives.Add(archive);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[ERROR] Failed to load archive {mpqPath}: {ex.Message}");
            }
        }

        foreach (string path in pathsToSearch)
        {
            if (Directory.Exists(path))
                ScanMapMpqArchives(path);

            string mapsDir = Path.Combine(path, "World", "Maps");
            if (Directory.Exists(mapsDir))
                ScanMapMpqArchives(mapsDir);
        }
    }

    private void ScanMapMpqArchives(string mapsDir)
    {
        if (!Directory.Exists(mapsDir)) return;

        try
        {
            string[] patterns = ["*.wdt.mpq", "*.WDT.MPQ", "*.wdl.mpq", "*.WDL.MPQ"];
            foreach (string pattern in patterns)
            {
                foreach (string file in Directory.GetFiles(mapsDir, pattern, SearchOption.AllDirectories))
                {
                    string? virtualPath = TryBuildScannedVirtualPath(file);
                    if (virtualPath != null)
                        _scannedArchives.TryAdd(virtualPath, file);
                }
            }
        }
        catch
        {
        }
    }

    private static string? TryBuildScannedVirtualPath(string mpqPath)
    {
        int mapsIndex = mpqPath.IndexOf("World\\Maps\\", StringComparison.OrdinalIgnoreCase);
        if (mapsIndex < 0)
            mapsIndex = mpqPath.IndexOf("World/Maps/", StringComparison.OrdinalIgnoreCase);

        if (mapsIndex < 0) return null;

        string relative = mpqPath[mapsIndex..];
        string virtualPath = relative.Replace('/', '\\').Replace(".mpq", "", StringComparison.OrdinalIgnoreCase);

        return virtualPath.TrimStart('\\');
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

        if (_scannedArchives.ContainsKey(normalized))
            return true;

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

        if (_scannedArchives.TryGetValue(normalized, out string? archivePath))
        {
            byte[]? data = ReadFromScannedArchive(archivePath);
            if (data != null) return data;
        }

        for (int i = _archives.Count - 1; i >= 0; i--)
        {
            var archive = _archives[i];
            var block = FindFileInArchive(archive, normalized);

            if (block != null)
            {
                if (block.FileSize == 0) continue;

                var data = ReadFileFromArchive(archive, block, Path.GetFileName(normalized));
                if (data != null && data.Length > 0)
                    return data;
            }
        }
        return null;
    }

    public List<string> ListFiles(string pattern = "*")
    {
        var all = ExtractInternalListfiles();
        if (pattern == "*" || pattern == "*.*") return all.ToList();
        return all.Where(f => DoesMatchPattern(f, pattern)).ToList();
    }

    public void LoadListfile(string path)
    {
        if (!File.Exists(path)) return;
        LoadListfileEntries(File.ReadLines(path));
    }

    public void LoadListfileEntries(IEnumerable<string> entries)
    {
        foreach (string rawLine in entries)
        {
            string name = rawLine.Trim();
            int semicolon = name.IndexOf(';');
            if (semicolon >= 0) name = name[..semicolon].Trim();
            if (name.Length == 0) continue;
            var hash = HashString(name.ToUpperInvariant(), HashFileKey);
            _knownFileHashes.Add(hash);
            _hashToName[hash] = name;
        }
    }

    public IReadOnlyList<string> ExtractInternalListfiles()
    {
        var allFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var archive in _archives)
        {
            var block = FindFileInArchive(archive, "(listfile)");
            if (block != null)
            {
                var data = ReadFileFromArchive(archive, block, "(listfile)");
                if (data != null)
                {
                    using var reader = new StreamReader(new MemoryStream(data), Encoding.UTF8);
                    string? line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        var trimmed = line.Trim();
                        if (trimmed.Length > 0)
                        {
                            allFiles.Add(trimmed);
                            var hash = HashString(trimmed.ToUpperInvariant(), HashFileKey);
                            _knownFileHashes.Add(hash);
                            _hashToName[hash] = trimmed;
                        }
                    }
                }
            }
        }

        return allFiles.ToList();
    }

    public IReadOnlyList<string> GetAllKnownFiles()
    {
        return _hashToName.Values.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
    }

    private static bool DoesMatchPattern(string filename, string pattern)
    {
        if (pattern == "*" || pattern == "*.*") return true;
        if (pattern.StartsWith("*") && pattern.EndsWith("*"))
            return filename.Contains(pattern.Trim('*'), StringComparison.OrdinalIgnoreCase);
        if (pattern.StartsWith("*"))
            return filename.EndsWith(pattern.TrimStart('*'), StringComparison.OrdinalIgnoreCase);
        if (pattern.EndsWith("*"))
            return filename.StartsWith(pattern.TrimEnd('*'), StringComparison.OrdinalIgnoreCase);

        return filename.Equals(pattern, StringComparison.OrdinalIgnoreCase);
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
        uint hashIndex = HashString(normalized, HashTableIndex) % (uint)archive.HashTable.Length;
        uint nameA = HashString(normalized, HashNameA);
        uint nameB = HashString(normalized, HashNameB);

        uint probeLimit = Math.Min((uint)archive.HashTable.Length, 256u);
        uint probes = 0;
        for (uint i = 0; i < archive.HashTable.Length && probes < probeLimit; i++)
        {
            var entry = archive.HashTable[(hashIndex + i) % archive.HashTable.Length];
            if (entry.BlockIndex == HashEntryEmpty)
            {
                probes++;
                continue;
            }
            if (entry.BlockIndex == HashEntryDeleted) continue;

            if (entry.Name1 == nameA && entry.Name2 == nameB && entry.BlockIndex < archive.BlockTable.Length)
            {
                var block = archive.BlockTable[entry.BlockIndex];
                if ((block.Flags & FlagExists) != 0)
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
            long baseOffset = archive.HeaderOffset + block.BlockOffset;
            fs.Position = baseOffset;
            return ReadFileData(reader, block, archive.Header.SectorSize, filename, baseOffset);
        }
        catch { return null; }
    }

    private byte[]? ReadFromScannedArchive(string archivePath)
    {
        try
        {
            using var fs = new FileStream(archivePath, FileMode.Open, FileAccess.Read, FileShare.Read);
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

            // Find the largest valid block (the data payload, not the 16-byte MD5)
            BlockEntry? largestBlock = null;
            foreach (var entry in hashTable)
            {
                if (entry.BlockIndex == HashEntryEmpty || entry.BlockIndex >= blockTable.Length)
                    continue;

                var block = blockTable[entry.BlockIndex];
                if ((block.Flags & FlagExists) != 0)
                {
                    if (largestBlock == null || block.FileSize > largestBlock.FileSize)
                        largestBlock = block;
                }
            }

            if (largestBlock == null)
                return null;

            fs.Position = headerOffset + largestBlock.BlockOffset;
            return ReadFileData(reader, largestBlock, header.SectorSize, "scanned", fs.Position);
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

        return new MpqHeader
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
    }

    private static HashEntry[] ReadHashTable(BinaryReader reader, uint entryCount)
    {
        var encryptedData = new uint[entryCount * 4];
        for (int i = 0; i < encryptedData.Length; i++) encryptedData[i] = reader.ReadUInt32();
        uint key = HashString("(hash table)", HashFileKey);
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
        uint key = HashString("(block table)", HashFileKey);
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
        if ((block.Flags & FlagEncrypted) != 0)
        {
            key = HashString(Path.GetFileName(filename), HashFileKey);
            if ((block.Flags & FlagFixKey) != 0) key = (key + block.BlockOffset) ^ block.FileSize;
        }

        // Small files can still use the normal sector-offset table layout.
        // Only bypass the sector table when the archive marks the block as single-unit.
        if ((block.Flags & FlagSingleUnit) != 0)
        {
            var data = reader.ReadBytes((int)block.BlockSize);
            if ((block.Flags & FlagEncrypted) != 0) DecryptData(data, key);
            if ((block.Flags & FlagCompressed) != 0 && block.BlockSize < block.FileSize) return DecompressData(data, block.FileSize);
            return data;
        }

        uint sectorCount = (block.FileSize + sectorSize - 1) / sectorSize;
        var offsetBytes = reader.ReadBytes(4 * ((int)sectorCount + 1));
        if ((block.Flags & FlagEncrypted) != 0) DecryptData(offsetBytes, key - 1);
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
            if ((block.Flags & FlagEncrypted) != 0) DecryptData(sectorData, key + i);
            if ((block.Flags & FlagCompressed) != 0 && compressedSize < uncompressedSize)
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
            uint v = (uint)(data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24));
            seed += CryptTable[0x400 + (key & 0xFF)];
            uint temp = v ^ (key + seed);
            key = ((~key << 0x15) + 0x11111111) | (key >> 0x0B);
            seed = temp + seed + (seed << 5) + 3;
            data[offset] = (byte)(temp & 0xFF);
            data[offset + 1] = (byte)((temp >> 8) & 0xFF);
            data[offset + 2] = (byte)((temp >> 16) & 0xFF);
            data[offset + 3] = (byte)((temp >> 24) & 0xFF);
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
        _knownFileHashes.Clear();
        _hashToName.Clear();
        _disposed = true;
    }

    private class MpqArchive
    {
        public string Path = "";
        public long HeaderOffset;
        public MpqHeader Header = new();
        public HashEntry[] HashTable = [];
        public BlockEntry[] BlockTable = [];
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
        public uint SectorSize => 512u << SectorSizeShift;
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
