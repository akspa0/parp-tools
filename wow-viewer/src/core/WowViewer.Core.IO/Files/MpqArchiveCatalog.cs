using System.IO.Compression;
using System.Text;
using ICSharpCode.SharpZipLib.BZip2;

namespace WowViewer.Core.IO.Files;

public sealed class MpqArchiveCatalog : IArchiveCatalog
{
    private const uint HashTableIndex = 0;
    private const uint HashNameA = 1;
    private const uint HashNameB = 2;
    private const uint HashFileKey = 3;

    private const uint HashEntryDeleted = 0xFFFFFFFE;
    private const uint HashEntryEmpty = 0xFFFFFFFF;

    private const uint FlagCompressed = 0x00000200;
    private const uint FlagEncrypted = 0x00010000;
    private const uint FlagFixKey = 0x00020000;
    private const uint FlagSingleUnit = 0x01000000;
    private const uint FlagExists = 0x80000000;

    private static readonly uint[] CryptTable = BuildCryptTable();

    private readonly List<MpqArchive> _archives = [];
    private readonly HashSet<ulong> _knownFileHashes = [];
    private readonly Dictionary<ulong, string> _hashToName = [];
    private readonly Dictionary<string, ScannedFileEntry> _scannedFiles = new(StringComparer.OrdinalIgnoreCase);
    private bool _disposed;

    public void LoadListfile(string path)
    {
        if (!File.Exists(path))
        {
            return;
        }

        LoadListfileEntries(File.ReadLines(path));
    }

    public void LoadListfileEntries(IEnumerable<string> entries)
    {
        ArgumentNullException.ThrowIfNull(entries);

        foreach (string rawLine in entries)
        {
            string name = rawLine;
            if (rawLine.Contains(';', StringComparison.Ordinal))
            {
                string[] parts = rawLine.Split(';');
                if (parts.Length > 1)
                {
                    name = parts[1];
                }
            }

            name = name.Trim();
            if (string.IsNullOrEmpty(name))
            {
                continue;
            }

            AddKnownFileName(name);
        }
    }

    public bool HasFile(string filename)
    {
        return FileExists(filename);
    }

    public IReadOnlyList<string> ExtractInternalListfiles()
    {
        HashSet<string> allFiles = new(StringComparer.OrdinalIgnoreCase);

        foreach (MpqArchive archive in _archives)
        {
            BlockEntry? listfileBlock = FindFileInArchive(archive, "(listfile)");
            if (listfileBlock is null || listfileBlock.FileSize == 0)
            {
                continue;
            }

            byte[]? listfileData = ReadFileFromArchive(archive, listfileBlock, "(listfile)");
            if (listfileData is null || listfileData.Length == 0)
            {
                continue;
            }

            string content = Encoding.UTF8.GetString(listfileData);
            foreach (string line in content.Split(['\n', '\r'], StringSplitOptions.RemoveEmptyEntries))
            {
                string normalizedPath = NormalizeVirtualPath(line.Trim()).TrimStart('\\');
                if (!string.IsNullOrEmpty(normalizedPath))
                {
                    allFiles.Add(normalizedPath);
                }
            }
        }

        return allFiles.ToList();
    }

    public IReadOnlyList<string> ScanWmoMpqArchives(string gamePath)
    {
        HashSet<string> foundWmos = new(StringComparer.OrdinalIgnoreCase);
        string[] searchPaths =
        [
            Path.Combine(gamePath, "Data"),
            Path.Combine(gamePath, "Data", "World"),
            Path.Combine(gamePath, "Data", "World", "wmo"),
            Path.Combine(gamePath, "Data", "World", "WMO"),
            Path.Combine(gamePath, "World"),
            Path.Combine(gamePath, "World", "wmo"),
            Path.Combine(gamePath, "World", "WMO"),
        ];

        foreach (string searchPath in searchPaths)
        {
            if (!Directory.Exists(searchPath))
            {
                continue;
            }

            try
            {
                List<string> wmoMpqFiles = Directory.GetFiles(searchPath, "*.wmo.mpq", SearchOption.AllDirectories)
                    .Concat(Directory.GetFiles(searchPath, "*.WMO.MPQ", SearchOption.AllDirectories))
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToList();

                foreach (string mpqPath in wmoMpqFiles)
                {
                    string mpqFileName = Path.GetFileNameWithoutExtension(mpqPath);
                    string mpqFileNameWithoutWmo = mpqFileName;
                    if (mpqFileNameWithoutWmo.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
                    {
                        mpqFileNameWithoutWmo = mpqFileNameWithoutWmo[..^4];
                    }

                    string virtualPath = NormalizeVirtualPath($"World\\wmo\\{mpqFileNameWithoutWmo}.wmo");
                    _scannedFiles.TryAdd(virtualPath, new ScannedFileEntry(mpqPath, 0));
                    foundWmos.Add(virtualPath);
                }
            }
            catch
            {
            }
        }

        return foundWmos.ToList();
    }

    public byte[]? ReadWmoMpqFile(string virtualPath)
    {
        if (!_scannedFiles.TryGetValue(virtualPath, out ScannedFileEntry mpqInfo) ||
            mpqInfo.BlockOffset != 0 ||
            !mpqInfo.ArchivePath.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        try
        {
            using FileStream fileStream = new(mpqInfo.ArchivePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using BinaryReader reader = new(fileStream);

            long headerOffset = FindMpqHeader(reader);
            if (headerOffset < 0)
            {
                return null;
            }

            fileStream.Position = headerOffset;
            MpqHeader? header = ReadMpqHeader(reader);
            if (header is null)
            {
                return null;
            }

            fileStream.Position = headerOffset + header.HashTableOffset;
            HashEntry[] hashTable = ReadHashTable(reader, header.HashTableEntries);

            fileStream.Position = headerOffset + header.BlockTableOffset;
            BlockEntry[] blockTable = ReadBlockTable(reader, header.BlockTableEntries);

            BlockEntry? file0Block = null;
            foreach (HashEntry entry in hashTable)
            {
                if (entry.BlockIndex == HashEntryEmpty || entry.BlockIndex >= blockTable.Length)
                {
                    continue;
                }

                BlockEntry block = blockTable[entry.BlockIndex];
                if ((block.Flags & FlagExists) != 0)
                {
                    file0Block = block;
                    break;
                }
            }

            if (file0Block is null)
            {
                return null;
            }

            fileStream.Position = headerOffset + file0Block.BlockOffset;
            return ReadFileData(reader, file0Block, header.SectorSize, "file_0", fileStream.Position);
        }
        catch
        {
            return null;
        }
    }

    public byte[]? ReadFile0FromPath(string mpqDiskPath)
    {
        try
        {
            using FileStream fileStream = new(mpqDiskPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using BinaryReader reader = new(fileStream);

            long headerOffset = FindMpqHeader(reader);
            if (headerOffset < 0)
            {
                return null;
            }

            fileStream.Position = headerOffset;
            MpqHeader? header = ReadMpqHeader(reader);
            if (header is null)
            {
                return null;
            }

            fileStream.Position = headerOffset + header.HashTableOffset;
            HashEntry[] hashTable = ReadHashTable(reader, header.HashTableEntries);

            fileStream.Position = headerOffset + header.BlockTableOffset;
            BlockEntry[] blockTable = ReadBlockTable(reader, header.BlockTableEntries);

            BlockEntry? file0Block = null;
            foreach (HashEntry entry in hashTable)
            {
                if (entry.BlockIndex == HashEntryEmpty || entry.BlockIndex >= blockTable.Length)
                {
                    continue;
                }

                BlockEntry block = blockTable[entry.BlockIndex];
                if ((block.Flags & FlagExists) != 0)
                {
                    file0Block = block;
                    break;
                }
            }

            if (file0Block is null)
            {
                return null;
            }

            fileStream.Position = headerOffset + file0Block.BlockOffset;
            return ReadFileData(reader, file0Block, header.SectorSize, "file_0", fileStream.Position);
        }
        catch
        {
            return null;
        }
    }

    public IReadOnlyList<string> GetAllKnownFiles()
    {
        HashSet<string> allFiles = new(StringComparer.OrdinalIgnoreCase);
        foreach ((_, string fileName) in _hashToName)
        {
            allFiles.Add(fileName);
        }

        foreach (string path in _scannedFiles.Keys)
        {
            allFiles.Add(path);
        }

        return allFiles.ToList();
    }

    public void LoadArchives(IEnumerable<string> searchPaths)
    {
        HashSet<string> pathsToSearch = new(StringComparer.OrdinalIgnoreCase);
        foreach (string path in searchPaths)
        {
            if (Directory.Exists(path))
            {
                pathsToSearch.Add(path);
            }

            string dataSubdirectory = Path.Combine(path, "Data");
            if (!Directory.Exists(dataSubdirectory))
            {
                continue;
            }

            pathsToSearch.Add(dataSubdirectory);
            foreach (string localeDirectory in Directory.GetDirectories(dataSubdirectory))
            {
                string localeName = Path.GetFileName(localeDirectory);
                if (localeName.Length == 4 &&
                    char.IsLetter(localeName[0]) &&
                    char.IsLetter(localeName[1]) &&
                    char.IsUpper(localeName[2]) &&
                    char.IsUpper(localeName[3]))
                {
                    pathsToSearch.Add(localeDirectory);
                }
            }
        }

        HashSet<string> allMpqFiles = new(StringComparer.OrdinalIgnoreCase);
        foreach (string path in pathsToSearch)
        {
            try
            {
                foreach (string mpqPath in Directory.EnumerateFiles(path, "*.mpq", SearchOption.AllDirectories))
                {
                    string lowerName = Path.GetFileName(mpqPath).ToLowerInvariant();
                    if (lowerName.EndsWith(".wmo.mpq", StringComparison.Ordinal) ||
                        lowerName.EndsWith(".wdt.mpq", StringComparison.Ordinal) ||
                        lowerName.EndsWith(".wdl.mpq", StringComparison.Ordinal))
                    {
                        continue;
                    }

                    allMpqFiles.Add(mpqPath);
                }
            }
            catch
            {
            }
        }

        List<string> orderedMpqFiles = allMpqFiles
            .OrderBy(static filePath => GetMpqPriority(Path.GetFileName(filePath)))
            .ThenBy(static filePath => filePath, StringComparer.OrdinalIgnoreCase)
            .ToList();

        foreach (string mpqPath in orderedMpqFiles)
        {
            try
            {
                MpqArchive? archive = LoadArchive(mpqPath);
                if (archive is not null)
                {
                    _archives.Add(archive);
                }
            }
            catch
            {
            }
        }
    }

    public bool FileExists(string virtualPath)
    {
        string normalized = NormalizeVirtualPath(virtualPath);
        for (int i = _archives.Count - 1; i >= 0; i--)
        {
            if (FindFileInArchive(_archives[i], normalized) is not null)
            {
                return true;
            }
        }

        return false;
    }

    public byte[]? ReadFile(string virtualPath)
    {
        string normalized = NormalizeVirtualPath(virtualPath);
        if (_scannedFiles.TryGetValue(normalized, out ScannedFileEntry scannedInfo) &&
            scannedInfo.BlockOffset == 0 &&
            File.Exists(scannedInfo.ArchivePath))
        {
            return File.ReadAllBytes(scannedInfo.ArchivePath);
        }

        for (int i = _archives.Count - 1; i >= 0; i--)
        {
            MpqArchive archive = _archives[i];
            BlockEntry? block = FindFileInArchive(archive, normalized);
            if (block is null)
            {
                continue;
            }

            if (block.FileSize == 0)
            {
                MpqDiagnostics.Increment("MpqPatchedDeleteHitCount");
                continue;
            }

            byte[]? data = ReadFileFromArchive(archive, block, normalized);
            if (data is { Length: > 0 })
            {
                return data;
            }
        }

        return null;
    }

    public byte[]? ReadScannedFile(string placeholderPath)
    {
        if (!_scannedFiles.TryGetValue(placeholderPath, out ScannedFileEntry info))
        {
            return null;
        }

        try
        {
            using FileStream fileStream = new(info.ArchivePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using BinaryReader reader = new(fileStream);

            MpqArchive? archive = _archives.FirstOrDefault(candidate => candidate.Path == info.ArchivePath);
            if (archive is null)
            {
                return null;
            }

            fileStream.Position = archive.HeaderOffset + info.BlockOffset;
            return ReadFileData(
                reader,
                new BlockEntry { BlockOffset = info.BlockOffset, BlockSize = 0, FileSize = 0, Flags = 0 },
                archive.Header.SectorSize,
                placeholderPath,
                archive.HeaderOffset + info.BlockOffset);
        }
        catch
        {
            return null;
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _archives.Clear();
        _knownFileHashes.Clear();
        _hashToName.Clear();
        _scannedFiles.Clear();
        _disposed = true;
    }

    private static string NormalizeVirtualPath(string path)
    {
        return path.Replace('/', '\\');
    }

    private void AddKnownFileName(string name)
    {
        string normalized = NormalizeVirtualPath(name);
        ulong hash = ((ulong)HashString(normalized, HashNameA) << 32) | HashString(normalized, HashNameB);
        if (_knownFileHashes.Add(hash))
        {
            _hashToName[hash] = name;
        }
    }

    private static int GetMpqPriority(string filename)
    {
        string lower = filename.ToLowerInvariant();
        if (lower.StartsWith("patch", StringComparison.OrdinalIgnoreCase))
        {
            string nameWithoutExtension = lower.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase)
                ? lower[..^4]
                : lower;
            string[] parts = nameWithoutExtension.Split('-', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length >= 1 && parts[0] == "patch")
            {
                bool isLocale = false;
                int suffixIndex = 1;
                if (parts.Length >= 2 && parts[1].Length == 4 && parts[1].All(char.IsLetter))
                {
                    isLocale = true;
                    suffixIndex = 2;
                }

                string? suffix = parts.Length > suffixIndex ? parts[suffixIndex] : null;
                int suffixRank = 0;
                if (!string.IsNullOrEmpty(suffix))
                {
                    if (int.TryParse(suffix, out int number))
                    {
                        suffixRank = Math.Clamp(number, 0, 499);
                    }
                    else if (suffix.Length == 1 && suffix[0] >= 'a' && suffix[0] <= 'z')
                    {
                        suffixRank = 500 + (suffix[0] - 'a' + 1);
                    }
                    else
                    {
                        suffixRank = 900;
                    }
                }

                return (isLocale ? 2000 : 1000) + suffixRank;
            }

            return 2900;
        }

        if (lower.Contains("enus", StringComparison.Ordinal) ||
            lower.Contains("engb", StringComparison.Ordinal) ||
            lower.Contains("dede", StringComparison.Ordinal) ||
            lower.Contains("locale", StringComparison.Ordinal))
        {
            return 500;
        }

        if (lower.StartsWith("expansion", StringComparison.Ordinal) || lower.StartsWith("lichking", StringComparison.Ordinal))
        {
            return 300;
        }

        if (lower == "common.mpq" || lower == "common-2.mpq")
        {
            return 100;
        }

        return 200;
    }

    private MpqArchive? LoadArchive(string mpqPath)
    {
        using FileStream fileStream = new(mpqPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        using BinaryReader reader = new(fileStream);

        long headerOffset = FindMpqHeader(reader);
        if (headerOffset < 0)
        {
            return null;
        }

        fileStream.Position = headerOffset;
        MpqHeader? header = ReadMpqHeader(reader);
        if (header is null)
        {
            return null;
        }

        fileStream.Position = headerOffset + header.HashTableOffset;
        HashEntry[] hashTable = ReadHashTable(reader, header.HashTableEntries);

        fileStream.Position = headerOffset + header.BlockTableOffset;
        BlockEntry[] blockTable = ReadBlockTable(reader, header.BlockTableEntries);

        return new MpqArchive
        {
            Path = mpqPath,
            HeaderOffset = headerOffset,
            Header = header,
            HashTable = hashTable,
            BlockTable = blockTable,
        };
    }

    private static BlockEntry? FindFileInArchive(MpqArchive archive, string filename)
    {
        string normalized = NormalizeVirtualPath(filename);
        uint hashIndex = HashString(normalized, HashTableIndex) % (uint)archive.HashTable.Length;
        uint nameA = HashString(normalized, HashNameA);
        uint nameB = HashString(normalized, HashNameB);

        for (uint i = 0; i < archive.HashTable.Length; i++)
        {
            HashEntry entry = archive.HashTable[(hashIndex + i) % archive.HashTable.Length];
            if (entry.BlockIndex == HashEntryEmpty)
            {
                break;
            }

            if (entry.BlockIndex == HashEntryDeleted)
            {
                continue;
            }

            if (entry.Name1 == nameA && entry.Name2 == nameB && entry.BlockIndex < archive.BlockTable.Length)
            {
                BlockEntry block = archive.BlockTable[entry.BlockIndex];
                if ((block.Flags & FlagExists) != 0)
                {
                    return block;
                }
            }
        }

        return null;
    }

    private static byte[]? ReadFileFromArchive(MpqArchive archive, BlockEntry block, string filename)
    {
        try
        {
            using FileStream fileStream = new(archive.Path, FileMode.Open, FileAccess.Read, FileShare.Read);
            using BinaryReader reader = new(fileStream);

            long baseOffset = archive.HeaderOffset + block.BlockOffset;
            fileStream.Position = baseOffset;

            byte[]? data = ReadFileData(reader, block, archive.Header.SectorSize, filename, baseOffset);
            if (data is { Length: > 0 })
            {
                return data;
            }

            if ((block.Flags & FlagEncrypted) != 0)
            {
                string baseName = Path.GetFileName(filename);
                if (!string.Equals(baseName, filename, StringComparison.OrdinalIgnoreCase))
                {
                    fileStream.Position = baseOffset;
                    data = ReadFileData(reader, block, archive.Header.SectorSize, baseName, baseOffset);
                    if (data is { Length: > 0 })
                    {
                        return data;
                    }
                }
            }

            return data;
        }
        catch
        {
            return null;
        }
    }

    private static long FindMpqHeader(BinaryReader reader)
    {
        if (reader.BaseStream.Length >= 4)
        {
            reader.BaseStream.Position = 0;
            if (reader.ReadUInt32() == 0x1A51504D)
            {
                return 0;
            }
        }

        for (long offset = 0x200; offset < reader.BaseStream.Length && offset < 0x100000; offset += 0x200)
        {
            reader.BaseStream.Position = offset;
            if (reader.BaseStream.Length - offset < 4)
            {
                break;
            }

            if (reader.ReadUInt32() == 0x1A51504D)
            {
                return offset;
            }
        }

        return -1;
    }

    private static MpqHeader? ReadMpqHeader(BinaryReader reader)
    {
        if (reader.ReadUInt32() != 0x1A51504D)
        {
            return null;
        }

        MpqHeader header = new()
        {
            HeaderSize = reader.ReadUInt32(),
            ArchiveSize = reader.ReadUInt32(),
            FormatVersion = reader.ReadUInt16(),
            SectorSizeShift = reader.ReadUInt16(),
            HashTableOffset = reader.ReadUInt32(),
            BlockTableOffset = reader.ReadUInt32(),
            HashTableEntries = reader.ReadUInt32(),
            BlockTableEntries = reader.ReadUInt32(),
        };

        header.SectorSize = 512u << header.SectorSizeShift;
        return header;
    }

    private static HashEntry[] ReadHashTable(BinaryReader reader, uint entryCount)
    {
        uint[] encryptedData = new uint[entryCount * 4];
        for (int i = 0; i < encryptedData.Length; i++)
        {
            encryptedData[i] = reader.ReadUInt32();
        }

        uint key = HashString("(hash table)", HashFileKey);
        DecryptBlock(encryptedData, key);

        HashEntry[] entries = new HashEntry[entryCount];
        for (uint i = 0; i < entryCount; i++)
        {
            entries[i] = new HashEntry
            {
                Name1 = encryptedData[i * 4],
                Name2 = encryptedData[i * 4 + 1],
                Locale = (ushort)(encryptedData[i * 4 + 2] & 0xFFFF),
                Platform = (ushort)((encryptedData[i * 4 + 2] >> 16) & 0xFFFF),
                BlockIndex = encryptedData[i * 4 + 3],
            };
        }

        return entries;
    }

    private static BlockEntry[] ReadBlockTable(BinaryReader reader, uint entryCount)
    {
        uint[] encryptedData = new uint[entryCount * 4];
        for (int i = 0; i < encryptedData.Length; i++)
        {
            encryptedData[i] = reader.ReadUInt32();
        }

        uint key = HashString("(block table)", HashFileKey);
        DecryptBlock(encryptedData, key);

        BlockEntry[] entries = new BlockEntry[entryCount];
        for (uint i = 0; i < entryCount; i++)
        {
            entries[i] = new BlockEntry
            {
                BlockOffset = encryptedData[i * 4],
                BlockSize = encryptedData[i * 4 + 1],
                FileSize = encryptedData[i * 4 + 2],
                Flags = encryptedData[i * 4 + 3],
            };
        }

        return entries;
    }

    private static byte[]? ReadFileData(BinaryReader reader, BlockEntry block, uint sectorSize, string filename, long fileBaseOffset)
    {
        uint key = 0;
        if ((block.Flags & FlagEncrypted) != 0)
        {
            key = HashString(NormalizeVirtualPath(filename), HashFileKey);
            if ((block.Flags & FlagFixKey) != 0)
            {
                key = (key + block.BlockOffset) ^ block.FileSize;
            }
        }

        if ((block.Flags & FlagSingleUnit) != 0)
        {
            byte[] data = reader.ReadBytes((int)block.BlockSize);
            if ((block.Flags & FlagEncrypted) != 0)
            {
                DecryptData(data, key);
            }

            if ((block.Flags & FlagCompressed) != 0 && block.BlockSize < block.FileSize)
            {
                return DecompressData(data, block.FileSize);
            }

            return data;
        }

        uint sectorCount = (block.FileSize + sectorSize - 1) / sectorSize;
        byte[] offsetBytes = reader.ReadBytes(4 * ((int)sectorCount + 1));
        if ((block.Flags & FlagEncrypted) != 0)
        {
            DecryptData(offsetBytes, key - 1);
        }

        uint[] sectorOffsets = new uint[sectorCount + 1];
        Buffer.BlockCopy(offsetBytes, 0, sectorOffsets, 0, offsetBytes.Length);
        if (sectorOffsets.Length == 0)
        {
            MpqDiagnostics.Increment("MpqSectorTableInvalidCount");
            return null;
        }

        uint previous = sectorOffsets[0];
        if (previous > block.BlockSize)
        {
            MpqDiagnostics.Increment("MpqSectorTableInvalidCount");
            return null;
        }

        for (int sectorIndex = 1; sectorIndex < sectorOffsets.Length; sectorIndex++)
        {
            uint current = sectorOffsets[sectorIndex];
            if (current < previous || current > block.BlockSize)
            {
                MpqDiagnostics.Increment("MpqSectorTableInvalidCount");
                return null;
            }

            previous = current;
        }

        using MemoryStream output = new();
        for (uint sectorIndex = 0; sectorIndex < sectorCount; sectorIndex++)
        {
            uint sectorStart = sectorOffsets[sectorIndex];
            uint sectorEnd = sectorOffsets[sectorIndex + 1];
            if (sectorEnd < sectorStart)
            {
                MpqDiagnostics.Increment("MpqSectorTableInvalidCount");
                return null;
            }

            uint compressedSize = sectorEnd - sectorStart;
            uint uncompressedSize = Math.Min(sectorSize, block.FileSize - (sectorIndex * sectorSize));
            if (fileBaseOffset + sectorStart > reader.BaseStream.Length || fileBaseOffset + sectorEnd > reader.BaseStream.Length)
            {
                MpqDiagnostics.Increment("MpqSectorTableInvalidCount");
                return null;
            }

            reader.BaseStream.Position = fileBaseOffset + sectorStart;
            byte[] sectorData = reader.ReadBytes((int)compressedSize);
            if ((block.Flags & FlagEncrypted) != 0)
            {
                DecryptData(sectorData, key + sectorIndex);
            }

            if ((block.Flags & FlagCompressed) != 0 && compressedSize < uncompressedSize)
            {
                byte[]? decompressed = DecompressData(sectorData, uncompressedSize);
                output.Write(decompressed ?? sectorData);
            }
            else
            {
                output.Write(sectorData);
            }
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
            uint value = (uint)(data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24));

            seed += CryptTable[0x400 + (key & 0xFF)];
            uint temp = value ^ (key + seed);
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
        if (data.Length == 0)
        {
            return null;
        }

        const byte MpqCompHuffman = 0x01;
        const byte MpqCompZlib = 0x02;
        const byte MpqCompPkware = 0x08;
        const byte MpqCompBzip2 = 0x10;
        const byte MpqCompLzma = 0x80;

        byte mask = data[0];
        byte[] payload = new byte[data.Length - 1];
        Array.Copy(data, 1, payload, 0, payload.Length);

        byte unsupported = (byte)(mask & ~(MpqCompHuffman | MpqCompZlib | MpqCompPkware | MpqCompBzip2 | MpqCompLzma));
        if (unsupported != 0 || (mask & MpqCompLzma) != 0 || (mask & MpqCompHuffman) != 0)
        {
            return null;
        }

        byte[] current = payload;
        if ((mask & MpqCompBzip2) != 0)
        {
            byte[]? bzip = DecompressBzip2(current);
            if (bzip is null)
            {
                return null;
            }

            current = bzip;
        }

        if ((mask & MpqCompZlib) != 0)
        {
            byte[]? zlib = DecompressZlib(current);
            if (zlib is null)
            {
                return null;
            }

            current = zlib;
        }

        if ((mask & MpqCompPkware) != 0)
        {
            byte[]? pkware = PkwareExplode.Decompress(current, expectedSize);
            if (pkware is null)
            {
                return null;
            }

            current = pkware;
        }

        return current;
    }

    private static byte[]? DecompressBzip2(byte[] data)
    {
        try
        {
            using MemoryStream input = new(data, writable: false);
            using BZip2InputStream bzip = new(input);
            using MemoryStream output = new();
            bzip.CopyTo(output);
            return output.ToArray();
        }
        catch
        {
            return null;
        }
    }

    private static byte[]? DecompressZlib(byte[] data)
    {
        try
        {
            using MemoryStream input = new(data);
            input.ReadByte();
            input.ReadByte();
            using DeflateStream deflate = new(input, CompressionMode.Decompress);
            using MemoryStream output = new();
            deflate.CopyTo(output);
            return output.ToArray();
        }
        catch
        {
            try
            {
                using MemoryStream input = new(data);
                using DeflateStream deflate = new(input, CompressionMode.Decompress);
                using MemoryStream output = new();
                deflate.CopyTo(output);
                return output.ToArray();
            }
            catch
            {
                return null;
            }
        }
    }

    private static uint HashString(string value, uint hashType)
    {
        uint seed1 = 0x7FED7FED;
        uint seed2 = 0xEEEEEEEE;
        foreach (char character in value.ToUpperInvariant())
        {
            uint ch = (byte)character;
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
        uint[] table = new uint[0x500];
        uint seed = 0x00100001;
        for (uint i = 0; i < 0x100; i++)
        {
            uint index = i;
            for (int j = 0; j < 5; j++, index += 0x100)
            {
                seed = (seed * 125 + 3) % 0x2AAAAB;
                uint temp1 = (seed & 0xFFFF) << 16;
                seed = (seed * 125 + 3) % 0x2AAAAB;
                uint temp2 = seed & 0xFFFF;
                table[index] = temp1 | temp2;
            }
        }

        return table;
    }

    private sealed record ScannedFileEntry(string ArchivePath, uint BlockOffset);

    private sealed class MpqArchive
    {
        public string Path { get; init; } = string.Empty;

        public long HeaderOffset { get; init; }

        public MpqHeader Header { get; init; } = new();

        public HashEntry[] HashTable { get; init; } = [];

        public BlockEntry[] BlockTable { get; init; } = [];
    }

    private sealed class MpqHeader
    {
        public uint HeaderSize { get; init; }

        public uint ArchiveSize { get; init; }

        public ushort FormatVersion { get; init; }

        public ushort SectorSizeShift { get; init; }

        public uint HashTableOffset { get; init; }

        public uint BlockTableOffset { get; init; }

        public uint HashTableEntries { get; init; }

        public uint BlockTableEntries { get; init; }

        public uint SectorSize { get; set; }
    }

    private sealed class BlockEntry
    {
        public uint BlockOffset { get; init; }

        public uint BlockSize { get; init; }

        public uint FileSize { get; init; }

        public uint Flags { get; init; }
    }

    private sealed class HashEntry
    {
        public uint Name1 { get; init; }

        public uint Name2 { get; init; }

        public ushort Locale { get; init; }

        public ushort Platform { get; init; }

        public uint BlockIndex { get; init; }
    }
}

public sealed class MpqArchiveCatalogFactory : IArchiveCatalogFactory
{
    public IArchiveCatalog Create()
    {
        return new MpqArchiveCatalog();
    }
}