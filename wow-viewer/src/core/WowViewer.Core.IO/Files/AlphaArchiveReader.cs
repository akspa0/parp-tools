using System.IO.Compression;

namespace WowViewer.Core.IO.Files;

public static class AlphaArchiveReader
{
    private const uint HashTableIndex = 0;
    private const uint HashNameA = 1;
    private const uint HashNameB = 2;
    private const uint HashFileKey = 3;

    private static readonly uint[] CryptTable = BuildCryptTable();

    public static byte[]? ReadFromMpq(string mpqPath)
    {
        return ReadFromMpq(mpqPath, Enumerable.Empty<string>());
    }

    public static byte[]? ReadFromMpq(string mpqPath, IEnumerable<string> internalNames)
    {
        if (!File.Exists(mpqPath))
            return null;

        try
        {
            using FileStream fileStream = new(mpqPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using BinaryReader reader = new(fileStream);

            long headerOffset = FindMpqHeader(reader);
            if (headerOffset < 0)
                return null;

            fileStream.Position = headerOffset;
            MpqHeader? header = ReadMpqHeader(reader);
            if (header is null)
                return null;

            fileStream.Position = headerOffset + header.BlockTableOffset;
            BlockEntry[] blockTable = ReadBlockTable(reader, header.BlockTableEntries);

            fileStream.Position = headerOffset + header.HashTableOffset;
            HashEntry[] hashTable = ReadHashTable(reader, header.HashTableEntries);

            BlockEntry? primaryBlock = TryGetPrimaryBlock(blockTable, hashTable, internalNames);
            if (primaryBlock is null)
                return null;

            fileStream.Position = headerOffset + primaryBlock.BlockOffset;
            byte[]? primaryData = ReadFileData(reader, primaryBlock, header.SectorSize);
            if (IsLikelyWdtOrWdl(primaryData))
                return primaryData;

            foreach (BlockEntry block in blockTable)
            {
                if (block == primaryBlock || block.FileSize == 0)
                    continue;

                fileStream.Position = headerOffset + block.BlockOffset;
                byte[]? blockData = ReadFileData(reader, block, header.SectorSize);
                if (IsLikelyWdtOrWdl(blockData))
                    return blockData;
            }

            BlockEntry? largestBlock = null;
            foreach (BlockEntry block in blockTable)
            {
                if (block.FileSize > 0 && (largestBlock is null || block.FileSize > largestBlock.FileSize))
                    largestBlock = block;
            }

            if (largestBlock is not null && largestBlock != primaryBlock)
            {
                fileStream.Position = headerOffset + largestBlock.BlockOffset;
                return ReadFileData(reader, largestBlock, header.SectorSize);
            }

            return primaryData;
        }
        catch
        {
            return null;
        }
    }

    public static byte[]? ReadWithMpqFallback(string filePath)
    {
        if (filePath.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
            return ReadFromMpq(filePath);

        if (File.Exists(filePath))
            return File.ReadAllBytes(filePath);

        foreach (string extension in new[] { ".MPQ", ".mpq" })
        {
            string mpqPath = filePath + extension;
            if (!File.Exists(mpqPath))
                continue;

            List<string> candidates = BuildInternalNameCandidates(filePath).ToList();
            return ReadFromMpq(mpqPath, candidates);
        }

        return null;
    }

    public static IEnumerable<string> BuildInternalNameCandidates(string filePath)
    {
        string? fileName = Path.GetFileName(filePath);
        if (string.IsNullOrEmpty(fileName))
            yield break;

        yield return fileName;

        string[] parts = filePath.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        int worldIndex = Array.FindIndex(parts, static part => string.Equals(part, "World", StringComparison.OrdinalIgnoreCase));
        if (worldIndex >= 0 && worldIndex + 2 < parts.Length && string.Equals(parts[worldIndex + 1], "Maps", StringComparison.OrdinalIgnoreCase))
        {
            string relative = string.Join('\\', parts.Skip(worldIndex));
            yield return relative;

            string? mapName = parts.ElementAtOrDefault(worldIndex + 2);
            if (!string.IsNullOrWhiteSpace(mapName))
            {
                yield return $"{mapName}\\{fileName}";
                yield return $"World\\Maps\\{mapName}\\{fileName}";
            }

            yield break;
        }

        string? inferredMapName = Path.GetFileNameWithoutExtension(fileName);
        if (!string.IsNullOrWhiteSpace(inferredMapName))
        {
            yield return $"{inferredMapName}\\{fileName}";
            yield return $"World\\Maps\\{inferredMapName}\\{fileName}";
        }
    }

    private static long FindMpqHeader(BinaryReader reader)
    {
        if (reader.BaseStream.Length >= 4)
        {
            reader.BaseStream.Position = 0;
            if (reader.ReadUInt32() == 0x1A51504D)
                return 0;
        }

        for (long offset = 0x200; offset < reader.BaseStream.Length && offset < 0x100000; offset += 0x200)
        {
            reader.BaseStream.Position = offset;
            if (reader.BaseStream.Length - offset < 4)
                break;

            if (reader.ReadUInt32() == 0x1A51504D)
                return offset;
        }

        return -1;
    }

    private static MpqHeader? ReadMpqHeader(BinaryReader reader)
    {
        if (reader.ReadUInt32() != 0x1A51504D)
            return null;

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

    private static BlockEntry[] ReadBlockTable(BinaryReader reader, uint entryCount)
    {
        uint[] encryptedData = new uint[entryCount * 4];
        for (int i = 0; i < encryptedData.Length; i++)
            encryptedData[i] = reader.ReadUInt32();

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

    private static HashEntry[] ReadHashTable(BinaryReader reader, uint entryCount)
    {
        uint[] encryptedData = new uint[entryCount * 4];
        for (int i = 0; i < encryptedData.Length; i++)
            encryptedData[i] = reader.ReadUInt32();

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

    private static BlockEntry? TryGetPrimaryBlock(BlockEntry[] blockTable, HashEntry[] hashTable, IEnumerable<string> internalNames)
    {
        const uint hashEntryDeleted = 0xFFFFFFFE;
        const uint hashEntryEmpty = 0xFFFFFFFF;

        foreach (string candidate in internalNames)
        {
            if (string.IsNullOrWhiteSpace(candidate))
                continue;

            string normalized = candidate.Replace('/', '\\');
            uint hashIndex = HashString(normalized, HashTableIndex) % (uint)hashTable.Length;
            uint nameA = HashString(normalized, HashNameA);
            uint nameB = HashString(normalized, HashNameB);

            for (uint i = 0; i < hashTable.Length; i++)
            {
                HashEntry entry = hashTable[(hashIndex + i) % hashTable.Length];
                if (entry.BlockIndex == hashEntryEmpty)
                    break;

                if (entry.BlockIndex == hashEntryDeleted)
                    continue;

                if (entry.Name1 == nameA && entry.Name2 == nameB && entry.BlockIndex < blockTable.Length)
                {
                    BlockEntry block = blockTable[entry.BlockIndex];
                    if (block.FileSize > 0)
                        return block;
                }
            }
        }

        if (blockTable.Length > 1 && blockTable[1].FileSize > 0)
            return blockTable[1];

        BlockEntry? largestBlock = null;
        foreach (BlockEntry block in blockTable)
        {
            if (block.FileSize > 0 && (largestBlock is null || block.FileSize > largestBlock.FileSize))
                largestBlock = block;
        }

        return largestBlock;
    }

    private static bool IsLikelyWdtOrWdl(byte[]? data)
    {
        if (data is null || data.Length < 8)
            return false;

        if (data[0] == (byte)'M' && data[1] == (byte)'V' && data[2] == (byte)'E' && data[3] == (byte)'R')
            return true;

        return data[0] == (byte)'R' && data[1] == (byte)'E' && data[2] == (byte)'V' && data[3] == (byte)'M';
    }

    private static byte[]? ReadFileData(BinaryReader reader, BlockEntry block, uint sectorSize)
    {
        const uint flagCompressed = 0x00000200;
        const uint flagSingleUnit = 0x01000000;

        if ((block.Flags & flagSingleUnit) != 0)
        {
            byte[] data = reader.ReadBytes((int)block.BlockSize);
            if ((block.Flags & flagCompressed) != 0 && block.BlockSize < block.FileSize)
                return DecompressData(data, block.FileSize);

            return data;
        }

        uint sectorCount = (block.FileSize + sectorSize - 1) / sectorSize;
        uint[] sectorOffsets = new uint[sectorCount + 1];
        for (uint i = 0; i <= sectorCount; i++)
            sectorOffsets[i] = reader.ReadUInt32();

        using MemoryStream output = new();
        for (uint i = 0; i < sectorCount; i++)
        {
            uint compressedSize = sectorOffsets[i + 1] - sectorOffsets[i];
            uint uncompressedSize = Math.Min(sectorSize, block.FileSize - (i * sectorSize));
            byte[] sectorData = reader.ReadBytes((int)compressedSize);

            if ((block.Flags & flagCompressed) != 0 && compressedSize < uncompressedSize)
            {
                byte[]? decompressed = DecompressData(sectorData, uncompressedSize);
                if (decompressed is not null)
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

        byte compressionType = data[0];
        byte[] compressedData = new byte[data.Length - 1];
        Array.Copy(data, 1, compressedData, 0, compressedData.Length);

        return compressionType switch
        {
            0x02 => DecompressZlib(compressedData),
            0x08 => PkwareExplode.Decompress(compressedData, expectedSize),
            _ => data,
        };
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

    private static uint HashString(string text, uint hashType)
    {
        uint seed1 = 0x7FED7FED;
        uint seed2 = 0xEEEEEEEE;

        foreach (char character in text.ToUpperInvariant())
        {
            uint value = (byte)character;
            seed1 = CryptTable[hashType * 0x100 + value] ^ (seed1 + seed2);
            seed2 = value + seed1 + seed2 + (seed2 << 5) + 3;
        }

        return seed1;
    }

    private static void DecryptBlock(uint[] data, uint key)
    {
        uint seed = 0xEEEEEEEE;

        for (int i = 0; i < data.Length; i++)
        {
            seed += CryptTable[0x400 + (key & 0xFF)];
            uint temporary = data[i] ^ (key + seed);
            key = ((~key << 0x15) + 0x11111111) | (key >> 0x0B);
            seed = temporary + seed + (seed << 5) + 3;
            data[i] = temporary;
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
                uint temporary1 = (seed & 0xFFFF) << 0x10;
                seed = (seed * 125 + 3) % 0x2AAAAB;
                uint temporary2 = seed & 0xFFFF;
                table[index] = temporary1 | temporary2;
            }
        }

        return table;
    }

    private sealed class MpqHeader
    {
        public uint HeaderSize { get; set; }
        public uint ArchiveSize { get; set; }
        public ushort FormatVersion { get; set; }
        public ushort SectorSizeShift { get; set; }
        public uint HashTableOffset { get; set; }
        public uint BlockTableOffset { get; set; }
        public uint HashTableEntries { get; set; }
        public uint BlockTableEntries { get; set; }
        public uint SectorSize { get; set; }
    }

    private sealed class BlockEntry
    {
        public uint BlockOffset { get; set; }
        public uint BlockSize { get; set; }
        public uint FileSize { get; set; }
        public uint Flags { get; set; }
    }

    private sealed class HashEntry
    {
        public uint Name1 { get; set; }
        public uint Name2 { get; set; }
        public ushort Locale { get; set; }
        public ushort Platform { get; set; }
        public uint BlockIndex { get; set; }
    }
}