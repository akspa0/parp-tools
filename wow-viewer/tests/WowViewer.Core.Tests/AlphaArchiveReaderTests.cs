using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

public sealed class AlphaArchiveReaderTests
{
    [Fact]
    public void BuildInternalNameCandidates_WorldMapsPath_ReturnsMapScopedVariants()
    {
        string filePath = @"C:\Games\Data\World\Maps\Azeroth\Azeroth_31_49.wdt";

        List<string> candidates = AlphaArchiveReader.BuildInternalNameCandidates(filePath).ToList();

        Assert.Equal(
        [
            "Azeroth_31_49.wdt",
            "World\\Maps\\Azeroth\\Azeroth_31_49.wdt",
            "Azeroth\\Azeroth_31_49.wdt",
            "World\\Maps\\Azeroth\\Azeroth_31_49.wdt",
        ], candidates);
    }

    [Fact]
    public void BuildInternalNameCandidates_WorldWmoPath_ReturnsWorldScopedVariants()
    {
        string filePath = @"C:\Games\Data\World\wmo\Azeroth\Buildings\Castle\castle01.wmo";

        List<string> candidates = AlphaArchiveReader.BuildInternalNameCandidates(filePath).ToList();

        Assert.Equal(
        [
            "castle01.wmo",
            "World\\wmo\\Azeroth\\Buildings\\Castle\\castle01.wmo",
            "wmo\\Azeroth\\Buildings\\Castle\\castle01.wmo",
            "Azeroth\\Buildings\\Castle\\castle01.wmo",
            "Buildings\\Castle\\castle01.wmo",
            "Castle\\castle01.wmo",
            "castle01\\castle01.wmo",
            "World\\Maps\\castle01\\castle01.wmo",
        ], candidates);
    }

    [Fact]
    public void ReadWithMpqFallback_ReadsDirectFileWhenPresent()
    {
        string tempFile = Path.GetTempFileName();
        try
        {
            byte[] expected = [1, 2, 3, 4];
            File.WriteAllBytes(tempFile, expected);

            byte[]? actual = AlphaArchiveReader.ReadWithMpqFallback(tempFile);

            Assert.Equal(expected, actual);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void ReadFromMpq_WithoutNames_PrefersSecondBlockForPerAssetArchives()
    {
        string tempDirectory = CreateTempDirectory();
        string archivePath = Path.Combine(tempDirectory, "sample.blp.MPQ");

        try
        {
            byte[] expected = "BLP0payload-data"u8.ToArray();
            CreateMpqArchive(
                archivePath,
                new MpqEntry("checksum.md5", "abcd"u8.ToArray()),
                new MpqEntry("texture.blp", expected));

            byte[]? actual = AlphaArchiveReader.ReadFromMpq(archivePath);

            Assert.Equal(expected, actual);
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void ReadWithMpqFallback_UsesCompanionArchiveAndInternalNameCandidates()
    {
        string tempDirectory = CreateTempDirectory();
        string worldDirectory = Path.Combine(tempDirectory, "World", "Maps", "Azeroth");
        Directory.CreateDirectory(worldDirectory);

        string filePath = Path.Combine(worldDirectory, "Azeroth_31_49.wdt");
        string archivePath = filePath + ".MPQ";
        byte[] expected = BuildChunkBytes("MVER", [18, 0, 0, 0]);

        try
        {
            CreateMpqArchive(
                archivePath,
                new MpqEntry("checksum.md5", "deadbeef"u8.ToArray()),
                new MpqEntry("World\\Maps\\Azeroth\\Azeroth_31_49.wdt", expected));

            byte[]? actual = AlphaArchiveReader.ReadWithMpqFallback(filePath);

            Assert.Equal(expected, actual);
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void ReadWithMpqFallback_DirectMpqPath_UsesInternalNameCandidates()
    {
        string tempDirectory = CreateTempDirectory();
        string archivePath = Path.Combine(tempDirectory, "World", "wmo", "Azeroth", "Buildings", "Castle", "castle01.wmo.MPQ");
        Directory.CreateDirectory(Path.GetDirectoryName(archivePath)!);
        byte[] expected = BuildChunkBytes("MVER", [17, 0, 0, 0]);

        try
        {
            CreateMpqArchive(
                archivePath,
                new MpqEntry("checksum.md5", "deadbeef"u8.ToArray()),
                new MpqEntry("World\\wmo\\Azeroth\\Buildings\\Castle\\castle01.wmo", expected));

            byte[]? actual = AlphaArchiveReader.ReadWithMpqFallback(archivePath);

            Assert.Equal(expected, actual);
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    private static byte[] BuildChunkBytes(string chunkId, byte[] payload)
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream);
        writer.Write(System.Text.Encoding.ASCII.GetBytes(chunkId));
        writer.Write(payload.Length);
        writer.Write(payload);
        return stream.ToArray();
    }

    private static string CreateTempDirectory()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"wowviewer-alpha-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static void CreateMpqArchive(string archivePath, params MpqEntry[] entries)
    {
        const uint flagExists = 0x80000000;
        const uint flagSingleUnit = 0x01000000;
        const uint hashEntryEmpty = 0xFFFFFFFF;
        const uint hashNameA = 1;
        const uint hashNameB = 2;
        const uint hashTableIndex = 0;
        const uint hashFileKey = 3;

        int hashTableEntries = NextPowerOfTwo(Math.Max(8, entries.Length * 2));
        int blockTableEntries = entries.Length;
        int headerSize = 32;
        int hashTableSize = hashTableEntries * 16;
        int blockTableSize = blockTableEntries * 16;
        int currentOffset = headerSize + hashTableSize + blockTableSize;

        List<byte[]> filePayloads = [];
        uint[] blockTable = new uint[blockTableEntries * 4];
        for (int i = 0; i < entries.Length; i++)
        {
            MpqEntry entry = entries[i];
            byte[] payload = entry.Data ?? [];

            blockTable[i * 4] = (uint)currentOffset;
            blockTable[i * 4 + 1] = (uint)payload.Length;
            blockTable[i * 4 + 2] = (uint)payload.Length;
            blockTable[i * 4 + 3] = flagExists | flagSingleUnit;

            filePayloads.Add(payload);
            currentOffset += payload.Length;
        }

        uint[] hashTable = new uint[hashTableEntries * 4];
        for (int i = 0; i < hashTableEntries; i++)
        {
            hashTable[i * 4 + 3] = hashEntryEmpty;
        }

        for (int blockIndex = 0; blockIndex < entries.Length; blockIndex++)
        {
            string normalizedName = entries[blockIndex].Name.Replace('/', '\\');
            uint slot = HashString(normalizedName, hashTableIndex) % (uint)hashTableEntries;
            while (hashTable[slot * 4 + 3] != hashEntryEmpty)
            {
                slot = (slot + 1) % (uint)hashTableEntries;
            }

            hashTable[slot * 4] = HashString(normalizedName, hashNameA);
            hashTable[slot * 4 + 1] = HashString(normalizedName, hashNameB);
            hashTable[slot * 4 + 2] = 0;
            hashTable[slot * 4 + 3] = (uint)blockIndex;
        }

        EncryptBlock(hashTable, HashString("(hash table)", hashFileKey));
        EncryptBlock(blockTable, HashString("(block table)", hashFileKey));

        using FileStream stream = new(archivePath, FileMode.Create, FileAccess.Write, FileShare.None);
        using BinaryWriter writer = new(stream);
        writer.Write(0x1A51504D);
        writer.Write((uint)headerSize);
        writer.Write((uint)currentOffset);
        writer.Write((ushort)0);
        writer.Write((ushort)3);
        writer.Write((uint)headerSize);
        writer.Write((uint)(headerSize + hashTableSize));
        writer.Write((uint)hashTableEntries);
        writer.Write((uint)blockTableEntries);

        foreach (uint value in hashTable)
        {
            writer.Write(value);
        }

        foreach (uint value in blockTable)
        {
            writer.Write(value);
        }

        foreach (byte[] payload in filePayloads)
        {
            writer.Write(payload);
        }
    }

    private static int NextPowerOfTwo(int value)
    {
        int result = 1;
        while (result < value)
        {
            result <<= 1;
        }

        return result;
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

    private static void EncryptBlock(uint[] data, uint key)
    {
        uint seed = 0xEEEEEEEE;
        for (int i = 0; i < data.Length; i++)
        {
            uint plain = data[i];
            seed += CryptTable[0x400 + (key & 0xFF)];
            data[i] = plain ^ (key + seed);
            key = ((~key << 0x15) + 0x11111111) | (key >> 0x0B);
            seed = plain + seed + (seed << 5) + 3;
        }
    }

    private static readonly uint[] CryptTable = BuildCryptTable();

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

    private sealed record MpqEntry(string Name, byte[]? Data);
}
