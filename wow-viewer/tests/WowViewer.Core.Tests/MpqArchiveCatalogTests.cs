using System.Text;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

public sealed class MpqArchiveCatalogTests
{
    [Fact]
    public void LoadArchives_PrefersHigherPriorityPatchData()
    {
        string tempDirectory = CreateTempDirectory();
        try
        {
            CreateMpqArchive(
                Path.Combine(tempDirectory, "common.mpq"),
                new MpqEntry("World\\Maps\\Azeroth\\Azeroth.wdt", Encoding.UTF8.GetBytes("base")),
                new MpqEntry("(listfile)", Encoding.UTF8.GetBytes("World\\Maps\\Azeroth\\Azeroth.wdt\n")));

            CreateMpqArchive(
                Path.Combine(tempDirectory, "patch-2.mpq"),
                new MpqEntry("World\\Maps\\Azeroth\\Azeroth.wdt", Encoding.UTF8.GetBytes("patch")));

            using MpqArchiveCatalog catalog = new();
            catalog.LoadArchives([tempDirectory]);

            byte[]? bytes = catalog.ReadFile("World/Maps/Azeroth/Azeroth.wdt");

            Assert.NotNull(bytes);
            Assert.Equal("patch", Encoding.UTF8.GetString(bytes));
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void ReadFile_FallsBackToBaseArchiveWhenPatchDeletesFile()
    {
        string tempDirectory = CreateTempDirectory();
        try
        {
            CreateMpqArchive(
                Path.Combine(tempDirectory, "common.mpq"),
                new MpqEntry("Interface\\GlueXML\\GlueXML.toc", Encoding.UTF8.GetBytes("base")));

            CreateMpqArchive(
                Path.Combine(tempDirectory, "patch.mpq"),
                new MpqEntry("Interface\\GlueXML\\GlueXML.toc", Data: null, Deleted: true));

            using MpqArchiveCatalog catalog = new();
            catalog.LoadArchives([tempDirectory]);

            byte[]? bytes = catalog.ReadFile("Interface\\GlueXML\\GlueXML.toc");

            Assert.NotNull(bytes);
            Assert.Equal("base", Encoding.UTF8.GetString(bytes));
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void ExtractInternalListfiles_ReadsListfileEntriesFromArchive()
    {
        string tempDirectory = CreateTempDirectory();
        try
        {
            CreateMpqArchive(
                Path.Combine(tempDirectory, "common.mpq"),
                new MpqEntry("(listfile)", Encoding.UTF8.GetBytes("World/Maps/Azeroth/Azeroth.wdt\nCreature/Wolf/Wolf.mdx\n")));

            using MpqArchiveCatalog catalog = new();
            catalog.LoadArchives([tempDirectory]);

            IReadOnlyList<string> files = catalog.ExtractInternalListfiles();

            Assert.Equal(
            [
                "Creature\\Wolf\\Wolf.mdx",
                "World\\Maps\\Azeroth\\Azeroth.wdt",
            ], files.OrderBy(static value => value, StringComparer.OrdinalIgnoreCase).ToArray());
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void ReadFile0FromPath_ReadsFirstStoredEntry()
    {
        string tempDirectory = CreateTempDirectory();
        string archivePath = Path.Combine(tempDirectory, "single.mpq");
        try
        {
            CreateMpqArchive(
                archivePath,
                new MpqEntry("World\\Maps\\Azeroth\\Azeroth.wdt", Encoding.UTF8.GetBytes("root")));

            using MpqArchiveCatalog catalog = new();

            byte[]? bytes = catalog.ReadFile0FromPath(archivePath);

            Assert.NotNull(bytes);
            Assert.Equal("root", Encoding.UTF8.GetString(bytes));
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    private static string CreateTempDirectory()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"wowviewer-mpq-{Guid.NewGuid():N}");
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
            uint blockOffset = (uint)currentOffset;
            uint blockSize = entry.Deleted ? 0u : (uint)payload.Length;
            uint fileSize = entry.Deleted ? 0u : (uint)payload.Length;

            blockTable[i * 4] = blockOffset;
            blockTable[i * 4 + 1] = blockSize;
            blockTable[i * 4 + 2] = fileSize;
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
            string normalizedName = NormalizeVirtualPath(entries[blockIndex].Name);
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

    private static string NormalizeVirtualPath(string value)
    {
        return value.Replace('/', '\\');
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

    private sealed record MpqEntry(string Name, byte[]? Data, bool Deleted = false);
}