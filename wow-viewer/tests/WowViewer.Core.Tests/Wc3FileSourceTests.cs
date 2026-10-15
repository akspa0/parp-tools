using System.Diagnostics;
using System.Text;
using WowViewer.Core.IO.Wc3;

namespace WowViewer.Core.Tests;

public sealed class Wc3FileSourceTests
{
    [Fact]
    public void LooseFilesKeepStoredNamesIncludeUnknownsAndReadWithCanonicalComparison()
    {
        using Wc3TestDirectory fixture = new();
        fixture.Write("source/Terrain/Short~1.BLP", [1, 2, 3, 0, 255]);
        fixture.Write("source/Build/Names.MAK", Encoding.UTF8.GetBytes("original -> SHORT~1"));
        fixture.Write("source/Units/data.slk", Encoding.UTF8.GetBytes("ID;PWXL"));
        fixture.Write("source/unknown.bin", [0, 255]);
        using Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"));

        Assert.Equal(Path.GetFullPath(fixture.PathOf("source")), source.Root);
        Assert.Equal(4, source.Files.Count);
        Assert.Contains(Path.Combine("Terrain", "Short~1.BLP"), source.Files);
        Assert.Contains(Path.Combine("Build", "Names.MAK"), source.Files);
        Assert.Equal(new byte[] { 1, 2, 3, 0, 255 }, source.ReadFile("terrain/SHORT~1.blp"));
        Assert.Equal(new byte[] { 0, 255 }, source.ReadFile("unknown.bin"));
        Assert.Contains("no archive bootstrap", source.Summary, StringComparison.OrdinalIgnoreCase);
        Assert.Throws<FileNotFoundException>(() => source.ReadFile("absent.txt"));
    }

    [Fact]
    public void EmptyMissingAndMissingListfileHaveActionableErrors()
    {
        using Wc3TestDirectory fixture = new();
        Directory.CreateDirectory(fixture.PathOf("empty"));
        Assert.Contains("No safe loose", Assert.Throws<NotSupportedException>(() => Wc3FileSource.Open(fixture.PathOf("empty"))).Message);
        Assert.Throws<DirectoryNotFoundException>(() => Wc3FileSource.Open(fixture.PathOf("absent")));
        fixture.Write("source/a.txt", [65]);
        Assert.Contains("listfile", Assert.Throws<FileNotFoundException>(() =>
            Wc3FileSource.Open(fixture.PathOf("source"), fixture.PathOf("missing.lst"))).Message);
    }

    [Fact]
    public void DisposeIsIdempotentAndReadAndExtractRejectDisposedSource()
    {
        using Wc3TestDirectory fixture = new();
        fixture.Write("source/a.txt", [65]);
        Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"));
        source.Dispose();
        source.Dispose();
        Assert.Throws<ObjectDisposedException>(() => source.ReadFile("a.txt"));
        Assert.Throws<ObjectDisposedException>(() => source.ExtractFile("a.txt", fixture.PathOf("out")));
    }

    [Fact]
    public void InternalAndSupplementalNamesAreConfirmedAndLooseBytesOverrideArchives()
    {
        using Wc3TestDirectory fixture = new();
        Directory.CreateDirectory(fixture.PathOf("source"));
        Wc3SyntheticMpq.Write(fixture.PathOf("source/war3.mpq"),
            ("(listfile)", Encoding.UTF8.GetBytes("Textures\\Tile.blp\r\nBuild\\missing.txt\r\n")),
            ("Textures\\Tile.blp", new byte[] { 10, 20 }),
            ("Build\\names.mak", Encoding.UTF8.GetBytes("source name")));
        fixture.Write("wc3.lst", Encoding.UTF8.GetBytes("Build/names.mak\nnever-present.txt\n"));
        using (Wc3FileSource internalOnly = Wc3FileSource.Open(fixture.PathOf("source")))
        {
            Assert.Contains("Textures\\Tile.blp", internalOnly.Files);
            Assert.DoesNotContain(internalOnly.Files, name => name.Replace('/', '\\').Equals("Build\\missing.txt", StringComparison.OrdinalIgnoreCase));
            Assert.DoesNotContain(internalOnly.Files, name => name.Replace('/', '\\').Equals("Build\\names.mak", StringComparison.OrdinalIgnoreCase));
            Assert.Equal(new byte[] { 10, 20 }, internalOnly.ReadFile("textures/tile.blp"));
        }
        fixture.Write("source/Textures/Tile.blp", [7, 8, 9]);
        using Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"), fixture.PathOf("wc3.lst"));
        Assert.Equal(new byte[] { 7, 8, 9 }, source.ReadFile("textures\\tile.blp"));
        Assert.Equal("source name", Encoding.UTF8.GetString(source.ReadFile("build/names.mak")));
        Assert.DoesNotContain("never-present.txt", source.Files);
        Assert.Contains("Supplemental listfile:", source.Summary);
        string output = source.ExtractFile("Build\\names.mak", fixture.PathOf("out"));
        Assert.Equal(source.ReadFile("Build/names.mak"), File.ReadAllBytes(output));
        Assert.False(File.Exists(fixture.PathOf("source/.listfile-cache.json")));
    }

    [Fact]
    public void ExistingCatalogPriorityIsNotReimplemented()
    {
        using Wc3TestDirectory fixture = new();
        Directory.CreateDirectory(fixture.PathOf("source"));
        Wc3SyntheticMpq.Write(fixture.PathOf("source/common.mpq"), ("a.txt", new byte[] { 1 }));
        Wc3SyntheticMpq.Write(fixture.PathOf("source/patch-2.mpq"), ("a.txt", new byte[] { 2 }));
        fixture.Write("wc3.lst", Encoding.UTF8.GetBytes("a.txt\n"));
        using Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"), fixture.PathOf("wc3.lst"));
        Assert.Equal(new byte[] { 2 }, source.ReadFile("A.TXT"));
        Assert.Contains("ordering is unchanged", source.Summary);
    }

    [Fact]
    public void UnsupportedMpqDoesNotInventArchivePayloads()
    {
        using Wc3TestDirectory fixture = new();
        fixture.Write("source/broken.mpq", [0, 1, 2, 3]);
        fixture.Write("wc3.lst", Encoding.UTF8.GetBytes("pretend.blp"));
        using Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"), fixture.PathOf("wc3.lst"));
        Assert.DoesNotContain("pretend.blp", source.Files);
        Assert.Contains("No archive payload names were confirmed", source.Summary);
        Assert.Equal(new byte[] { 0, 1, 2, 3 }, source.ReadFile("broken.mpq"));
    }

    [Fact]
    public void ReparseDirectoriesAreSkippedAndReplacingAnIndexedDirectoryBlocksReads()
    {
        using Wc3TestDirectory fixture = new();
        fixture.Write("source/safe.txt", [65]);
        fixture.Write("source/indexed/a.txt", [66]);
        fixture.Write("outside/a.txt", [67]);
        using (fixture.LinkDirectory("source/link", "outside"))
        {
            using Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"));
            Assert.DoesNotContain(source.Files, path => path.Contains("link", StringComparison.OrdinalIgnoreCase));
            Directory.Delete(fixture.PathOf("source/indexed"), recursive: true);
            using (fixture.LinkDirectory("source/indexed", "outside"))
                Assert.Throws<IOException>(() => source.ReadFile("indexed/a.txt"));
        }
    }

    [Fact]
    public void ReparseSourceRootAndAncestorAreRejected()
    {
        using Wc3TestDirectory fixture = new();
        fixture.Write("physical/nested/a.txt", [65]);
        using (fixture.LinkDirectory("link", "physical"))
        {
            Assert.Throws<IOException>(() => Wc3FileSource.Open(fixture.PathOf("link")));
            Assert.Throws<IOException>(() => Wc3FileSource.Open(fixture.PathOf("link/nested")));
        }
    }

    [Fact]
    public void SharedArchiveScanIsNotAllowedThroughAReparseDataDirectory()
    {
        using Wc3TestDirectory fixture = new();
        Directory.CreateDirectory(fixture.PathOf("source"));
        Directory.CreateDirectory(fixture.PathOf("outside"));
        Wc3SyntheticMpq.Write(fixture.PathOf("source/war3.mpq"), ("a.txt", new byte[] { 1 }));
        Wc3SyntheticMpq.Write(fixture.PathOf("outside/patch.mpq"), ("a.txt", new byte[] { 2 }));
        fixture.Write("wc3.lst", Encoding.UTF8.GetBytes("a.txt"));
        using (fixture.LinkDirectory("source/Data", "outside"))
        {
            using Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"), fixture.PathOf("wc3.lst"));
            Assert.DoesNotContain("a.txt", source.Files);
            Assert.Contains("archive directories", source.Summary);
        }
    }
}

// All test outputs are below the project's existing bin directory, never OS temp.
internal sealed class Wc3TestDirectory : IDisposable
{
    internal string Root { get; } = Path.Combine(AppContext.BaseDirectory, "wc3-tests", Guid.NewGuid().ToString("N"));
    internal Wc3TestDirectory() => Directory.CreateDirectory(Root);
    internal string PathOf(string relative) => Path.Combine(Root, relative.Replace('/', Path.DirectorySeparatorChar));
    internal void Write(string relative, byte[] bytes)
    {
        string path = PathOf(relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, bytes);
    }
    internal IDisposable LinkDirectory(string link, string target)
    {
        string path = PathOf(link);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        if (OperatingSystem.IsWindows())
        {
            // Directory junctions do not require the symbolic-link privilege.
            ProcessStartInfo info = new("cmd.exe", $"/c mklink /J \"{path}\" \"{PathOf(target)}\"")
                { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true };
            using Process process = Process.Start(info)!;
            string output = process.StandardOutput.ReadToEnd() + process.StandardError.ReadToEnd();
            process.WaitForExit();
            Assert.True(process.ExitCode == 0, output);
        }
        else
            Directory.CreateSymbolicLink(path, PathOf(target));
        Assert.True((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0);
        return new DirectoryLink(path);
    }
    public void Dispose() => Directory.Delete(Root, recursive: true);
    private sealed class DirectoryLink(string path) : IDisposable
    {
        public void Dispose() => Directory.Delete(path);
    }
}

// Minimal synthetic single-unit MPQ fixture, matching the existing catalog test
// pattern. This is not a beta inventory, external probe, or new product reader.
internal static class Wc3SyntheticMpq
{
    private static readonly uint[] CryptTable = BuildCryptTable();
    internal static void Write(string path, params (string Name, byte[] Bytes)[] entries)
    {
        int slots = 8;
        while (slots < entries.Length * 2) slots *= 2;
        uint[] hashes = new uint[slots * 4];
        uint[] blocks = new uint[entries.Length * 4];
        for (int i = 0; i < slots; i++) hashes[i * 4 + 3] = uint.MaxValue;
        int offset = 32 + hashes.Length * 4 + blocks.Length * 4;
        for (int i = 0; i < entries.Length; i++)
        {
            string name = entries[i].Name.Replace('/', '\\');
            uint slot = Hash(name, 0) % (uint)slots;
            while (hashes[slot * 4 + 3] != uint.MaxValue) slot = (slot + 1) % (uint)slots;
            hashes[slot * 4] = Hash(name, 1);
            hashes[slot * 4 + 1] = Hash(name, 2);
            hashes[slot * 4 + 3] = (uint)i;
            blocks[i * 4] = (uint)offset;
            blocks[i * 4 + 1] = blocks[i * 4 + 2] = (uint)entries[i].Bytes.Length;
            blocks[i * 4 + 3] = 0x81000000;
            offset += entries[i].Bytes.Length;
        }
        Encrypt(hashes, Hash("(hash table)", 3));
        Encrypt(blocks, Hash("(block table)", 3));
        using BinaryWriter writer = new(File.Create(path));
        writer.Write(0x1a51504d);
        writer.Write(32u);
        writer.Write((uint)offset);
        writer.Write((ushort)0);
        writer.Write((ushort)3);
        writer.Write(32u);
        writer.Write((uint)(32 + hashes.Length * 4));
        writer.Write((uint)slots);
        writer.Write((uint)entries.Length);
        foreach (uint value in hashes) writer.Write(value);
        foreach (uint value in blocks) writer.Write(value);
        foreach (var entry in entries) writer.Write(entry.Bytes);
    }
    private static uint Hash(string value, uint kind)
    {
        uint a = 0x7fed7fed, b = 0xeeeeeeee;
        foreach (char c in value.ToUpperInvariant())
        {
            uint ch = (byte)c;
            a = CryptTable[kind * 256 + ch] ^ (a + b);
            b = ch + a + b + (b << 5) + 3;
        }
        return a;
    }
    private static void Encrypt(uint[] values, uint key)
    {
        uint seed = 0xeeeeeeee;
        for (int i = 0; i < values.Length; i++)
        {
            uint plain = values[i];
            seed += CryptTable[0x400 + (key & 255)];
            values[i] = plain ^ (key + seed);
            key = ((~key << 21) + 0x11111111) | (key >> 11);
            seed = plain + seed + (seed << 5) + 3;
        }
    }
    private static uint[] BuildCryptTable()
    {
        uint[] table = new uint[0x500];
        uint seed = 0x100001;
        for (uint i = 0; i < 256; i++)
        for (uint j = 0, index = i; j < 5; j++, index += 256)
        {
            seed = (seed * 125 + 3) % 0x2aaaab;
            uint high = (seed & 65535) << 16;
            seed = (seed * 125 + 3) % 0x2aaaab;
            table[index] = high | (seed & 65535);
        }
        return table;
    }
}
