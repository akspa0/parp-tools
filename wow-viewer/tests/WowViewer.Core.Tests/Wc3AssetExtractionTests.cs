using WowViewer.Core.IO.Wc3;

namespace WowViewer.Core.Tests;

public sealed class Wc3AssetExtractionTests
{
    [Fact]
    public void ExtractionPreservesEveryByteAndStoredNameAndNeverOverwrites()
    {
        using Wc3TestDirectory fixture = new();
        byte[] bytes = Enumerable.Range(0, 256).Select(static i => (byte)i).ToArray();
        fixture.Write("source/Build/SHORT~1.MAK", bytes);
        using Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"));
        string result = source.ExtractFile("build/short~1.mak", fixture.PathOf("out"));
        Assert.Equal(fixture.PathOf("out/Build/SHORT~1.MAK"), result);
        Assert.Equal(bytes, File.ReadAllBytes(result));
        Assert.Equal(bytes, File.ReadAllBytes(fixture.PathOf("source/Build/SHORT~1.MAK")));
        Assert.Throws<IOException>(() => source.ExtractFile("Build\\SHORT~1.MAK", fixture.PathOf("out")));
        Assert.Equal(bytes, File.ReadAllBytes(result));
        Assert.Empty(Directory.EnumerateFiles(fixture.PathOf("out"), "*.partial", SearchOption.AllDirectories));
    }

    [Theory]
    [InlineData("../a.txt")]
    [InlineData("x/../../a.txt")]
    [InlineData("x\\..\\a.txt")]
    [InlineData("/a.txt")]
    [InlineData("\\a.txt")]
    [InlineData("C:\\a.txt")]
    [InlineData("C:a.txt")]
    [InlineData("\\\\server\\share\\a.txt")]
    [InlineData("//server/share/a.txt")]
    [InlineData("a.txt:stream")]
    [InlineData("a.txt::$DATA")]
    [InlineData("a.txt.")]
    [InlineData("a.txt ")]
    [InlineData("./a.txt")]
    [InlineData("x//a.txt")]
    [InlineData("NUL.txt")]
    [InlineData("COM1.txt")]
    [InlineData("a\0.txt")]
    public void UnsafeNamesAreRejectedForReadsAndExtraction(string name)
    {
        using Wc3TestDirectory fixture = new();
        fixture.Write("source/a.txt", [65]);
        using Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"));
        Assert.Throws<ArgumentException>(() => source.ReadFile(name));
        Assert.Throws<ArgumentException>(() => source.ExtractFile(name, fixture.PathOf("out")));
        Assert.False(Directory.Exists(fixture.PathOf("out")));
    }

    [Fact]
    public void SourceTreeOutputAndCanonicalizedDescendantsAreRejected()
    {
        using Wc3TestDirectory fixture = new();
        fixture.Write("source/a.txt", [65]);
        using Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"));
        Assert.Throws<IOException>(() => source.ExtractFile("a.txt", fixture.PathOf("source")));
        Assert.Throws<IOException>(() => source.ExtractFile("a.txt", fixture.PathOf("source/new/output")));
        Assert.Throws<IOException>(() => source.ExtractFile("a.txt", fixture.PathOf("other/../source/output")));
        // A sibling sharing the source's string prefix is not inside the source.
        string output = source.ExtractFile("a.txt", fixture.PathOf("source-export"));
        Assert.Equal(new byte[] { 65 }, File.ReadAllBytes(output));
        Assert.False(Directory.Exists(fixture.PathOf("source/new")));
    }

    [Fact]
    public void OutputAncestorOutsideSourceCannotWriteBackIntoSource()
    {
        using Wc3TestDirectory fixture = new();
        fixture.Write("source/source/b.txt", [1]);
        using Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"));
        Assert.Throws<IOException>(() => source.ExtractFile("source/b.txt", fixture.Root));
        Assert.False(File.Exists(fixture.PathOf("source/b.txt")));
    }

    [Fact]
    public void ExistingFileDirectoryAndCaseInsensitiveCollisionAreNeverOverwritten()
    {
        using Wc3TestDirectory fixture = new();
        fixture.Write("source/a.txt", [65]);
        fixture.Write("out/A.TXT", [66]);
        Directory.CreateDirectory(fixture.PathOf("out2/a.txt"));
        using Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"));
        Assert.Throws<IOException>(() => source.ExtractFile("a.txt", fixture.PathOf("out")));
        Assert.Throws<IOException>(() => source.ExtractFile("a.txt", fixture.PathOf("out2")));
        Assert.Equal(new byte[] { 66 }, File.ReadAllBytes(fixture.PathOf("out/A.TXT")));
        Assert.Empty(Directory.EnumerateFiles(fixture.PathOf("out"), "*.partial"));
    }

    [Fact]
    public void ReparseOutputRootAncestorAndDescendantAreRejectedWithoutEscapedWrites()
    {
        using Wc3TestDirectory fixture = new();
        fixture.Write("source/folder/a.txt", [65]);
        Directory.CreateDirectory(fixture.PathOf("outside"));
        using Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"));
        using (fixture.LinkDirectory("link", "outside"))
        {
            Assert.Throws<IOException>(() => source.ExtractFile("folder/a.txt", fixture.PathOf("link")));
            Assert.Throws<IOException>(() => source.ExtractFile("folder/a.txt", fixture.PathOf("link/new")));
        }
        using (fixture.LinkDirectory("out/folder", "outside"))
            Assert.Throws<IOException>(() => source.ExtractFile("folder/a.txt", fixture.PathOf("out")));
        using (fixture.LinkDirectory("source-alias", "source"))
            Assert.Throws<IOException>(() => source.ExtractFile("folder/a.txt", fixture.PathOf("source-alias")));
        Assert.Empty(Directory.EnumerateFileSystemEntries(fixture.PathOf("outside")));
    }

    [Fact]
    public void MissingSourceAndBlockedOutputLeaveNoPartialFiles()
    {
        using Wc3TestDirectory fixture = new();
        fixture.Write("source/a.txt", [65]);
        fixture.Write("blocked", [66]);
        using Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"));
        Assert.Throws<IOException>(() => source.ExtractFile("a.txt", fixture.PathOf("blocked/child")));
        File.Delete(fixture.PathOf("source/a.txt"));
        Assert.Throws<FileNotFoundException>(() => source.ExtractFile("a.txt", fixture.PathOf("out")));
        Assert.False(Directory.Exists(fixture.PathOf("out")));
        Assert.Empty(Directory.EnumerateFiles(fixture.Root, "*.partial", SearchOption.AllDirectories));
    }

    [Fact]
    public void CancelledExtractionCreatesNoOutputOrPartialFile()
    {
        using Wc3TestDirectory fixture = new();
        fixture.Write("source/a.txt", [65]);
        using Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"));
        using CancellationTokenSource cancel = new();
        cancel.Cancel();
        Assert.Throws<OperationCanceledException>(() => source.ExtractFile("a.txt", fixture.PathOf("out"), cancel.Token));
        Assert.False(Directory.Exists(fixture.PathOf("out")));
        Assert.Equal(new byte[] { 65 }, source.ReadFile("a.txt"));
    }

    [Fact]
    public void EmptyLooseFileExtractionIsByteIdentical()
    {
        using Wc3TestDirectory fixture = new();
        fixture.Write("source/empty.txt", []);
        using Wc3FileSource source = Wc3FileSource.Open(fixture.PathOf("source"));
        Assert.Empty(File.ReadAllBytes(source.ExtractFile("empty.txt", fixture.PathOf("out"))));
    }
}
