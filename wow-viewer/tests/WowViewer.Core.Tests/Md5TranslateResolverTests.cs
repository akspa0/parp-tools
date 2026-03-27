using System.Text;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

public sealed class Md5TranslateResolverTests
{
    [Fact]
    public void TryLoad_MapSpecificArchiveCandidate_BuildsExpectedMappings()
    {
        byte[] contents = Encoding.UTF8.GetBytes("map00_00.blp\t7b3d4f.blp\n");
        Dictionary<string, byte[]> archive = new(StringComparer.OrdinalIgnoreCase)
        {
            ["World\\Maps\\Azeroth\\md5translate.trs"] = contents,
        };

        bool loaded = Md5TranslateResolver.TryLoad(
            searchPaths: Array.Empty<string>(),
            archiveFileExists: path => archive.ContainsKey(path),
            archiveReadFile: path => archive.TryGetValue(path, out byte[]? bytes) ? bytes : null,
            index: out Md5TranslateIndex? index,
            extraCandidates: ["World/Maps/Azeroth/md5translate.trs"]);

        Assert.True(loaded);
        Assert.NotNull(index);

        string plainPath = "textures/minimap/azeroth/map00_00.blp";
        Assert.True(index!.PlainToHash.TryGetValue(plainPath, out string? hashedPath));
        Assert.Equal("textures/minimap/7b3d4f.blp", hashedPath);
    }

    [Fact]
    public void TryLoad_DiskCandidateWithDirDirective_UsesDirectoryContext()
    {
        string root = Path.Combine(Path.GetTempPath(), $"wowviewer-md5translate-{Guid.NewGuid():N}");
        try
        {
            string minimapDirectory = Path.Combine(root, "textures", "minimap");
            Directory.CreateDirectory(minimapDirectory);
            string filePath = Path.Combine(minimapDirectory, "md5translate.trs");
            File.WriteAllText(filePath, "dir:Kalimdor\nmap12_34.blp\tabcd1234.blp\n", Encoding.UTF8);

            bool loaded = Md5TranslateResolver.TryLoad(
                searchPaths: [root],
                archiveFileExists: static _ => false,
                archiveReadFile: static _ => null,
                index: out Md5TranslateIndex? index);

            Assert.True(loaded);
            Assert.NotNull(index);
            Assert.Equal(
                "textures/minimap/abcd1234.blp",
                index!.PlainToHash["textures/minimap/kalimdor/map12_34.blp"]);
        }
        finally
        {
            if (Directory.Exists(root))
                Directory.Delete(root, recursive: true);
        }
    }
}