using WowViewer.Core.Anim;

namespace WowViewer.Core.Anim.Tests;

public sealed class M2PoseSourceLoaderTests
{
    [Fact]
    public void LoadFromFile_RejectsStaleHColonClients()
    {
        Assert.Throws<InvalidOperationException>(() =>
            M2PoseSourceLoader.LoadFromFile(@"H:\CLIENTS\data\model.m2"));
    }

    [Fact]
    public void LoadFromFile_ThrowsOnNonExistentFile()
    {
        string nonexistent = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".m2");
        Assert.Throws<FileNotFoundException>(() => M2PoseSourceLoader.LoadFromFile(nonexistent));
    }

    [Fact]
    public void LoadFromFile_ThrowsOnBadMagic()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".m2");
        try
        {
            byte[] garbage = new byte[4096];
            for (int i = 0; i < garbage.Length; i++)
                garbage[i] = (byte)(i & 0xFF);
            File.WriteAllBytes(tempFile, garbage);

            Assert.Throws<InvalidDataException>(() => M2PoseSourceLoader.LoadFromFile(tempFile));
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Fact]
    public void LoadFromVirtualFile_RejectsStaleHColonClients()
    {
        Assert.Throws<InvalidOperationException>(() =>
            M2PoseSourceLoader.LoadFromVirtualFile(@"H:\CLIENTS\data\model.m2", new[] { "C:\\staging" }));
    }

    [Fact]
    public void LoadFromVirtualFile_ThrowsWhenNoArchiveRootHasFile()
    {
        string missing = "completely/missing/path/" + Guid.NewGuid().ToString("N") + ".m2";
        string emptyRoot = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(emptyRoot);
        try
        {
            Assert.Throws<FileNotFoundException>(() =>
                M2PoseSourceLoader.LoadFromVirtualFile(missing, new[] { emptyRoot }));
        }
        finally
        {
            Directory.Delete(emptyRoot, recursive: true);
        }
    }
}

