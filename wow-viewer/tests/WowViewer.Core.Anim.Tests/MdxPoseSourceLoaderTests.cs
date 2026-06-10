using WowViewer.Core.Anim;

namespace WowViewer.Core.Anim.Tests;

public sealed class MdxPoseSourceLoaderTests
{
    [Fact]
    public void LoadFromFile_RejectsStaleHColonClients()
    {
        Assert.Throws<InvalidOperationException>(() =>
            MdxPoseSourceLoader.LoadFromFile(@"H:\CLIENTS\data\model.mdx"));
    }

    [Fact]
    public void LoadFromFile_ThrowsOnNonExistentFile()
    {
        string nonexistent = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".mdx");
        Assert.Throws<FileNotFoundException>(() => MdxPoseSourceLoader.LoadFromFile(nonexistent));
    }

    [Fact]
    public void LoadFromFile_ThrowsOnBadMagic()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".mdx");
        try
        {
            byte[] garbage = new byte[4096];
            for (int i = 0; i < garbage.Length; i++)
                garbage[i] = (byte)((i + 1) & 0xFF);
            File.WriteAllBytes(tempFile, garbage);

            Assert.Throws<InvalidDataException>(() => MdxPoseSourceLoader.LoadFromFile(tempFile));
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
            MdxPoseSourceLoader.LoadFromVirtualFile(@"H:\CLIENTS\data\model.mdx", new[] { "C:\\staging" }));
    }
}
