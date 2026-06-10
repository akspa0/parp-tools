using WowViewer.Core.Anim;

namespace WowViewer.Core.Anim.Tests;

public sealed class PathNormalizerTests
{
    [Fact]
    public void NormalizeForOutput_ReplacesBackslashesWithForwardSlashes()
    {
        string result = PathNormalizer.NormalizeForOutput(@"C:\Game\Data\Model.m2");
        Assert.Equal("c:/game/data/model.m2", result);
    }

    [Fact]
    public void NormalizeForOutput_LowercasesEntirePath()
    {
        string result = PathNormalizer.NormalizeForOutput(@"C:\Game\Data\Creature\Orc\ORC.M2");
        Assert.Equal("c:/game/data/creature/orc/orc.m2", result);
    }

    [Fact]
    public void NormalizeForOutput_PreservesExistingForwardSlashes()
    {
        string result = PathNormalizer.NormalizeForOutput("c:/game/data/model.m2");
        Assert.Equal("c:/game/data/model.m2", result);
    }

    [Fact]
    public void NormalizeForOutput_ThrowsOnNullOrEmpty()
    {
        Assert.Throws<ArgumentException>(() => PathNormalizer.NormalizeForOutput(""));
        Assert.Throws<ArgumentNullException>(() => PathNormalizer.NormalizeForOutput(null!));
    }

    [Fact]
    public void AssertNoStalePath_RejectsHColonClients()
    {
        Assert.Throws<InvalidOperationException>(() =>
            PathNormalizer.AssertNoStalePath(@"H:\CLIENTS\Data\model.m2"));
    }

    [Fact]
    public void AssertNoStalePath_RejectsHColonClients_CaseInsensitive()
    {
        Assert.Throws<InvalidOperationException>(() =>
            PathNormalizer.AssertNoStalePath(@"h:\clients\data\model.m2"));
    }

    [Fact]
    public void AssertNoStalePath_RejectsHColonClients_ForwardSlashes()
    {
        Assert.Throws<InvalidOperationException>(() =>
            PathNormalizer.AssertNoStalePath("H:/CLIENTS/data/model.m2"));
    }

    [Fact]
    public void AssertNoStalePath_AllowsStagedClientRoot()
    {
        string allowed = @"I:\parp\parp-tools\output\tmp\wowarchive-clients\3.3.5\Creature\Orc.m2";
        PathNormalizer.AssertNoStalePath(allowed);
    }

    [Fact]
    public void AssertNoStalePath_NoOpForNullOrEmpty()
    {
        PathNormalizer.AssertNoStalePath(null!);
        PathNormalizer.AssertNoStalePath("");
        PathNormalizer.AssertNoStalePath("   ");
    }

    [Fact]
    public void AssertNoStalePath_DoesNotFalsePositiveOnSimilarPrefixes()
    {
        PathNormalizer.AssertNoStalePath(@"C:\HCLIENTS\model.m2");
        PathNormalizer.AssertNoStalePath(@"H:\DATA\model.m2");
    }

    [Fact]
    public void NormalizeAndAssert_ComposesBoth()
    {
        string result = PathNormalizer.NormalizeAndAssert(@"C:\Game\Data\model.m2");
        Assert.Equal("c:/game/data/model.m2", result);
    }

    [Fact]
    public void NormalizeAndAssert_RejectsStaleBeforeNormalizing()
    {
        Assert.Throws<InvalidOperationException>(() =>
            PathNormalizer.NormalizeAndAssert(@"H:\CLIENTS\model.m2"));
    }
}
