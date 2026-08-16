using WowViewer.Core.IO.Wtf;

namespace WowViewer.Core.Tests;

/// <summary>
/// Pins WtfLineClassifier against the two statement shapes confirmed from real client data: SET (from
/// 0.5.3.3368's Config.wtf) and bind (from 0.5.3.3368's WTF\DefaultBindings.wtf, including the exact
/// real "bind ALT-P TOGGLEPERFORMANCEDISPLAY" line).
/// </summary>
public sealed class WtfLineClassifierTests
{
    [Fact]
    public void SetStatement_IsRecognizedWithNameAndValue()
    {
        WtfLine line = WtfLineClassifier.Classify("SET gxColorBits \"24\"");

        Assert.Equal(WtfLineKind.Set, line.Kind);
        Assert.Equal("gxColorBits", line.Name);
        Assert.Equal("24", line.Value);
    }

    [Fact]
    public void SetStatement_WithEmptyValue_IsRecognized()
    {
        // Real line from 0.5.3.3368's Config.wtf: SET time ""
        WtfLine line = WtfLineClassifier.Classify("SET time \"\"");

        Assert.Equal(WtfLineKind.Set, line.Kind);
        Assert.Equal("time", line.Name);
        Assert.Equal("", line.Value);
    }

    [Fact]
    public void SetStatement_WithUnquotedValue_IsRecognized()
    {
        // Real line from 2.0.0.5610's realmlist.wtf: lowercase keyword, no quotes.
        WtfLine line = WtfLineClassifier.Classify("set realmlist beta.us.logon.worldofwarcraft.com");

        Assert.Equal(WtfLineKind.Set, line.Kind);
        Assert.Equal("realmlist", line.Name);
        Assert.Equal("beta.us.logon.worldofwarcraft.com", line.Value);
    }

    [Fact]
    public void BindStatement_IsRecognizedWithKeyAndAction()
    {
        // The real, measured line that confirmed Spec 158's Alt+P assumption.
        WtfLine line = WtfLineClassifier.Classify("bind ALT-P TOGGLEPERFORMANCEDISPLAY");

        Assert.Equal(WtfLineKind.Bind, line.Kind);
        Assert.Equal("ALT-P", line.Name);
        Assert.Equal("TOGGLEPERFORMANCEDISPLAY", line.Value);
    }

    [Fact]
    public void BindStatement_WithMultiModifierKey_IsRecognized()
    {
        // Real line: bind CTRL-SHIFT-PAGEDOWN COMBATLOGBOTTOM
        WtfLine line = WtfLineClassifier.Classify("bind CTRL-SHIFT-PAGEDOWN COMBATLOGBOTTOM");

        Assert.Equal(WtfLineKind.Bind, line.Kind);
        Assert.Equal("CTRL-SHIFT-PAGEDOWN", line.Name);
    }

    [Theory]
    [InlineData("worldport 0 1234.5 -678.9 12.3")]
    [InlineData("teleport 45.0 -12.0 100.0")]
    [InlineData("# a comment")]
    [InlineData("garbage line with no recognizable shape")]
    public void UnrecognizedLine_KeepsExactOriginalText(string rawLine)
    {
        WtfLine line = WtfLineClassifier.Classify(rawLine);

        Assert.Equal(WtfLineKind.Unrecognized, line.Kind);
        Assert.Equal(rawLine, line.RawText);
    }

    [Fact]
    public void ClassifyFile_SkipsBlankLines_ClassifiesRest()
    {
        string text = "SET a \"1\"\n\n\nbind W MOVEFORWARD\n   \nunrecognized here";

        IReadOnlyList<WtfLine> lines = WtfLineClassifier.ClassifyFile(text);

        Assert.Equal(3, lines.Count);
        Assert.Equal(WtfLineKind.Set, lines[0].Kind);
        Assert.Equal(WtfLineKind.Bind, lines[1].Kind);
        Assert.Equal(WtfLineKind.Unrecognized, lines[2].Kind);
    }
}
