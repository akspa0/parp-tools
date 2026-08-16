using WowViewer.Core.IO.Wtf;

namespace WowViewer.Core.Tests;

/// <summary>Pins the honesty properties of a build-level WTF survey: unreadable files don't count as
/// zero-unrecognized, and repeated unrecognized shapes collapse to one entry without losing uniques.</summary>
public sealed class WtfBuildSurveyTests
{
    [Fact]
    public void UnreadableFile_IsNotCountedInLineTotals()
    {
        WtfBuildSurvey survey = new("build", [
            new WtfFileSurvey("a.wtf", WtfFileSource.Loose, [], "truncated"),
        ]);

        Assert.Equal(1, survey.FilesUnreadable);
        Assert.Equal(0, survey.TotalLines);
    }

    [Fact]
    public void DistinctUnrecognizedLines_CollapsesRepeatsButKeepsUniques()
    {
        WtfLine repeated = new("SAME LINE", WtfLineKind.Unrecognized);
        WtfLine unique = new("ONLY ONCE", WtfLineKind.Unrecognized);

        WtfBuildSurvey survey = new("build", [
            new WtfFileSurvey("a.wtf", WtfFileSource.Archive, [repeated, repeated]),
            new WtfFileSurvey("b.wtf", WtfFileSource.Archive, [repeated, unique]),
        ]);

        Assert.Equal(2, survey.DistinctUnrecognizedLines.Count);
        Assert.Contains("SAME LINE", survey.DistinctUnrecognizedLines);
        Assert.Contains("ONLY ONCE", survey.DistinctUnrecognizedLines);
    }

    [Fact]
    public void RecognizedLines_AreExcludedFromUnrecognizedTotal()
    {
        WtfFileSurvey file = new("a.wtf", WtfFileSource.Loose, [
            new WtfLine("SET x \"1\"", WtfLineKind.Set, "x", "1"),
            new WtfLine("bind W MOVEFORWARD", WtfLineKind.Bind, "W", "MOVEFORWARD"),
            new WtfLine("weird", WtfLineKind.Unrecognized),
        ]);

        Assert.Equal(2, file.RecognizedCount);
        Assert.Equal(1, file.UnrecognizedCount);
    }
}
