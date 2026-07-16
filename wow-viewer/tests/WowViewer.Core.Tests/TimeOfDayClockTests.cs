using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class TimeOfDayClockTests
{
    [Theory]
    [InlineData("1215", 12, 15)]
    [InlineData("0015", 0, 15)]
    [InlineData("23:59", 23, 59)]
    [InlineData("9:05", 9, 5)]
    [InlineData("12", 12, 0)]
    [InlineData("12.25", 12, 15)]
    [InlineData("23.999", 23, 59)]
    public void TryParse_AcceptsCompactColonAndLegacyDecimalForms(string input, int expectedHour, int expectedMinute)
    {
        bool parsed = TimeOfDayClock.TryParse(input, out TimeOfDayClock clock);

        Assert.True(parsed);
        Assert.Equal(expectedHour, clock.Hour);
        Assert.Equal(expectedMinute, clock.Minute);
    }

    [Theory]
    [InlineData("2400")]
    [InlineData("1260")]
    [InlineData("24:00")]
    [InlineData("12:60")]
    [InlineData("-1")]
    [InlineData("not-a-time")]
    public void TryParse_RejectsOutOfDayAndMalformedValues(string input)
    {
        Assert.False(TimeOfDayClock.TryParse(input, out _));
    }

    [Fact]
    public void FromHours_FormatsMinutePreciseClockText()
    {
        TimeOfDayClock clock = TimeOfDayClock.FromHours(12.25f);

        Assert.Equal("12:15", clock.ToString());
        Assert.Equal("1215", clock.CompactText);
        Assert.Equal(12.25f, clock.Hours);
    }
}
