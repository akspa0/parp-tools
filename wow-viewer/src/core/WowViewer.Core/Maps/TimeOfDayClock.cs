using System.Globalization;

namespace WowViewer.Core.Maps;

/// <summary>
/// A minute-precise clock time within one client day.
/// </summary>
public readonly record struct TimeOfDayClock(int Hour, int Minute)
{
    public const int MinutesPerDay = 24 * 60;

    public float Hours => (Hour * 60 + Minute) / 60f;

    public string CompactText => $"{Hour:00}{Minute:00}";

    public override string ToString() => $"{Hour:00}:{Minute:00}";

    public static bool TryParse(string? value, out TimeOfDayClock clock)
    {
        clock = default;
        if (string.IsNullOrWhiteSpace(value))
            return false;

        string trimmed = value.Trim();
        if (TryParseCompact(trimmed, out clock) || TryParseColonSeparated(trimmed, out clock))
            return true;

        if (!float.TryParse(trimmed, NumberStyles.Float, CultureInfo.InvariantCulture, out float decimalHours)
            || !float.IsFinite(decimalHours)
            || decimalHours < 0f
            || decimalHours >= 24f)
        {
            return false;
        }

        int totalMinutes = (int)MathF.Round(decimalHours * 60f, MidpointRounding.AwayFromZero);
        if (totalMinutes < 0)
            return false;
        if (totalMinutes >= MinutesPerDay)
            totalMinutes = MinutesPerDay - 1;

        clock = new TimeOfDayClock(totalMinutes / 60, totalMinutes % 60);
        return true;
    }

    public static TimeOfDayClock FromHours(float hours)
    {
        if (!float.IsFinite(hours) || hours < 0f || hours >= 24f)
            throw new ArgumentOutOfRangeException(nameof(hours), "Time must be within [0, 24).");

        int totalMinutes = (int)MathF.Round(hours * 60f, MidpointRounding.AwayFromZero);
        if (totalMinutes >= MinutesPerDay)
            totalMinutes = MinutesPerDay - 1;

        return new TimeOfDayClock(totalMinutes / 60, totalMinutes % 60);
    }

    private static bool TryParseCompact(string value, out TimeOfDayClock clock)
    {
        clock = default;
        if (value.Length != 4 || !value.All(char.IsAsciiDigit))
            return false;

        int hour = int.Parse(value.AsSpan(0, 2), CultureInfo.InvariantCulture);
        int minute = int.Parse(value.AsSpan(2, 2), CultureInfo.InvariantCulture);
        return TryCreate(hour, minute, out clock);
    }

    private static bool TryParseColonSeparated(string value, out TimeOfDayClock clock)
    {
        clock = default;
        string[] parts = value.Split(':', StringSplitOptions.None);
        if (parts.Length != 2
            || !int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out int hour)
            || !int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out int minute))
        {
            return false;
        }

        return TryCreate(hour, minute, out clock);
    }

    private static bool TryCreate(int hour, int minute, out TimeOfDayClock clock)
    {
        clock = default;
        if (hour is < 0 or >= 24 || minute is < 0 or >= 60)
            return false;

        clock = new TimeOfDayClock(hour, minute);
        return true;
    }
}
