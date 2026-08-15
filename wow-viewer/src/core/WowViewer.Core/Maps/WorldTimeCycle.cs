namespace WowViewer.Core.Maps;

/// <summary>
/// Converts real elapsed time to the original Alpha 0.5.3 world clock.
/// The client day is 2,880 time units and completes in 24 real minutes.
/// </summary>
public static class WorldTimeCycle
{
    public const int TimeUnitsPerDay = 2880;
    public const double RealSecondsPerDay = 24d * 60d;

    /// <summary>
    /// Advances a normalized game time in [0, 1) by real elapsed seconds.
    /// </summary>
    public static float AdvanceNormalized(float normalizedTime, double realSeconds)
    {
        normalizedTime = Normalize(normalizedTime);
        if (!double.IsFinite(realSeconds) || realSeconds <= 0d)
            return normalizedTime;

        return Normalize(normalizedTime + (float)(realSeconds / RealSecondsPerDay));
    }

    /// <summary>Converts a normalized game time to the Light/LIT 0..2880 clock.</summary>
    public static int ToTimeUnits(float normalizedTime)
    {
        normalizedTime = Normalize(normalizedTime);
        return Math.Clamp(
            (int)MathF.Round(normalizedTime * TimeUnitsPerDay, MidpointRounding.AwayFromZero),
            0,
            TimeUnitsPerDay);
    }

    /// <summary>Converts Light/LIT time units to normalized game time.</summary>
    public static float FromTimeUnits(int timeUnits)
    {
        int wrapped = timeUnits % TimeUnitsPerDay;
        if (wrapped < 0)
            wrapped += TimeUnitsPerDay;

        return wrapped / (float)TimeUnitsPerDay;
    }

    public static float Normalize(float normalizedTime)
    {
        if (!float.IsFinite(normalizedTime))
            return 0.5f;

        float wrapped = normalizedTime - MathF.Floor(normalizedTime);
        return wrapped >= 1f ? 0f : wrapped;
    }
}
