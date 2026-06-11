namespace WoWViewer.Logging;

/// <summary>
/// Profiling switches and accumulators for PM4 hot paths (click pick, per-frame
/// research-info walk, per-frame scene-graph rebuild). Default off; toggle with
/// Pm4Profiling.Enabled = true from the dev console, or set the env var
/// WOWVIEWER_PM4_PROFILE=1 before launch. Accumulators are static so they
/// survive across frames and between the world-scene (where the work happens)
/// and the viewer-app (where the timing logs are written).
/// </summary>
public static class Pm4Profiling
{
    public static bool Enabled
    {
        get
        {
            if (s_enabled.HasValue) return s_enabled.Value;
            string? env = System.Environment.GetEnvironmentVariable("WOWVIEWER_PM4_PROFILE");
            s_enabled = !string.IsNullOrEmpty(env) && (env == "1" || env.Equals("true", System.StringComparison.OrdinalIgnoreCase));
            return s_enabled.Value;
        }
        set => s_enabled = value;
    }

    private static bool? s_enabled;
}
