using System.Text.Json;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services.Config;

public static class RangeConfigLoader
{
    public static RangeConfig LoadFromJson(string path)
    {
        var json = File.ReadAllText(path);
        var cfg = JsonSerializer.Deserialize<RangeConfig>(json, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        });
        if (cfg == null) throw new InvalidOperationException($"Failed to parse config: {path}");
        if (string.IsNullOrWhiteSpace(cfg.Mode)) cfg.Mode = "keep";
        return cfg;
    }

    public static void ApplyCliOverrides(RangeConfig config, IEnumerable<string> keepRanges, IEnumerable<string> dropRanges)
    {
        foreach (var spec in keepRanges)
        {
            if (TryParseRange(spec, out var rr)) config.IncludeRanges.Add(rr);
        }
        foreach (var spec in dropRanges)
        {
            if (TryParseRange(spec, out var rr)) config.ExcludeRanges.Add(rr);
        }
    }

    private static bool TryParseRange(string spec, out RangeRule rule)
    {
        rule = new RangeRule(0, 0);
        var parts = spec.Split(':', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length != 2) return false;
        if (!uint.TryParse(parts[0], out var min)) return false;
        if (!uint.TryParse(parts[1], out var max)) return false;
        if (min > max) (min, max) = (max, min);
        rule = new RangeRule(min, max);
        return true;
    }
}
