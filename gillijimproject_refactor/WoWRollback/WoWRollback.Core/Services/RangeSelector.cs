using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services;

public static class RangeSelector
{
    public static bool ShouldRemove(RangeConfig cfg, uint uniqueId)
    {
        // Exclude ranges take precedence: if id in exclude -> remove
        if (cfg.ExcludeRanges != null && cfg.ExcludeRanges.Count > 0)
        {
            if (IsInRanges(cfg.ExcludeRanges!, uniqueId)) return true;
        }

        // Include ranges behavior depends on mode
        var mode = (cfg.Mode ?? "keep").Trim().ToLowerInvariant();
        var haveInclude = cfg.IncludeRanges != null && cfg.IncludeRanges.Count > 0;

        if (!haveInclude)
        {
            // No include ranges provided: default to keep everything
            return false;
        }

        var inInclude = haveInclude && IsInRanges(cfg.IncludeRanges!, uniqueId);

        return mode switch
        {
            // keep: remove if NOT in include ranges
            "keep" => !inInclude,
            // drop: remove if in include ranges
            "drop" => inInclude,
            // fallback: treat unknown as keep
            _ => !inInclude
        };
    }

    private static bool IsInRanges(List<RangeRule> ranges, uint id)
        => ranges.Any(r => id >= r.Min && id <= r.Max);
}
