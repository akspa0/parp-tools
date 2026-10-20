using System.Text.RegularExpressions;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Services;

public static class Pm4DesignKitHints
{
    public static IReadOnlyList<Pm4DesignKitSourceHint> Merge(
        IReadOnlyList<Pm4DesignKitSourceHint>? existing, IReadOnlyList<Pm4DesignKitSourceHint> imported)
    {
        // Keep confirmations and sources absent from this import; never erase operator work.
        var result = (existing ?? []).ToDictionary(h => h.SourcePath, StringComparer.OrdinalIgnoreCase);
        foreach (var hint in imported)
            if (!result.TryGetValue(hint.SourcePath, out var old) || !old.OperatorConfirmed)
                result[hint.SourcePath] = hint;
        return result.Values.OrderBy(h => h.SourcePath, StringComparer.Ordinal).ToArray();
    }

    public static IReadOnlyList<Pm4DesignKitAlias> OperatorProvidedAliases { get; } = Array.AsReadOnly(new[]
    {
        new Pm4DesignKitAlias("HFjord", "Howling Fjord", "Northrend"),
        new Pm4DesignKitAlias("Howling Fjord", "Howling Fjord", "Northrend"),
        new Pm4DesignKitAlias("Grizzly", "Grizzly Hills", "Northrend"),
    });

    public static IReadOnlyList<Pm4DesignKitSourceHint> FromSourceNames(
        IEnumerable<string> paths, IReadOnlyList<Pm4DesignKitAlias>? aliases = null)
    {
        ArgumentNullException.ThrowIfNull(paths);
        var rules = (aliases ?? OperatorProvidedAliases).Select(a =>
        {
            ArgumentNullException.ThrowIfNull(a);
            ArgumentException.ThrowIfNullOrWhiteSpace(a.Token);
            ArgumentException.ThrowIfNullOrWhiteSpace(a.Zone);
            ArgumentException.ThrowIfNullOrWhiteSpace(a.DesignKit);
            return (Alias: a, Pattern: new Regex(@"(?<![a-z0-9])" + Regex.Escape(a.Token) + @"(?![a-z0-9])",
                RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, TimeSpan.FromSeconds(1)));
        }).ToArray();
        return paths.Order(StringComparer.Ordinal).Select(path =>
        {
            string file = path.Replace('\\', '/').Split('/')[^1];
            var matched = rules.Where(r => r.Pattern.IsMatch(file)).Select(r => r.Alias).ToArray();
            return new Pm4DesignKitSourceHint(path.Replace('\\', '/'), matched.Select(a => a.Token).Distinct().ToArray(),
                matched.Select(a => a.Zone).Distinct().ToArray(), matched.Select(a => a.DesignKit).Distinct().ToArray(),
                "Filename hint only; exact tile/footprint association not established.", false);
        }).ToArray();
    }
}
