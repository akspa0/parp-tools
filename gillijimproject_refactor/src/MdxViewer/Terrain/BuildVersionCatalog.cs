using System.Text.RegularExpressions;

namespace MdxViewer.Terrain;

public readonly record struct ClientBuildOption(string Label, string BuildVersion);

public static class BuildVersionCatalog
{
    private static readonly Regex FullBuildRegex = new(@"\b(\d+\.\d+\.\d+\.\d+)\b", RegexOptions.Compiled);
    private static readonly Regex ShortBuildRegex = new(@"\b(\d+\.\d+\.\d+)\b", RegexOptions.Compiled);

    public static List<ClientBuildOption> LoadOptionsFromMapDbd(string dbdDefinitionsDir)
    {
        var mapDbdPath = Path.Combine(dbdDefinitionsDir, "Map.dbd");
        if (!File.Exists(mapDbdPath))
            return new List<ClientBuildOption>();

        var builds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var line in File.ReadLines(mapDbdPath))
        {
            var trimmed = line.Trim();
            if (!trimmed.StartsWith("BUILD ", StringComparison.OrdinalIgnoreCase))
                continue;

            var entries = trimmed[6..].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            foreach (var entry in entries)
            {
                var rangeParts = entry.Split('-', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                foreach (var part in rangeParts)
                {
                    if (TryParseBuild(part, out _))
                        builds.Add(part);
                }
            }
        }

        var parsed = new List<(BuildKey Key, string Build)>(builds.Count);
        foreach (var build in builds)
        {
            if (TryParseBuild(build, out var key))
                parsed.Add((key, build));
        }

        parsed.Sort(static (a, b) =>
        {
            int major = a.Key.Major.CompareTo(b.Key.Major);
            if (major != 0) return major;
            int minor = a.Key.Minor.CompareTo(b.Key.Minor);
            if (minor != 0) return minor;
            int patch = a.Key.Patch.CompareTo(b.Key.Patch);
            if (patch != 0) return patch;
            return a.Key.Build.CompareTo(b.Key.Build);
        });

        var result = new List<ClientBuildOption>(parsed.Count);
        foreach (var (key, build) in parsed)
        {
            string family = GetMajorFamilyLabel(key.Major);
            string label = $"{family} ({key.Major}.x) - {build}";
            result.Add(new ClientBuildOption(label, build));
        }

        return result;
    }

    public static bool TryInferBuildIndexFromPath(IReadOnlyList<ClientBuildOption> options, string gamePath, out int index)
    {
        index = -1;
        if (options.Count == 0 || string.IsNullOrWhiteSpace(gamePath))
            return false;

        var byBuild = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < options.Count; i++)
            byBuild[options[i].BuildVersion] = i;

        foreach (Match match in FullBuildRegex.Matches(gamePath))
        {
            string candidate = match.Groups[1].Value;
            if (byBuild.TryGetValue(candidate, out index))
                return true;
        }

        foreach (Match match in ShortBuildRegex.Matches(gamePath))
        {
            string prefix = match.Groups[1].Value + ".";
            for (int i = options.Count - 1; i >= 0; i--)
            {
                if (options[i].BuildVersion.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                {
                    index = i;
                    return true;
                }
            }
        }

        return false;
    }

    private static string GetMajorFamilyLabel(int major)
    {
        return major switch
        {
            0 => "Alpha",
            1 => "Classic",
            2 => "Burning Crusade",
            3 => "Wrath",
            4 => "Cataclysm",
            5 => "Mists",
            6 => "Warlords",
            7 => "Legion",
            8 => "Battle for Azeroth",
            9 => "Shadowlands",
            10 => "Dragonflight",
            11 => "The War Within",
            _ => "Major"
        };
    }

    private static bool TryParseBuild(string? build, out BuildKey key)
    {
        key = default;
        if (string.IsNullOrWhiteSpace(build))
            return false;

        string[] parts = build.Split('.');
        if (parts.Length != 4)
            return false;

        if (!int.TryParse(parts[0], out int major)
            || !int.TryParse(parts[1], out int minor)
            || !int.TryParse(parts[2], out int patch)
            || !int.TryParse(parts[3], out int revision))
        {
            return false;
        }

        key = new BuildKey(major, minor, patch, revision);
        return true;
    }

    private readonly record struct BuildKey(int Major, int Minor, int Patch, int Build);
}