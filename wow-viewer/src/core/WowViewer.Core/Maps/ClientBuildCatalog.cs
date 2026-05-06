using System.Text.RegularExpressions;

namespace WowViewer.Core.Maps;

public readonly record struct ClientBuildKey(int Major, int Minor, int Patch, int Build)
{
    public static ClientBuildKey FromVersion(string version) =>
        TryParse(version, out var key) ? key : default;

    public static bool TryParse(string? version, out ClientBuildKey key)
    {
        key = default;
        if (string.IsNullOrWhiteSpace(version))
            return false;

        string[] parts = version.Split('.');
        if (parts.Length < 4)
            return false;

        return int.TryParse(parts[0], out int major)
            && int.TryParse(parts[1], out int minor)
            && int.TryParse(parts[2], out int patch)
            && int.TryParse(parts[3], out int build)
            && (key = new ClientBuildKey(major, minor, patch, build)) is var k && true;
    }

    public string ToVersionString() => $"{Major}.{Minor}.{Patch}.{Build}";

    public string MajorFamilyLabel => Major switch
    {
        0 => "Alpha",
        2 => "Classic",
        3 => "TBC/WotLK",
        4 => "Cataclysm",
        5 => "MoP",
        6 => "WoD",
        7 => "Legion",
        8 => "BfA",
        9 => "Shadowlands",
        10 => "Dragonflight",
        _ => $"Unknown({Major})"
    };

    public bool IsAlpha => Major == 0;
    public bool IsWotLK => Major == 3 && Minor >= 0 && Minor <= 3;
    public bool IsCata => Major == 4;
}

public readonly record struct ClientBuildOption(string Label, string BuildVersion);

public static class ClientBuildCatalog
{
    private static readonly Regex FullBuildRegex = new(@"\b(\d+\.\d+\.\d+\.\d+)\b", RegexOptions.Compiled);

    public static List<ClientBuildOption> LoadFromMapDbd(string dbdDefinitionsDir)
    {
        var mapDbdPath = Path.Combine(dbdDefinitionsDir, "Map.dbd");
        if (!File.Exists(mapDbdPath))
            return new List<ClientBuildOption>();

        var builds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string line in File.ReadLines(mapDbdPath))
        {
            string trimmed = line.Trim();
            if (!trimmed.StartsWith("BUILD ", StringComparison.OrdinalIgnoreCase))
                continue;

            string[] entries = trimmed[6..].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            foreach (string entry in entries)
            {
                string[] rangeParts = entry.Split('-', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                foreach (string part in rangeParts)
                {
                    if (ClientBuildKey.TryParse(part, out _))
                        builds.Add(part);
                }
            }
        }

        var parsed = new List<(ClientBuildKey Key, string Build)>(builds.Count);
        foreach (string build in builds)
        {
            if (ClientBuildKey.TryParse(build, out var key))
                parsed.Add((key, build));
        }

        parsed.Sort(static (a, b) =>
        {
            int c = a.Key.Major.CompareTo(b.Key.Major);
            if (c != 0) return c;
            c = a.Key.Minor.CompareTo(b.Key.Minor);
            if (c != 0) return c;
            c = a.Key.Patch.CompareTo(b.Key.Patch);
            if (c != 0) return c;
            return a.Key.Build.CompareTo(b.Key.Build);
        });

        var result = new List<ClientBuildOption>(parsed.Count);
        foreach (var (key, build) in parsed)
        {
            string label = $"{key.MajorFamilyLabel} ({key.Major}.x) - {build}";
            result.Add(new ClientBuildOption(label, build));
        }
        return result;
    }

    public static bool TryInferBuildFromPath(IReadOnlyList<ClientBuildOption> options, string gamePath, out int index)
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

        return false;
    }
}