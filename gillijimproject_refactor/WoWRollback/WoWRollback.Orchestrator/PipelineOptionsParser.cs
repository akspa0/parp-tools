using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace WoWRollback.Orchestrator;

internal static class PipelineOptionsParser
{
    private const string DefaultOutputRoot = "parp_out";
    private static readonly string DefaultDbdRelative = Path.Combine("..", "lib", "WoWDBDefs", "definitions");
    private const string DefaultNoggitClientPath = "H:/WoWDev/modernwow";
    private const int DefaultPort = 8080;

    public static bool TryParse(string[] args, out PipelineOptions? options, out string? error)
    {
        options = null;
        error = null;

        if (args.Length == 0)
        {
            error = "Missing required arguments.";
            return false;
        }

        var parsed = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var flags = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        for (var i = 0; i < args.Length; i++)
        {
            var token = args[i];
            if (!token.StartsWith("--", StringComparison.Ordinal))
            {
                error = $"Unexpected argument '{token}'.";
                return false;
            }

            var key = token.Substring(2);
            if (string.Equals(key, "serve", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(key, "verbose", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(key, "verify", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(key, "run-verifier", StringComparison.OrdinalIgnoreCase))
            {
                flags.Add(key);
                continue;
            }

            if (i + 1 >= args.Length)
            {
                error = $"Missing value for '{token}'.";
                return false;
            }

            var value = args[++i];
            parsed[key] = value;
        }

        if (!parsed.TryGetValue("maps", out var mapsValue) || string.IsNullOrWhiteSpace(mapsValue))
        {
            error = "--maps is required.";
            return false;
        }

        if (!parsed.TryGetValue("versions", out var versionsValue) || string.IsNullOrWhiteSpace(versionsValue))
        {
            error = "--versions is required.";
            return false;
        }

        if (!parsed.TryGetValue("alpha-root", out var alphaRoot) || string.IsNullOrWhiteSpace(alphaRoot))
        {
            error = "--alpha-root is required.";
            return false;
        }

        var maps = SplitList(mapsValue);
        var versions = SplitList(versionsValue);

        if (maps.Count == 0)
        {
            error = "At least one map must be specified.";
            return false;
        }

        if (versions.Count == 0)
        {
            error = "At least one version must be specified.";
            return false;
        }

        var outputRoot = parsed.TryGetValue("output", out var outputValue) && !string.IsNullOrWhiteSpace(outputValue)
            ? outputValue
            : DefaultOutputRoot;

        var dbdDir = parsed.TryGetValue("dbd-dir", out var dbdValue) && !string.IsNullOrWhiteSpace(dbdValue)
            ? dbdValue
            : DefaultDbdRelative;

        parsed.TryGetValue("lk-dbc-dir", out var lkDbcDir);
        parsed.TryGetValue("community-listfile", out var communityListfile);
        parsed.TryGetValue("lk-listfile", out var lkListfile);
        parsed.TryGetValue("noggit-client-path", out var noggitClientPathRaw);
        parsed.TryGetValue("area-overrides", out var areaOverrides);
        var noggitClientPath = !string.IsNullOrWhiteSpace(noggitClientPathRaw)
            ? noggitClientPathRaw
            : DefaultNoggitClientPath;
        if (areaOverrides != null && string.IsNullOrWhiteSpace(areaOverrides))
        {
            areaOverrides = null;
        }

        var port = DefaultPort;
        if (parsed.TryGetValue("port", out var portValue) && !string.IsNullOrWhiteSpace(portValue))
        {
            if (!int.TryParse(portValue, NumberStyles.Integer, CultureInfo.InvariantCulture, out port) || port <= 0)
            {
                error = "--port must be a positive integer.";
                return false;
            }
        }

        options = new PipelineOptions(
            Maps: maps,
            Versions: versions,
            AlphaRoot: alphaRoot,
            OutputRoot: outputRoot,
            DbdDirectory: dbdDir,
            LkDbcDirectory: lkDbcDir,
            CommunityListfile: communityListfile,
            LkListfile: lkListfile,
            NoggitClientPath: noggitClientPath,
            RunVerifier: flags.Contains("verify") || flags.Contains("run-verifier"),
            AreaOverrideDirectory: areaOverrides,
            Serve: flags.Contains("serve"),
            Port: port,
            Verbose: flags.Contains("verbose"));

        return true;
    }

    private static IReadOnlyList<string> SplitList(string value)
    {
        return value
            .Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(v => v.Trim())
            .Where(v => !string.IsNullOrWhiteSpace(v))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }
}
