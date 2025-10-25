using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace WoWRollback.ViewerModule;

public static class DefaultPathsService
{
    public sealed class Defaults
    {
        public string? Wdt { get; init; }
        public string? CrosswalkDir { get; init; }
        public string? LkDbcDir { get; init; }
        public string? RepoRoot { get; init; }
    }

    public static Defaults Discover()
    {
        var baseDir = AppContext.BaseDirectory;
        var rootsToTry = Ascend(baseDir, 6).ToList();

        string? repoRoot = rootsToTry
            .FirstOrDefault(r => Directory.Exists(Path.Combine(r, "WoWRollback")) ||
                                 File.Exists(Path.Combine(r, ".git")) ||
                                 Directory.Exists(Path.Combine(r, ".git")));

        // Fallback: current working directory
        repoRoot ??= Directory.GetCurrentDirectory();

        var testData = FindFirstExisting(
            rootsToTry.Select(r => Path.Combine(r, "test_data")).Append(Path.Combine(repoRoot!, "test_data"))
        );

        string? wdt = null;
        string? lkDbcDir = null;
        string? crosswalkDir = null;

        if (testData != null)
        {
            wdt = TryFindAny(testData, "*.wdt");
            // Prefer common maps if available
            wdt = PreferIfPresent(wdt, TryFindExact(testData, Path.Combine("World", "Maps", "Kalimdor", "Kalimdor.wdt")));
            wdt = PreferIfPresent(wdt, TryFindExact(testData, Path.Combine("World", "Maps", "Azeroth", "Azeroth.wdt")));

            // DBC dir: look for DBFilesClient or dbc folder
            lkDbcDir = FindFirstExisting(new[]
            {
                TryFindDirectory(testData, "DBFilesClient"),
                TryFindDirectory(testData, "dbc"),
            });
        }

        // Crosswalks: prefer repo crosswalks/ then test_data/crosswalks
        crosswalkDir = FirstNonNull(
            Directory.Exists(Path.Combine(repoRoot!, "crosswalks")) ? Path.Combine(repoRoot!, "crosswalks") : null,
            testData != null && Directory.Exists(Path.Combine(testData, "crosswalks")) ? Path.Combine(testData, "crosswalks") : null
        );

        return new Defaults
        {
            Wdt = wdt,
            CrosswalkDir = crosswalkDir,
            LkDbcDir = lkDbcDir,
            RepoRoot = repoRoot
        };
    }

    private static IEnumerable<string> Ascend(string start, int maxLevels)
    {
        var cur = new DirectoryInfo(start);
        for (int i = 0; i < maxLevels && cur != null; i++)
        {
            yield return cur.FullName;
            cur = cur.Parent;
        }
    }

    private static string? TryFindAny(string root, string pattern)
    {
        try
        {
            var matches = Directory.EnumerateFiles(root, pattern, SearchOption.AllDirectories);
            return matches.FirstOrDefault();
        }
        catch { return null; }
    }

    private static string? TryFindExact(string root, string relative)
    {
        try
        {
            var full = Path.Combine(root, relative);
            return File.Exists(full) ? full : null;
        }
        catch { return null; }
    }

    private static string? TryFindDirectory(string root, string name)
    {
        try
        {
            // Search exact dir name anywhere under root (shallow heuristic)
            var dirs = Directory.EnumerateDirectories(root, name, SearchOption.AllDirectories);
            return dirs.FirstOrDefault();
        }
        catch { return null; }
    }

    private static string? FindFirstExisting(IEnumerable<string?> candidates)
    {
        foreach (var c in candidates)
        {
            if (!string.IsNullOrWhiteSpace(c) && Directory.Exists(c))
                return c;
        }
        return null;
    }

    private static string? FirstNonNull(params string?[] items)
    {
        foreach (var x in items)
        {
            if (!string.IsNullOrWhiteSpace(x)) return x;
        }
        return null;
    }

    private static string? PreferIfPresent(string? current, string? preferred)
    {
        return !string.IsNullOrWhiteSpace(preferred) ? preferred : current;
    }
}
