using System.Text;
using System.Text.Json;
using WowViewer.Core.IO.AssetReferences;
using WowViewer.Core.IO.Files;

namespace WowViewer.Tool.Inspect;

/// <summary>
/// Thin CLI surface over <see cref="AssetReferenceSweeper"/>. All analysis lives in the library; this
/// parses arguments, prints, and writes reports.
/// </summary>
public static class AssetReferenceCommandSupport
{
    public static void Run(string[] args)
    {
        if (args.Length == 0)
        {
            ShowUsage();
            return;
        }

        string command = args[0].ToLowerInvariant();
        string[] tail = args.Skip(1).ToArray();
        switch (command)
        {
            case "refs":
                RunRefs(tail);
                break;
            case "sweep":
                RunSweep(tail);
                break;
            default:
                Console.Error.WriteLine($"Unknown assets command '{command}'.");
                ShowUsage();
                Environment.ExitCode = 1;
                break;
        }
    }

    private static void ShowUsage()
    {
        Console.WriteLine("Asset reference commands:");
        Console.WriteLine("  assets refs --archive-root <game|data dir> --virtual-path <path/to/asset> [--listfile <listfile.txt>]");
        Console.WriteLine("  assets sweep --archive-root <game|data dir> --build <label> [--listfile <listfile.txt>] [--output <report.json>] [--limit <n>] [--missing-only]");
    }

    private static void RunRefs(string[] args)
    {
        string? archiveRoot = GetOption(args, "--archive-root", "-r");
        string? virtualPath = GetOption(args, "--virtual-path", "-v");
        if (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath))
        {
            Console.Error.WriteLine("Error: assets refs requires --archive-root and --virtual-path.");
            Environment.ExitCode = 1;
            return;
        }

        AssetReferenceSweeper sweeper = CreateSweeper(archiveRoot, GetOption(args, "--listfile", "--listfile"));
        ReferencingAssetResult result = AssetReferenceSweeper.IsWorldObject(virtualPath)
            ? sweeper.SweepWorldObject(virtualPath)
            : sweeper.SweepModel(virtualPath);

        Console.WriteLine($"ASSET: {result.Path}");
        Console.WriteLine($"STATE: {result.State}{(result.FailureDetail is null ? "" : $"  ({result.FailureDetail})")}");
        Console.WriteLine($"REFERENCES: {result.References.Count}");
        foreach (AssetReference reference in result.References)
            Console.WriteLine($"  [{reference.Resolution,-10}] {reference.Kind,-19} {reference.TargetPath}");

        int missing = result.References.Count(static r => r.Resolution == AssetResolution.Absent);
        if (missing > 0)
            Console.WriteLine($"UNRESOLVED: {missing}");
    }

    private static void RunSweep(string[] args)
    {
        string? archiveRoot = GetOption(args, "--archive-root", "-r");
        string? buildLabel = GetOption(args, "--build", "-b");
        if (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(buildLabel))
        {
            Console.Error.WriteLine("Error: assets sweep requires --archive-root and --build.");
            Environment.ExitCode = 1;
            return;
        }

        bool missingOnly = HasFlag(args, "--missing-only");
        string? outputPath = GetOption(args, "--output", "-o");
        AssetReferenceSweeper sweeper = CreateSweeper(archiveRoot, GetOption(args, "--listfile", "--listfile"));
        BuildIdentity build = new(buildLabel!, archiveRoot!);

        Console.WriteLine($"Sweeping {buildLabel} from {archiveRoot}");
        int lastPercent = -1;
        SweepReport report = sweeper.Sweep(build, (examined, total) =>
        {
            if (total == 0)
                return;

            int percent = (int)(100L * examined / total);
            if (percent == lastPercent)
                return;

            lastPercent = percent;
            Console.Write($"\r  {examined}/{total} ({percent}%)   ");
        });

        Console.WriteLine();
        PrintReport(report, missingOnly);

        if (!string.IsNullOrWhiteSpace(outputPath))
        {
            WriteReport(report, outputPath!);
            Console.WriteLine($"Report written to {outputPath}");
        }
    }

    private static void PrintReport(SweepReport report, bool missingOnly)
    {
        Console.WriteLine();
        Console.WriteLine($"BUILD:            {report.Build.Label}");
        Console.WriteLine($"World objects:    {report.WorldObjectsExamined}");
        Console.WriteLine($"Models:           {report.ModelsExamined}");
        Console.WriteLine($"References:       {report.ReferenceCount}");
        Console.WriteLine($"  via extension substitution: {report.SubstitutedReferenceCount}");
        Console.WriteLine($"Unresolved refs:  {report.UnresolvedReferenceCount}");
        Console.WriteLine($"Missing assets:   {report.DistinctMissingTargets.Count} distinct");
        Console.WriteLine($"Unreadable:       {report.AssetsUnreadable}");
        Console.WriteLine($"Blocked routes:   {report.BlockedRoutes.Count}");
        Console.WriteLine($"COMPLETE:         {report.Complete}");

        if (!report.Complete)
        {
            Console.WriteLine();
            Console.WriteLine("WARNING: this sweep is incomplete. Assets that could not be read contribute no");
            Console.WriteLine("references, so a low missing count here does NOT mean nothing is missing.");
        }

        Console.WriteLine();
        Console.WriteLine("Missing assets (distinct, referenced but not obtainable):");
        foreach (string target in report.DistinctMissingTargets.Take(missingOnly ? int.MaxValue : 50))
            Console.WriteLine($"  {target}");

        if (!missingOnly && report.DistinctMissingTargets.Count > 50)
            Console.WriteLine($"  ... {report.DistinctMissingTargets.Count - 50} more (use --missing-only or --output)");

        if (report.AssetsUnreadable > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Unreadable referencing assets (their references are UNKNOWN, not absent):");
            foreach (ReferencingAssetResult result in report.Results
                .Where(static r => r.State == ReferencingAssetState.Unreadable)
                .Take(20))
            {
                Console.WriteLine($"  {result.Path}  -- {result.FailureDetail}");
            }

            if (report.AssetsUnreadable > 20)
                Console.WriteLine($"  ... {report.AssetsUnreadable - 20} more");
        }
    }

    private static void WriteReport(SweepReport report, string outputPath)
    {
        var payload = new
        {
            build = new { label = report.Build.Label, rootLabel = report.Build.RootLabel },
            worldObjectsExamined = report.WorldObjectsExamined,
            modelsExamined = report.ModelsExamined,
            referenceCount = report.ReferenceCount,
            unresolvedReferenceCount = report.UnresolvedReferenceCount,
            assetsUnreadable = report.AssetsUnreadable,
            blockedRoutes = report.BlockedRoutes,
            referenceKindsSwept = report.ReferenceKindsSwept.Select(static k => k.ToString()),
            complete = report.Complete,
            missingTargets = report.DistinctMissingTargets,
            unreadableAssets = report.Results
                .Where(static r => r.State == ReferencingAssetState.Unreadable)
                .Select(static r => new { path = r.Path, detail = r.FailureDetail }),
            references = report.Results
                .Where(static r => r.References.Count > 0)
                .Select(static r => new
                {
                    source = r.Path,
                    references = r.References.Select(static reference => new
                    {
                        kind = reference.Kind.ToString(),
                        target = reference.TargetPath,
                        resolution = reference.Resolution.ToString(),
                        resolvedPath = reference.ResolvedPath,
                    }),
                }),
        };

        string? directory = Path.GetDirectoryName(Path.GetFullPath(outputPath));
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        File.WriteAllText(
            outputPath,
            JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }),
            Encoding.UTF8);
    }

    private static AssetReferenceSweeper CreateSweeper(string archiveRoot, string? listfilePath)
    {
        ArchiveCatalogSession session = ArchiveCatalogSessionCache.GetOrCreate(
            [archiveRoot],
            new ArchiveCatalogBootstrapOptions(ExternalListfilePath: listfilePath));

        return new AssetReferenceSweeper(session);
    }

    private static string? GetOption(string[] args, string longName, string shortName)
    {
        for (int index = 0; index < args.Length - 1; index++)
        {
            if (string.Equals(args[index], longName, StringComparison.OrdinalIgnoreCase)
                || string.Equals(args[index], shortName, StringComparison.OrdinalIgnoreCase))
            {
                return args[index + 1];
            }
        }

        return null;
    }

    private static bool HasFlag(string[] args, string name)
        => args.Any(arg => string.Equals(arg, name, StringComparison.OrdinalIgnoreCase));
}
