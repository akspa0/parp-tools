using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Wtf;

namespace WowViewer.Tool.Inspect;

/// <summary>Thin CLI surface over <see cref="WtfSweeper"/>. All analysis lives in the library.</summary>
public static class WtfCommandSupport
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
            case "sweep":
                RunSweep(tail);
                break;
            case "probe":
                RunProbe(tail);
                break;
            default:
                Console.Error.WriteLine($"Unknown wtf command '{command}'.");
                ShowUsage();
                Environment.ExitCode = 1;
                break;
        }
    }

    private static void ShowUsage()
    {
        Console.WriteLine("WTF commands:");
        Console.WriteLine("  wtf sweep --archive-root <game dir> --build <label>");
        Console.WriteLine("  wtf probe --archive-root <game dir> --build <label> --name <candidate1> [--name <candidate2> ...]");
    }

    private static void RunSweep(string[] args)
    {
        string? archiveRoot = GetOption(args, "--archive-root", "-r");
        string? buildLabel = GetOption(args, "--build", "-b");
        if (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(buildLabel))
        {
            Console.Error.WriteLine("Error: wtf sweep requires --archive-root and --build.");
            Environment.ExitCode = 1;
            return;
        }

        WtfSweeper sweeper = CreateSweeper(archiveRoot);
        WtfBuildSurvey survey = sweeper.Sweep(buildLabel);
        PrintSurvey(survey);
    }

    private static void RunProbe(string[] args)
    {
        string? archiveRoot = GetOption(args, "--archive-root", "-r");
        string? buildLabel = GetOption(args, "--build", "-b");
        List<string> candidates = GetAllOptions(args, "--name", "-n");
        if (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(buildLabel) || candidates.Count == 0)
        {
            Console.Error.WriteLine("Error: wtf probe requires --archive-root, --build, and at least one --name.");
            Environment.ExitCode = 1;
            return;
        }

        WtfSweeper sweeper = CreateSweeper(archiveRoot);
        Console.WriteLine($"BUILD: {buildLabel}");
        foreach (string candidate in candidates)
        {
            WtfCandidateProbeResult result = sweeper.ProbeCandidate(candidate);
            if (!result.Resolved)
            {
                Console.WriteLine($"  NOT FOUND: {candidate}");
                continue;
            }

            Console.WriteLine($"  FOUND ({result.Survey!.Source}): {candidate}");
            PrintFile(result.Survey, indent: "    ");
        }
    }

    private static void PrintSurvey(WtfBuildSurvey survey)
    {
        Console.WriteLine($"BUILD:              {survey.BuildLabel}");
        Console.WriteLine($"Files found:        {survey.FilesFound}");
        Console.WriteLine($"Files unreadable:   {survey.FilesUnreadable}");
        Console.WriteLine($"Total lines:        {survey.TotalLines}");
        Console.WriteLine($"Recognized:         {survey.TotalRecognized}");
        Console.WriteLine($"Unrecognized:       {survey.TotalUnrecognized}");
        Console.WriteLine();

        foreach (WtfFileSurvey file in survey.Files)
            PrintFile(file, indent: "  ");

        if (survey.DistinctUnrecognizedLines.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine($"Distinct unrecognized lines ({survey.DistinctUnrecognizedLines.Count}):");
            foreach (string line in survey.DistinctUnrecognizedLines)
                Console.WriteLine($"  {line}");
        }
    }

    private static void PrintFile(WtfFileSurvey file, string indent)
    {
        if (!file.Readable)
        {
            Console.WriteLine($"{indent}{file.Path} [{file.Source}] -- UNREADABLE: {file.FailureDetail}");
            return;
        }

        Console.WriteLine($"{indent}{file.Path} [{file.Source}] -- {file.Lines.Count} lines " +
            $"({file.RecognizedCount} recognized, {file.UnrecognizedCount} unrecognized)");
    }

    private static WtfSweeper CreateSweeper(string archiveRoot)
    {
        ArchiveCatalogSession session = ArchiveCatalogSessionCache.GetOrCreate([archiveRoot]);
        return new WtfSweeper(session);
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

    private static List<string> GetAllOptions(string[] args, string longName, string shortName)
    {
        List<string> values = [];
        for (int index = 0; index < args.Length - 1; index++)
        {
            if (string.Equals(args[index], longName, StringComparison.OrdinalIgnoreCase)
                || string.Equals(args[index], shortName, StringComparison.OrdinalIgnoreCase))
            {
                values.Add(args[index + 1]);
            }
        }

        return values;
    }
}
