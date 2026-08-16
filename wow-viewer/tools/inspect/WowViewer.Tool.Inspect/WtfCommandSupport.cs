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
        Console.WriteLine("  wtf sweep --archive-root <game dir> --build <label> [--listfile <listfile.txt>]");
        Console.WriteLine("  wtf probe --archive-root <game dir> --build <label> (--name <candidate> ... | --names-file <list.txt>) [--found-only] [--listfile <listfile.txt>]");
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

        WtfSweeper sweeper = CreateSweeper(archiveRoot, GetOption(args, "--listfile", "--listfile"));
        WtfBuildSurvey survey = sweeper.Sweep(buildLabel);
        PrintSurvey(survey);
    }

    private static void RunProbe(string[] args)
    {
        string? archiveRoot = GetOption(args, "--archive-root", "-r");
        string? buildLabel = GetOption(args, "--build", "-b");
        List<string> candidates = GetAllOptions(args, "--name", "-n");

        // A real candidate list (e.g. every WTF\ name in the community listfile) is thousands of
        // entries — far past what repeated --name flags can carry.
        string? namesFile = GetOption(args, "--names-file", "-f");
        if (!string.IsNullOrWhiteSpace(namesFile))
        {
            if (!File.Exists(namesFile))
            {
                Console.Error.WriteLine($"Error: names file '{namesFile}' does not exist.");
                Environment.ExitCode = 1;
                return;
            }

            foreach (string line in File.ReadLines(namesFile))
            {
                string trimmed = line.Trim();
                if (trimmed.Length > 0)
                    candidates.Add(trimmed);
            }
        }

        if (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(buildLabel) || candidates.Count == 0)
        {
            Console.Error.WriteLine("Error: wtf probe requires --archive-root, --build, and at least one --name or a --names-file.");
            Environment.ExitCode = 1;
            return;
        }

        bool foundOnly = args.Any(static a => string.Equals(a, "--found-only", StringComparison.OrdinalIgnoreCase));
        WtfSweeper sweeper = CreateSweeper(archiveRoot, GetOption(args, "--listfile", "--listfile"));
        Console.WriteLine($"BUILD: {buildLabel}");
        Console.WriteLine($"Candidates tested: {candidates.Count}");

        int foundCount = 0;
        foreach (string candidate in candidates)
        {
            WtfCandidateProbeResult result = sweeper.ProbeCandidate(candidate);
            if (!result.Resolved)
            {
                if (!foundOnly)
                    Console.WriteLine($"  NOT FOUND: {candidate}");

                continue;
            }

            foundCount++;
            Console.WriteLine($"  FOUND ({result.Survey!.Source}): {candidate}");
            PrintFile(result.Survey, indent: "    ");
        }

        Console.WriteLine();
        Console.WriteLine($"RESOLVED: {foundCount} of {candidates.Count}");
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

    private static WtfSweeper CreateSweeper(string archiveRoot, string? listfilePath)
    {
        ArchiveCatalogSession session = ArchiveCatalogSessionCache.GetOrCreate(
            [archiveRoot],
            new ArchiveCatalogBootstrapOptions(ExternalListfilePath: listfilePath));
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
