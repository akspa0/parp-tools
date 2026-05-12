using WowViewer.Core.IO.Wmo;

namespace WowViewer.Tool.Converter;

internal static class WmoV14ToV17Command
{
    public static void Run(string[] args)
    {
        string? inputRoot = GetOption(args, "--input-root", "-i")
            ?? GetOption(args, "--input", "-i")
            ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? outputPath = GetOption(args, "--output", "-o");
        string? groupsDirectory = GetOption(args, "--groups-dir", "-g");
        bool verbose = HasOption(args, "--verbose", "-v");

        if (string.IsNullOrWhiteSpace(inputRoot) || string.IsNullOrWhiteSpace(outputPath))
        {
            Console.Error.WriteLine("Error: convert-wmo-v14-to-v17 requires --input-root <root.wmo> and --output <output_root.wmo>.");
            Environment.ExitCode = 1;
            return;
        }

        string fullInputRoot = Path.GetFullPath(inputRoot);
        string fullOutputPath = Path.GetFullPath(outputPath);
        string? fullGroupsDirectory = string.IsNullOrWhiteSpace(groupsDirectory)
            ? null
            : Path.GetFullPath(groupsDirectory);

        if (!File.Exists(fullInputRoot))
        {
            Console.Error.WriteLine($"Error: input root WMO not found: {fullInputRoot}");
            Environment.ExitCode = 1;
            return;
        }

        if (fullGroupsDirectory is not null && !Directory.Exists(fullGroupsDirectory))
        {
            Console.Error.WriteLine($"Error: groups directory not found: {fullGroupsDirectory}");
            Environment.ExitCode = 1;
            return;
        }

        try
        {
            Console.WriteLine("WowViewer.Tool.Converter convert-wmo-v14-to-v17 report");
            Console.WriteLine($"  InputRoot:  {fullInputRoot}");
            Console.WriteLine($"  Output:     {fullOutputPath}");
            Console.WriteLine($"  GroupsDir:  {fullGroupsDirectory ?? Path.GetDirectoryName(fullOutputPath) ?? "."}");

            WmoV14ToV17Converter.Convert(fullInputRoot, fullOutputPath, fullGroupsDirectory);

            Console.WriteLine("  Status:     completed");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            if (verbose)
                Console.Error.WriteLine(ex);
            Environment.ExitCode = 1;
        }
    }

    private static string? GetOption(string[] args, string longName, string shortName)
    {
        for (int index = 0; index < args.Length; index++)
        {
            string arg = args[index];
            if (!arg.Equals(longName, StringComparison.OrdinalIgnoreCase)
                && !arg.Equals(shortName, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (index + 1 >= args.Length)
                return null;

            return args[index + 1];
        }

        return null;
    }

    private static bool HasOption(string[] args, string longName, string shortName)
    {
        return args.Any(arg => arg.Equals(longName, StringComparison.OrdinalIgnoreCase)
            || arg.Equals(shortName, StringComparison.OrdinalIgnoreCase));
    }
}