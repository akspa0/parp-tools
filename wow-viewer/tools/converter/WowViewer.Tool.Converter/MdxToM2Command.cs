using WowViewer.Core.IO.M2;
using WowViewer.Core.M2;

namespace WowViewer.Tool.Converter;

internal static class MdxToM2Command
{
    public static void Run(string[] args)
    {
        string? inputPath = GetOption(args, "--input", "-i")
            ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? outputPath = GetOption(args, "--output", "-o");
        bool verbose = HasOption(args, "--verbose", "-v");

        if (string.IsNullOrWhiteSpace(inputPath) || string.IsNullOrWhiteSpace(outputPath))
        {
            Console.Error.WriteLine("Error: convert-mdx-to-m2 requires --input <model.mdx> and --output <output.m2>.");
            Environment.ExitCode = 1;
            return;
        }

        string fullInputPath = Path.GetFullPath(inputPath);
        string fullOutputPath = Path.GetFullPath(outputPath);

        if (!File.Exists(fullInputPath))
        {
            Console.Error.WriteLine($"Error: input MDX not found: {fullInputPath}");
            Environment.ExitCode = 1;
            return;
        }

        try
        {
            string skinPath = M2ModelIdentity.FromPath(fullOutputPath).BuildSkinPath(0);

            Console.WriteLine("WowViewer.Tool.Converter convert-mdx-to-m2 report");
            Console.WriteLine($"  Input:   {fullInputPath}");
            Console.WriteLine($"  Output:  {fullOutputPath}");
            Console.WriteLine($"  Skin:    {skinPath}");

            MdxToM2Converter.Convert(fullInputPath, fullOutputPath);

            Console.WriteLine("  Status:  completed");
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