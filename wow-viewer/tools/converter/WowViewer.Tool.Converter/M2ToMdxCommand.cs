using WowViewer.Core.IO.M2;

namespace WowViewer.Tool.Converter;

internal static class M2ToMdxCommand
{
    public static void Run(string[] args)
    {
        string? inputPath = GetOption(args, "--input", "-i")
            ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? skinPath = GetOption(args, "--skin", "-s");
        string? outputPath = GetOption(args, "--output", "-o");
        bool verbose = HasOption(args, "--verbose", "-v");

        if (string.IsNullOrWhiteSpace(inputPath)
            || string.IsNullOrWhiteSpace(skinPath)
            || string.IsNullOrWhiteSpace(outputPath))
        {
            Console.Error.WriteLine("Error: convert-m2-to-mdx requires --input <model.m2>, --skin <model00.skin>, and --output <output.mdx>.");
            Environment.ExitCode = 1;
            return;
        }

        string fullInputPath = Path.GetFullPath(inputPath);
        string fullSkinPath = Path.GetFullPath(skinPath);
        string fullOutputPath = Path.GetFullPath(outputPath);

        if (!File.Exists(fullInputPath))
        {
            Console.Error.WriteLine($"Error: input M2 not found: {fullInputPath}");
            Environment.ExitCode = 1;
            return;
        }

        if (!File.Exists(fullSkinPath))
        {
            Console.Error.WriteLine($"Error: input skin not found: {fullSkinPath}");
            Environment.ExitCode = 1;
            return;
        }

        try
        {
            Console.WriteLine("WowViewer.Tool.Converter convert-m2-to-mdx report");
            Console.WriteLine($"  Input:   {fullInputPath}");
            Console.WriteLine($"  Skin:    {fullSkinPath}");
            Console.WriteLine($"  Output:  {fullOutputPath}");

            M2ToMdxConverter.Convert(fullInputPath, fullSkinPath, fullOutputPath);

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