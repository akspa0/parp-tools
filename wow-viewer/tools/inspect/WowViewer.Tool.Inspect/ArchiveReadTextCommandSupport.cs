using System.Text;
using WowViewer.Core.IO.Files;

namespace WowViewer.Tool.Inspect;

/// <summary>
/// Dumps one file's raw text content from a build, whether it is loose on disk or packed inside an MPQ
/// archive. A small, generic complement to the format-specific inspect commands, for exactly the case of
/// "what is actually in this file" when no dedicated reader exists yet — e.g. WTF files (Spec 159).
/// </summary>
public static class ArchiveReadTextCommandSupport
{
    public static void Run(string[] args)
    {
        string? archiveRoot = GetOption(args, "--archive-root", "-r");
        string? virtualPath = GetOption(args, "--virtual-path", "-v");
        if (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath))
        {
            Console.Error.WriteLine("Error: archive read-text requires --archive-root and --virtual-path.");
            Environment.ExitCode = 1;
            return;
        }

        byte[] bytes;
        try
        {
            bytes = ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], (ArchiveCatalogBootstrapOptions?)null);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: could not read '{virtualPath}' from '{archiveRoot}': {ex.Message}");
            Environment.ExitCode = 1;
            return;
        }

        Console.WriteLine($"PATH:   {virtualPath}");
        Console.WriteLine($"BYTES:  {bytes.Length}");
        Console.WriteLine("---");
        Console.WriteLine(Encoding.UTF8.GetString(bytes));
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
}
