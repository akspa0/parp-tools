using GillijimProject.Next.Core.Adapters.Dbcd;
using GillijimProject.Next.Core.Services;
using GillijimProject.Next.Core.IO;
using GillijimProject.Next.Core.Transform;
using GillijimProject.Next.Core.Adapters.WarcraftNet;
using GillijimProject.Next.Cli.Commands;
using System;
using System.Linq;

namespace GillijimProject.Next.Cli;

public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help")
        {
            PrintHelp();
            return 0;
        }

        var command = args[0].ToLowerInvariant();
        var rest = args.Skip(1).ToArray();

        return command switch
        {
            "convert" => ConvertCommand.Run(rest),
            "analyze" => AnalyzeCommand.Run(rest),
            "fix-areaids" => FixAreaIdsCommand.Run(rest),
            "gen-alpha-wdt" => GenAlphaWdtCommand.Run(rest),
            _ => Unknown(command)
        };
    }

    private static void PrintHelp()
    {
        Console.WriteLine("GillijimProject.Next CLI\n");
        Console.WriteLine("Commands:");
        Console.WriteLine("  convert      Convert Alpha → LK ADTs");
        Console.WriteLine("  analyze      Analyze UniqueIDs and assets");
        Console.WriteLine("  fix-areaids  Re-emit ADTs with corrected AreaIDs");
        Console.WriteLine("  gen-alpha-wdt Generate Alpha WDT from LK ADTs (reverse)");
        Console.WriteLine("\nUse: dotnet run --project next/src/GillijimProject.Next.Cli -- <command> [options]\n");
    }

    private static int Unknown(string cmd)
    {
        Console.Error.WriteLine($"Unknown command: {cmd}");
        PrintHelp();
        return 1;
    }
}
