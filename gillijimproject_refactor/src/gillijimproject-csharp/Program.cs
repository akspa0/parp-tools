using System;
using System.IO;
using GillijimProject.Utilities;

namespace GillijimProject;

/// <summary>
/// [PORT] Minimal CLI stub. Will mirror lib/gillijimproject/main.cpp flow as port progresses.
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.WriteLine("Usage: gillijimproject-csharp <WDT_ALPHA_PATH>");
            return 1;
        }

        var path = args[0];
        if (!File.Exists(path))
        {
            Console.Error.WriteLine($"File not found: {path}");
            return 2;
        }

        try
        {
            var bytes = Utilities.GetWholeFile(path);
            Console.WriteLine($"[INFO] Read {bytes.Length} bytes from {path}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 3;
        }
    }
}
