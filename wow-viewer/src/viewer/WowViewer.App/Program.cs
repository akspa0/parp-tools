namespace WowViewer.App;

internal static class Program
{
    private const string ProductName = "WoWAlphaViewer";

    private static int Main(string[] args)
    {
        if (args.Length == 0 || HasFlag(args, "--help", "-h"))
        {
            ShowUsage();
            return 0;
        }

        string command = args[0].Trim().ToLowerInvariant();
        return command switch
        {
            "status" => RunStatus(),
            "layer0-status" => RunLayer0Status(),
            "viewer" => RunViewerStub(),
            _ => RunUnknownCommand(command),
        };
    }

    private static int RunStatus()
    {
        Console.WriteLine($"{ProductName} reset baseline is active.");
        Console.WriteLine("Scope: fresh app scaffold only, runtime features not yet ported.");
        Console.WriteLine("Reference target: 0.5.3-first behavior with multi-era compatibility up through 4.0.0.");
        return 0;
    }

    private static int RunViewerStub()
    {
        Console.WriteLine($"{ProductName} viewer shell has been reset.");
        Console.WriteLine("Next slice will introduce the new world-session bootstrap and renderer pipeline from scratch.");
        return 0;
    }

    private static int RunLayer0Status()
    {
        IReadOnlyList<IWoWAlphaViewerLayerModule> modules = WoWAlphaViewerLayerRegistry.GetRegisteredModules();
        Console.WriteLine($"{ProductName} Layer 0 readiness report");
        Console.WriteLine($"Registered modules: {modules.Count}");

        foreach (IWoWAlphaViewerLayerModule module in modules)
        {
            Console.WriteLine($"- {(int)module.Layer}:{module.Layer} | {module.Name} | ready={module.IsReady()}");
            Console.WriteLine($"  {module.DescribeStatus()}");
        }

        bool allReady = modules.All(static module => module.IsReady());
        Console.WriteLine($"All ready: {allReady}");
        return allReady ? 0 : 2;
    }

    private static int RunUnknownCommand(string command)
    {
        Console.Error.WriteLine($"Unknown command: {command}");
        Console.Error.WriteLine("Use --help to see available commands.");
        return 1;
    }

    private static void ShowUsage()
    {
        Console.WriteLine(ProductName);
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  wowalphaviewer status");
        Console.WriteLine("  wowalphaviewer layer0-status");
        Console.WriteLine("  wowalphaviewer viewer");
        Console.WriteLine("  wowalphaviewer --help");
    }

    private static bool HasFlag(string[] args, params string[] flags)
    {
        return args.Any(arg => flags.Contains(arg, StringComparer.OrdinalIgnoreCase));
    }
}
