using WoWRollback.Orchestrator;

if (!PipelineOptionsParser.TryParse(args, out var options, out var error))
{
    PrintUsage(error);
    return 1;
}

if (options is null)
{
    PrintUsage("Failed to parse arguments.");
    return 1;
}

var verbose = options.Verbose;

try
{
    var orchestrator = new PipelineOrchestrator();
    var result = orchestrator.Run(options);

    Console.WriteLine($"✓ Session initialized at: {result.Session.Root}");
    Console.WriteLine($"  Manifest path: {result.Session.ManifestPath}");

    if (options.Serve)
    {
        Console.WriteLine("Viewer hosting is not yet implemented. Skipping --serve step.");
    }

    return result.Success ? 0 : 2;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Pipeline failed: {ex.Message}");
    if (verbose)
    {
        Console.Error.WriteLine(ex);
    }

    return 1;
}

static void PrintUsage(string? error)
{
    if (!string.IsNullOrWhiteSpace(error))
    {
        Console.Error.WriteLine(error);
        Console.Error.WriteLine();
    }

    Console.WriteLine("WoWRollback Orchestrator");
    Console.WriteLine("Usage:");
    Console.WriteLine("  dotnet run --project WoWRollback.Orchestrator -- --maps <map1,map2> --versions <v1,v2> --alpha-root <path> [options]");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine("  --maps <names>          Comma-separated list of map names (required)");
    Console.WriteLine("  --versions <versions>    Comma-separated list of versions (required)");
    Console.WriteLine("  --alpha-root <path>      Path to Alpha data root (required)");
    Console.WriteLine("  --output <path>          Output root directory (default: ./parp_out)");
    Console.WriteLine("  --dbd-dir <path>         WoWDBDefs directory (default: ../lib/WoWDBDefs/definitions)");
    Console.WriteLine("  --lk-dbc-dir <path>      Lich King DBC directory (optional)");
    Console.WriteLine("  --serve                  Start viewer server after pipeline (not yet implemented)");
    Console.WriteLine("  --port <number>          Viewer server port (default: 8080)");
    Console.WriteLine("  --verbose                Enable detailed error output");
}
