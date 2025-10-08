using WoWRollback.Orchestrator;
using WoWRollback.ViewerModule;

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

    if (result.Success)
    {
        Console.WriteLine("=== PIPELINE COMPLETE ===");
        Console.WriteLine($"Session: {result.Session.Root}");
    }
    else
    {
        Console.WriteLine("=== PIPELINE FINISHED WITH ERRORS ===");
    }
    Console.WriteLine();

    if (options.Serve)
    {
        var viewerDir = result.Session.Paths.ViewerDir;
        var port = options.Port;

        using var server = new ViewerServer();
        
        try
        {
            server.Start(viewerDir, port);
            Console.WriteLine($"✓ Viewer server started at http://localhost:{port}");
            Console.WriteLine($"  Serving: {viewerDir}");
            Console.WriteLine();
            Console.WriteLine("Press Ctrl+C to stop the server...");
            
            // Block until Ctrl+C
            var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) =>
            {
                e.Cancel = true;
                cts.Cancel();
            };
            
            cts.Token.WaitHandle.WaitOne();
            
            Console.WriteLine();
            Console.WriteLine("Shutting down server...");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Failed to start viewer server: {ex.Message}");
            if (verbose)
            {
                Console.Error.WriteLine(ex);
            }
            return 3;
        }
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
    Console.WriteLine("  --noggit-client-path <path>  Optional Noggit client path to embed in project files");
    Console.WriteLine("  --serve                  Start viewer server after pipeline");
    Console.WriteLine("  --port <number>          Viewer server port (default: 8080)");
    Console.WriteLine("  --verbose                Enable detailed error output");
}
