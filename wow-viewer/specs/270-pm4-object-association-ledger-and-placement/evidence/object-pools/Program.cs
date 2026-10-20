using System.Globalization;
using System.Text.Json;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

if (args.Length == 0 || args[0] == "--help")
{
    Console.WriteLine("init <manifest.json> <map-name>");
    Console.WriteLine("pool <manifest.json> <pool-id> <design-kit> <object-class> <m2|wmo|both> <path-prefix> [additional-prefixes...]");
    Console.WriteLine("assign <manifest.json> <pool-id> <min-x> <min-y> <max-x> <max-y> <operator-note>");
    Console.WriteLine("filter <manifest.json> <surface-db.json> <map-name> <tile-x> <tile-y> <new-output-db.json>");
    Console.WriteLine("hints <manifest.json> <wl-map-directory>  (source-name hints only; no automatic tile assignments)");
    return;
}
switch (args[0])
{
    case "init" when args.Length == 3:
        if (File.Exists(args[1])) throw new IOException("Manifest exists; refusing to replace it.");
        Pm4ObjectPoolManifestStore.Write(args[1], new Pm4ObjectPoolManifest(1, args[2], [], []));
        break;
    case "pool" when args.Length >= 7:
    {
        var m = Pm4ObjectPoolManifestStore.Read(args[1]);
        if (m.Pools.Any(p => p.Id.Equals(args[2], StringComparison.OrdinalIgnoreCase)))
            throw new ArgumentException("Pool exists; edit its explicit manifest entry rather than silently replacing it.");
        string[] kinds = args[5] == "both" ? ["m2", "wmo"] : [args[5]];
        var p = new Pm4ObjectPool(args[2], args[3], [args[4]], kinds, [], args[6..], "Operator-authored expectation, not a confirmed match.");
        Pm4ObjectPoolManifestStore.Write(args[1], m with { Pools = [.. m.Pools, p] });
        break;
    }
    case "assign" when args.Length == 8:
    {
        var m = Pm4ObjectPoolManifestStore.Read(args[1]);
        var a = new Pm4ObjectPoolAssignment(args[2], Number(args[3]), Number(args[4]), Number(args[5]), Number(args[6]), args[7]);
        Pm4ObjectPoolManifestStore.Write(args[1], m with { Assignments = [.. m.Assignments, a] });
        break;
    }
    case "hints" when args.Length == 3:
    {
        var m = Pm4ObjectPoolManifestStore.Read(args[1]);
        string[] names = Directory.EnumerateFiles(args[2]).Where(p => Path.GetExtension(p).StartsWith(".wl", StringComparison.OrdinalIgnoreCase))
            .Select(p => Path.GetFileName(p)).ToArray();
        var hints = Pm4DesignKitHints.FromSourceNames(names, m.HintAliases);
        Pm4ObjectPoolManifestStore.Write(args[1], m with { SourceHints = Pm4DesignKitHints.Merge(m.SourceHints, hints) });
        Console.WriteLine($"Retained {hints.Count} WL-family source hints, {hints.Count(h => h.MatchedAliases.Count > 0)} recognized; no tile assignments created.");
        break;
    }
    case "filter" when args.Length == 7:
    {
        if (File.Exists(args[6]) || File.Exists(args[6] + ".context.json"))
            throw new IOException("Filtered output or context sidecar exists; refusing overwrite.");
        var options = new JsonSerializerOptions { IncludeFields = true, PropertyNameCaseInsensitive = true, WriteIndented = true };
        var db = JsonSerializer.Deserialize<SurfaceCorrelationDatabase>(File.ReadAllText(args[2]), options)
            ?? throw new InvalidDataException("Empty surface database.");
        var selection = Pm4ObjectPoolFilter.Select(Pm4ObjectPoolManifestStore.Read(args[1]), db, args[3], Number(args[4]), Number(args[5]));
        string output = Path.GetFullPath(args[6]);
        Directory.CreateDirectory(Path.GetDirectoryName(output)!);
        // Complete both serializations before publishing; do not leave a DB without its context.
        byte[] databaseBytes = JsonSerializer.SerializeToUtf8Bytes(selection.Database, options);
        byte[] contextBytes = JsonSerializer.SerializeToUtf8Bytes(new { selection.MapName, selection.TileX, selection.TileY, selection.Restricted,
            selection.ActivePools, selection.InputCandidates, selection.EligibleCandidates }, options);
        bool createdDatabase = false, createdContext = false;
        try
        {
            using (var stream = new FileStream(output, FileMode.CreateNew, FileAccess.Write))
            { createdDatabase = true; stream.Write(databaseBytes); }
            using (var stream = new FileStream(output + ".context.json", FileMode.CreateNew, FileAccess.Write))
            { createdContext = true; stream.Write(contextBytes); }
        }
        catch
        {
            if (createdContext) File.Delete(output + ".context.json");
            if (createdDatabase) File.Delete(output);
            throw;
        }
        Console.WriteLine($"Restricted={selection.Restricted}; candidates {selection.InputCandidates} -> {selection.EligibleCandidates}; pools={string.Join(',', selection.ActivePools)}");
        break;
    }
    default:
        throw new ArgumentException("Invalid arguments; use --help.");
}

static int Number(string value) => int.Parse(value, CultureInfo.InvariantCulture);
