using WowViewer.Core.IO.Archive;
using WowViewer.Core.IO.Blp;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.M2;
using WowViewer.Core.Maps;
using WowViewer.Core.Wmo;

namespace WowViewer.Tool.V22Enrich;

/// <summary>
/// CLI tool that reads a V18 Zarr store, walks its placements for unique
/// M2/WMO paths, decodes each from the staged client, and writes a
/// stable-path-keyed enrichment stream.
///
/// Usage:
///   WowViewer.Tool.V22Enrich --v18-store <path> --client-root <path>
///       --output <stream-path> --build-key <key>
///
/// Options:
///   --v18-store       Path to the V18 Zarr store.
///   --client-root     Staged client root.
///   --output          Output enrichment stream path.
///   --build-key       Build key for stream metadata (e.g. "3_3_5_12340").
///   --limit           Optional. Limit unique assets per kind.
///   --verbose         Optional. Print per-asset progress to stderr.
/// </summary>
static class Program
{
    static int Main(string[] args)
    {
        Environment.ExitCode = 0;

        if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
        {
            ShowUsage();
            return 0;
        }

        try
        {
            return Run(args);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Fatal: {ex.Message}");
            Console.Error.WriteLine(ex.StackTrace);
            Environment.ExitCode = 1;
            return 1;
        }
    }

    static int Run(string[] args)
    {
        string? v18Store = GetOption(args, "--v18-store");
        string? clientRoot = GetOption(args, "--client-root");
        string? output = GetOption(args, "--output");
        string? buildKey = GetOption(args, "--build-key");
        int? limit = GetIntOption(args, "--limit");
        bool verbose = HasFlag(args, "--verbose");

        if (string.IsNullOrWhiteSpace(v18Store) || string.IsNullOrWhiteSpace(clientRoot)
            || string.IsNullOrWhiteSpace(output) || string.IsNullOrWhiteSpace(buildKey))
        {
            Console.Error.WriteLine("Error: --v18-store, --client-root, --output, and --build-key are required.");
            ShowUsage();
            return 1;
        }

        // Resolve client root (handle nested "World of Warcraft" directory)
        clientRoot = ResolveGameClientRoot(clientRoot);

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"Error: client root not found: {clientRoot}");
            return 1;
        }

        if (!Directory.Exists(v18Store))
        {
            Console.Error.WriteLine($"Error: V18 store not found: {v18Store}");
            return 1;
        }

        // ── 1. Read placements from V18 store ──────────────
        var inventory = V18StorePlacementsReader.ReadPlacements(v18Store);
        if (verbose)
        {
            Console.Error.WriteLine($"Unique M2 paths: {inventory.UniqueM2Paths.Count}");
            Console.Error.WriteLine($"Unique WMO paths: {inventory.UniqueWmoPaths.Count}");
            Console.Error.WriteLine($"Unique BLP paths: {inventory.UniqueBlpPaths.Count}");
        }

        if (inventory.UniqueM2Paths.Count == 0 && inventory.UniqueWmoPaths.Count == 0)
        {
            Console.Error.WriteLine("Warning: no unique asset paths found in placements.parquet. Enrichment stream will be empty.");
        }

        // ── 2. Open MPQ archives from client root ─────────
        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        var internalListfilePaths = new HashSet<string>(
            catalog.ExtractInternalListfiles().Select(NormalizeVirtualPath),
            StringComparer.OrdinalIgnoreCase);

        if (verbose)
            Console.Error.WriteLine($"Internal listfile paths: {internalListfilePaths.Count}");

        // assetReader: resolves a virtual path to raw bytes from MPQ
        Func<string, byte[]?> assetReader = path => catalog.ReadFile(path);

        // ── 3. Write enrichment stream ─────────────────────
        string? outputDir = Path.GetDirectoryName(output);
        if (!string.IsNullOrWhiteSpace(outputDir))
            Directory.CreateDirectory(outputDir);
        using var fs = File.Create(output);
        var writer = new EnrichmentStreamWriter(fs);
        writer.WriteHeader();

        int m2Processed = 0;
        int wmoProcessed = 0;
        int blpProcessed = 0;
        int errors = 0;

        // ── 4. Decode M2 models ────────────────────────────
        foreach (string m2Path in inventory.UniqueM2Paths)
        {
            bool sourceInListfile = internalListfilePaths.Contains(NormalizeVirtualPath(m2Path));
            if (limit.HasValue && m2Processed >= limit.Value)
            {
                if (verbose)
                    Console.Error.WriteLine($"[M2] limit {limit} reached, stopping.");
                break;
            }

            try
            {
                byte[]? m2Bytes = assetReader(m2Path);
                if (m2Bytes is null || m2Bytes.Length < 16)
                {
                    writer.WriteEntry(BuildFailureEntry(m2Path, AssetKind.M2, sourceInListfile));
                    errors++;
                    if (verbose)
                        Console.Error.WriteLine($"[M2] {m2Path}: not found in archives");
                    continue;
                }

                using var m2Ms = new MemoryStream(m2Bytes, writable: false);
                M2GeometryDocument geoDoc = M2GeometryReader.Read(m2Ms, m2Path);

                // Build skin path
                string skinPath = M2ModelIdentity.FromPath(m2Path).BuildSkinPath(0);
                byte[]? skinBytes = assetReader(skinPath);

                if (skinBytes is null || skinBytes.Length < 8)
                {
                    writer.WriteEntry(BuildFailureEntry(m2Path, AssetKind.M2, sourceInListfile));
                    errors++;
                    if (verbose)
                        Console.Error.WriteLine($"[M2] {m2Path}: companion .skin not found");
                    continue;
                }

                using var skinMs = new MemoryStream(skinBytes, writable: false);
                M2SkinDocument skinDoc = M2SkinReader.Read(skinMs, skinPath);

                // Build enrichment entry arrays
                var entry = AddSourceProvenance(
                    M2EnrichmentBuilder.BuildEntry(m2Path, geoDoc, skinDoc),
                    sourceInListfile);
                writer.WriteEntry(entry);
                m2Processed++;

                if (verbose)
                    Console.Error.WriteLine($"[M2] {m2Path}: OK");
            }
            catch (Exception ex)
            {
                writer.WriteEntry(BuildFailureEntry(m2Path, AssetKind.M2, sourceInListfile));
                errors++;
                if (verbose)
                    Console.Error.WriteLine($"[M2] {m2Path}: ERROR {ex.Message}");
            }
        }

        // ── 5. Decode WMO models ───────────────────────────
        foreach (string wmoPath in inventory.UniqueWmoPaths)
        {
            bool sourceInListfile = internalListfilePaths.Contains(NormalizeVirtualPath(wmoPath));
            if (limit.HasValue && wmoProcessed >= limit.Value)
            {
                if (verbose)
                    Console.Error.WriteLine($"[WMO] limit {limit} reached, stopping.");
                break;
            }

            try
            {
                byte[]? wmoBytes = assetReader(wmoPath);
                if (wmoBytes is null || wmoBytes.Length < 16)
                {
                    writer.WriteEntry(BuildFailureEntry(wmoPath, AssetKind.Wmo, sourceInListfile));
                    errors++;
                    if (verbose)
                        Console.Error.WriteLine($"[WMO] {wmoPath}: not found in archives");
                    continue;
                }

                using var wmoMs = new MemoryStream(wmoBytes, writable: false);
                // WMO reader needs an assetReader for external groups/doodads
                WmoRenderDocument wmoDoc = WmoRenderDocumentReader.Read(wmoMs, wmoPath, assetReader);

                var entry = AddSourceProvenance(
                    WmoEnrichmentBuilder.BuildEntry(wmoPath, wmoDoc),
                    sourceInListfile);
                writer.WriteEntry(entry);
                wmoProcessed++;

                if (verbose)
                    Console.Error.WriteLine($"[WMO] {wmoPath}: OK");
            }
            catch (Exception ex)
            {
                writer.WriteEntry(BuildFailureEntry(wmoPath, AssetKind.Wmo, sourceInListfile));
                errors++;
                if (verbose)
                    Console.Error.WriteLine($"[WMO] {wmoPath}: ERROR {ex.Message}");
            }
        }

        // ── 6. Decode BLP tilesets ──────────────────────────────
        foreach (string blpPath in inventory.UniqueBlpPaths)
        {
            bool sourceInListfile = internalListfilePaths.Contains(NormalizeVirtualPath(blpPath));
            if (limit.HasValue && blpProcessed >= limit.Value)
            {
                if (verbose)
                    Console.Error.WriteLine($"[BLP] limit {limit} reached, stopping.");
                break;
            }

            try
            {
                byte[]? blpBytes = assetReader(blpPath);
                if (blpBytes is null || blpBytes.Length < 4)
                {
                    writer.WriteEntry(BuildFailureEntry(blpPath, AssetKind.Blp, sourceInListfile));
                    errors++;
                    if (verbose)
                        Console.Error.WriteLine($"[BLP] {blpPath}: not found in archives");
                    continue;
                }

                BlpRgbResult rgb = BlpRgbReader.ReadRgb(blpBytes, blpPath);
                if (rgb.LoadError != 0 || rgb.Rgb is null || rgb.Width <= 0 || rgb.Height <= 0)
                {
                    writer.WriteEntry(BuildFailureEntry(blpPath, AssetKind.Blp, sourceInListfile));
                    errors++;
                    if (verbose)
                        Console.Error.WriteLine($"[BLP] {blpPath}: decode failed");
                    continue;
                }

                var arrays = new List<EnrichmentArray>
                {
                    new("texture_rgb", [rgb.Height, rgb.Width, 3], typeof(byte), rgb.Rgb),
                    new("texture_shape", [2], typeof(int),
                        EnrichmentArrayHelper.FlattenInts([rgb.Height, rgb.Width])),
                };
                arrays.Add(BuildSourceProvenanceArray(sourceInListfile));

                writer.WriteEntry(new EnrichmentEntry(blpPath, AssetKind.Blp, 0, arrays));
                blpProcessed++;

                if (verbose)
                    Console.Error.WriteLine($"[BLP] {blpPath}: OK {rgb.Width}x{rgb.Height}");
            }
            catch (Exception ex)
            {
                writer.WriteEntry(BuildFailureEntry(blpPath, AssetKind.Blp, sourceInListfile));
                errors++;
                if (verbose)
                    Console.Error.WriteLine($"[BLP] {blpPath}: ERROR {ex.Message}");
            }
        }

        writer.WriteEnds();

        if (verbose)
        {
            Console.Error.WriteLine($"Enrichment done. M2={m2Processed} WMO={wmoProcessed} BLP={blpProcessed} Errors={errors}");
            Console.Error.WriteLine($"Output: {output}");
        }

        return errors > 0 ? 2 : 0;
    }

    static void ShowUsage()
    {
        Console.WriteLine("""
            WowViewer.Tool.V22Enrich — V22 enrichment stream builder

            Reads a V18 Zarr store's placements, decodes each unique M2 and WMO
            from the staged client, and writes a stable-path-keyed enrichment stream.

            Usage:
              WowViewer.Tool.V22Enrich --v18-store <path> --client-root <path>
                  --output <stream-path> --build-key <build_name>
                  [--limit <N>] [--verbose]

            Required:
              --v18-store     Path to the V18 Zarr store directory.
              --client-root   Staged client root directory.
              --output        Output enrichment stream file path.
              --build-key     Build identifier for metadata (e.g. "3_3_5_12340").

            Optional:
              --limit <N>     Max unique assets per kind to decode.
              --verbose       Print per-asset progress to stderr.
              --help, -h      Show this message.
            """);
    }

    static string? GetOption(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i].Equals(name, StringComparison.OrdinalIgnoreCase))
                return args[i + 1];
        }
        return null;
    }

    static int? GetIntOption(string[] args, string name)
    {
        string? val = GetOption(args, name);
        return val is not null && int.TryParse(val, out int n) ? n : null;
    }

    static EnrichmentEntry AddSourceProvenance(EnrichmentEntry entry, bool sourceInListfile)
    {
        var arrays = new List<EnrichmentArray>(entry.Arrays)
        {
            BuildSourceProvenanceArray(sourceInListfile),
        };
        return entry with { Arrays = arrays };
    }

    static EnrichmentEntry BuildFailureEntry(string path, AssetKind kind, bool sourceInListfile)
        => new(path, kind, 1, [BuildSourceProvenanceArray(sourceInListfile)]);

    static EnrichmentArray BuildSourceProvenanceArray(bool sourceInListfile)
        => new(
            "source_in_listfile",
            [1],
            typeof(byte),
            EnrichmentArrayHelper.FlattenBytes([sourceInListfile ? (byte)1 : (byte)0]));

    static bool HasFlag(string[] args, string name)
    {
        return args.Any(a => a.Equals(name, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Resolve a staged client root: if the path points to a parent directory
    /// and a nested "World of Warcraft" exists, use the nested path.
    /// </summary>
    static string ResolveGameClientRoot(string root)
    {
        string nested = Path.Combine(root, "World of Warcraft");
        if (Directory.Exists(nested))
            return nested;

        // Check for other common nested patterns
        string[] possible = Directory.GetDirectories(root);
        foreach (string dir in possible)
        {
            string name = Path.GetFileName(dir);
            if (name.StartsWith("World of Warcraft", StringComparison.OrdinalIgnoreCase)
                || name.StartsWith("WoW", StringComparison.OrdinalIgnoreCase))
            {
                // Verify it has a Data subdirectory
                string dataDir = Path.Combine(dir, "Data");
                if (Directory.Exists(dataDir))
                    return dir;
            }
        }

        return root;
    }

    static string NormalizeVirtualPath(string path)
        => path.Trim().Replace('/', '\\').TrimStart('\\');
}
