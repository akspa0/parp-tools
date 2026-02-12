using CASCLib;
using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Globalization;

namespace BlpResizer;

/// <summary>
/// BLP Tileset Resizer - Downscales BLP textures for Alpha client compatibility.
/// Alpha 0.5.3 through 10.0 supports max 256x256 tilesets.
/// </summary>
class Program
{
    static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] == "--help" || args[0] == "-h")
        {
            PrintUsage();
            return 0;
        }

        var parsed = ParseArgs(args);

        // Check for CASC mode
        if (parsed.TryGetValue("--casc", out var cascPath) && !string.IsNullOrWhiteSpace(cascPath))
        {
            return RunCascMode(parsed, cascPath);
        }

        if (!parsed.TryGetValue("--input", out var input) || string.IsNullOrWhiteSpace(input))
        {
            Console.WriteLine("Error: --input is required (or use --casc for CASC mode)");
            return 1;
        }

        if (!parsed.TryGetValue("--output", out var output) || string.IsNullOrWhiteSpace(output))
        {
            Console.WriteLine("Error: --output is required");
            return 1;
        }

        int maxSize = 256;
        if (parsed.TryGetValue("--max-size", out var maxSizeStr) && 
            int.TryParse(maxSizeStr, NumberStyles.Integer, CultureInfo.InvariantCulture, out var ms))
        {
            maxSize = ms;
        }

        // Default to BLP2 since Alpha 0.5.3 actually uses BLP2, not BLP1
        bool blp1 = parsed.ContainsKey("--blp1");
        bool recursive = parsed.ContainsKey("--recursive") || parsed.ContainsKey("-r");
        bool verbose = parsed.ContainsKey("--verbose") || parsed.ContainsKey("-v");
        bool dryRun = parsed.ContainsKey("--dry-run");

        Console.WriteLine($"BLP Tileset Resizer");
        Console.WriteLine($"  Input:     {input}");
        Console.WriteLine($"  Output:    {output}");
        Console.WriteLine($"  Max Size:  {maxSize}x{maxSize}");
        Console.WriteLine($"  Format:    {(blp1 ? "BLP1 (WC3)" : "BLP2 (WoW Alpha+)")}");
        Console.WriteLine($"  Recursive: {recursive}");
        Console.WriteLine();

        try
        {
            if (File.Exists(input))
            {
                // Single file mode
                ProcessFile(input, output, maxSize, blp1, verbose, dryRun);
            }
            else if (Directory.Exists(input))
            {
                // Directory mode
                ProcessDirectory(input, output, maxSize, blp1, recursive, verbose, dryRun);
            }
            else
            {
                Console.WriteLine($"Error: Input not found: {input}");
                return 2;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 3;
        }

        return 0;
    }

    static int RunCascMode(Dictionary<string, string> parsed, string cascPath)
    {
        if (!parsed.TryGetValue("--output", out var output) || string.IsNullOrWhiteSpace(output))
        {
            Console.WriteLine("Error: --output is required");
            return 1;
        }

        int maxSize = 256;
        if (parsed.TryGetValue("--max-size", out var maxSizeStr) && 
            int.TryParse(maxSizeStr, NumberStyles.Integer, CultureInfo.InvariantCulture, out var ms))
        {
            maxSize = ms;
        }

        bool blp1 = parsed.ContainsKey("--blp1");
        bool verbose = parsed.ContainsKey("--verbose") || parsed.ContainsKey("-v");
        bool dryRun = parsed.ContainsKey("--dry-run");

        // Pattern filter (default: tileset BLPs)
        string pattern = "tileset";
        if (parsed.TryGetValue("--pattern", out var p) && !string.IsNullOrWhiteSpace(p))
            pattern = p.ToLowerInvariant();

        // Listfile path (optional, will try to download if not provided)
        string? listfilePath = null;
        if (parsed.TryGetValue("--listfile", out var lf) && !string.IsNullOrWhiteSpace(lf))
            listfilePath = lf;

        Console.WriteLine($"BLP Tileset Resizer - CASC Mode");
        Console.WriteLine($"  CASC Path: {cascPath}");
        Console.WriteLine($"  Output:    {output}");
        Console.WriteLine($"  Pattern:   *{pattern}*.blp");
        Console.WriteLine($"  Max Size:  {maxSize}x{maxSize}");
        Console.WriteLine($"  Format:    {(blp1 ? "BLP1 (WC3)" : "BLP2 (WoW Alpha+)")}");
        Console.WriteLine();

        try
        {
            // Load listfile (FileDataID -> filename mapping)
            Console.WriteLine("Loading listfile...");
            var listfile = LoadListfile(listfilePath);
            Console.WriteLine($"Loaded {listfile.Count} entries from listfile");

            Console.WriteLine("Opening CASC archive...");
            // Don't load Download handler - it can have duplicate key issues and we don't need it
            CASCConfig.LoadFlags = LoadFlags.All & ~LoadFlags.Download;
            CASCConfig.ValidateData = false;
            CASCConfig.ThrowOnFileNotFound = false;

            var config = CASCConfig.LoadLocalStorageConfig(cascPath, "wow");
            var casc = CASCHandler.OpenStorage(config);

            Console.WriteLine($"CASC opened. Build: {config.BuildName}");
            Console.WriteLine();

            // Get all files matching pattern using listfile
            var root = casc.Root as WowRootHandler;
            if (root == null)
            {
                Console.WriteLine("Error: Could not get WoW root handler");
                return 4;
            }

            var blpFiles = new List<(int fileDataId, string path)>();

            foreach (var entry in root.GetAllEntriesWithFileDataId())
            {
                int fdid = entry.FileDataId;
                if (listfile.TryGetValue(fdid, out var path))
                {
                    var pathLower = path.ToLowerInvariant();
                    if (pathLower.EndsWith(".blp") && pathLower.Contains(pattern))
                    {
                        blpFiles.Add((fdid, path));
                    }
                }
            }

            Console.WriteLine($"Found {blpFiles.Count} BLP files matching pattern '*{pattern}*.blp'");
            Console.WriteLine();

            int processed = 0;
            int resized = 0;
            int skipped = 0;
            int errors = 0;

            foreach (var (fileDataId, filePath) in blpFiles)
            {
                try
                {
                    using var stream = casc.OpenFile(fileDataId);
                    if (stream == null)
                    {
                        if (verbose) Console.WriteLine($"  SKIP: {filePath} (could not open)");
                        skipped++;
                        continue;
                    }

                    // Determine output path - keep same relative structure
                    var outPath = Path.Combine(output, filePath);

                    var result = ProcessCascFile(stream, outPath, maxSize, blp1, verbose, dryRun);
                    processed++;
                    if (result == ProcessResult.Resized) resized++;
                    else if (result == ProcessResult.Copied) skipped++;
                }
                catch (Exception ex)
                {
                    errors++;
                    if (verbose) Console.WriteLine($"  ERROR: {filePath} - {ex.Message}");
                }
            }

            Console.WriteLine();
            Console.WriteLine($"Summary: {processed} processed, {resized} resized, {skipped} copied/skipped, {errors} errors");
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            if (verbose) Console.WriteLine(ex.StackTrace);
            return 5;
        }
    }

    /// <summary>
    /// Load listfile (FileDataID -> filename mapping).
    /// Tries local file first, then downloads from wow-listfile repo.
    /// </summary>
    static Dictionary<int, string> LoadListfile(string? path)
    {
        var result = new Dictionary<int, string>();

        // Try provided path first
        if (!string.IsNullOrEmpty(path) && File.Exists(path))
        {
            return ParseListfile(path);
        }

        // Try local listfile.csv
        if (File.Exists("listfile.csv"))
        {
            return ParseListfile("listfile.csv");
        }

        // Download from GitHub
        Console.WriteLine("Downloading listfile from GitHub...");
        try
        {
            using var client = new HttpClient();
            var url = "https://github.com/wowdev/wow-listfile/releases/latest/download/community-listfile.csv";
            var content = client.GetStringAsync(url).Result;
            File.WriteAllText("listfile.csv", content);
            return ParseListfile("listfile.csv");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Could not download listfile: {ex.Message}");
            Console.WriteLine("CASC mode requires a listfile. Please provide one with --listfile <path>");
            return result;
        }
    }

    static Dictionary<int, string> ParseListfile(string path)
    {
        var result = new Dictionary<int, string>();
        foreach (var line in File.ReadLines(path))
        {
            var parts = line.Split(';', 2);
            if (parts.Length == 2 && int.TryParse(parts[0], out int fdid))
            {
                result[fdid] = parts[1];
            }
        }
        return result;
    }

    static ProcessResult ProcessCascFile(Stream stream, string outputPath, int maxSize, bool blp1, bool verbose, bool dryRun)
    {
        // Ensure output directory exists
        var outDir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outDir) && !dryRun)
        {
            Directory.CreateDirectory(outDir);
        }

        // Read the BLP from stream
        using var blp = new BlpFile(stream);
        using var image = blp.GetImage(0); // Mip 0 = full resolution

        var origW = image.Width;
        var origH = image.Height;

        // Check if resize is needed
        if (origW <= maxSize && origH <= maxSize)
        {
            if (verbose)
                Console.WriteLine($"  SKIP: {Path.GetFileName(outputPath)} ({origW}x{origH} already ≤ {maxSize})");

            if (!dryRun)
            {
                // Re-encode at same size (can't just copy stream since we already consumed it)
                bool hasAlpha = HasMeaningfulAlpha(image);
                if (blp1)
                    BlpWriter.WriteBlp1(outputPath, image, hasAlpha, generateMipmaps: true);
                else
                    BlpWriter.WriteBlp2(outputPath, image, hasAlpha, generateMipmaps: true);
            }
            return ProcessResult.Copied;
        }

        // Calculate new dimensions (maintain aspect ratio)
        float scale = Math.Min((float)maxSize / origW, (float)maxSize / origH);
        int newW = (int)(origW * scale);
        int newH = (int)(origH * scale);

        // Ensure power of 2 (required for BLP)
        newW = NextPowerOfTwo(newW);
        newH = NextPowerOfTwo(newH);

        // Clamp to maxSize
        newW = Math.Min(newW, maxSize);
        newH = Math.Min(newH, maxSize);

        if (verbose)
            Console.WriteLine($"  RESIZE: {Path.GetFileName(outputPath)} ({origW}x{origH} → {newW}x{newH})");
        else
            Console.WriteLine($"  {Path.GetFileName(outputPath)}: {origW}x{origH} → {newW}x{newH}");

        if (dryRun)
            return ProcessResult.Resized;

        // Resize the image
        image.Mutate(x => x.Resize(newW, newH));

        // Detect if image has meaningful alpha
        bool hasAlpha2 = HasMeaningfulAlpha(image);

        // Write the resized BLP
        if (blp1)
            BlpWriter.WriteBlp1(outputPath, image, hasAlpha2, generateMipmaps: true);
        else
            BlpWriter.WriteBlp2(outputPath, image, hasAlpha2, generateMipmaps: true);

        return ProcessResult.Resized;
    }

    static void ProcessDirectory(string inputDir, string outputDir, int maxSize, bool blp1, bool recursive, bool verbose, bool dryRun)
    {
        var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        var blpFiles = Directory.EnumerateFiles(inputDir, "*.blp", searchOption).ToList();

        Console.WriteLine($"Found {blpFiles.Count} BLP files");

        int processed = 0;
        int resized = 0;
        int skipped = 0;
        int errors = 0;

        foreach (var blpPath in blpFiles)
        {
            var relativePath = Path.GetRelativePath(inputDir, blpPath);
            var outPath = Path.Combine(outputDir, relativePath);

            try
            {
                var result = ProcessFile(blpPath, outPath, maxSize, blp1, verbose, dryRun);
                processed++;
                if (result == ProcessResult.Resized) resized++;
                else if (result == ProcessResult.Copied) skipped++;
            }
            catch (Exception ex)
            {
                errors++;
                Console.WriteLine($"  ERROR: {relativePath} - {ex.Message}");
            }
        }

        Console.WriteLine();
        Console.WriteLine($"Summary: {processed} processed, {resized} resized, {skipped} copied (already ≤{maxSize}), {errors} errors");
    }

    enum ProcessResult { Resized, Copied, Error }

    static ProcessResult ProcessFile(string inputPath, string outputPath, int maxSize, bool blp1, bool verbose, bool dryRun)
    {
        // Ensure output directory exists
        var outDir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outDir) && !dryRun)
        {
            Directory.CreateDirectory(outDir);
        }

        // Read the BLP
        using var fs = File.OpenRead(inputPath);
        using var blp = new BlpFile(fs);
        using var image = blp.GetImage(0); // Mip 0 = full resolution

        var origW = image.Width;
        var origH = image.Height;

        // Check if resize is needed
        if (origW <= maxSize && origH <= maxSize)
        {
            if (verbose)
                Console.WriteLine($"  SKIP: {Path.GetFileName(inputPath)} ({origW}x{origH} already ≤ {maxSize})");

            if (!dryRun)
            {
                // Just copy the file as-is
                File.Copy(inputPath, outputPath, overwrite: true);
            }
            return ProcessResult.Copied;
        }

        // Calculate new dimensions (maintain aspect ratio)
        float scale = Math.Min((float)maxSize / origW, (float)maxSize / origH);
        int newW = (int)(origW * scale);
        int newH = (int)(origH * scale);

        // Ensure power of 2 (required for BLP)
        newW = NextPowerOfTwo(newW);
        newH = NextPowerOfTwo(newH);

        // Clamp to maxSize
        newW = Math.Min(newW, maxSize);
        newH = Math.Min(newH, maxSize);

        if (verbose)
            Console.WriteLine($"  RESIZE: {Path.GetFileName(inputPath)} ({origW}x{origH} → {newW}x{newH})");
        else
            Console.WriteLine($"  {Path.GetFileName(inputPath)}: {origW}x{origH} → {newW}x{newH}");

        if (dryRun)
            return ProcessResult.Resized;

        // Resize the image
        image.Mutate(x => x.Resize(newW, newH));

        // Detect if image has meaningful alpha
        bool hasAlpha = HasMeaningfulAlpha(image);

        // Write the resized BLP
        if (blp1)
            BlpWriter.WriteBlp1(outputPath, image, hasAlpha, generateMipmaps: true);
        else
            BlpWriter.WriteBlp2(outputPath, image, hasAlpha, generateMipmaps: true);

        return ProcessResult.Resized;
    }

    /// <summary>
    /// Check if an image has meaningful alpha (not all 255).
    /// </summary>
    static bool HasMeaningfulAlpha(Image<Rgba32> image)
    {
        bool hasAlpha = false;
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height && !hasAlpha; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                {
                    if (row[x].A < 255)
                    {
                        hasAlpha = true;
                        break;
                    }
                }
            }
        });
        return hasAlpha;
    }

    /// <summary>
    /// Round up to the next power of 2.
    /// </summary>
    static int NextPowerOfTwo(int v)
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }

    static Dictionary<string, string> ParseArgs(string[] args)
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i].StartsWith("--") || args[i].StartsWith("-"))
            {
                var key = args[i];
                string value = "";
                if (i + 1 < args.Length && !args[i + 1].StartsWith("-"))
                {
                    value = args[++i];
                }
                result[key] = value;
            }
        }
        return result;
    }

    static void PrintUsage()
    {
        Console.WriteLine(@"
BLP Tileset Resizer - Downscale BLP textures for Alpha client compatibility

Usage:
  BlpResizer --input <path> --output <path> [options]
  BlpResizer --casc <wow_path> --output <path> [options]

Options:
  --input <path>      Input BLP file or directory
  --output <path>     Output BLP file or directory
  --casc <path>       WoW install path (CASC mode - extracts from game files)
  --listfile <path>   Path to listfile.csv (auto-downloads if not provided)
  --pattern <str>     Filter pattern for CASC mode (default: 'tileset')
  --max-size <n>      Maximum dimension (default: 256)
  --blp1              Output BLP1 format (WC3) instead of BLP2 (default: BLP2)
  --recursive, -r     Process subdirectories (file mode only)
  --verbose, -v       Show detailed output
  --dry-run           Show what would be done without writing files

Examples:
  # Resize a single tileset to 256x256 for Alpha client
  BlpResizer --input Tileset/Grass01.blp --output out/Tileset/Grass01.blp

  # Resize all tilesets in a directory
  BlpResizer --input Tileset --output out/Tileset -r

  # Extract and resize tilesets from WoW 12.0 CASC
  BlpResizer --casc ""I:\wow12\World of Warcraft"" --output out/Tileset

  # Extract with custom pattern
  BlpResizer --casc ""I:\wow12\World of Warcraft"" --output out --pattern ""tileset/generic""

Notes:
  - Alpha 0.5.3 through 10.0 supports max 256x256 tilesets
  - 11.0+ supports up to 4096x4096 but defaults to 512x512
  - Output dimensions are always power-of-2
  - BLP2 is the default (used by WoW Alpha 0.5.3+)
  - BLP2 uses DXT1 (no alpha) or DXT5 (with alpha)
  - CASC mode preserves original file paths
");
    }
}
