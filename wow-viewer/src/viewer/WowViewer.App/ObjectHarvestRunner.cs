using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WowViewer.App;

internal sealed record ObjectHarvestOptions
{
    public string ClientRoot { get; init; } = string.Empty;
    public string ClientVersion { get; init; } = string.Empty;
    public string OutputDir { get; init; } = string.Empty;
    public string? ModelList { get; init; }
    public int RenderSize { get; init; } = 256;
    public int Workers { get; init; } = 1;
}

internal sealed class ObjectHarvestEntry
{
    [JsonPropertyName("virtual_path")]
    public string VirtualPath { get; init; } = string.Empty;

    [JsonPropertyName("hash")]
    public string Hash { get; init; } = string.Empty;

    [JsonPropertyName("signature_path")]
    public string SignaturePath { get; init; } = string.Empty;

    [JsonPropertyName("model_type")]
    public string ModelType { get; init; } = string.Empty;

    [JsonPropertyName("version")]
    public string Version { get; init; } = string.Empty;
}

internal sealed class ObjectHarvestManifest
{
    [JsonPropertyName("client_version")]
    public string ClientVersion { get; init; } = string.Empty;

    [JsonPropertyName("client_root")]
    public string ClientRoot { get; init; } = string.Empty;

    [JsonPropertyName("count")]
    public int Count { get; init; }

    [JsonPropertyName("entries")]
    public List<ObjectHarvestEntry> Entries { get; init; } = [];
}

internal static class ObjectHarvestRunner
{
    private static readonly string[] WmoRootSearchPatterns = ["*.wmo"];
    private static readonly string[] MdxSearchPatterns = ["*.mdx"];
    private static readonly string WorldDirectory = "World";
    private static readonly string OutputExtension = ".png";
    private static readonly System.Text.RegularExpressions.Regex WmoGroupSuffix = new(@"_\d{3}\.wmo$", System.Text.RegularExpressions.RegexOptions.IgnoreCase);

    public static int Run(ObjectHarvestOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (string.IsNullOrWhiteSpace(options.ClientRoot))
            throw new ArgumentException("Provide --client-root <game dir> for object-harvest.");
        if (string.IsNullOrWhiteSpace(options.OutputDir))
            throw new ArgumentException("Provide --output-dir <dir> for object-harvest.");
        if (options.RenderSize < 16 || options.RenderSize > 4096)
            throw new ArgumentOutOfRangeException(nameof(options.RenderSize), "--render-size must be in range 16..4096.");

        string clientRoot = Path.GetFullPath(options.ClientRoot);
        string outputDir = Path.GetFullPath(options.OutputDir);
        Directory.CreateDirectory(outputDir);

        List<string> discoveredPaths = DiscoverModels(clientRoot, options.ModelList);

        var entries = new List<ObjectHarvestEntry>();
        int succeeded = 0;
        int failed = 0;

        for (int index = 0; index < discoveredPaths.Count; index++)
        {
            string virtualPath = discoveredPaths[index];
            string modelType = DetectModelType(virtualPath);
            string hash = ComputePathHash(virtualPath);
            string signatureFileName = $"{hash}__{options.ClientVersion}{OutputExtension}";
            string outputPath = Path.Combine(outputDir, signatureFileName);

            Console.Error.Write($"\r[{index + 1}/{discoveredPaths.Count}] {virtualPath}");

            try
            {
                if (modelType == "wmo_root")
                {
                    WmoPreviewLoadRequest request = new()
                    {
                        ArchiveRoot = clientRoot,
                        VirtualPath = virtualPath,
                        VisualWidth = options.RenderSize,
                        VisualHeight = options.RenderSize,
                        Camera = new PreviewCameraSettings
                        {
                            Mode = PreviewCameraMode.Orbit,
                            PresetName = "top",
                        },
                    };
                    WmoGpuPreviewCaptureRunner.Capture(request, outputPath);
                }
                else
                {
                    MdxPreviewLoadRequest request = new()
                    {
                        ArchiveRoot = clientRoot,
                        VirtualPath = virtualPath,
                        VisualWidth = options.RenderSize,
                        VisualHeight = options.RenderSize,
                        Camera = new PreviewCameraSettings
                        {
                            Mode = PreviewCameraMode.Orbit,
                            PresetName = "top",
                        },
                    };
                    MdxGpuPreviewCaptureRunner.Capture(request, outputPath);
                }

                entries.Add(new ObjectHarvestEntry
                {
                    VirtualPath = virtualPath,
                    Hash = hash,
                    SignaturePath = signatureFileName,
                    ModelType = modelType,
                    Version = options.ClientVersion,
                });
                succeeded++;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"\n  FAILED: {ex.Message}");
                failed++;
            }
        }

        Console.Error.WriteLine();
        Console.Error.WriteLine($"Harvest complete: {succeeded} succeeded, {failed} failed");

        WriteManifest(outputDir, options, entries);
        return failed > 0 && succeeded == 0 ? 1 : 0;
    }

    private static List<string> DiscoverModels(string clientRoot, string? modelListPath)
    {
        var paths = new List<string>();

        if (!string.IsNullOrWhiteSpace(modelListPath))
        {
            if (!File.Exists(modelListPath))
                throw new FileNotFoundException($"Model list file not found: {modelListPath}", modelListPath);

            foreach (string line in File.ReadAllLines(modelListPath))
            {
                string trimmedLine = line.Trim();
                if (trimmedLine.Length == 0 || trimmedLine.StartsWith('#') || trimmedLine.StartsWith("//"))
                    continue;

                paths.Add(trimmedLine);
            }

            return paths;
        }

        DiscoverLooseFiles(clientRoot, WorldDirectory, paths);
        return paths;
    }

    private static void DiscoverLooseFiles(string clientRoot, string worldDir, List<string> paths)
    {
        string worldPath = Path.Combine(clientRoot, worldDir);
        if (!Directory.Exists(worldPath))
        {
            Console.Error.WriteLine($"World directory not found: {worldPath}");
            return;
        }

        foreach (string pattern in WmoRootSearchPatterns)
        {
            foreach (string file in Directory.EnumerateFiles(worldPath, pattern, SearchOption.AllDirectories))
            {
                if (WmoGroupSuffix.IsMatch(file))
                    continue;

                string virtualPath = GetVirtualPath(clientRoot, file);
                paths.Add(virtualPath);
            }
        }

        foreach (string pattern in MdxSearchPatterns)
        {
            foreach (string file in Directory.EnumerateFiles(worldPath, pattern, SearchOption.AllDirectories))
            {
                string virtualPath = GetVirtualPath(clientRoot, file);
                paths.Add(virtualPath);
            }
        }
    }

    private static string GetVirtualPath(string clientRoot, string fullPath)
    {
        string normalizedClientRoot = clientRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string normalizedFullPath = fullPath;
        if (normalizedFullPath.StartsWith(normalizedClientRoot, StringComparison.OrdinalIgnoreCase))
            normalizedFullPath = normalizedFullPath[(normalizedClientRoot.Length + 1)..];

        return normalizedFullPath.Replace('/', '\\');
    }

    private static string DetectModelType(string virtualPath)
    {
        string extension = Path.GetExtension(virtualPath).ToLowerInvariant();
        return extension switch
        {
            ".wmo" => "wmo_root",
            ".mdx" => "mdx",
            _ => "unknown",
        };
    }

    private static string ComputePathHash(string virtualPath)
    {
        byte[] hashBytes = MD5.HashData(Encoding.UTF8.GetBytes(virtualPath));
        return Convert.ToHexString(hashBytes).ToLowerInvariant();
    }

    private static void WriteManifest(string outputDir, ObjectHarvestOptions options, List<ObjectHarvestEntry> entries)
    {
        var manifest = new ObjectHarvestManifest
        {
            ClientVersion = options.ClientVersion,
            ClientRoot = Path.GetFullPath(options.ClientRoot),
            Count = entries.Count,
            Entries = entries,
        };

        string manifestPath = Path.Combine(outputDir, $"object-harvest-index__{options.ClientVersion}.json");
        string json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(manifestPath, json);
        Console.Error.WriteLine($"Manifest: {manifestPath}");
    }
}
