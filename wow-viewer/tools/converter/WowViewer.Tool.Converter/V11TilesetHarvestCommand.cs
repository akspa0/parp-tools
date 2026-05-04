using System.IO.Compression;
using System.Text.Json;
using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WoWMapConverter.Core.Services;

namespace WowViewer.Tool.Converter;

public static class V11TilesetHarvestCommand
{
    public static void Run(string[] args)
    {
        var cliArgs = ParseArgs(args);
        string manifestPath = cliArgs.GetValueOrDefault("--input", cliArgs.GetValueOrDefault("-i", ""));
        string outputDir = cliArgs.GetValueOrDefault("--output", cliArgs.GetValueOrDefault("-o", ""));
        string clientRoot = cliArgs.GetValueOrDefault("--client-root", cliArgs.GetValueOrDefault("-c", ""));

        if (string.IsNullOrWhiteSpace(manifestPath) || string.IsNullOrWhiteSpace(outputDir))
        {
            Console.Error.WriteLine("Usage: harvest-tilesets --input <manifest.json> --output <dir> [--client-root <path>]");
            Environment.ExitCode = 1;
            return;
        }

        string output = Path.GetFullPath(outputDir);
        Directory.CreateDirectory(output);

        // Collect unique MCLY texture names from manifest
        var clientTextures = new Dictionary<string, HashSet<string>>(StringComparer.OrdinalIgnoreCase);
        string manifestJson = File.ReadAllText(manifestPath);
        using var doc = System.Text.Json.JsonDocument.Parse(manifestJson);
        var entries = doc.RootElement.GetProperty("entries");

        foreach (var entry in entries.EnumerateArray())
        {
            string? datasetKey = null;
            if (entry.TryGetProperty("dataset_key", out var dk))
                datasetKey = dk.GetString() ?? "";

            // also try reading from shard metadata
            string? shardPath = null;
            if (entry.TryGetProperty("shard_path", out var sp))
                shardPath = sp.GetString() ?? "";

            if (string.IsNullOrWhiteSpace(shardPath) || !File.Exists(shardPath))
                continue;

            try
            {
                // Read sidecar metadata
                string sidecar = Path.ChangeExtension(shardPath!, null) + "_metadata.json";
                if (!File.Exists(sidecar)) continue;

                var meta = JsonDocument.Parse(File.ReadAllText(sidecar));
                if (!meta.RootElement.TryGetProperty("mcly_texture_names", out var names)) continue;
                foreach (var name in names.EnumerateArray())
                {
                    string textureName = (name.GetString() ?? "").Trim();
                    if (string.IsNullOrWhiteSpace(textureName)) continue;

                    string client = string.IsNullOrWhiteSpace(datasetKey) ? "default" : datasetKey;
                    if (!clientTextures.ContainsKey(client))
                        clientTextures[client] = new(StringComparer.OrdinalIgnoreCase);
                    clientTextures[client].Add(textureName);
                }
            }
            catch { }
        }

        // Flatten to unique set
        var allTextures = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in clientTextures)
        foreach (var tex in kvp.Value)
            allTextures.Add(tex);

        Console.WriteLine($"Unique textures to harvest: {allTextures.Count}");
        Console.WriteLine($"Clients: {string.Join(", ", clientTextures.Keys)}");

        // If client roots provided, resolve the client path and load archives
        Dictionary<string, NativeMpqService> services = new(StringComparer.OrdinalIgnoreCase);

        if (!string.IsNullOrWhiteSpace(clientRoot))
        {
            string root = Path.GetFullPath(clientRoot);
            var svc = new NativeMpqService();
            svc.LoadArchives([root]);
            string? listfile = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "MdxViewer", "community-listfile-withcapitals.csv");
            if (File.Exists(listfile))
                svc.LoadListfile(listfile);
            services[root] = svc;
        }
        else
        {
            // Try to find client roots from staged copies
            foreach (string clientKey in clientTextures.Keys)
            {
                    // Parse clientkey like "3_3_5_12340__azeroth" -> "3_3_5_12340"
                    string clientBuild = clientKey.Split("__")[0];
                    string candidateRoot = Path.Combine(Environment.CurrentDirectory, "output", "tmp", "wowarchive-clients", clientBuild, "World of Warcraft");
                    if (!Directory.Exists(candidateRoot))
                        continue;

                    if (services.ContainsKey(candidateRoot)) continue;

                    Console.WriteLine($"Loading client: {candidateRoot}");
                var svc = new NativeMpqService();
                svc.LoadArchives([candidateRoot]);
                string? listfile = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "MdxViewer", "community-listfile-withcapitals.csv");
                if (File.Exists(listfile))
                    svc.LoadListfile(listfile);
                services[candidateRoot] = svc;
            }

            if (services.Count == 0)
            {
                Console.Error.WriteLine("No client roots found. Use --client-root <path> to specify.");
                Environment.ExitCode = 1;
                return;
            }
        }

        // Harvest: for each unique texture name, try to read from each loaded service
        int harvested = 0;
        int missing = 0;
        var harvestedIndex = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        foreach (string textureName in allTextures)
        {
            byte[]? blpData = null;
            foreach (var svc in services.Values)
            {
                blpData = svc.ReadFile(textureName);
                if (blpData is { Length: > 0 }) break;
            }

            if (blpData is null or { Length: 0 })
            {
                missing++;
                if (missing % 100 == 0)
                    Console.WriteLine($"  Missing: {missing}...");
                continue;
            }

            try
            {
                var blp = new BlpFile(new MemoryStream(blpData));
                var pixels = blp.GetPixels(0, out int w, out int h, bgra: true);
                if (pixels is null || w <= 0 || h <= 0) continue;

                // pixels are BGRA, convert to RGBA for ImageSharp
                for (int i = 0; i < pixels.Length; i += 4)
                {
                    (pixels[i], pixels[i + 2]) = (pixels[i + 2], pixels[i]);
                }

                string safeName = textureName.Replace('\\', '_').Replace('/', '_').Replace(':', '_');
                string pngPath = Path.Combine(output, safeName + ".png");
                using var img = Image.LoadPixelData<Rgba32>(pixels, w, h);
                img.SaveAsPng(pngPath);
                harvestedIndex[textureName] = pngPath;
                harvested++;

                if (harvested % 50 == 0)
                    Console.WriteLine($"  Harvested: {harvested}/{allTextures.Count}");
            }
            catch (Exception ex)
            {
                missing++;
                if (harvested % 50 == 0)
                    Console.WriteLine($"  Error: {textureName}: {ex.Message}");
            }
        }

        // Write index
        var indexPath = Path.Combine(output, "tileset_index.json");
        File.WriteAllText(indexPath, System.Text.Json.JsonSerializer.Serialize(new
        {
            harvested,
            missing,
            output_dir = output,
            textures = harvestedIndex,
        }, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));

        Console.WriteLine($"\nDone: {harvested} harvested, {missing} missing");
        Console.WriteLine($"Index: {indexPath}");

        foreach (var svc in services.Values)
            svc.Dispose();
    }

    private static Dictionary<string, string> ParseArgs(string[] args)
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i].StartsWith('-') && !args[i + 1].StartsWith('-'))
                result[args[i]] = args[i + 1];
        }
        return result;
    }
}
