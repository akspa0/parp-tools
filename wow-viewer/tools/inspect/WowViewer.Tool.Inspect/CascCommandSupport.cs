using System.Diagnostics;
using System.Security.Cryptography;
using WowViewer.Core.IO.Casc;

namespace WowViewer.Tool.Inspect;

/// <summary>Spec 238: thin CLI surface over <see cref="CascStorage"/>. All CASC logic lives in the library.</summary>
public static class CascCommandSupport
{
    public static void Run(string[] args)
    {
        if (args.Length == 0)
        {
            ShowUsage();
            return;
        }

        string command = args[0].ToLowerInvariant();
        string[] tail = args.Skip(1).ToArray();
        switch (command)
        {
            case "products":
                RunProducts(tail);
                break;
            case "read":
                RunRead(tail);
                break;
            case "exists":
                RunExists(tail);
                break;
            case "wmo":
                RunWmo(tail);
                break;
            default:
                Console.Error.WriteLine($"Unknown casc command '{command}'.");
                ShowUsage();
                Environment.ExitCode = 1;
                break;
        }
    }

    private static void ShowUsage()
    {
        Console.WriteLine("CASC commands:");
        Console.WriteLine("  casc products --install <wow install dir>");
        Console.WriteLine("  casc read --install <dir> --product <product> --cache <dir> (--id <fileDataId> | --path <virtual path> --listfile <id;path csv>...) [--out <file>]");
        Console.WriteLine("  casc exists --install <dir> --product <product> --cache <dir> --paths-file <one path per line> --listfile <id;path csv>... [--show-missing]");
    }

    private static void RunExists(string[] args)
    {
        string? install = GetOption(args, "--install");
        string? product = GetOption(args, "--product");
        string? cache = GetOption(args, "--cache");
        string? pathsFile = GetOption(args, "--paths-file");
        List<string> listfiles = GetOptions(args, "--listfile");
        bool showMissing = args.Contains("--show-missing", StringComparer.OrdinalIgnoreCase);
        if (install is null || product is null || cache is null || pathsFile is null || listfiles.Count == 0)
        {
            ShowUsage();
            Environment.ExitCode = 1;
            return;
        }

        CommunityListfile listfile = CommunityListfile.Load(listfiles);
        List<string> products = GetOptions(args, "--product");
        if (products.Count > 1)
        {
            RunExistsLayered(install, products, cache, pathsFile, listfile);
            return;
        }

        CascStorage storage = CascStorage.OpenLocal(install, product, cache);
        var counts = new SortedDictionary<string, SortedDictionary<string, int>>(StringComparer.OrdinalIgnoreCase);
        var missing = new List<string>();
        foreach (string raw in File.ReadLines(pathsFile))
        {
            string path = raw.Trim();
            if (path.Length == 0)
                continue;

            string outcome;
            if (!listfile.TryGetFileDataId(path, out uint fileDataId))
            {
                outcome = "NoListfileId";
                missing.Add($"no-id\t{path}");
            }
            else
            {
                outcome = storage.TryReadFile(fileDataId, out _).ToString();
                if (outcome != nameof(CascReadStatus.Ok))
                    missing.Add($"{outcome}\t{fileDataId}\t{path}");
            }

            string extension = Path.GetExtension(path);
            if (!counts.TryGetValue(extension, out SortedDictionary<string, int>? byOutcome))
                counts[extension] = byOutcome = new SortedDictionary<string, int>(StringComparer.Ordinal);
            byOutcome[outcome] = byOutcome.GetValueOrDefault(outcome) + 1;
        }

        Console.WriteLine($"{storage.Product.Product} {storage.Product.Version}");
        foreach ((string extension, SortedDictionary<string, int> byOutcome) in counts)
            Console.WriteLine($"  {extension}\ttotal={byOutcome.Values.Sum()}\t{string.Join('\t', byOutcome.Select(static kv => $"{kv.Key}={kv.Value}"))}");

        if (showMissing)
        {
            foreach (string line in missing)
                Console.WriteLine($"  {line}");
        }
    }

    private static void RunProducts(string[] args)
    {
        string? install = GetOption(args, "--install");
        if (install is null)
        {
            ShowUsage();
            Environment.ExitCode = 1;
            return;
        }

        foreach (CascProductInfo product in CascStorage.ListProducts(install))
            Console.WriteLine($"{product.Product}\t{product.Version}\tbuild={product.BuildConfig}\tcdn={product.CdnConfig}\tpath={product.CdnPath}");
    }

    private static void RunRead(string[] args)
    {
        string? install = GetOption(args, "--install");
        string? product = GetOption(args, "--product");
        string? cache = GetOption(args, "--cache");
        string? idText = GetOption(args, "--id");
        string? path = GetOption(args, "--path");
        string? output = GetOption(args, "--out");
        List<string> listfiles = GetOptions(args, "--listfile");
        if (install is null || product is null || cache is null || (idText is null && path is null))
        {
            ShowUsage();
            Environment.ExitCode = 1;
            return;
        }

        uint fileDataId;
        if (idText is not null)
        {
            fileDataId = uint.Parse(idText);
        }
        else
        {
            CommunityListfile listfile = CommunityListfile.Load(listfiles);
            if (!listfile.TryGetFileDataId(path!, out fileDataId))
            {
                Console.Error.WriteLine($"'{path}' is not in the listfile ({listfile.Count} entries loaded).");
                Environment.ExitCode = 2;
                return;
            }
        }

        var timer = Stopwatch.StartNew();
        bool cdnFill = args.Contains("--cdn-fill", StringComparer.OrdinalIgnoreCase);
        CascStorage storage = CascStorage.OpenLocal(install, product, cache, cdnFill);
        Console.WriteLine($"opened {storage.Product.Product} {storage.Product.Version} build={storage.Product.BuildConfig} in {timer.ElapsedMilliseconds} ms");

        CascReadStatus status = storage.TryReadFile(fileDataId, out byte[]? data);
        if (status != CascReadStatus.Ok || data is null)
        {
            Console.Error.WriteLine($"fdid {fileDataId}: {status}");
            Environment.ExitCode = 2;
            return;
        }

        string magic = data.Length >= 4 ? System.Text.Encoding.ASCII.GetString(data, 0, 4) : string.Empty;
        Console.WriteLine($"fdid {fileDataId}: {data.Length} bytes, magic '{magic}', sha256 {Convert.ToHexString(SHA256.HashData(data))}");
        if (output is not null)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            File.WriteAllBytes(output, data);
            Console.WriteLine($"wrote {output}");
        }
    }

    /// <summary>Spec 239: parses a v17 WMO from CASC (groups via GFID, names via MOMT/MODI FileDataIDs) and reports resolution.</summary>
    private static void RunWmo(string[] args)
    {
        string? install = GetOption(args, "--install");
        string? cache = GetOption(args, "--cache");
        string? path = GetOption(args, "--path");
        List<string> products = GetOptions(args, "--product");
        List<string> listfiles = GetOptions(args, "--listfile");
        if (install is null || cache is null || path is null || products.Count == 0 || listfiles.Count == 0)
        {
            Console.WriteLine("  casc wmo --install <dir> --product <p> [--product <p2>] --cache <dir> --listfile <csv> --path <root.wmo>");
            Environment.ExitCode = 1;
            return;
        }

        CommunityListfile listfile = CommunityListfile.Load(listfiles);
        WowViewer.Core.IO.Files.FileDataIdPaths.Resolver = listfile.GetPath;
        List<CascStorage> storages = products.Select(p => CascStorage.OpenLocal(install, p, cache)).ToList();

        byte[]? Read(string virtualPath)
        {
            uint id;
            if (!WowViewer.Core.IO.Files.FileDataIdPaths.TryParse(virtualPath, out id) && !listfile.TryGetFileDataId(virtualPath, out id))
                return null;
            foreach (CascStorage storage in storages)
            {
                if (storage.TryReadFile(id, out byte[]? bytes) == CascReadStatus.Ok)
                    return bytes;
            }
            return null;
        }

        byte[]? root = Read(path);
        if (root is null)
        {
            Console.Error.WriteLine($"root not readable: {path}");
            Environment.ExitCode = 2;
            return;
        }

        uint[] groupIds = WowViewer.Core.IO.Converters.WmoV17ToV14Converter.ReadGroupFileDataIds(root);
        var groups = new List<byte[]>();
        foreach (uint groupId in groupIds)
        {
            byte[]? groupBytes = Read(WowViewer.Core.IO.Files.FileDataIdPaths.Resolve(groupId));
            if (groupBytes is null)
                break;
            groups.Add(groupBytes);
        }

        var model = new WowViewer.Core.IO.Converters.WmoV17ToV14Converter().ParseV17ToModel(root, groups);
        Console.WriteLine($"GFID groups={groupIds.Length} loaded={groups.Count} parsedGroups={model.Groups.Count} vertices={model.Groups.Sum(g => g.Vertices.Count)}");

        var textures = model.Materials.SelectMany(m => new[] { m.Texture1Name, m.Texture2Name, m.Texture3Name }).Where(n => !string.IsNullOrEmpty(n)).Distinct().ToList();
        Console.WriteLine($"material textures: {textures.Count} distinct, readable={textures.Count(t => Read(t) is not null)}");
        foreach (string texture in textures.Take(4))
            Console.WriteLine($"  {texture}");

        var doodadNames = model.DoodadDefs.Select(d => ResolveOffset(model.DoodadNamesRaw, d.NameIndex)).Where(n => n.Length > 0).Distinct().ToList();
        Console.WriteLine($"doodad defs={model.DoodadDefs.Count}, distinct doodad models={doodadNames.Count}, readable={doodadNames.Count(n => Read(n) is not null)}");
        foreach (string name in doodadNames.Take(4))
            Console.WriteLine($"  {name}");
    }

    private static string ResolveOffset(byte[] blob, uint offset)
    {
        if (offset >= blob.Length)
            return string.Empty;
        int end = Array.IndexOf(blob, (byte)0, (int)offset);
        return System.Text.Encoding.ASCII.GetString(blob, (int)offset, (end < 0 ? blob.Length : end) - (int)offset);
    }

    /// <summary>Opens several products in one process (as the viewer does) and reports which product serves each path.</summary>
    private static void RunExistsLayered(string install, List<string> products, string cache, string pathsFile, CommunityListfile listfile)
    {
        var storages = new List<CascStorage>();
        foreach (string product in products)
        {
            try
            {
                storages.Add(CascStorage.OpenLocal(install, product, cache));
                Console.WriteLine($"opened {product}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"FAILED to open {product}: {ex.GetType().Name}: {ex.Message}");
            }
        }

        var servedBy = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (string raw in File.ReadLines(pathsFile))
        {
            string path = raw.Trim();
            if (path.Length == 0)
                continue;

            string outcome = "unresolved";
            if (listfile.TryGetFileDataId(path, out uint fileDataId))
            {
                foreach (CascStorage storage in storages)
                {
                    if (storage.TryReadFile(fileDataId, out _) == CascReadStatus.Ok)
                    {
                        outcome = storage.Product.Product;
                        break;
                    }
                }
            }

            string key = $"{Path.GetExtension(path).ToUpperInvariant()} <- {outcome}";
            servedBy[key] = servedBy.GetValueOrDefault(key) + 1;
        }

        foreach ((string key, int count) in servedBy)
            Console.WriteLine($"  {key}: {count}");
    }

    private static string? GetOption(string[] args, string name)
    {
        int index = Array.FindIndex(args, a => string.Equals(a, name, StringComparison.OrdinalIgnoreCase));
        return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
    }

    private static List<string> GetOptions(string[] args, string name)
    {
        var values = new List<string>();
        for (int i = 0; i + 1 < args.Length; i++)
        {
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
                values.Add(args[i + 1]);
        }

        return values;
    }
}
