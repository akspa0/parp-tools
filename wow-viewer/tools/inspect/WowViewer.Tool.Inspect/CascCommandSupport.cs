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
            case "wmo-survey":
                RunWmoSurvey(tail);
                break;
            case "map-survey":
                RunMapSurvey(tail);
                break;
            case "db2":
                RunDb2(tail);
                break;
            case "m2":
                RunM2(tail);
                break;
            case "adt-heights":
                RunAdtHeights(tail);
                break;
            case "bench":
                RunBench(tail);
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
        Console.WriteLine("  casc wmo-survey --install <dir> --product <p> --cache <dir> --listfile <csv> [--limit <n>]");
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

    /// <summary>
    /// Spec 239: parses every listfile-named WMO root present locally in one product and tallies
    /// parse failures by message, plus the MOGP sub-chunk sequence of each group that fails alone.
    /// Local data only (no CDN), so the survey measures the install as-is.
    /// </summary>
    private static void RunWmoSurvey(string[] args)
    {
        string? install = GetOption(args, "--install");
        string? product = GetOption(args, "--product");
        string? cache = GetOption(args, "--cache");
        List<string> listfiles = GetOptions(args, "--listfile");
        int limit = int.TryParse(GetOption(args, "--limit"), out int parsedLimit) ? parsedLimit : int.MaxValue;
        if (install is null || product is null || cache is null || listfiles.Count == 0)
        {
            Console.WriteLine("  casc wmo-survey --install <dir> --product <p> --cache <dir> --listfile <csv> [--limit <n>]");
            Environment.ExitCode = 1;
            return;
        }

        CommunityListfile listfile = CommunityListfile.Load(listfiles);
        WowViewer.Core.IO.Files.FileDataIdPaths.Resolver = listfile.GetPath;
        CascStorage storage = CascStorage.OpenLocal(install, product, cache);
        var groupSuffix = new System.Text.RegularExpressions.Regex(@"_\d{3}\.wmo$|_lod\d\.wmo$", System.Text.RegularExpressions.RegexOptions.IgnoreCase);

        List<KeyValuePair<uint, string>> roots = listfile.Entries
            .Where(e => e.Value.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase) && !groupSuffix.IsMatch(e.Value) && storage.FileExists(e.Key))
            .OrderBy(static e => e.Key)
            .Take(limit)
            .ToList();

        int ok = 0, notLocal = 0, nonV17 = 0, failed = 0;
        var materialShapes = new Dictionary<string, (int Count, string Example)>(StringComparer.Ordinal);
        var momxSizes = new Dictionary<string, (int Count, string Example)>(StringComparer.Ordinal);
        long batchTotal = 0, batchLargeFlag = 0, batchLargeDiffers = 0;
        int maxLargeMaterialId = 0;
        string? largeExample = null;
        var failureMessages = new Dictionary<string, int>(StringComparer.Ordinal);
        var failingGroupLayouts = new Dictionary<string, (int Count, string Example)>(StringComparer.Ordinal);
        var parser = new WowViewer.Core.IO.Converters.WmoV17ToV14Converter();
        Stopwatch stopwatch = Stopwatch.StartNew();
        foreach ((uint rootId, string rootPath) in roots)
        {
            if (storage.TryReadFile(rootId, out byte[]? root) != CascReadStatus.Ok || root is null)
            {
                notLocal++;
                continue;
            }

            if (root.Length < 12 || BitConverter.ToUInt32(root, 8) != 17)
            {
                nonV17++;
                continue;
            }

            TallyMaterials(root, rootPath, materialShapes, momxSizes);

            var groups = new List<byte[]>();
            bool groupMissing = false;
            foreach (uint groupId in WowViewer.Core.IO.Converters.WmoV17ToV14Converter.ReadGroupFileDataIds(root))
            {
                if (storage.TryReadFile(groupId, out byte[]? groupBytes) != CascReadStatus.Ok || groupBytes is null)
                {
                    groupMissing = true;
                    break;
                }

                groups.Add(groupBytes);
            }

            if (groupMissing)
            {
                notLocal++;
                continue;
            }

            TallyBatches(groups, ref batchTotal, ref batchLargeFlag, ref batchLargeDiffers, ref maxLargeMaterialId, ref largeExample, rootPath);

            try
            {
                parser.ParseV17ToModel(root, groups);
                ok++;
            }
            catch (Exception ex)
            {
                failed++;
                string message = System.Text.RegularExpressions.Regex.Replace(ex.GetType().Name + ": " + ex.Message, @"0x[0-9A-Fa-f]+", "0x?");
                failureMessages[message] = failureMessages.GetValueOrDefault(message) + 1;
                foreach (byte[] group in groups)
                {
                    try
                    {
                        parser.ParseV17ToModel(root, [group]);
                    }
                    catch (Exception)
                    {
                        string layout = DescribeMogpLayout(group);
                        (int count, string example) = failingGroupLayouts.GetValueOrDefault(layout, (0, rootPath));
                        failingGroupLayouts[layout] = (count + 1, example);
                    }
                }
            }
        }

        Console.WriteLine($"{product}: {roots.Count} WMO roots in listfile+root; ok={ok} failed={failed} notLocal={notLocal} nonV17={nonV17} in {stopwatch.Elapsed.TotalSeconds:F1}s");
        foreach ((string message, int count) in failureMessages.OrderByDescending(static kv => kv.Value))
            Console.WriteLine($"  {count,6}  {message}");
        Console.WriteLine($"MOBA batches: {batchTotal}; flag 0x2 (material_id_large) set on {batchLargeFlag}; large id differs from material_id on {batchLargeDiffers}; max large id {maxLargeMaterialId}{(largeExample is null ? "" : $" e.g. {largeExample}")}");
        Console.WriteLine("MOMT materials (shader | blend | texture slots set 1/2/3 | MOMX present):");
        foreach ((string shape, (int count, string example)) in materialShapes.OrderByDescending(static kv => kv.Value.Count).Take(40))
            Console.WriteLine($"  {count,7}  {shape}   e.g. {example}");
        Console.WriteLine("MOMX size relative to MOMT count:");
        foreach ((string shape, (int count, string example)) in momxSizes.OrderByDescending(static kv => kv.Value.Count).Take(10))
            Console.WriteLine($"  {count,7}  {shape}   e.g. {example}");
        Console.WriteLine("failing group layouts (flags | sub-chunks):");
        foreach ((string layout, (int count, string example)) in failingGroupLayouts.OrderByDescending(static kv => kv.Value.Count).Take(25))
            Console.WriteLine($"  {count,6}  {layout}   e.g. {example}");
    }

    private static void TallyBatches(List<byte[]> groups, ref long total, ref long largeFlag, ref long largeDiffers, ref int maxLarge, ref string? example, string rootPath)
    {
        foreach (byte[] group in groups)
        {
            // MOGP payload: 68-byte header then sub-chunks.
            if (group.Length < 12 + 8 + 68)
                continue;
            int mogpSize = BitConverter.ToInt32(group, 12 + 4);
            int end = Math.Min(group.Length, 12 + 8 + mogpSize);
            for (int position = 12 + 8 + 68; position + 8 <= end;)
            {
                string id = new string(System.Text.Encoding.ASCII.GetString(group, position, 4).Reverse().ToArray());
                int size = BitConverter.ToInt32(group, position + 4);
                if (size < 0)
                    break;
                if (id == "MOBA")
                {
                    for (int record = position + 8; record + 24 <= Math.Min(end, position + 8 + size); record += 24)
                    {
                        total++;
                        ushort large = BitConverter.ToUInt16(group, record + 10);
                        byte flags = group[record + 22];
                        byte materialId = group[record + 23];
                        if ((flags & 0x2) != 0)
                        {
                            largeFlag++;
                            maxLarge = Math.Max(maxLarge, large);
                            if (large != materialId)
                            {
                                largeDiffers++;
                                example ??= rootPath;
                            }
                        }
                    }
                }

                position += 8 + size;
            }
        }
    }

    private static void TallyMaterials(
        byte[] root,
        string rootPath,
        Dictionary<string, (int Count, string Example)> materialShapes,
        Dictionary<string, (int Count, string Example)> momxSizes)
    {
        int momt = -1, momtSize = 0, momxSize = -1;
        for (int position = 0; position + 8 <= root.Length;)
        {
            string id = new string(System.Text.Encoding.ASCII.GetString(root, position, 4).Reverse().ToArray());
            int size = BitConverter.ToInt32(root, position + 4);
            if (size < 0 || position + 8L + size > root.Length)
                break;
            if (id == "MOMT") { momt = position + 8; momtSize = size; }
            if (id == "MOMX") momxSize = size;
            position += 8 + size;
        }

        if (momt < 0)
            return;

        int materialCount = momtSize / 64;
        if (momxSize >= 0)
        {
            string key = materialCount == 0 ? $"size={momxSize}" : $"bytes/material={(double)momxSize / materialCount:0.##} (size {momxSize % Math.Max(1, materialCount) == 0})";
            (int c, string e) = momxSizes.GetValueOrDefault(key, (0, rootPath));
            momxSizes[key] = (c + 1, e);
        }

        for (int i = 0; i < materialCount; i++)
        {
            int o = momt + i * 64;
            uint shader = BitConverter.ToUInt32(root, o + 4);
            uint blend = BitConverter.ToUInt32(root, o + 8);
            uint t1 = BitConverter.ToUInt32(root, o + 12);
            uint t2 = BitConverter.ToUInt32(root, o + 24);
            uint t3 = BitConverter.ToUInt32(root, o + 36);
            string shape = $"shader {shader,2} | blend {blend} | {(t1 != 0 ? 1 : 0)}{(t2 != 0 ? 1 : 0)}{(t3 != 0 ? 1 : 0)} | momx {(momxSize >= 0 ? "y" : "n")}";
            (int c, string e) = materialShapes.GetValueOrDefault(shape, (0, rootPath));
            materialShapes[shape] = (c + 1, e);
        }
    }

    private static string DescribeMogpLayout(byte[] group)
    {
        var parts = new List<string>();
        for (int position = 0; position + 8 <= group.Length;)
        {
            string id = new string(System.Text.Encoding.ASCII.GetString(group, position, 4).Reverse().ToArray());
            int size = BitConverter.ToInt32(group, position + 4);
            if (size < 0)
                break;

            if (id == "MOGP" && position + 8 + 68 <= group.Length)
            {
                uint flags = BitConverter.ToUInt32(group, position + 8 + 8);
                var sub = new List<string>();
                for (int inner = position + 8 + 68; inner + 8 <= Math.Min(group.Length, position + 8 + size);)
                {
                    string subId = new string(System.Text.Encoding.ASCII.GetString(group, inner, 4).Reverse().ToArray());
                    int subSize = BitConverter.ToInt32(group, inner + 4);
                    if (subSize < 0)
                        break;
                    sub.Add(subId);
                    inner += 8 + subSize;
                }

                parts.Add($"0x{flags:X8} | {string.Join(' ', sub)}");
            }

            position += 8 + size;
        }

        return string.Join(" ; ", parts);
    }

    /// <summary>
    /// Spec 239: surveys every MAID tile of a WDT in one product: tex0 texture tables (MTEX vs MDID/MHID),
    /// MCLY layer counts, and MDDF/MODF flag bits that mark FileDataID name references.
    /// </summary>
    private static void RunMapSurvey(string[] args)
    {
        string? install = GetOption(args, "--install");
        string? product = GetOption(args, "--product");
        string? cache = GetOption(args, "--cache");
        string? wdtIdText = GetOption(args, "--wdt-id");
        if (install is null || product is null || cache is null || wdtIdText is null)
        {
            Console.WriteLine("  casc map-survey --install <dir> --product <p> --cache <dir> --wdt-id <fileDataId> [--cdn-fill]");
            Environment.ExitCode = 1;
            return;
        }

        bool cdnFill = args.Contains("--cdn-fill", StringComparer.OrdinalIgnoreCase);
        CascStorage storage = CascStorage.OpenLocal(install, product, cache, cdnFill);
        byte[]? Read(uint id) => id != 0 && storage.TryReadFile(id, out byte[]? bytes) == CascReadStatus.Ok ? bytes : null;

        byte[] wdt = Read(uint.Parse(wdtIdText)) ?? throw new InvalidOperationException("WDT not readable");
        var chunks = TopChunks(wdt).ToDictionary(static c => c.Id, static c => c, StringComparer.Ordinal);
        uint mphdFlags = BitConverter.ToUInt32(wdt, chunks["MPHD"].Offset);
        Console.WriteLine($"WDT MPHD flags=0x{mphdFlags:X}; top-level chunks: {string.Join(' ', TopChunks(wdt).Select(static c => $"{c.Id}({c.Size})"))}");
        if (!chunks.TryGetValue("MAID", out var maid))
        {
            Console.WriteLine("no MAID chunk");
            return;
        }

        int tiles = 0, rootOk = 0, tex0Ok = 0, obj0Ok = 0, withMtex = 0, withMdid = 0, withMhid = 0;
        var layerHistogram = new SortedDictionary<int, int>();
        var tex0Chunks = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var obj0Chunks = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var mddfFlags = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var modfFlags = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var tileList = new List<string>();
        var zeroExamples = new SortedSet<string>(StringComparer.Ordinal);
        var mclyFlags = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var layerRefs = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var mcalSizesPerLayerCount = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var textureIds = new HashSet<uint>();
        var doodadIds = new HashSet<uint>();
        var wmoIds = new HashSet<uint>();
        for (int slot = 0; slot < maid.Size / 32; slot++)
        {
            uint rootId = BitConverter.ToUInt32(wdt, maid.Offset + slot * 32);
            uint obj0Id = BitConverter.ToUInt32(wdt, maid.Offset + slot * 32 + 4);
            uint tex0Id = BitConverter.ToUInt32(wdt, maid.Offset + slot * 32 + 12);
            if (rootId == 0 && obj0Id == 0 && tex0Id == 0)
                continue;

            tiles++;
            tileList.Add($"{slot % 64}_{slot / 64}");
            rootOk += Read(rootId) is null ? 0 : 1;

            if (Read(tex0Id) is { } tex0)
            {
                tex0Ok++;
                var top = TopChunks(tex0).ToList();
                foreach (string id in top.Select(static c => c.Id).Distinct())
                    tex0Chunks[id] = tex0Chunks.GetValueOrDefault(id) + 1;
                withMtex += top.Any(static c => c.Id == "MTEX") ? 1 : 0;
                withMdid += top.Any(static c => c.Id == "MDID") ? 1 : 0;
                withMhid += top.Any(static c => c.Id == "MHID") ? 1 : 0;
                foreach (var c in top.Where(static c => c.Id == "MDID"))
                {
                    for (int p = c.Offset; p + 4 <= c.Offset + c.Size; p += 4)
                    {
                        uint textureId = BitConverter.ToUInt32(tex0, p);
                        if (textureId != 0)
                            textureIds.Add(textureId);
                    }
                }
                uint[] mdid = top.Where(static c => c.Id == "MDID").Select(c => Enumerable.Range(0, c.Size / 4).Select(i => BitConverter.ToUInt32(tex0, c.Offset + i * 4)).ToArray()).FirstOrDefault() ?? [];
                foreach (var mcnk in top.Where(static c => c.Id == "MCNK"))
                {
                    var sub = TopChunks(tex0, mcnk.Offset, mcnk.Offset + mcnk.Size).ToList();
                    var mcly = sub.FirstOrDefault(static c => c.Id == "MCLY");
                    int layers = mcly.Id is null ? 0 : mcly.Size / 16;
                    layerHistogram[layers] = layerHistogram.GetValueOrDefault(layers) + 1;
                    var mcal = sub.FirstOrDefault(static c => c.Id == "MCAL");
                    for (int l = 0; l < layers; l++)
                    {
                        int textureIndex = BitConverter.ToInt32(tex0, mcly.Offset + l * 16);
                        uint layerFlags = BitConverter.ToUInt32(tex0, mcly.Offset + l * 16 + 4);
                        string flagKey = $"0x{layerFlags & ~0x7u:X}";
                        mclyFlags[flagKey] = mclyFlags.GetValueOrDefault(flagKey) + 1;
                        string reference = textureIndex < 0 || textureIndex >= mdid.Length ? "index-out-of-MDID" : mdid[textureIndex] == 0 ? "MDID-zero" : "MDID-ok";
                        if (reference == "MDID-zero" && zeroExamples.Count < 6)
                            zeroExamples.Add($"tile {slot % 64}_{slot / 64} tex0={tex0Id} MDID[{textureIndex}]=0 (MDID length {mdid.Length}, MHID[{textureIndex}]={(top.FirstOrDefault(static c => c.Id == "MHID") is var mh && mh.Id is not null && textureIndex * 4 + 4 <= mh.Size ? BitConverter.ToUInt32(tex0, mh.Offset + textureIndex * 4) : 0)})");
                        layerRefs[reference] = layerRefs.GetValueOrDefault(reference) + 1;
                    }

                    if (layers > 1)
                    {
                        string mcalKey = mcal.Id is null ? "none" : $"{mcal.Size}";
                        mcalSizesPerLayerCount[$"{layers}L:{mcalKey}"] = mcalSizesPerLayerCount.GetValueOrDefault($"{layers}L:{mcalKey}") + 1;
                    }
                }
            }

            if (Read(obj0Id) is { } obj0)
            {
                obj0Ok++;
                var top = TopChunks(obj0).ToList();
                foreach (string id in top.Select(static c => c.Id).Distinct())
                    obj0Chunks[id] = obj0Chunks.GetValueOrDefault(id) + 1;
                foreach (var c in top.Where(static c => c.Id == "MDDF"))
                {
                    for (int p = c.Offset; p + 36 <= c.Offset + c.Size; p += 36)
                    {
                        ushort flags = BitConverter.ToUInt16(obj0, p + 34);
                        string key = $"0x{flags:X}";
                        mddfFlags[key] = mddfFlags.GetValueOrDefault(key) + 1;
                        if ((flags & 0x40) != 0)
                            doodadIds.Add(BitConverter.ToUInt32(obj0, p));
                    }
                }
                foreach (var c in top.Where(static c => c.Id == "MODF"))
                {
                    for (int p = c.Offset; p + 64 <= c.Offset + c.Size; p += 64)
                    {
                        ushort flags = BitConverter.ToUInt16(obj0, p + 56);
                        string key = $"0x{flags:X}";
                        modfFlags[key] = modfFlags.GetValueOrDefault(key) + 1;
                        if ((flags & 0x8) != 0)
                            wmoIds.Add(BitConverter.ToUInt32(obj0, p));
                    }
                }
            }
        }

        Console.WriteLine($"tiles={tiles} rootReadable={rootOk} tex0Readable={tex0Ok} obj0Readable={obj0Ok}");
        Console.WriteLine($"tex0: MTEX={withMtex} MDID={withMdid} MHID={withMhid}; top-level chunk file counts {string.Join(' ', tex0Chunks.Select(static kv => $"{kv.Key}:{kv.Value}"))}");
        Console.WriteLine($"MCLY layers per MCNK: {string.Join(' ', layerHistogram.Select(static kv => $"{kv.Key}:{kv.Value}"))}");
        Console.WriteLine($"MCLY flags (low 3 bits masked): {string.Join(' ', mclyFlags.Select(static kv => $"{kv.Key}:{kv.Value}"))}");
        Console.WriteLine($"MCLY texture references: {string.Join(' ', layerRefs.Select(static kv => $"{kv.Key}:{kv.Value}"))}");
        foreach (string example in zeroExamples)
            Console.WriteLine($"MDID-zero example: {example}");
        Console.WriteLine($"MCAL size by layer count (top 12): {string.Join(' ', mcalSizesPerLayerCount.OrderByDescending(static kv => kv.Value).Take(12).Select(static kv => $"{kv.Key}={kv.Value}"))}");
        Console.WriteLine($"obj0 top-level chunk file counts {string.Join(' ', obj0Chunks.Select(static kv => $"{kv.Key}:{kv.Value}"))}");
        Console.WriteLine($"MDDF flags: {string.Join(' ', mddfFlags.Select(static kv => $"{kv.Key}:{kv.Value}"))}");
        Console.WriteLine($"MODF flags: {string.Join(' ', modfFlags.Select(static kv => $"{kv.Key}:{kv.Value}"))}");
        Console.WriteLine($"tiles (x_y from MAID slot): {string.Join(' ', tileList.Take(40))}{(tileList.Count > 40 ? " ..." : "")}");
        var wmoSequences = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var wmoParse = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (uint wmoId in wmoIds)
        {
            if (Read(wmoId) is not { } wmoRoot)
                continue;
            string sequence = string.Join(' ', TopChunks(wmoRoot).Select(static c => c.Id));
            wmoSequences[sequence] = wmoSequences.GetValueOrDefault(sequence) + 1;

            string outcome;
            try
            {
                var groups = WowViewer.Core.IO.Converters.WmoV17ToV14Converter.ReadGroupFileDataIds(wmoRoot)
                    .Select(Read).TakeWhile(static g => g is not null).Select(static g => g!).ToList();
                var model = new WowViewer.Core.IO.Converters.WmoV17ToV14Converter().ParseV17ToModel(wmoRoot, groups);
                outcome = model.Groups.Count > 0 && model.Groups.Sum(static g => g.Vertices.Count) > 0 ? "ok (geometry)" : "ok (no geometry)";
            }
            catch (Exception ex)
            {
                outcome = $"FAILED {ex.GetType().Name}: {ex.Message}";
            }

            wmoParse[outcome] = wmoParse.GetValueOrDefault(outcome) + 1;
        }

        Console.WriteLine("WMO parse outcomes:");
        foreach ((string outcome, int count) in wmoParse)
            Console.WriteLine($"  [{count}] {outcome}");

        Console.WriteLine("WMO root chunk sequences:");
        foreach ((string sequence, int count) in wmoSequences)
            Console.WriteLine($"  [{count}] {sequence}");

        var m2Outcomes = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (uint doodadId in doodadIds)
        {
            string outcome = TryBuildNativeM2(Read, doodadId, $"fdid_{doodadId}.m2", out _);
            m2Outcomes[outcome] = m2Outcomes.GetValueOrDefault(outcome) + 1;
        }

        Console.WriteLine("Native M2 build outcomes:");
        foreach ((string outcome, int count) in m2Outcomes.OrderByDescending(static kv => kv.Value))
            Console.WriteLine($"  [{count}] {outcome}");

        foreach ((string label, HashSet<uint> ids) in new[] { ("MDID textures", textureIds), ("MDDF models (0x40)", doodadIds), ("MODF WMOs (0x8)", wmoIds) })
        {
            var statuses = ids.GroupBy(id => storage.TryReadFile(id, out _)).ToDictionary(static g => g.Key, static g => g.Count());
            Console.WriteLine($"{label}: {ids.Count} distinct; {string.Join(' ', statuses.Select(static kv => $"{kv.Key}={kv.Value}"))}");
        }
    }

    /// <summary>Spec 239: decodes a DB2 table from CASC through DBCD + WoWDBDefs for the product's own build.</summary>
    private static void RunDb2(string[] args)
    {
        string? install = GetOption(args, "--install");
        string? product = GetOption(args, "--product");
        string? cache = GetOption(args, "--cache");
        string? defs = GetOption(args, "--defs");
        List<string> tables = GetOptions(args, "--table");
        List<string> listfiles = GetOptions(args, "--listfile");
        List<long> findValues = GetOptions(args, "--find").Select(static v => long.Parse(v, System.Globalization.CultureInfo.InvariantCulture)).ToList();
        if (install is null || product is null || cache is null || defs is null || tables.Count == 0 || listfiles.Count == 0)
        {
            Console.WriteLine("  casc db2 --install <dir> --product <p> --cache <dir> --defs <WoWDBDefs/definitions> --listfile <csv> --table <name> [--table ...] [--find <integer> ...]");
            Environment.ExitCode = 1;
            return;
        }

        CommunityListfile listfile = CommunityListfile.Load(listfiles);
        CascStorage storage = CascStorage.OpenLocal(install, product, cache);
        var provider = new CascDbcProvider(storage, listfile);
        foreach (string table in tables)
        {
            try
            {
                DBCD.IDBCDStorage rows = WowViewer.Core.IO.Dbc.DbcTableLoader.Load(provider, defs, storage.Product.Version, table);
                Console.WriteLine($"{table} @ {storage.Product.Version}: {rows.Count} rows; columns: {string.Join(", ", rows.AvailableColumns.Take(12))}{(rows.AvailableColumns.Length > 12 ? ", ..." : "")}");
                if (findValues.Count > 0)
                {
                    // Scan every column (arrays element-wise) for the requested integers; dictionary keys are the real row ids.
                    long maxKey = rows.Keys.Count > 0 ? rows.Keys.Max() : 0;
                    Console.WriteLine($"  row ids {rows.Keys.Min()}..{maxKey}");
                    foreach (long wanted in findValues)
                    {
                        int[] below = rows.Keys.Where(k => k < wanted).OrderByDescending(static k => k).Take(3).OrderBy(static k => k).ToArray();
                        int[] above = rows.Keys.Where(k => k > wanted).OrderBy(static k => k).Take(3).ToArray();
                        Console.WriteLine($"  nearest row ids around {wanted}: [{string.Join(", ", below)}] .. [{string.Join(", ", above)}]{(rows.ContainsKey((int)Math.Min(wanted, int.MaxValue)) ? " (row exists)" : "")}");
                    }
                    foreach (int key in rows.Keys)
                    {
                        DBCD.DBCDRow row = rows[key];
                        foreach (string column in rows.AvailableColumns)
                        {
                            object? value = row[column];
                            IEnumerable<object?> values = value is Array array ? array.Cast<object?>() : [value];
                            int index = 0;
                            foreach (object? element in values)
                            {
                                if (element is IConvertible convertible and not string)
                                {
                                    long number;
                                    try { number = Convert.ToInt64(convertible, System.Globalization.CultureInfo.InvariantCulture); }
                                    catch (Exception) { index++; continue; }
                                    if (findValues.Contains(number))
                                    {
                                        string label = rows.AvailableColumns.Contains("Directory") ? $" Directory={row["Directory"]}" : string.Empty;
                                        string name = rows.AvailableColumns.Contains("MapName_lang") ? $" MapName={row["MapName_lang"]}" : rows.AvailableColumns.Contains("AreaName_lang") ? $" AreaName={row["AreaName_lang"]}" : string.Empty;
                                        Console.WriteLine($"  match {number}: row {key} column {column}{(value is Array ? $"[{index}]" : string.Empty)}{label}{name}");
                                    }
                                }

                                index++;
                            }
                        }
                    }
                }

                if (string.Equals(table, "Map", StringComparison.OrdinalIgnoreCase))
                {
                    foreach (DBCD.DBCDRow row in rows.Values)
                    {
                        string directory = row["Directory"]?.ToString() ?? string.Empty;
                        if (directory.Equals("development", StringComparison.OrdinalIgnoreCase) || directory.Equals("Azeroth", StringComparison.OrdinalIgnoreCase))
                            Console.WriteLine($"  Map row: ID={row["ID"]} Directory={directory} MapName={row["MapName_lang"]}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"{table} @ {storage.Product.Version}: FAILED {ex.GetType().Name}: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Spec 239: runs the viewer's native static M2 build (geometry reader + SFID skin + skin profile
    /// runtime + static render model) on models from CASC. Paths from --path, or every model placed on
    /// a WDT's first N tiles via --wdt-id.
    /// </summary>
    private static void RunM2(string[] args)
    {
        string? install = GetOption(args, "--install");
        string? product = GetOption(args, "--product");
        string? cache = GetOption(args, "--cache");
        List<string> listfiles = GetOptions(args, "--listfile");
        List<string> paths = GetOptions(args, "--path");
        if (install is null || product is null || cache is null || listfiles.Count == 0 || paths.Count == 0)
        {
            Console.WriteLine("  casc m2 --install <dir> --product <p> --cache <dir> --listfile <csv> --path <model.m2> [--path ...]");
            Environment.ExitCode = 1;
            return;
        }

        CommunityListfile listfile = CommunityListfile.Load(listfiles);
        WowViewer.Core.IO.Files.FileDataIdPaths.Resolver = listfile.GetPath;
        CascStorage storage = CascStorage.OpenLocal(install, product, cache);
        byte[]? Read(uint id) => storage.TryReadFile(id, out byte[]? b) == CascReadStatus.Ok ? b : null;

        foreach (string path in paths)
        {
            try
            {
                if (!listfile.TryGetFileDataId(path, out uint id) || Read(id) is not { } model)
                {
                    Console.WriteLine($"{path}: model not readable");
                    continue;
                }

                if (!WowViewer.Core.IO.M2.M2ChunkedFileIds.TryRead(model, out var ids) || ids.SkinFileDataIds.Length == 0 || Read(ids.SkinFileDataIds[0]) is not { } skin)
                {
                    Console.WriteLine($"{path}: no readable SFID skin");
                    continue;
                }

                using var modelStream = new MemoryStream(WowViewer.Core.IO.M2.M2ChunkedFileIds.GetMd20Payload(model), writable: false);
                var geometry = ids.ApplyTextureNames(WowViewer.Core.IO.M2.M2GeometryReader.Read(modelStream, path));
                using var skinStream = new MemoryStream(skin, writable: false);
                var skinDocument = WowViewer.Core.IO.M2.M2SkinReader.Read(skinStream, WowViewer.Core.IO.Files.FileDataIdPaths.Resolve(ids.SkinFileDataIds[0]).Replace('/', '\\'));
                var selection = new WowViewer.Core.M2.M2SkinProfileSelection(0, skinDocument.SourcePath);
                var chosen = new WowViewer.Core.Runtime.M2.M2SkinProfileRuntimeState(geometry.Model, selection, WowViewer.Core.Runtime.M2.M2SkinProfileStage.Chosen, loadedSkin: null, activeSkinProfile: null);
                var initialized = WowViewer.Core.Runtime.M2.M2SkinProfileRuntime.Initialize(WowViewer.Core.Runtime.M2.M2SkinProfileRuntime.Load(chosen, skinDocument));
                var render = WowViewer.Core.Runtime.M2.M2StaticRenderModelBuilder.Build(geometry, initialized);
                Console.WriteLine($"{path}: OK md20v={geometry.Model.Version} vertices={geometry.Vertices.Count} sections={render.Sections.Count} textures=[{string.Join(", ", geometry.Textures.Select(static t => t.Filename ?? $"<replaceable {t.ReplaceableId}>"))}]");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"{path}: FAILED {ex.GetType().Name}: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Load-path timing on real data: listfile load, per-extension list filtering (what
    /// CascDataSource.GetFileList does per call), and reading every WMO root/group and M2/skin placed on
    /// a map, sequentially and with parallel threads.
    /// </summary>
    private static void RunBench(string[] args)
    {
        string? install = GetOption(args, "--install");
        string? product = GetOption(args, "--product");
        string? cache = GetOption(args, "--cache");
        string? wdtIdText = GetOption(args, "--wdt-id");
        List<string> listfiles = GetOptions(args, "--listfile");
        int threads = int.Parse(GetOption(args, "--threads") ?? "8");
        if (install is null || product is null || cache is null || wdtIdText is null || listfiles.Count == 0)
        {
            Console.WriteLine("  casc bench --install <dir> --product <p> --cache <dir> --wdt-id <id> --listfile <csv> [--threads 8]");
            Environment.ExitCode = 1;
            return;
        }

        var sw = Stopwatch.StartNew();
        CommunityListfile listfile = CommunityListfile.Load(listfiles);
        Console.WriteLine($"listfile load: {sw.ElapsedMilliseconds} ms ({listfile.Count} entries)");

        sw.Restart();
        CascStorage storage = CascStorage.OpenLocal(install, product, cache, args.Contains("--cdn-fill", StringComparer.OrdinalIgnoreCase));
        Console.WriteLine($"storage open: {sw.ElapsedMilliseconds} ms (cdn fill {storage.AllowsCdnFill})");

        sw.Restart();
        var present = listfile.Entries.Where(e => storage.FileExists(e.Key)).Select(static e => e.Value).ToList();
        Console.WriteLine($"GetFileList base build (FileExists over listfile): {sw.ElapsedMilliseconds} ms ({present.Count} present)");
        sw.Restart();
        int blp = present.Count(static p => p.EndsWith(".blp", StringComparison.OrdinalIgnoreCase));
        Console.WriteLine($"one GetFileList(\".blp\") filter pass: {sw.ElapsedMilliseconds} ms ({blp} matches)");

        byte[]? Read(uint id) => id != 0 && storage.TryReadFile(id, out byte[]? b) == CascReadStatus.Ok ? b : null;
        byte[] wdt = Read(uint.Parse(wdtIdText))!;
        var maid = TopChunks(wdt).First(static c => c.Id == "MAID");
        var modelIds = new HashSet<uint>();
        var wmoIds = new HashSet<uint>();
        for (int slot = 0; slot < maid.Size / 32; slot++)
        {
            if (Read(BitConverter.ToUInt32(wdt, maid.Offset + slot * 32 + 4)) is not { } obj0)
                continue;
            foreach (var c in TopChunks(obj0))
            {
                if (c.Id == "MDDF")
                    for (int p = c.Offset; p + 36 <= c.Offset + c.Size; p += 36)
                        modelIds.Add(BitConverter.ToUInt32(obj0, p));
                else if (c.Id == "MODF")
                    for (int p = c.Offset; p + 64 <= c.Offset + c.Size; p += 64)
                        wmoIds.Add(BitConverter.ToUInt32(obj0, p));
            }
        }

        // Expand to the full file set a load touches: WMO roots + GFID groups, M2s + SFID skins.
        var files = new List<uint>();
        foreach (uint wmoId in wmoIds)
        {
            files.Add(wmoId);
            if (Read(wmoId) is { } root)
                files.AddRange(WowViewer.Core.IO.Converters.WmoV17ToV14Converter.ReadGroupFileDataIds(root));
        }
        int wmoFileCount = files.Count;
        foreach (uint modelId in modelIds)
        {
            files.Add(modelId);
            if (Read(modelId) is { } model && WowViewer.Core.IO.M2.M2ChunkedFileIds.TryRead(model, out var ids))
                files.AddRange(ids.SkinFileDataIds.Take(1));
        }
        Console.WriteLine($"file set: {wmoIds.Count} WMO roots + groups = {wmoFileCount} files, {modelIds.Count} M2 + skins; total {files.Count} (all already read once above: warm)");

        sw.Restart();
        long bytes = 0;
        foreach (uint id in files)
            bytes += Read(id)?.Length ?? 0;
        Console.WriteLine($"sequential read (warm OS cache): {sw.ElapsedMilliseconds} ms, {bytes / 1_048_576.0:F1} MB, {files.Count / Math.Max(0.001, sw.Elapsed.TotalSeconds):F0} files/s");

        sw.Restart();
        long parallelBytes = 0;
        Parallel.ForEach(files, new ParallelOptions { MaxDegreeOfParallelism = threads }, id => Interlocked.Add(ref parallelBytes, Read(id)?.Length ?? 0));
        Console.WriteLine($"parallel read x{threads} through CascStorage (single read lock): {sw.ElapsedMilliseconds} ms");

        // Concurrency proof: every file read in parallel without the lock must hash identically to a
        // sequential locked read. Includes CDN-cached textures when --cdn-fill is set.
        var sequentialHashes = new Dictionary<uint, string>();
        foreach (uint id in files.Distinct())
            sequentialHashes[id] = Read(id) is { } b ? Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(b)) : "-";
        storage.SerializeReads = false;
        var parallelHashes = new System.Collections.Concurrent.ConcurrentDictionary<uint, string>();
        sw.Restart();
        Parallel.ForEach(files.Distinct().Concat(files.Distinct()), new ParallelOptions { MaxDegreeOfParallelism = threads }, id =>
        {
            string hash = Read(id) is { } b ? Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(b)) : "-";
            parallelHashes.AddOrUpdate(id, hash, (_, existing) => existing == hash ? hash : "MISMATCH");
        });
        int mismatches = sequentialHashes.Count(kv => !parallelHashes.TryGetValue(kv.Key, out string? h) || h != kv.Value);
        Console.WriteLine($"lock-free parallel x{threads} (each file twice): {sw.ElapsedMilliseconds} ms; hash mismatches vs sequential: {mismatches} of {sequentialHashes.Count}");

        sw.Restart();
        var wmoGroups = files.Take(wmoFileCount).Select(Read).Where(static b => b is not null).ToList();
        Console.WriteLine($"WMO bytes only ({wmoFileCount} files): {sw.ElapsedMilliseconds} ms, {wmoGroups.Sum(static b => b!.Length) / 1_048_576.0:F1} MB");

        // WMO parse + convert (what WorldAssetManager.LoadWmoDataModel does), and material texture decode
        // the way WmoRenderer/TerrainRenderer do (full mip 0 to RGBA via SereniaBLPLib).
        WowViewer.Core.IO.Files.FileDataIdPaths.Resolver = listfile.GetPath;
        var textureNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        sw.Restart();
        foreach (uint wmoId in wmoIds)
        {
            if (Read(wmoId) is not { } root)
                continue;
            var groups = WowViewer.Core.IO.Converters.WmoV17ToV14Converter.ReadGroupFileDataIds(root).Select(Read).TakeWhile(static g => g is not null).Select(static g => g!).ToList();
            var model = new WowViewer.Core.IO.Converters.WmoV17ToV14Converter().ParseV17ToModel(root, groups);
            foreach (var material in model.Materials)
            {
                foreach (string? name in new[] { material.Texture1Name, material.Texture2Name })
                {
                    if (!string.IsNullOrEmpty(name))
                        textureNames.Add(name);
                }
            }
        }
        Console.WriteLine($"WMO read+parse ({wmoIds.Count} WMOs incl. groups): {sw.ElapsedMilliseconds} ms; {textureNames.Count} distinct material textures");

        // Cold, parallel texture fetch (what CascDataSource's prefetch pool does). Only meaningful with an
        // empty --cache and --cdn-fill; run before the sequential loop so that loop measures warm reads.
        sw.Restart();
        Parallel.ForEach(textureNames, new ParallelOptions { MaxDegreeOfParallelism = threads }, name =>
        {
            if (WowViewer.Core.IO.Files.FileDataIdPaths.TryParse(name, out uint tid))
                Read(tid);
            else if (listfile.TryGetFileDataId(name, out uint lid))
                Read(lid);
        });
        Console.WriteLine($"WMO textures parallel fetch x{threads} (first pass; cold if --cache is empty): {sw.ElapsedMilliseconds} ms");

        long decodePixels = 0;
        int decoded = 0, missing = 0;
        var sizes = new SortedDictionary<string, int>(StringComparer.Ordinal);
        long readMs = 0, decodeMs = 0;
        foreach (string name in textureNames)
        {
            var t = Stopwatch.StartNew();
            byte[]? blpBytes = WowViewer.Core.IO.Files.FileDataIdPaths.TryParse(name, out uint tid) ? Read(tid)
                : listfile.TryGetFileDataId(name, out uint lid) ? Read(lid) : null;
            readMs += t.ElapsedMilliseconds;
            if (blpBytes is null)
            {
                missing++;
                continue;
            }

            t.Restart();
            using var stream = new MemoryStream(blpBytes);
            using var blpFile = new SereniaBLPLib.BlpFile(stream);
            using var image = blpFile.GetImage(0);
            decodeMs += t.ElapsedMilliseconds;
            decodePixels += (long)image.Width * image.Height;
            decoded++;
            string key = $"{image.Width}x{image.Height}";
            sizes[key] = sizes.GetValueOrDefault(key) + 1;
        }
        Console.WriteLine($"WMO textures: decoded {decoded}, not readable {missing}; read {readMs} ms, decode mip0->RGBA {decodeMs} ms, {decodePixels * 4 / 1_048_576.0:F0} MB RGBA; sizes {string.Join(' ', sizes.Select(static kv => $"{kv.Key}:{kv.Value}"))}");
    }

    /// <summary>
    /// Writes absolute outer-vertex heights (MCNK position.z + MCVT) of a rectangle of ADT tiles from a WDT's
    /// MAID as a CSV (tileX,tileY,row,col,height) for offline comparison with DAT v26 grids.
    /// </summary>
    private static void RunAdtHeights(string[] args)
    {
        string? install = GetOption(args, "--install");
        string? product = GetOption(args, "--product");
        string? cache = GetOption(args, "--cache");
        string? wdtIdText = GetOption(args, "--wdt-id");
        string? output = GetOption(args, "--out");
        int x0 = int.Parse(GetOption(args, "--x0") ?? "0"), x1 = int.Parse(GetOption(args, "--x1") ?? "63");
        int y0 = int.Parse(GetOption(args, "--y0") ?? "0"), y1 = int.Parse(GetOption(args, "--y1") ?? "63");
        if (install is null || product is null || cache is null || wdtIdText is null || output is null)
        {
            Console.WriteLine("  casc adt-heights --install <dir> --product <p> --cache <dir> --wdt-id <id> --x0 --x1 --y0 --y1 --out <csv>");
            Environment.ExitCode = 1;
            return;
        }

        CascStorage storage = CascStorage.OpenLocal(install, product, cache);
        byte[]? Read(uint id) => id != 0 && storage.TryReadFile(id, out byte[]? b) == CascReadStatus.Ok ? b : null;
        byte[] wdt = Read(uint.Parse(wdtIdText)) ?? throw new InvalidOperationException("WDT not readable");
        var maid = TopChunks(wdt).First(static c => c.Id == "MAID");

        using var writer = new StreamWriter(output);
        writer.WriteLine("tileX,tileY,row,col,height");
        int tiles = 0;
        for (int y = y0; y <= y1; y++)
        {
            for (int x = x0; x <= x1; x++)
            {
                uint rootId = BitConverter.ToUInt32(wdt, maid.Offset + (y * 64 + x) * 32);
                if (Read(rootId) is not { } root)
                    continue;

                tiles++;
                int chunkIndex = 0;
                foreach (var mcnk in TopChunks(root).Where(static c => c.Id == "MCNK"))
                {
                    int chunkX = BitConverter.ToInt32(root, mcnk.Offset + 4);
                    int chunkY = BitConverter.ToInt32(root, mcnk.Offset + 8);
                    float baseZ = BitConverter.ToSingle(root, mcnk.Offset + 0x70);
                    var mcvt = TopChunks(root, mcnk.Offset + 0x80, mcnk.Offset + mcnk.Size).FirstOrDefault(static c => c.Id == "MCVT");
                    if (mcvt.Id is null)
                        continue;

                    for (int outerRow = 0; outerRow < 9; outerRow++)
                    {
                        for (int c = 0; c < 9; c++)
                        {
                            float h = baseZ + BitConverter.ToSingle(root, mcvt.Offset + (outerRow * 17 + c) * 4);
                            writer.WriteLine($"{x},{y},{chunkY * 8 + outerRow},{chunkX * 8 + c},{h:0.###}");
                        }
                    }

                    chunkIndex++;
                }
            }
        }

        Console.WriteLine($"wrote {tiles} tiles to {output}");
    }

    /// <summary>Same steps as WowViewerM2RuntimeBridge.BuildStaticRenderModel; returns "ok" or a failure category.</summary>
    private static string TryBuildNativeM2(Func<uint, byte[]?> read, uint modelId, string path, out string detail)
    {
        detail = string.Empty;
        try
        {
            if (read(modelId) is not { } model)
                return "model not readable";
            if (!WowViewer.Core.IO.M2.M2ChunkedFileIds.TryRead(model, out var ids))
                return "not MD21";
            if (ids.SkinFileDataIds.Length == 0)
                return "no SFID";
            if (read(ids.SkinFileDataIds[0]) is not { } skin)
                return "SFID skin not readable";

            using var modelStream = new MemoryStream(WowViewer.Core.IO.M2.M2ChunkedFileIds.GetMd20Payload(model), writable: false);
            var geometry = ids.ApplyTextureNames(WowViewer.Core.IO.M2.M2GeometryReader.Read(modelStream, path));
            using var skinStream = new MemoryStream(skin, writable: false);
            var skinDocument = WowViewer.Core.IO.M2.M2SkinReader.Read(skinStream, $"fdid_{ids.SkinFileDataIds[0]}.skin");
            var selection = new WowViewer.Core.M2.M2SkinProfileSelection(0, skinDocument.SourcePath);
            var chosen = new WowViewer.Core.Runtime.M2.M2SkinProfileRuntimeState(geometry.Model, selection, WowViewer.Core.Runtime.M2.M2SkinProfileStage.Chosen, loadedSkin: null, activeSkinProfile: null);
            var render = WowViewer.Core.Runtime.M2.M2StaticRenderModelBuilder.Build(geometry, WowViewer.Core.Runtime.M2.M2SkinProfileRuntime.Initialize(WowViewer.Core.Runtime.M2.M2SkinProfileRuntime.Load(chosen, skinDocument)));
            detail = $"v{geometry.Model.Version} vertices={geometry.Vertices.Count} sections={render.Sections.Count}";
            return render.Sections.Count > 0 ? $"ok (md20 v{geometry.Model.Version})" : $"ok but 0 sections (v{geometry.Model.Version})";
        }
        catch (Exception ex)
        {
            string message = ex.Message.Length > 110 ? ex.Message[..110] : ex.Message;
            return $"FAILED {ex.GetType().Name}: {System.Text.RegularExpressions.Regex.Replace(message, @"'[^']*'", "'…'")}";
        }
    }

    private sealed class CascDbcProvider(CascStorage storage, CommunityListfile listfile) : DBCD.Providers.IDBCProvider
    {
        public Stream StreamForTableName(string tableName, string build)
        {
            foreach (string path in WowViewer.Core.IO.Files.DbClientFileReader.EnumerateTablePaths(tableName))
            {
                if (listfile.TryGetFileDataId(path, out uint id) && storage.TryReadFile(id, out byte[]? data) == CascReadStatus.Ok && data is not null)
                    return new MemoryStream(data);
            }

            throw new FileNotFoundException($"{tableName} not readable from {storage.Product.Product}");
        }
    }

    private static IEnumerable<(string Id, int Offset, int Size)> TopChunks(byte[] data, int start = 0, int end = -1)
    {
        end = end < 0 ? data.Length : end;
        int position = start;
        while (position + 8 <= end)
        {
            string id = new(System.Text.Encoding.ASCII.GetString(data, position, 4).Reverse().ToArray());
            int size = BitConverter.ToInt32(data, position + 4);
            if (size < 0 || position + 8L + size > end)
                yield break;
            yield return (id, position + 8, size);
            position += 8 + size;
        }
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
