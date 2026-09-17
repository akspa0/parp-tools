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
            case "map-survey":
                RunMapSurvey(tail);
                break;
            case "db2":
                RunDb2(tail);
                break;
            case "m2":
                RunM2(tail);
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
        if (install is null || product is null || cache is null || defs is null || tables.Count == 0 || listfiles.Count == 0)
        {
            Console.WriteLine("  casc db2 --install <dir> --product <p> --cache <dir> --defs <WoWDBDefs/definitions> --listfile <csv> --table <name> [--table ...]");
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
