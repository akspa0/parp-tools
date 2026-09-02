using System.Buffers.Binary;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Maps;

/// <summary>
/// Histograms the second uint16 of every MH2O SMLiquidInstance across a client's ADTs.
/// </summary>
/// <remarks>
/// In Cataclysm and later that field is <c>liquid_object_or_lvf</c>: values below 42 are a liquid
/// vertex format (0-3), values 42 and above are a <c>LiquidObject.dbc</c> id and the real vertex
/// format has to be resolved through it. <see cref="WowViewer.Core.IO.Maps.AdtLiquidReader"/> casts
/// the field straight to a vertex-format enum and switches on it with no default, so an id lands on
/// no case, the height array stays null, and the mesh falls back to a flat plane at the layer's
/// minimum height.
/// <para>
/// This command exists to establish whether that actually happens in the operator's data before
/// anything is changed on the strength of it. The reported histogram is the evidence; a run that
/// shows only values 0-3 refutes the hypothesis outright.
/// </para>
/// </remarks>
internal static class AdtLiquidFormatSupport
{
    private const int ChunkHeaderSize = 8;
    private const int Mh2oChunkHeaderCount = 256;
    private const int Mh2oChunkHeaderSize = 12;
    private const int LayerSize = 24;

    /// <summary>Values at or above this are a LiquidObject.dbc id, not a vertex format.</summary>
    private const int LiquidObjectThreshold = 42;

    public static void Run(string[] args)
    {
        string? clientRoot = GetOption(args, "--client");
        if (string.IsNullOrWhiteSpace(clientRoot))
        {
            Console.Error.WriteLine("Usage: adt liquid-formats --client <client-dir> [--map <name>] [--limit <n>] [--build <version>] [--defs <WoWDBDefs/definitions>]");
            Environment.ExitCode = 1;
            return;
        }

        string? mapFilter = GetOption(args, "--map");
        string? buildVersion = GetOption(args, "--build");
        string? definitionsDir = GetOption(args, "--defs") ?? ResolveDefinitionsDirectory();
        int limit = int.TryParse(GetOption(args, "--limit"), out int parsedLimit) ? parsedLimit : 200;

        using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
        ArchiveCatalogBootstrapResult bootstrap = ArchiveCatalogBootstrapper.Bootstrap(
            archiveCatalog, [clientRoot], new ArchiveCatalogBootstrapOptions());

        IEnumerable<string> candidates = bootstrap.AllFiles
            .Where(static path => path.EndsWith(".adt", StringComparison.OrdinalIgnoreCase));

        if (!string.IsNullOrWhiteSpace(mapFilter))
            candidates = candidates.Where(path => path.Contains(mapFilter, StringComparison.OrdinalIgnoreCase));

        // Split ADTs carry no MH2O; only the root tile does. Filtering here keeps the scanned count
        // meaningful rather than counting three-quarters misses.
        candidates = candidates.Where(static path =>
            !path.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase)
            && !path.EndsWith("_obj1.adt", StringComparison.OrdinalIgnoreCase)
            && !path.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase)
            && !path.EndsWith("_tex1.adt", StringComparison.OrdinalIgnoreCase));

        var formatCounts = new SortedDictionary<int, int>();
        var typeByFormat = new SortedDictionary<int, SortedSet<int>>();
        var detail = new SortedDictionary<int, FormatDetail>();
        int scanned = 0;
        int withWater = 0;
        int layers = 0;

        foreach (string path in candidates.OrderBy(static p => p, StringComparer.OrdinalIgnoreCase))
        {
            if (scanned >= limit)
                break;

            byte[]? bytes = archiveCatalog.ReadFile(path);
            if (bytes is null)
                continue;

            scanned++;
            if (!TryFindChunk(bytes, "MH2O", out int payloadOffset, out int payloadLength))
                continue;

            withWater++;
            layers += ScanMh2o(bytes, payloadOffset, payloadLength, formatCounts, typeByFormat, detail);
        }

        Console.WriteLine("WowViewer.Tool.Inspect MH2O liquid vertex-format histogram");
        Console.WriteLine($"client={clientRoot} map={mapFilter ?? "(all)"}");
        Console.WriteLine($"root ADTs scanned={scanned} withMh2o={withWater} liquidLayers={layers}");
        Console.WriteLine();
        Console.WriteLine("liquid_object_or_lvf  count      meaning");

        int objectIdLayers = 0;
        foreach ((int value, int count) in formatCounts)
        {
            string meaning = value >= LiquidObjectThreshold
                ? $"LiquidObject.dbc id -> NOT a vertex format (liquidTypes: {string.Join(',', typeByFormat[value])})"
                : value switch
                {
                    0 => "LVF 0 height+depth",
                    1 => "LVF 1 height+uv",
                    2 => "LVF 2 depth only",
                    3 => "LVF 3 height+uv+depth",
                    _ => "unknown, below the LiquidObject threshold",
                };

            if (value >= LiquidObjectThreshold)
                objectIdLayers += count;

            Console.WriteLine($"{value,20}  {count,9}  {meaning}");
        }

        Console.WriteLine();
        Console.WriteLine("value   layers  withVertexData  sloped  maxSpread  sizes(w x h)");
        foreach ((int value, FormatDetail d) in detail)
        {
            string sizes = string.Join(" ", d.Sizes.OrderBy(static s => s.Width).ThenBy(static s => s.Height)
                .Select(static s => $"{s.Width}x{s.Height}").Distinct().Take(6));
            Console.WriteLine($"{value,5}  {d.Layers,6}  {d.WithVertexData,14}  {d.Sloped,6}  {d.MaxSpread,9:0.00}  {sizes}");
            Console.WriteLine($"        vertexBlocks: plausibleHeights={d.HeightsPlausible} implausible={d.HeightsImplausible}"
                + $" varying={d.HeightsVary} maxSpread={d.MaxVertexSpread:0.00} disagreeWithHeader={d.HeightsDisagreeWithHeader}");
        }

        Console.WriteLine();
        if (layers == 0)
        {
            Console.WriteLine("VERDICT: no liquid layers found. Nothing measured; widen --map or --limit.");
            return;
        }

        double share = 100.0 * objectIdLayers / layers;
        Console.WriteLine(objectIdLayers > 0
            ? $"VERDICT: {objectIdLayers} of {layers} layers ({share:0.0}%) carry a LiquidObject id. "
              + "AdtLiquidReader's switch has no case for these, so their heights parse as null and "
              + "the mesh falls back to a flat plane. Hypothesis CONFIRMED."
            : "VERDICT: every layer uses a vertex format in 0-3. The LiquidObject hypothesis is "
              + "REFUTED for this data; the flat-plane cause is elsewhere.");

        if (!string.IsNullOrWhiteSpace(buildVersion))
            ReportChainResolution(archiveCatalog, definitionsDir, buildVersion!, detail);
        else
        {
            Console.WriteLine();
            Console.WriteLine("Pass --build <version> to resolve each value through LiquidObject -> LiquidType -> LiquidMaterial.");
        }
    }

    /// <summary>
    /// Spec 205 Phase 1 gate: resolve every observed value through the DBC chain and cross-check the
    /// answer against the vertex-block plausibility probe.
    /// </summary>
    /// <remarks>
    /// The probe is an instrument, never the decoder (research R6: it misread 18 of 6,194 ocean
    /// layers). Its only job here is disagreement detection: where the DBC says "heights" and the
    /// block reads as garbage, or the DBC says "depth only" and the block reads as clean varying
    /// floats, one of the two is wrong and this report says which layers to look at.
    /// </remarks>
    private static void ReportChainResolution(
        IArchiveCatalog archiveCatalog,
        string? definitionsDirectory,
        string buildVersion,
        SortedDictionary<int, FormatDetail> detail)
    {
        Console.WriteLine();
        Console.WriteLine("=== DBC chain resolution (spec 205 Phase 1 gate) ===");

        if (string.IsNullOrWhiteSpace(definitionsDirectory) || !Directory.Exists(definitionsDirectory))
        {
            Console.WriteLine($"WoWDBDefs definitions not found (looked for '{definitionsDirectory ?? "<null>"}'). Pass --defs.");
            return;
        }

        Console.WriteLine($"build={buildVersion} defs={definitionsDirectory}");

        ArchiveReaderDbcProvider provider = new(archiveCatalog);
        LiquidVertexFormatChain chain = LiquidVertexFormatChain.Load(
            provider, definitionsDirectory, buildVersion, out IReadOnlyList<string> diagnostics);

        foreach (string line in diagnostics)
            Console.WriteLine($"  {line}");

        DumpTables(provider, definitionsDirectory, buildVersion, detail);

        Console.WriteLine();
        Console.WriteLine("value  layers  ->liquidType  ->material  ->LVF                 source        probe agreement");

        int resolved = 0;
        int unresolved = 0;
        foreach ((int value, FormatDetail d) in detail)
        {
            if (value is < 0 or > ushort.MaxValue)
                continue;

            LiquidVertexFormatResolution r = chain.Resolve((ushort)value);
            if (r.Resolved)
                resolved += d.Layers;
            else
                unresolved += d.Layers;

            string lvf = r.Resolved ? $"{(int)r.Format} {r.Format}" : "-";
            string source = r.Resolved ? r.Source.ToString() : "UNRESOLVED";

            // Does the chain's answer match what the bytes look like?
            bool chainSaysHeights = r.Resolved && r.Format != AdtLiquidVertexFormat.DepthOnly;
            bool bytesLookLikeHeights = d.HeightsPlausible > d.HeightsImplausible;
            string agreement = !r.Resolved
                ? "n/a"
                : chainSaysHeights == bytesLookLikeHeights
                    ? "agree"
                    : $"DISAGREE (chain={(chainSaysHeights ? "heights" : "depth")}, bytes={(bytesLookLikeHeights ? "heights" : "depth")})";

            Console.WriteLine($"{value,5}  {d.Layers,6}  {(r.LiquidTypeId == 0 ? "-" : r.LiquidTypeId.ToString()),12}  {(r.MaterialId == 0 ? "-" : r.MaterialId.ToString()),9}  {lvf,-20} {source,-13} {agreement}");

            if (!r.Resolved)
                Console.WriteLine($"       reason: {r.FailureReason}");
        }

        Console.WriteLine();
        int total = resolved + unresolved;
        Console.WriteLine(total == 0
            ? "GATE: nothing to resolve."
            : unresolved == 0
                ? $"GATE PASS: all {total} layers resolve through the DBC chain."
                : $"GATE: {resolved} of {total} layers resolve ({100.0 * resolved / total:0.0}%); {unresolved} do not and would keep today's flat fallback.");
    }

    /// <summary>
    /// Print the raw contents of each link so a wrong answer can be blamed on the right thing.
    /// Without this, a table that loaded with the wrong layout is indistinguishable from a client
    /// that genuinely does not use the documented chain.
    /// </summary>
    private static void DumpTables(
        ArchiveReaderDbcProvider provider,
        string definitionsDirectory,
        string buildVersion,
        SortedDictionary<int, FormatDetail> detail)
    {
        DumpOne(provider, definitionsDirectory, buildVersion, "LiquidObject", detail.Keys);
        DumpOne(provider, definitionsDirectory, buildVersion, "LiquidMaterial", null);
        DumpOne(provider, definitionsDirectory, buildVersion, "LiquidType", null);
    }

    private static void DumpOne(
        ArchiveReaderDbcProvider provider,
        string definitionsDirectory,
        string buildVersion,
        string tableName,
        IEnumerable<int>? idsOfInterest)
    {
        Console.WriteLine();
        try
        {
            DBCD.Providers.FilesystemDBDProvider dbd = new(definitionsDirectory);
            DBCD.DBCD dbcd = new(provider, dbd);
            DBCD.IDBCDStorage storage;
            try { storage = dbcd.Load(tableName, buildVersion, DBCD.Locale.EnUS); }
            catch { storage = dbcd.Load(tableName, buildVersion, DBCD.Locale.None); }

            // Key by the ID COLUMN, not DBCDRow.ID: for these WDB2 tables the latter is positional.
            string? idColumn = null;
            foreach (string candidate in new[] { "ID", "Id" })
            {
                foreach (string column in storage.AvailableColumns)
                {
                    if (string.Equals(column, candidate, StringComparison.OrdinalIgnoreCase))
                    {
                        idColumn = column;
                        break;
                    }
                }

                if (idColumn is not null)
                    break;
            }

            Dictionary<int, DBCD.DBCDRow> byRealId = [];
            foreach (DBCD.DBCDRow row in storage.Values)
            {
                int realId = row.ID;
                if (idColumn is not null)
                {
                    try { realId = Convert.ToInt32(row[idColumn]); }
                    catch { realId = row.ID; }
                }

                byRealId[realId] = row;
            }

            int[] ids = byRealId.Keys.OrderBy(static k => k).ToArray();
            Console.WriteLine($"--- {tableName}: {ids.Length} rows, REAL id range {(ids.Length == 0 ? "-" : ids[0] + ".." + ids[^1])} (idColumn '{idColumn ?? "<none>"}'), positional keys {storage.Keys.Count}");
            Console.WriteLine($"    columns: {string.Join(", ", storage.AvailableColumns)}");

            if (ids.Length <= 12)
            {
                foreach (int id in ids)
                    Console.WriteLine($"    [{id}] {RowToString(byRealId[id], storage.AvailableColumns)}");
                return;
            }

            Console.WriteLine($"    lowest ids: {string.Join(", ", ids.Take(12))}");
            foreach (int id in (idsOfInterest ?? []).Distinct().OrderBy(static k => k))
            {
                Console.WriteLine(byRealId.TryGetValue(id, out DBCD.DBCDRow? found)
                    ? $"    [{id}] {RowToString(found, storage.AvailableColumns)}"
                    : $"    [{id}] ABSENT from {tableName}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"--- {tableName}: FAILED: {ex.Message}");
        }
    }

    private static string RowToString(DBCD.DBCDRow row, string[] columns)
    {
        List<string> parts = [];
        foreach (string column in columns.Take(10))
        {
            try { parts.Add($"{column}={row[column]}"); }
            catch { parts.Add($"{column}=?"); }
        }

        return string.Join(" ", parts);
    }

    /// <summary>Walk up from the binary looking for the vendored WoWDBDefs definitions.</summary>
    private static string? ResolveDefinitionsDirectory()
    {
        DirectoryInfo? dir = new(AppDomain.CurrentDomain.BaseDirectory);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, "libs", "wowdev", "WoWDBDefs", "definitions");
            if (Directory.Exists(candidate))
                return candidate;

            dir = dir.Parent;
        }

        return null;
    }

    private static int ScanMh2o(
        byte[] bytes,
        int payloadOffset,
        int payloadLength,
        SortedDictionary<int, int> formatCounts,
        SortedDictionary<int, SortedSet<int>> typeByFormat,
        SortedDictionary<int, FormatDetail> detail)
    {
        int layers = 0;
        ReadOnlySpan<byte> payload = bytes.AsSpan(payloadOffset, payloadLength);

        for (int chunkIndex = 0; chunkIndex < Mh2oChunkHeaderCount; chunkIndex++)
        {
            int headerOffset = chunkIndex * Mh2oChunkHeaderSize;
            if (headerOffset + Mh2oChunkHeaderSize > payload.Length)
                break;

            uint offsetInstances = BinaryPrimitives.ReadUInt32LittleEndian(payload[headerOffset..]);
            uint layerCount = BinaryPrimitives.ReadUInt32LittleEndian(payload[(headerOffset + 4)..]);
            if (layerCount == 0 || offsetInstances == 0)
                continue;

            for (int layer = 0; layer < layerCount; layer++)
            {
                int layerOffset = (int)offsetInstances + (layer * LayerSize);
                if (layerOffset < 0 || layerOffset + LayerSize > payload.Length)
                    break;

                int liquidType = BinaryPrimitives.ReadUInt16LittleEndian(payload[layerOffset..]);
                int format = BinaryPrimitives.ReadUInt16LittleEndian(payload[(layerOffset + 2)..]);

                // Geometry, so the fix can be written from evidence rather than from the wiki.
                // A layer whose min and max height differ is a SLOPED surface: flattening it to
                // minHeight is visibly wrong and is what produces the chunk-boundary steps.
                float minHeight = BitConverter.ToSingle(payload[(layerOffset + 4)..]);
                float maxHeight = BitConverter.ToSingle(payload[(layerOffset + 8)..]);
                int width = payload[layerOffset + 14];
                int height = payload[layerOffset + 15];
                uint vertexDataOffset = BinaryPrimitives.ReadUInt32LittleEndian(payload[(layerOffset + 20)..]);

                if (!detail.TryGetValue(format, out FormatDetail? d))
                {
                    d = new FormatDetail();
                    detail[format] = d;
                }

                d.Layers++;
                if (vertexDataOffset != 0)
                    d.WithVertexData++;
                if (maxHeight - minHeight > 0.001f)
                {
                    d.Sloped++;
                    d.MaxSpread = Math.Max(d.MaxSpread, maxHeight - minHeight);
                }

                d.Sizes.Add((width, height));

                // Read the vertex block as if it were an LVF-0/1/3 heightmap. If those floats vary
                // and sit in a plausible world-Z range, they are heights the current reader is
                // discarding — which is the whole question. If they are garbage, the layer is
                // genuinely depth-only and flat is correct.
                if (vertexDataOffset != 0)
                {
                    int vertexCount = (width + 1) * (height + 1);
                    int vo = (int)vertexDataOffset;
                    if (vo >= 0 && vo + (vertexCount * 4) <= payload.Length)
                    {
                        float lo = float.MaxValue;
                        float hi = float.MinValue;
                        bool plausible = true;
                        for (int v = 0; v < vertexCount; v++)
                        {
                            float value = BitConverter.ToSingle(payload[(vo + (v * 4))..]);
                            if (float.IsNaN(value) || float.IsInfinity(value) || Math.Abs(value) > 20000f)
                            {
                                plausible = false;
                                break;
                            }

                            lo = Math.Min(lo, value);
                            hi = Math.Max(hi, value);
                        }

                        if (plausible)
                        {
                            d.HeightsPlausible++;
                            float spread = hi - lo;
                            if (spread > 0.01f)
                            {
                                d.HeightsVary++;
                                d.MaxVertexSpread = Math.Max(d.MaxVertexSpread, spread);
                            }

                            // Does the vertex block agree with the header's declared level?
                            if (Math.Abs(lo - minHeight) > 0.5f)
                                d.HeightsDisagreeWithHeader++;
                        }
                        else
                        {
                            d.HeightsImplausible++;
                        }
                    }
                }

                formatCounts[format] = formatCounts.TryGetValue(format, out int existing) ? existing + 1 : 1;
                if (!typeByFormat.TryGetValue(format, out SortedSet<int>? types))
                {
                    types = [];
                    typeByFormat[format] = types;
                }

                types.Add(liquidType);
                layers++;
            }
        }

        return layers;
    }

    /// <summary>Walks the top-level chunk list. ADT chunk sizes are exact; no padding is applied.</summary>
    private static bool TryFindChunk(byte[] bytes, string magic, out int payloadOffset, out int payloadLength)
    {
        payloadOffset = 0;
        payloadLength = 0;

        // Chunk magics are stored reversed on disk.
        Span<char> reversed = stackalloc char[4];
        for (int i = 0; i < 4; i++)
            reversed[i] = magic[3 - i];

        int offset = 0;
        while (offset + ChunkHeaderSize <= bytes.Length)
        {
            bool match = true;
            for (int i = 0; i < 4; i++)
            {
                if (bytes[offset + i] != (byte)reversed[i])
                {
                    match = false;
                    break;
                }
            }

            uint size = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offset + 4, 4));
            if (size > int.MaxValue || offset + ChunkHeaderSize + (long)size > bytes.Length)
                return false;

            if (match)
            {
                payloadOffset = offset + ChunkHeaderSize;
                payloadLength = (int)size;
                return payloadLength > 0;
            }

            offset += ChunkHeaderSize + (int)size;
        }

        return false;
    }

    private sealed class FormatDetail
    {
        public int Layers;
        public int WithVertexData;

        /// <summary>Layers whose max height exceeds their min height — a surface with actual slope.</summary>
        public int Sloped;
        public float MaxSpread;

        /// <summary>Vertex blocks that read as finite, in-range floats — i.e. look like heights.</summary>
        public int HeightsPlausible;
        public int HeightsImplausible;

        /// <summary>Vertex blocks whose values actually differ across the 9x9 grid: a sloped surface.</summary>
        public int HeightsVary;
        public float MaxVertexSpread;

        /// <summary>Blocks whose lowest vertex differs from the header's minHeight by more than 0.5.</summary>
        public int HeightsDisagreeWithHeader;
        public HashSet<(int Width, int Height)> Sizes { get; } = [];
    }

    private static string? GetOption(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
                return args[i + 1];
        }

        return null;
    }
}
