using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using GillijimProject.WowFiles.Alpha;
using GillijimProject.WowFiles.LichKing;

#if USE_DBCD
using DBCD;
using DBCD.Providers;
#endif

namespace AlphaWdtAnalyzer.Core.Export;

public static class AdtExportPipeline
{
    public sealed class Options
    {
        public string? SingleWdtPath { get; init; }
        public string? InputRoot { get; init; }
        public string? CommunityListfilePath { get; init; }
        public string? LkListfilePath { get; init; }
        public required string ExportDir { get; init; }
        public required string FallbackTileset { get; init; }
        public required string FallbackNonTilesetBlp { get; init; }
        public required string FallbackWmo { get; init; }
        public required string FallbackM2 { get; init; }
        public bool ConvertToMh2o { get; init; } = true;
        public bool AssetFuzzy { get; init; } = true;
        public string? AreaAlphaPath { get; init; }
        public string? AreaLkPath { get; init; }
        public string? DbdDefinitionsDir { get; init; }
        public string? DbdBuild { get; init; }
    }

    public static void ExportSingle(Options opts)
    {
        if (string.IsNullOrWhiteSpace(opts.SingleWdtPath)) throw new ArgumentException("SingleWdtPath required", nameof(opts.SingleWdtPath));
        Directory.CreateDirectory(opts.ExportDir);

        var resolver = MultiListfileResolver.FromFiles(opts.LkListfilePath, opts.CommunityListfilePath);
        var fixup = new AssetFixupPolicy(
            resolver,
            opts.FallbackTileset,
            opts.FallbackNonTilesetBlp,
            opts.FallbackWmo,
            opts.FallbackM2,
            opts.AssetFuzzy);

        var presentTiles = new HashSet<(int tx, int ty)>();

        var wdt = new WdtAlphaScanner(opts.SingleWdtPath!);
        var adtScanner = new AdtScanner();
        var result = adtScanner.Scan(wdt);

        WriteTileDiagnostics(opts.ExportDir, wdt.MapName, result.Placements, wdt.AdtNumbers, wdt.AdtMhdrOffsets);

        var mapDir = Path.Combine(opts.ExportDir, "World", "Maps", wdt.MapName);
        Directory.CreateDirectory(mapDir);

        // Build union of tiles from placements and from WDT offsets (non-zero)
        var placementsByTile = result.Placements
            .GroupBy(p => (p.TileX, p.TileY))
            .ToDictionary(g => g.Key, g => (IEnumerable<PlacementRecord>)g);

        var candidateTiles = new HashSet<(int tx, int ty)>(placementsByTile.Keys);
        for (int adtNum = 0; adtNum < wdt.AdtMhdrOffsets.Count; adtNum++)
        {
            if (wdt.AdtMhdrOffsets[adtNum] > 0)
            {
                int x = adtNum % 64;
                int y = adtNum / 64;
                candidateTiles.Add((x, y));
            }
        }

        foreach (var (x, y) in candidateTiles.OrderBy(t => t.tx).ThenBy(t => t.ty))
        {
            var hasGroup = placementsByTile.TryGetValue((x, y), out var group);
            var g = hasGroup ? group! : Array.Empty<PlacementRecord>();

            int adtNum = (y * 64) + x;
            int offset = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
            if (offset > 0)
            {
                presentTiles.Add((x, y));

                var alpha = new AdtAlpha(wdt.WdtPath, offset, adtNum);

                // Analysis-only: fetch raw alpha ADT MCNK area IDs (no decoding)
                var rawAlphaAreaIds = alpha.GetAlphaMcnkAreaIds();

                var fixedM2 = wdt.MdnmFiles.Select(n => fixup.Resolve(AssetType.MdxOrM2, n)).ToList();
                var fixedWmo = wdt.MonmFiles.Select(n => fixup.Resolve(AssetType.Wmo, n)).ToList();

                var adtLk = alpha.ToAdtLk(fixedM2, fixedWmo);

                var alphaTextures = alpha.GetMtexTextureNames();
                var fixedTextures = alphaTextures
                    .Select(t => fixup.ResolveTexture(t))
                    .ToList();

                var outFile = Path.Combine(mapDir, $"{wdt.MapName}_{x}_{y}.adt");
                adtLk.ToFile(outFile);

                ReplaceMtexOnDisk(outFile, fixedTextures);

                // Write placeholder summary with raw area IDs for per-MCNK CSV (mapping disabled)
                var ctx = new AdtWotlkWriter.WriteContext
                {
                    ExportDir = mapDir,
                    MapName = wdt.MapName,
                    TileX = x,
                    TileY = y,
                    Placements = g,
                    Fixup = fixup,
                    ConvertToMh2o = opts.ConvertToMh2o,
                    AreaMapper = null,
                    AlphaAreaIds = rawAlphaAreaIds
                };
                AdtWotlkWriter.WritePlaceholder(ctx);
            }
            else
            {
                // Still emit placeholder row for tiles without embedded ADT for completeness (no area IDs available)
                var ctx = new AdtWotlkWriter.WriteContext
                {
                    ExportDir = mapDir,
                    MapName = wdt.MapName,
                    TileX = x,
                    TileY = y,
                    Placements = g,
                    Fixup = fixup,
                    ConvertToMh2o = opts.ConvertToMh2o,
                    AreaMapper = null,
                    AlphaAreaIds = null
                };
                AdtWotlkWriter.WritePlaceholder(ctx);
            }
        }

        WriteMinimalWdtLk(Path.Combine(mapDir, $"{wdt.MapName}.wdt"), presentTiles);

        // Alpha DBC CSVs via DBCD (with official definitions)
        if (!string.IsNullOrWhiteSpace(opts.AreaAlphaPath))
        {
            WriteAlphaDbcDump(opts.AreaAlphaPath!, opts.DbdDefinitionsDir, Path.Combine(mapDir, "alpha_AreaTable.dump.csv"), opts.DbdBuild);
            WriteAlphaNormalizedCsv(opts.AreaAlphaPath!, opts.DbdDefinitionsDir, Path.Combine(mapDir, "alpha_AreaTable.normalized.csv"), opts.DbdBuild);
            WriteAlphaAreaHierarchyCsv(opts.AreaAlphaPath!, opts.DbdDefinitionsDir, Path.Combine(mapDir, "alpha_AreaTable.hierarchy.csv"), opts.DbdBuild);
        }
    }

    public static void ExportBatch(Options opts)
    {
        if (string.IsNullOrWhiteSpace(opts.InputRoot)) throw new ArgumentException("InputRoot required", nameof(opts.InputRoot));
        Directory.CreateDirectory(opts.ExportDir);

        var resolver = MultiListfileResolver.FromFiles(opts.LkListfilePath, opts.CommunityListfilePath);
        var fixup = new AssetFixupPolicy(
            resolver,
            opts.FallbackTileset,
            opts.FallbackNonTilesetBlp,
            opts.FallbackWmo,
            opts.FallbackM2,
            opts.AssetFuzzy);

        var wdts = Directory.EnumerateFiles(opts.InputRoot!, "*.wdt", SearchOption.AllDirectories)
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase);

        foreach (var wdtPath in wdts)
        {
            try
            {
                var wdt = new WdtAlphaScanner(wdtPath);
                var adtScanner = new AdtScanner();
                var result = adtScanner.Scan(wdt);

                WriteTileDiagnostics(opts.ExportDir, wdt.MapName, result.Placements, wdt.AdtNumbers, wdt.AdtMhdrOffsets);

                var mapDir = Path.Combine(opts.ExportDir, "World", "Maps", wdt.MapName);
                Directory.CreateDirectory(mapDir);

                var presentTiles = new HashSet<(int tx, int ty)>();

                // Build union of tiles from placements and from WDT offsets (non-zero)
                var placementsByTile = result.Placements
                    .GroupBy(p => (p.TileX, p.TileY))
                    .ToDictionary(g => g.Key, g => (IEnumerable<PlacementRecord>)g);

                var candidateTiles = new HashSet<(int tx, int ty)>(placementsByTile.Keys);
                for (int adtNum = 0; adtNum < wdt.AdtMhdrOffsets.Count; adtNum++)
                {
                    if (wdt.AdtMhdrOffsets[adtNum] > 0)
                    {
                        int x = adtNum % 64;
                        int y = adtNum / 64;
                        candidateTiles.Add((x, y));
                    }
                }

                foreach (var (x, y) in candidateTiles.OrderBy(t => t.tx).ThenBy(t => t.ty))
                {
                    var hasGroup = placementsByTile.TryGetValue((x, y), out var group);
                    var g = hasGroup ? group! : Array.Empty<PlacementRecord>();

                    int adtNum = (y * 64) + x;
                    int offset = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
                    if (offset > 0)
                    {
                        presentTiles.Add((x, y));

                        var alpha = new AdtAlpha(wdt.WdtPath, offset, adtNum);

                        // Analysis-only: fetch raw alpha ADT MCNK area IDs (no decoding)
                        var rawAlphaAreaIds = alpha.GetAlphaMcnkAreaIds();

                        var fixedM2 = wdt.MdnmFiles.Select(n => fixup.Resolve(AssetType.MdxOrM2, n)).ToList();
                        var fixedWmo = wdt.MonmFiles.Select(n => fixup.Resolve(AssetType.Wmo, n)).ToList();

                        var adtLk = alpha.ToAdtLk(fixedM2, fixedWmo);

                        var alphaTextures = alpha.GetMtexTextureNames();
                        var fixedTextures = alphaTextures
                            .Select(t => fixup.ResolveTexture(t))
                            .ToList();

                        var outFile = Path.Combine(mapDir, $"{wdt.MapName}_{x}_{y}.adt");
                        adtLk.ToFile(outFile);

                        ReplaceMtexOnDisk(outFile, fixedTextures);

                        var ctx = new AdtWotlkWriter.WriteContext
                        {
                            ExportDir = mapDir,
                            MapName = wdt.MapName,
                            TileX = x,
                            TileY = y,
                            Placements = g,
                            Fixup = fixup,
                            ConvertToMh2o = opts.ConvertToMh2o,
                            AreaMapper = null,
                            AlphaAreaIds = rawAlphaAreaIds
                        };
                        AdtWotlkWriter.WritePlaceholder(ctx);
                    }
                    else
                    {
                        var ctx = new AdtWotlkWriter.WriteContext
                        {
                            ExportDir = mapDir,
                            MapName = wdt.MapName,
                            TileX = x,
                            TileY = y,
                            Placements = g,
                            Fixup = fixup,
                            ConvertToMh2o = opts.ConvertToMh2o,
                            AreaMapper = null,
                            AlphaAreaIds = null
                        };
                        AdtWotlkWriter.WritePlaceholder(ctx);
                    }
                }

                WriteMinimalWdtLk(Path.Combine(mapDir, $"{wdt.MapName}.wdt"), presentTiles);

                // Alpha DBC CSVs via DBCD (with official definitions)
                if (!string.IsNullOrWhiteSpace(opts.AreaAlphaPath))
                {
                    WriteAlphaDbcDump(opts.AreaAlphaPath!, opts.DbdDefinitionsDir, Path.Combine(mapDir, "alpha_AreaTable.dump.csv"), opts.DbdBuild);
                    WriteAlphaNormalizedCsv(opts.AreaAlphaPath!, opts.DbdDefinitionsDir, Path.Combine(mapDir, "alpha_AreaTable.normalized.csv"), opts.DbdBuild);
                    WriteAlphaAreaHierarchyCsv(opts.AreaAlphaPath!, opts.DbdDefinitionsDir, Path.Combine(mapDir, "alpha_AreaTable.hierarchy.csv"), opts.DbdBuild);
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Export failed for {wdtPath}: {ex.Message}");
            }
        }
    }

    // Minimal LK WDT writer for present tiles
    private static void WriteMinimalWdtLk(string outPath, IReadOnlyCollection<(int tx, int ty)> presentTiles)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);
            using var fs = new FileStream(outPath, FileMode.Create, FileAccess.Write, FileShare.Read);
            void WriteChunk(string fourCC, byte[] data)
            {
                var rev = new string(new[] { fourCC[3], fourCC[2], fourCC[1], fourCC[0] });
                var four = Encoding.ASCII.GetBytes(rev);
                fs.Write(four, 0, 4);
                fs.Write(BitConverter.GetBytes(data.Length), 0, 4);
                fs.Write(data, 0, data.Length);
                if ((data.Length & 1) == 1) fs.WriteByte(0);
            }

            // MVER 0x12
            WriteChunk("MVER", new byte[] { 0x12, 0x00, 0x00, 0x00 });
            // MPHD 32 bytes zero
            WriteChunk("MPHD", new byte[32]);
            // MAIN tile presence flags
            var main = new byte[4096 * 8];
            foreach (var (tx, ty) in presentTiles)
            {
                int index = ty * 64 + tx;
                int byteIndex = index * 8;
                var one = BitConverter.GetBytes(1);
                Buffer.BlockCopy(one, 0, main, byteIndex, 4);
            }
            WriteChunk("MAIN", main);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warn: WriteMinimalWdtLk failed for '{outPath}': {ex.Message}");
        }
    }

    // Replace MTEX string block with updated texture names (keeping size)
    private static void ReplaceMtexOnDisk(string adtPath, IReadOnlyList<string> names)
    {
        try
        {
            using var fs = new FileStream(adtPath, FileMode.Open, FileAccess.ReadWrite, FileShare.None);

            static (string fourCC, int size, long startPos) ReadChunkHeader(FileStream fs)
            {
                Span<byte> hdr = stackalloc byte[8];
                if (fs.Read(hdr) != 8) throw new EndOfStreamException();
                var disk = Encoding.ASCII.GetString(hdr.Slice(0, 4));
                var fourCC = new string(new[] { disk[3], disk[2], disk[1], disk[0] });
                int size = BitConverter.ToInt32(hdr.Slice(4, 4));
                return (fourCC, size, fs.Position - 8);
            }

            fs.Position = 0;
            var (c1, s1, p1) = ReadChunkHeader(fs); fs.Position = p1 + 8 + s1 + ((s1 & 1) == 1 ? 1 : 0);
            var (c2, s2, p2) = ReadChunkHeader(fs); fs.Position = p2 + 8 + s2 + ((s2 & 1) == 1 ? 1 : 0);
            var (c3, s3, p3) = ReadChunkHeader(fs); fs.Position = p3 + 8 + s3 + ((s3 & 1) == 1 ? 1 : 0);

            var (c4, s4, p4) = ReadChunkHeader(fs);
            if (!string.Equals(c4, "MTEX", StringComparison.Ordinal)) return;
            long mtexDataPos = p4 + 8;
            int mtexSize = s4;

            using var ms = new MemoryStream();
            foreach (var n in names)
            {
                var norm = (n ?? string.Empty).Replace('\\', '/');
                var b = Encoding.ASCII.GetBytes(norm);
                ms.Write(b, 0, b.Length);
                ms.WriteByte(0);
            }
            var newData = ms.ToArray();

            if (newData.Length <= mtexSize)
            {
                fs.Position = mtexDataPos;
                fs.Write(newData, 0, newData.Length);
                int remaining = mtexSize - newData.Length;
                if (remaining > 0)
                {
                    Span<byte> zero = stackalloc byte[256];
                    while (remaining > 0)
                    {
                        int chunk = Math.Min(remaining, zero.Length);
                        fs.Write(zero.Slice(0, chunk));
                        remaining -= chunk;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warn: ReplaceMtexOnDisk failed for '{adtPath}': {ex.Message}");
        }
    }

    // Alpha AreaTable DBC CSV via DBCD (with fallback to raw parser)
    private static void WriteAlphaDbcDump(string dbcPath, string? dbdDir, string outCsvPath, string? dbdBuild = null)
    {
        try
        {
#if USE_DBCD
            var dbcDir = Path.GetDirectoryName(dbcPath)!;
            var dbc = new DBCD.DBCD(new FilesystemDBCProvider(dbcDir), new FilesystemDBDProvider(dbdDir ?? Path.Combine(dbcDir, "..")));
            var build = ResolveDbdBuild(dbdBuild, dbcPath, dbcDir);
            var storage = dbc.Load("AreaTable", build: build, locale: DBCD.Locale.EnUS);
            var columns = storage.AvailableColumns;
            Directory.CreateDirectory(Path.GetDirectoryName(outCsvPath)!);
            using var sw = new StreamWriter(outCsvPath, false, Encoding.UTF8);
            sw.WriteLine("id," + string.Join(',', columns));
            var dict = storage.ToDictionary();
            foreach (var kvp in dict.OrderBy(k => k.Key))
            {
                var row = kvp.Value;
                var cells = new List<string> { kvp.Key.ToString() };
                foreach (var col in columns)
                {
                    var val = row[col];
                    string s = FormatDbcdValue(val);
                    cells.Add(s);
                }
                sw.WriteLine(string.Join(',', cells));
            }
            return;
#endif
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warn: WriteAlphaDbcDump DBCD failed for '{dbcPath}': {ex.Message}, falling back to raw parser.");
        }
        // Fallback: raw
        try
        {
            var t = AlphaWdtAnalyzer.Core.Dbc.RawDbcParser.Parse(dbcPath);
            Directory.CreateDirectory(Path.GetDirectoryName(outCsvPath)!);
            using var sw = new StreamWriter(outCsvPath, false, Encoding.UTF8);
            // Determine which columns have any guessed string
            var stringCols = new HashSet<int>();
            for (int i = 0; i < t.Rows.Count; i++)
            {
                var row = t.Rows[i];
                for (int f = 0; f < t.FieldCount; f++)
                {
                    if (!string.IsNullOrWhiteSpace(row.GuessedStrings[f])) stringCols.Add(f);
                }
            }
            var headers = new List<string> { "row" };
            headers.AddRange(Enumerable.Range(0, t.FieldCount).Select(i => $"col{i}"));
            headers.AddRange(stringCols.OrderBy(i => i).Select(i => $"col{i}_str"));
            sw.WriteLine(string.Join(',', headers));
            for (int i = 0; i < t.Rows.Count; i++)
            {
                var row = t.Rows[i];
                var cells = new List<string> { i.ToString() };
                for (int f = 0; f < t.FieldCount; f++) cells.Add(unchecked((int)row.Fields[f]).ToString());
                foreach (var sc in stringCols.OrderBy(i2 => i2))
                {
                    var s = row.GuessedStrings[sc] ?? string.Empty;
                    cells.Add($"\"{EscapeCsv(s)}\"");
                }
                sw.WriteLine(string.Join(',', cells));
            }
        }
        catch (Exception ex2)
        {
            Console.Error.WriteLine($"Warn: WriteAlphaDbcDump raw failed for '{dbcPath}': {ex2.Message}");
        }
    }

    private static void WriteAlphaNormalizedCsv(string alphaDbcPath, string? dbdDir, string outCsvPath, string? dbdBuild = null)
    {
        try
        {
#if USE_DBCD
            var dbcDir = Path.GetDirectoryName(alphaDbcPath)!;
            var dbc = new DBCD.DBCD(new FilesystemDBCProvider(dbcDir), new FilesystemDBDProvider(dbdDir ?? Path.Combine(dbcDir, "..")));
            var build = ResolveDbdBuild(dbdBuild, alphaDbcPath, dbcDir);
            var storage = dbc.Load("AreaTable", build: build, locale: DBCD.Locale.EnUS);
            var columns = storage.AvailableColumns;
            string? parentCol = columns.FirstOrDefault(c => string.Equals(c, "ParentAreaID", StringComparison.OrdinalIgnoreCase))
                                ?? columns.FirstOrDefault(c => c.Contains("Parent", StringComparison.OrdinalIgnoreCase));
            string? nameCol = columns.FirstOrDefault(c => c.EndsWith("_lang", StringComparison.OrdinalIgnoreCase))
                              ?? columns.FirstOrDefault(c => c.IndexOf("Name", StringComparison.OrdinalIgnoreCase) >= 0);

            Directory.CreateDirectory(Path.GetDirectoryName(outCsvPath)!);
            using var sw = new StreamWriter(outCsvPath, false, Encoding.UTF8);
            sw.WriteLine("row_id,area_id,parent_area_id,name,parent_col,name_col");
            var dict = storage.ToDictionary();
            foreach (var kvp in dict.OrderBy(k => k.Key))
            {
                int id = kvp.Key;
                var row = kvp.Value;
                int parent = 0;
                if (!string.IsNullOrEmpty(parentCol))
                {
                    var pv = row[parentCol!];
                    parent = TryToInt(pv);
                }
                string name = string.Empty;
                if (!string.IsNullOrEmpty(nameCol))
                {
                    var nv = row[nameCol!];
                    name = FormatDbcdString(nv);
                }
                sw.WriteLine($"{id},{id},{parent},\"{EscapeCsv(name)}\",{(parentCol ?? string.Empty)},{(nameCol ?? string.Empty)}");
            }
            return;
#endif
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warn: WriteAlphaNormalizedCsv DBCD failed for '{alphaDbcPath}': {ex.Message}, falling back to raw parser.");
        }
        // Fallback to raw
        try
        {
            var t = AlphaWdtAnalyzer.Core.Dbc.RawDbcParser.Parse(alphaDbcPath);
            int areaIdIndex = 1, parentIdIndex = 3, nameIndex = 14;
            Directory.CreateDirectory(Path.GetDirectoryName(outCsvPath)!);
            using var sw = new StreamWriter(outCsvPath, false, Encoding.UTF8);
            sw.WriteLine("row_id,area_id,parent_area_id,name,area_id_col,parent_id_col,name_col");
            for (int i = 0; i < t.Rows.Count; i++)
            {
                var row = t.Rows[i];
                int rowId = unchecked((int)row.Fields[0]);
                int areaId = (areaIdIndex < t.FieldCount) ? unchecked((int)row.Fields[areaIdIndex]) : 0;
                int parentId = (parentIdIndex < t.FieldCount) ? unchecked((int)row.Fields[parentIdIndex]) : 0;
                string name = (nameIndex < t.FieldCount) ? (row.GuessedStrings[nameIndex] ?? string.Empty) : string.Empty;
                sw.WriteLine($"{rowId},{areaId},{parentId},\"{EscapeCsv(name)}\",{areaIdIndex},{parentIdIndex},{nameIndex}");
            }
        }
        catch (Exception ex2)
        {
            Console.Error.WriteLine($"Warn: WriteAlphaNormalizedCsv raw failed for '{alphaDbcPath}': {ex2.Message}");
        }
    }

    private static void WriteAlphaAreaHierarchyCsv(string alphaDbcPath, string? dbdDir, string outCsvPath, string? dbdBuild = null)
    {
        try
        {
#if USE_DBCD
            var dbcDir = Path.GetDirectoryName(alphaDbcPath)!;
            var dbc = new DBCD.DBCD(new FilesystemDBCProvider(dbcDir), new FilesystemDBDProvider(dbdDir ?? Path.Combine(dbcDir, "..")));
            var build = ResolveDbdBuild(dbdBuild, alphaDbcPath, dbcDir);
            var storage = dbc.Load("AreaTable", build: build, locale: DBCD.Locale.EnUS);
            var columns = storage.AvailableColumns;
            string? parentCol = columns.FirstOrDefault(c => string.Equals(c, "ParentAreaID", StringComparison.OrdinalIgnoreCase))
                                ?? columns.FirstOrDefault(c => c.Contains("Parent", StringComparison.OrdinalIgnoreCase));
            string? nameCol = columns.FirstOrDefault(c => c.EndsWith("_lang", StringComparison.OrdinalIgnoreCase))
                              ?? columns.FirstOrDefault(c => c.IndexOf("Name", StringComparison.OrdinalIgnoreCase) >= 0);

            // Build name map by raw area id (ID)
            var dict = storage.ToDictionary();
            var namesById = dict.OrderBy(k => k.Key).ToDictionary(k => k.Key, k =>
            {
                if (string.IsNullOrEmpty(nameCol)) return string.Empty;
                return FormatDbcdString(k.Value[nameCol!]);
            });

            Directory.CreateDirectory(Path.GetDirectoryName(outCsvPath)!);
            using var sw = new StreamWriter(outCsvPath, false, Encoding.UTF8);
            sw.WriteLine("area_id,parent_area_id,area_name,parent_name");
            foreach (var kvp in dict.OrderBy(k => k.Key))
            {
                int id = kvp.Key;
                int parent = 0;
                if (!string.IsNullOrEmpty(parentCol)) parent = TryToInt(kvp.Value[parentCol!]);
                var areaName = namesById.TryGetValue(id, out var an) ? an : string.Empty;
                var parentName = namesById.TryGetValue(parent, out var pn) ? pn : string.Empty;
                sw.WriteLine($"{id},{parent},\"{EscapeCsv(areaName)}\",\"{EscapeCsv(parentName)}\"");
            }
            return;
#endif
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warn: WriteAlphaAreaHierarchyCsv DBCD failed for '{alphaDbcPath}': {ex.Message}, falling back to raw parser.");
        }
        // Fallback
        try
        {
            var t = AlphaWdtAnalyzer.Core.Dbc.RawDbcParser.Parse(alphaDbcPath);
            int areaIdIndex = 1, parentIdIndex = 3, nameIndex = 14;
            var namesByAreaId = new Dictionary<int, string>();
            for (int i = 0; i < t.Rows.Count; i++)
            {
                var row = t.Rows[i];
                int areaId = (areaIdIndex < t.FieldCount) ? unchecked((int)row.Fields[areaIdIndex]) : 0;
                string name = (nameIndex < t.FieldCount) ? (row.GuessedStrings[nameIndex] ?? string.Empty) : string.Empty;
                if (!namesByAreaId.ContainsKey(areaId)) namesByAreaId[areaId] = name;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(outCsvPath)!);
            using var sw = new StreamWriter(outCsvPath, false, Encoding.UTF8);
            sw.WriteLine("area_id,parent_area_id,area_name,parent_name");
            for (int i = 0; i < t.Rows.Count; i++)
            {
                var row = t.Rows[i];
                int areaId = (areaIdIndex < t.FieldCount) ? unchecked((int)row.Fields[areaIdIndex]) : 0;
                int parentId = (parentIdIndex < t.FieldCount) ? unchecked((int)row.Fields[parentIdIndex]) : 0;
                string areaName = (nameIndex < t.FieldCount) ? (row.GuessedStrings[nameIndex] ?? string.Empty) : string.Empty;
                namesByAreaId.TryGetValue(parentId, out var parentName);
                sw.WriteLine($"{areaId},{parentId},\"{EscapeCsv(areaName)}\",\"{EscapeCsv(parentName ?? string.Empty)}\"");
            }
        }
        catch (Exception ex2)
        {
            Console.Error.WriteLine($"Warn: WriteAlphaAreaHierarchyCsv raw failed for '{alphaDbcPath}': {ex2.Message}");
        }
    }

    private static string FormatDbcdValue(object? val)
    {
        if (val is null) return string.Empty;
        if (val is string s) return $"\"{EscapeCsv(s)}\"";
        if (val is Array arr)
        {
            var parts = new List<string>();
            foreach (var o in arr) parts.Add(o?.ToString() ?? string.Empty);
            return string.Join('|', parts);
        }
        return val.ToString() ?? string.Empty;
    }

    private static string FormatDbcdString(object? val)
    {
        if (val is string s) return s;
        if (val is Array arr && arr.Length > 0 && arr.GetValue(0) is string s2) return s2;
        return val?.ToString() ?? string.Empty;
    }

    private static int TryToInt(object? val)
    {
        try
        {
            return Convert.ToInt32(val);
        }
        catch
        {
            return 0;
        }
    }

    // Write a 64x64 tile matrix CSV for diagnostics and emit warnings on suspicious conditions.
    private static void WriteTileDiagnostics(
        string exportDir,
        string mapName,
        IEnumerable<PlacementRecord> placements,
        List<int> adtNumbers,
        List<int> adtOffsets)
    {
        try
        {
            var csvDir = Path.Combine(exportDir, "csv", mapName);
            Directory.CreateDirectory(csvDir);
            var csvPath = Path.Combine(csvDir, "wdt_tile_matrix.csv");
            using var sw = new StreamWriter(csvPath, false, Encoding.UTF8);
            sw.WriteLine("tile_x,tile_y,adt_num,in_main_flags,have_offset,offset,placements_count,wrote_adt");

            // Placements per tile
            var placementCounts = placements
                .GroupBy(p => (p.TileX, p.TileY))
                .ToDictionary(g => g.Key, g => g.Count());

            // MAIN flags present set
            var presentSet = new HashSet<int>(adtNumbers);

            for (int y = 0; y < 64; y++)
            {
                for (int x = 0; x < 64; x++)
                {
                    int adtNum = y * 64 + x;
                    bool inMain = presentSet.Contains(adtNum);
                    int offset = (adtNum >= 0 && adtNum < adtOffsets.Count) ? adtOffsets[adtNum] : 0;
                    bool haveOffset = offset > 0;
                    int pcount = placementCounts.TryGetValue((x, y), out var c) ? c : 0;
                    bool wrote = haveOffset; // current pipeline writes only when offset > 0

                    // Emit warnings for suspicious discrepancies
                    if (inMain && !haveOffset)
                    {
                        Console.Error.WriteLine($"Warn: {mapName} tile ({x},{y}) adt_num={adtNum}: MAIN indicates present but ADT offset is missing/zero.");
                    }
                    if (pcount > 0 && !haveOffset)
                    {
                        Console.Error.WriteLine($"Warn: {mapName} tile ({x},{y}) adt_num={adtNum}: placements={pcount} but no alpha ADT offset; ADT will not be written.");
                    }

                    sw.WriteLine($"{x},{y},{adtNum},{(inMain?1:0)},{(haveOffset?1:0)},{offset},{pcount},{(wrote?1:0)}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warn: WriteTileDiagnostics failed for '{mapName}': {ex.Message}");
        }
    }

    private static string EscapeCsv(string s)
    {
        if (s.IndexOfAny(new[] { '"', ',', '\n', '\r' }) >= 0)
        {
            return s.Replace("\"", "\"\"");
        }
        return s;
    }

#if USE_DBCD
    private static string? ResolveDbdBuild(string? explicitBuild, string dbFilePath, string dbcDir)
    {
        if (!string.IsNullOrWhiteSpace(explicitBuild))
        {
            Console.WriteLine($"DBCD build resolved (explicit): {explicitBuild}");
            return explicitBuild;
        }
        // Try to infer from parent folders like ...\0.5.5\tree\...
        static bool LooksLikeXyz(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return false;
            var dotCount = s.Count(c => c == '.');
            if (dotCount != 2) return false;
            var parts = s.Split('.');
            if (parts.Length != 3) return false;
            return parts.All(p => int.TryParse(p, out _));
        }

        var set = new LinkedList<string>();
        foreach (var p in new[] { dbFilePath, dbcDir })
        {
            try
            {
                var parts = Path.GetFullPath(p).Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                foreach (var part in parts)
                {
                    if (LooksLikeXyz(part))
                    {
                        var v = part;
                        set.AddLast(v + ".0");
                        set.AddLast(v + ".99999");
                    }
                }
            }
            catch { }
        }
        if (set.Count > 0)
        {
            var unique = set.Distinct().ToList();
            Console.WriteLine($"DBCD build candidates inferred: {string.Join(",", unique)}");
            return unique.First();
        }
        Console.Error.WriteLine("Warn: Could not infer DBCD build from paths; DBCD may fail and fall back to raw parser.");
        return null;
    }
#endif
}
