using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using GillijimProject.WowFiles.Alpha;
using GillijimProject.WowFiles.LichKing;

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
            }

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

        WriteMinimalWdtLk(Path.Combine(mapDir, $"{wdt.MapName}.wdt"), presentTiles);
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
                    }

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

                WriteMinimalWdtLk(Path.Combine(mapDir, $"{wdt.MapName}.wdt"), presentTiles);
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
}
