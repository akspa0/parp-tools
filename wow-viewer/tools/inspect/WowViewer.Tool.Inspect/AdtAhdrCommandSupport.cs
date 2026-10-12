using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.Maps.AdtAhdr;

namespace WowViewer.Tool.Inspect;

/// <summary>Spec 237: thin CLI surface over <see cref="AdtAhdrReader"/> / <see cref="AdtAhdrTileSlicer"/>.</summary>
public static class AdtAhdrCommandSupport
{
    public static void Run(string[] args)
    {
        bool isCheck = args.Length > 0 && string.Equals(args[0], "check", StringComparison.OrdinalIgnoreCase);
        bool isObjects = args.Length > 0 && string.Equals(args[0], "objects", StringComparison.OrdinalIgnoreCase);
        bool isRoundTrip = args.Length > 0 && string.Equals(args[0], "roundtrip", StringComparison.OrdinalIgnoreCase);
        bool isExportLk = args.Length > 0 && string.Equals(args[0], "export-lk", StringComparison.OrdinalIgnoreCase);
        if (!isCheck && !isObjects && !isRoundTrip && !isExportLk)
        {
            Console.WriteLine("adt-ahdr commands:");
            Console.WriteLine("  adt-ahdr check --root <folder of AHDR-family files, any names/extensions>");
            Console.WriteLine("  adt-ahdr objects --root <folder> [--list]   resolve ACDO placements and measure them against the terrain");
            Console.WriteLine("  adt-ahdr roundtrip --root <folder>          decode then re-encode every file (experimental writer) and compare bytes");
            Console.WriteLine("  adt-ahdr export-lk --root <folder> [--out <dir>] [--map <name>] [--transpose]");
            Console.WriteLine("                     [--format lk|alpha|lk+alpha]  (default lk)");
            Console.WriteLine("                                              convert every DAT tile to LK v18 ADT + WDT and write a loss manifest");
            Environment.ExitCode = args.Length == 0 ? 0 : 1;
            return;
        }

        int rootIndex = Array.FindIndex(args, a => string.Equals(a, "--root", StringComparison.OrdinalIgnoreCase));
        if (rootIndex < 0 || rootIndex + 1 >= args.Length)
        {
            Console.Error.WriteLine("--root is required.");
            Environment.ExitCode = 1;
            return;
        }

        if (isExportLk)
            RunExportLk(args, args[rootIndex + 1]);
        else if (isRoundTrip)
            RunRoundTrip(args[rootIndex + 1]);
        else if (isObjects)
            RunObjects(args[rootIndex + 1], args.Contains("--list", StringComparer.OrdinalIgnoreCase));
        else
            RunCheck(args[rootIndex + 1]);
    }

    /// <summary>
    /// Resolves every ACDO through <see cref="AdtAhdrTileSlicer.ResolveObjectGridPosition"/> and compares the
    /// object's height with the bilinear terrain height (outer grid) under it. Also reports ADST rows and
    /// whether their uniqueIds match any ACDO.
    /// </summary>
    private static void RunObjects(string root, bool list)
    {
        var tiles = new List<AdtAhdrTile>();
        foreach (string path in Directory.EnumerateFiles(root))
        {
            byte[] data = File.ReadAllBytes(path);
            if (AdtAhdrReader.IsAhdrFamily(data))
                tiles.Add(AdtAhdrReader.Read(data, path));
        }

        var errors = new List<double>();
        var uniqueIds = new HashSet<uint>();
        int objects = 0, m2 = 0, wmo = 0, badIndex = 0, adst = 0;
        var trailing = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (AdtAhdrTile tile in tiles)
        {
            adst += tile.ModelFileReferences.Count;
            foreach (AdtAhdrChunk chunk in tile.Chunks)
            {
                foreach (AdtAhdrObjectDefinition obj in chunk.Objects)
                {
                    objects++;
                    uniqueIds.Add(obj.UniqueId);
                    if ((uint)obj.ModelIndex >= (uint)tile.ModelNames.Count)
                    {
                        badIndex++;
                        continue;
                    }

                    string model = tile.ModelNames[obj.ModelIndex];
                    if (model.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase)) wmo++; else m2++;
                    string key = $"field34={obj.Field34} trailing=[{string.Join(",", obj.TrailingValues)}]";
                    trailing[key] = trailing.GetValueOrDefault(key) + 1;

                    (float column, float row, float height) = AdtAhdrTileSlicer.ResolveObjectGridPosition(tile, chunk, obj);
                    if (column is >= 0 and <= 128 && row is >= 0 and <= 128 && tile.OuterHeights.Length == 129 * 129)
                        errors.Add(Math.Abs(height - SampleOuter(tile, column, row)));

                    if (list)
                    {
                        Console.WriteLine($"  tile {tile.TileX},{tile.TileY} chunk {chunk.IndexX},{chunk.IndexY} uid {obj.UniqueId} {model} " +
                            $"grid ({column:F2},{row:F2}) h {height / 36f:F2} yd rot ({obj.RotationDegrees.X:F1},{obj.RotationDegrees.Y:F1},{obj.RotationDegrees.Z:F1}) scale {obj.Scale:F3}");
                    }
                }
            }
        }

        errors.Sort();
        int adstMatches = tiles.SelectMany(static t => t.ModelFileReferences).Count(r => uniqueIds.Contains(r.UniqueId));
        Console.WriteLine($"files {tiles.Count}; ACDO {objects} ({m2} M2, {wmo} WMO, {badIndex} invalid model index)");
        if (errors.Count > 0)
        {
            Console.WriteLine($"height vs terrain (inches): median {errors[errors.Count / 2]:F2}, p90 {errors[(int)(errors.Count * 0.9)]:F2}, " +
                $"within 0.5 in {errors.Count(static e => e < 0.5)}/{errors.Count}");
        }
        Console.WriteLine($"ADST rows {adst}; uniqueIds matching an ACDO: {adstMatches}");
        foreach ((string key, int count) in trailing)
            Console.WriteLine($"  {count,6}  {key}");
    }

    /// <summary>Decode with <see cref="AdtAhdrReader"/>, re-encode with <see cref="AdtAhdrWriter"/>, compare byte for byte.</summary>
    private static void RunRoundTrip(string root)
    {
        int files = 0, identical = 0;
        foreach (string path in Directory.EnumerateFiles(root))
        {
            byte[] original = File.ReadAllBytes(path);
            if (!AdtAhdrReader.IsAhdrFamily(original))
                continue;

            files++;
            byte[] rewritten = AdtAhdrWriter.Write(AdtAhdrReader.Read(original, path));
            if (rewritten.AsSpan().SequenceEqual(original))
            {
                identical++;
                continue;
            }

            int firstDifference = 0;
            int limit = Math.Min(original.Length, rewritten.Length);
            while (firstDifference < limit && original[firstDifference] == rewritten[firstDifference])
                firstDifference++;
            Console.WriteLine($"  {Path.GetFileName(path)}: differs at byte {firstDifference} (original {original.Length} bytes, rewritten {rewritten.Length})");
        }

        Console.WriteLine($"round trip: {identical}/{files} files byte-identical");
        if (identical != files)
            Environment.ExitCode = 2;
    }

    private static float SampleOuter(AdtAhdrTile tile, float column, float row)
    {
        int c0 = Math.Min((int)column, 127), r0 = Math.Min((int)row, 127);
        float ax = column - c0, ay = row - r0;
        float[] h = tile.OuterHeights;
        return h[r0 * 129 + c0] * (1 - ax) * (1 - ay) + h[r0 * 129 + c0 + 1] * ax * (1 - ay)
            + h[(r0 + 1) * 129 + c0] * (1 - ax) * ay + h[(r0 + 1) * 129 + c0 + 1] * ax * ay;
    }

    private static void RunCheck(string root)
    {
        var tiles = new Dictionary<(int X, int Y), AdtAhdrTile>();
        int files = 0, ahdr = 0, withDiagnostics = 0, duplicates = 0, indexMismatches = 0;
        var versions = new SortedDictionary<uint, int>();
        foreach (string path in Directory.EnumerateFiles(root))
        {
            files++;
            byte[] data = File.ReadAllBytes(path);
            if (!AdtAhdrReader.IsAhdrFamily(data))
                continue;

            ahdr++;
            AdtAhdrTile tile = AdtAhdrReader.Read(data, path);
            versions[tile.Version] = versions.GetValueOrDefault(tile.Version) + 1;
            if (tile.Diagnostics.Count > 0)
            {
                withDiagnostics++;
                Console.WriteLine($"  {Path.GetFileName(path)}: {string.Join("; ", tile.Diagnostics)}");
            }

            for (int i = 0; i < tile.Chunks.Count; i++)
            {
                if (tile.Chunks[i].IndexX != i % 16 || tile.Chunks[i].IndexY != i / 16)
                    indexMismatches++;
            }

            if (tile.TileX is int x && tile.TileY is int y && !tiles.TryAdd((x, y), tile))
                duplicates++;
        }

        // Seam check through the slicer: last outer column of chunk 15 in tile (x, y) vs first outer
        // column of chunk 0 in tile (x + 1, y), and the same for rows along Y.
        int xPairs = 0, xExact = 0, yPairs = 0, yExact = 0;
        foreach (((int x, int y), AdtAhdrTile tile) in tiles)
        {
            if (tiles.TryGetValue((x + 1, y), out AdtAhdrTile? right))
            {
                xPairs++;
                bool exact = true;
                for (int cy = 0; cy < 16 && exact; cy++)
                    exact = OuterColumn(AdtAhdrTileSlicer.SliceHeights(tile, 15, cy), 8).SequenceEqual(OuterColumn(AdtAhdrTileSlicer.SliceHeights(right, 0, cy), 0));
                xExact += exact ? 1 : 0;
            }

            if (tiles.TryGetValue((x, y + 1), out AdtAhdrTile? below))
            {
                yPairs++;
                bool exact = true;
                for (int cx = 0; cx < 16 && exact; cx++)
                    exact = OuterRow(AdtAhdrTileSlicer.SliceHeights(tile, cx, 15), 8).SequenceEqual(OuterRow(AdtAhdrTileSlicer.SliceHeights(below, cx, 0), 0));
                yExact += exact ? 1 : 0;
            }
        }

        Console.WriteLine($"files={files} ahdr={ahdr} versions={string.Join(',', versions.Select(static kv => $"{kv.Key}:{kv.Value}"))} unique-tiles={tiles.Count} duplicate-tiles={duplicates}");
        Console.WriteLine($"files-with-diagnostics={withDiagnostics} acnk-index-mismatches={indexMismatches}");
        Console.WriteLine($"slicer seams: x-neighbours {xExact}/{xPairs} exact, y-neighbours {yExact}/{yPairs} exact");
    }

    /// <summary>
    /// Spec 247 US3: convert every AHDR-family DAT tile in a folder to LK v18 ADT + WDT with a loss manifest.
    /// The walk and the manifest live in <see cref="DatToLkAdtFolderExporter"/> so the viewer's menu item and
    /// this command run exactly the same conversion.
    /// </summary>
    private static void RunExportLk(string[] args, string root)
    {
        string? map = ArgValue(args, "--map");
        string? outDir = ArgValue(args, "--out");
        bool transpose = args.Contains("--transpose", StringComparer.OrdinalIgnoreCase);

        // --format lk | alpha | lk+alpha (default lk)
        string formats = (ArgValue(args, "--format") ?? "lk").ToLowerInvariant();
        var targets = new List<MapConversionTargetFormat>();
        if (formats.Contains("lk"))
            targets.Add(MapConversionTargetFormat.LkAdtV18);
        if (formats.Contains("alpha"))
            targets.Add(MapConversionTargetFormat.AlphaWdt053);
        if (targets.Count == 0)
        {
            Console.Error.WriteLine("--format must name lk, alpha, or lk+alpha.");
            Environment.ExitCode = 1;
            return;
        }

        DatFolderExportResult result = DatToLkAdtFolderExporter.Export(
            root, outDir, map, new DatToLkConversionOptions { TransposeChunks = transpose }, targets);

        foreach (string reason in result.SkipReasons)
            Console.WriteLine($"  skipped {reason}");

        if (result.TilesWritten.Count == 0)
        {
            Console.Error.WriteLine($"No AHDR-family tiles could be placed in {result.SourceRoot}.");
            Environment.ExitCode = 1;
            return;
        }

        DatToLkConversionReport report = result.Report;
        Console.WriteLine($"files={result.FilesSeen} skipped={result.SkipReasons.Count} tiles={result.TilesWritten.Count} "
            + $"versions={result.VersionSummary}");
        Console.WriteLine($"chunks: source={report.ChunksInSource} written={report.ChunksWritten} empty-filled={report.ChunksSynthesizedEmpty}");
        Console.WriteLine($"layers={report.LayersWritten} alpha-maps={report.AlphaMapsWritten} dropped-no-alpha={report.LayersDroppedNoAlpha}");
        Console.WriteLine($"shadows={report.ShadowMapsCarried} area-ids={report.AreaIdsCarried} mccv-chunks={report.VertexColourChunks} "
            + $"objects={report.ObjectsPlaced} (skipped {report.ObjectsSkippedUnnamed})");
        Console.WriteLine($"adst-dropped={report.AdstRowsDropped} negative-uniqueids={report.ObjectsWithNegativeUniqueId}");
        Console.WriteLine($"targets={string.Join(" + ", result.TargetsWritten.Select(MapConversionFormats.GetDisplayName))}");
        if (result.AlphaWdtPath is not null)
            Console.WriteLine($"alpha-wdt={result.AlphaWdtPath}");
        Console.WriteLine($"-> {result.OutputDirectory}");
    }

    private static string? ArgValue(string[] args, string name)
    {
        int i = Array.FindIndex(args, a => string.Equals(a, name, StringComparison.OrdinalIgnoreCase));
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }

    /// <summary>Outer vertex at (outerRow, column) of the 145-entry layout.</summary>
    private static float OuterAt(float[] heights, int outerRow, int column) => heights[outerRow * 17 + column];

    private static IEnumerable<float> OuterColumn(float[] heights, int column) =>
        Enumerable.Range(0, 9).Select(r => OuterAt(heights, r, column));

    private static IEnumerable<float> OuterRow(float[] heights, int outerRow) =>
        Enumerable.Range(0, 9).Select(c => OuterAt(heights, outerRow, c));
}
