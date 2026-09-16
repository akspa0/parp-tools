using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps.AdtAhdr;

namespace WowViewer.Tool.Inspect;

/// <summary>Spec 237: thin CLI surface over <see cref="AdtAhdrReader"/> / <see cref="AdtAhdrTileSlicer"/>.</summary>
public static class AdtAhdrCommandSupport
{
    public static void Run(string[] args)
    {
        if (args.Length == 0 || !string.Equals(args[0], "check", StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine("adt-ahdr commands:");
            Console.WriteLine("  adt-ahdr check --root <folder of AHDR-family files, any names/extensions>");
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

        RunCheck(args[rootIndex + 1]);
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

    /// <summary>Outer vertex at (outerRow, column) of the 145-entry layout.</summary>
    private static float OuterAt(float[] heights, int outerRow, int column) => heights[outerRow * 17 + column];

    private static IEnumerable<float> OuterColumn(float[] heights, int column) =>
        Enumerable.Range(0, 9).Select(r => OuterAt(heights, r, column));

    private static IEnumerable<float> OuterRow(float[] heights, int outerRow) =>
        Enumerable.Range(0, 9).Select(c => OuterAt(heights, outerRow, c));
}
