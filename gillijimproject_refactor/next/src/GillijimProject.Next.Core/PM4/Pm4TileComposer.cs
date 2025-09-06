using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace GillijimProject.Next.Core.PM4;

public static class Pm4TileComposer
{
    /// <summary>
    /// Exports per-tile OBJs by composing triangles from global MSUR surfaces (which reference global MSVI ranges)
    /// and routing them to tiles using per-tile index ranges from GlobalTileLoader. No per-tile loads performed.
    /// </summary>
    public static void ExportTilesFromObjects(string inputPath, bool invertX, string tilesOutDir)
    {
        Directory.CreateDirectory(tilesOutDir);

        // Resolve first tile path and region directory/prefix
        string firstTilePath = File.Exists(inputPath) ? inputPath : FirstOrThrow(inputPath, "*.pm4");
        string dir = Path.GetDirectoryName(firstTilePath) ?? ".";
        var name = Path.GetFileNameWithoutExtension(firstTilePath);
        var parts = name.Split('_');
        if (parts.Length < 3)
        {
            throw new InvalidOperationException("Input filename must follow <prefix>_XX_YY.pm4 for region composition.");
        }
        string prefix = string.Join("_", parts.Take(parts.Length - 2));

        // Load region via GlobalTileLoader to get per-tile offsets/counts and globally-adjusted surfaces
        var global = ParpToolbox.Services.PM4.Pm4GlobalTileLoader.LoadRegion(dir, $"{prefix}_*.pm4", applyMscnRemap: true);
        var scene = ParpToolbox.Services.PM4.Pm4GlobalTileLoader.ToStandardScene(global);

        // Build tile ranges from scene dictionaries
        var tileRanges = new List<TileRange>();
        foreach (var kv in scene.TileIndexOffsetByTileId)
        {
            int tileId = kv.Key;
            int idxStart = kv.Value;
            int idxCount = scene.TileIndexCountByTileId.TryGetValue(tileId, out var c) ? c : 0;
            if (idxCount <= 0) continue;
            int x = tileId % 64;
            int y = tileId / 64;
            string stem = $"{prefix}_{x:D2}_{y:D2}";
            tileRanges.Add(new TileRange
            {
                TileId = tileId,
                FileStem = stem,
                IndexStart = idxStart,
                IndexCount = idxCount
            });
        }
        tileRanges = tileRanges.OrderBy(tr => tr.TileId / 64).ThenBy(tr => tr.TileId % 64).ToList();

        // Prepare tile aggregators
        var aggregators = tileRanges.ToDictionary(
            tr => tr,
            tr => new TileAggregator(tr.FileStem, invertX)
        );

        // Route triangles from surfaces using global MSVI ranges
        if (scene.Indices is not null && scene.Surfaces is not null)
        {
            foreach (var surface in scene.Surfaces)
            {
                int first = (int)surface.MsviFirstIndex;
                int count = (int)surface.IndexCount;
                if (first < 0 || count <= 0) continue;
                int endExclusive = first + count;
                for (int g = first; g + 2 < endExclusive; g += 3)
                {
                    int a = scene.Indices[g];
                    int b = scene.Indices[g + 1];
                    int c = scene.Indices[g + 2];
                    var tile = FindTileForIndex(tileRanges, g);
                    if (tile is null) continue;
                    aggregators[tile].AddTriangle(scene, (a, b, c));
                }
            }
        }

        // Emit per-tile OBJs
        foreach (var kv in aggregators)
        {
            var tr = kv.Key;
            var ag = kv.Value;
            if (ag.FaceCount == 0) continue;
            var path = Path.Combine(tilesOutDir, tr.FileStem + ".obj");
            ag.WriteObj(path);
        }
    }

    private static TileRange? FindTileForIndex(List<TileRange> ranges, int globalIndex)
    {
        foreach (var tr in ranges)
        {
            if (globalIndex >= tr.IndexStart && globalIndex < tr.IndexStart + tr.IndexCount)
                return tr;
        }
        return null;
    }

    private static string FirstOrThrow(string dir, string pattern)
    {
        var files = Directory.GetFiles(dir, pattern);
        if (files.Length == 0)
            throw new FileNotFoundException($"No files matching '{pattern}' in '{dir}'");
        return files[0];
    }

    private sealed record TileRange
    {
        public int TileId { get; init; }
        public string FileStem { get; init; } = string.Empty;
        public int IndexStart { get; init; }
        public int IndexCount { get; init; }
    }

    private sealed class TileAggregator
    {
        private readonly string _stem;
        private readonly bool _invertX;
        private readonly List<(float X, float Y, float Z)> _verts = new();
        private readonly List<(int A, int B, int C)> _faces = new();
        private readonly Dictionary<int, int> _vmap = new();

        public TileAggregator(string stem, bool invertX)
        {
            _stem = stem; _invertX = invertX;
        }
        public int FaceCount => _faces.Count;

        public void AddTriangle(ParpToolbox.Formats.PM4.Pm4Scene src, (int A, int B, int C) tri)
        {
            int a = MapVertex(src, tri.A);
            int b = MapVertex(src, tri.B);
            int c = MapVertex(src, tri.C);
            if (_invertX)
                _faces.Add((a, c, b));
            else
                _faces.Add((a, b, c));
        }

        private int MapVertex(ParpToolbox.Formats.PM4.Pm4Scene src, int idx)
        {
            if (_vmap.TryGetValue(idx, out var local)) return local;
            var v = src.Vertices[idx];
            float x = _invertX ? -v.X : v.X;
            _verts.Add((x, v.Y, v.Z));
            int li = _verts.Count; // 1-based for OBJ indexing later
            _vmap[idx] = li;
            return li;
        }

        public void WriteObj(string path)
        {
            using var sw = new StreamWriter(path, false);
            sw.NewLine = "\n";
            sw.WriteLine($"# Tile {_stem} (surface-routed)");
            foreach (var v in _verts)
            {
                sw.WriteLine($"v {v.X.ToString("G9", CultureInfo.InvariantCulture)} {v.Y.ToString("G9", CultureInfo.InvariantCulture)} {v.Z.ToString("G9", CultureInfo.InvariantCulture)}");
            }
            foreach (var f in _faces)
            {
                sw.WriteLine($"f {f.A} {f.B} {f.C}");
            }
        }
    }
}
